#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/particle_shapes.hpp"

#include <vector>

namespace user {
  using namespace ntt;

  // Max number of sinusoidal modes in the bx2 spectrum.
  // Kept as a fixed-size array so the struct stays trivially copyable
  // and can be captured by value into Kokkos device kernels.
  inline constexpr std::size_t MAX_BX2_MODES { 8 };

  /// InitFields — B + E satisfying ∇·E = ρ_a at t=0.
  ///
  /// Axion always propagates along x₁:  a(t, x₁) = cos(k·x₁ - ω·t)
  ///
  /// B field (2D Minkowski, x₁-x₂ plane):
  ///   Bx1 = B0                          (constant, along axion propagation)
  ///   Bx2 = Σᵢ Aᵢ · sin(kᵢ · x₁)        (in-plane transverse spectrum)
  ///   Bx3 = 0                           (out-of-plane)
  ///
  /// Gauss law: ρ_a = -ε·B·∇a = ε·B0·k·sin(k·x₁)
  ///   (Bx2 does not contribute since ∂_x₂ a = 0)
  /// → Ex1 = -ε·B0·cos(k·x₁),  Ex2 = Ex3 = 0
  ///
  template <Dimension D>
  struct InitFields {
    const real_t B0;
    const real_t epsilon, k;
    // C-style arrays avoid std::array::operator[] device-availability issues.
    real_t           bx2_wavenumbers[MAX_BX2_MODES];
    real_t           bx2_amplitudes[MAX_BX2_MODES];
    std::size_t      nmodes;

    InitFields(real_t Bmag,
               real_t eps,
               real_t k_in,
               const std::vector<real_t>& wavenumbers,
               const std::vector<real_t>& amplitudes)
      : B0 { Bmag }
      , epsilon { eps }
      , k { k_in }
      , bx2_wavenumbers {}
      , bx2_amplitudes {}
      , nmodes { wavenumbers.size() } {
      raise::ErrorIf(wavenumbers.size() != amplitudes.size(),
        "bx2_wavenumbers and bx2_amplitudes must have the same length", HERE);
      raise::ErrorIf(wavenumbers.size() > MAX_BX2_MODES,
        "bx2 spectrum exceeds MAX_BX2_MODES", HERE);
      for (std::size_t i = 0; i < nmodes; ++i) {
        bx2_wavenumbers[i] = wavenumbers[i];
        bx2_amplitudes[i]  = amplitudes[i];
      }
    }

    // B field
    Inline auto bx1(const coord_t<D>&) const -> real_t { return B0; }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      real_t b { ZERO };
      for (std::size_t i = 0; i < nmodes; ++i) {
        b += bx2_amplitudes[i] * math::sin(bx2_wavenumbers[i] * x_Ph[0]);
      }
      return b;
    }

    // E field — axion-driven, matching ext_current normalization
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      if (cmp::AlmostZero(k)) {
        return ZERO;
      }
      return -epsilon * B0 * math::cos(k * x_Ph[0]);
    }
  };

  /// Axion external current: J_a = ε · ∂_t a · B  (read B from the actual
  /// evolved EM field at (i1, i2)). The ∇a × E term is intentionally not
  /// included — same approximation as the 1D predecessor.
  ///
  /// Expressions are inlined (no phase/adot temporaries) to spare GPU
  /// registers / cache.
  ///
  template <Dimension D>
  struct AxionExternalCurrent {
    const real_t epsilon, omega, k;
    const real_t coef; // skindepth0^2 / larmor0 — compensates Ampere normalization

    AxionExternalCurrent(real_t eps, real_t omega_in, real_t k_in, real_t c)
      : epsilon { eps }
      , omega { omega_in }
      , k { k_in }
      , coef { c } {}

    template <class Context>
    Inline auto jx1(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      return coef * epsilon * omega
           * math::sin(k * ctx.x_Ph[0] - omega * ctx.time)
           * ctx.em(ctx.i1, ctx.i2, em::bx1);
    }

    template <class Context>
    Inline auto jx2(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      return coef * epsilon * omega
           * math::sin(k * ctx.x_Ph[0] - omega * ctx.time)
           * ctx.em(ctx.i1, ctx.i2, em::bx2);
    }

    template <class Context>
    Inline auto jx3(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      return coef * epsilon * omega
           * math::sin(k * ctx.x_Ph[0] - omega * ctx.time)
           * ctx.em(ctx.i1, ctx.i2, em::bx3);
    }
  };

  template <Dimension D>
  struct UnitWeightDistribution {
    Inline auto operator()(const coord_t<D>&) const -> Kokkos::pair<real_t, real_t> {
      return { ONE, ONE };
    }
  };

  /// PGen — 2D Axion-PIC with transverse bx2 spectrum.
  ///
  /// Initial E satisfies the modified Gauss law. Evolution uses ext_current
  /// so the axion current is added through the same Ampere path as plasma
  /// current. Dynamic pair injection is disabled in this version; initial
  /// plasma can still be loaded by setting setup.density > 0.
  ///
  template <SimEngine::type S, class M>
  struct PGen {

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D> {}
    };

    static constexpr auto D { M::Dim };

    const SimulationParams& params;
    const Metadomain<S, M>& metadomain;

    const real_t B0, omega, k, epsilon, dt;
    const real_t temperature, density, ppc0;

    AxionExternalCurrent<D> ext_current;
    InitFields<D>           init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , B0 { p.template get<real_t>("setup.B0", ONE) }
      , omega { p.template get<real_t>("setup.omega_ratio")
          / p.template get<real_t>("scales.skindepth0", ONE) }
      , k { p.template get<real_t>("setup.k", ZERO) }
      , epsilon { p.template get<real_t>("setup.epsilon") }
      , dt { p.template get<real_t>("algorithms.timestep.dt") }
      , temperature { p.template get<real_t>("setup.temperature", ZERO) }
      , density { p.template get<real_t>("setup.density", ZERO) }
      , ppc0 { p.template get<real_t>("particles.ppc0") }
      , ext_current { epsilon, omega, k,
          SQR(p.template get<real_t>("scales.skindepth0", ONE))
          / p.template get<real_t>("scales.larmor0", ONE) }
      , init_flds {
          B0,
          epsilon,
          k,
          p.template get<std::vector<real_t>>("setup.bx2_wavenumbers",
                                              std::vector<real_t> {}),
          p.template get<std::vector<real_t>>("setup.bx2_amplitudes",
                                              std::vector<real_t> {}) } {
      raise::ErrorIf(omega * dt >= ONE,
        "omega*dt >= 1: cannot resolve axion oscillation", HERE);
      raise::ErrorIf(
        not p.template get<bool>("particles.use_weights", false),
        "axion pgen requires particles.use_weights = true", HERE);
    }

    void InitPrtls(Domain<S, M>& domain) {
      if (density <= ZERO) {
        return;
      }
      const auto maxw = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(), temperature);
      const auto uniform = UnitWeightDistribution<D> {};
      arch::InjectNonUniform<S, M, decltype(maxw), decltype(maxw), decltype(uniform)>(
        params, domain, { 1, 2 }, { maxw, maxw }, uniform, density, true);
    }

  };

} // namespace user

#endif
