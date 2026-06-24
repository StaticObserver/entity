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

namespace user {
  using namespace ntt;

  /// InitFields — B + E satisfying ∇·E = ρ_a at t=0.
  ///
  /// Axion always propagates along x₁:  a(t, x₁) = cos(k·x₁ - ω·t)
  ///
  /// B in x-z plane at angle θ from x₁:
  ///   B = B₀ (cosθ, 0, sinθ),   θ=0 → B ∥ x₁
  ///
  template <Dimension D>
  struct InitFields {
    const real_t B0;
    const real_t b_para;   // B₀ cosθ — along x₁
    const real_t b_perp;   // B₀ sinθ — along x₃
    const real_t epsilon, k;
    const real_t larmor0;

    InitFields(real_t Bmag, real_t theta_deg,
               real_t eps, real_t k_in, real_t larmor)
      : B0 { Bmag }
      , b_para { Bmag * math::cos(theta_deg * static_cast<real_t>(convert::deg2rad)) }
      , b_perp { Bmag * math::sin(theta_deg * static_cast<real_t>(convert::deg2rad)) }
      , epsilon { eps }
      , k { k_in }
      , larmor0 { larmor } {}

    // B field — B = B₀ (cosθ, 0, sinθ) in xz plane
    Inline auto bx1(const coord_t<D>&) const -> real_t { return b_para; }
    Inline auto bx3(const coord_t<D>&) const -> real_t { return b_perp; }

    // E field — axion-driven, divided by larmor0
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      if (cmp::AlmostZero(k)) { return ZERO; }
      return -epsilon * b_para * math::cos(k * x_Ph[0]) / larmor0;
    }
  };

  template <Dimension D>
  struct AxionExternalCurrent {
    const real_t epsilon, omega, k;

    AxionExternalCurrent(real_t eps, real_t omega_in, real_t k_in)
      : epsilon { eps }
      , omega { omega_in }
      , k { k_in } {}

    template <class Context>
    Inline auto jx1(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      const auto phase = k * ctx.x_Ph[0] - omega * ctx.time;
      const auto adot  = epsilon * omega * math::sin(phase);
      return adot * ctx.em(ctx.i1, em::bx1);
    }

    template <class Context>
    Inline auto jx2(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      const auto phase = k * ctx.x_Ph[0] - omega * ctx.time;
      const auto adot  = epsilon * omega * math::sin(phase);
      return adot * ctx.em(ctx.i1, em::bx2);
    }

    template <class Context>
    Inline auto jx3(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      const auto phase = k * ctx.x_Ph[0] - omega * ctx.time;
      const auto adot  = epsilon * omega * math::sin(phase);
      return adot * ctx.em(ctx.i1, em::bx3);
    }
  };

  /// PGen — Stage 2 Axion-PIC (plasma + axion background)
  ///
  /// Initial E satisfies the modified Gauss law. Evolution uses ext_current
  /// so the axion current is added through the same Ampere path as plasma
  /// current.
  template <SimEngine::type S, class M>
  struct PGen {

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D> {}
    };

    static constexpr auto D { M::Dim };

    const SimulationParams& params;
    const Metadomain<S, M>& metadomain;

    const real_t B0, omega, k, epsilon, dt, theta;
    const real_t temperature, density;

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
      , theta { p.template get<real_t>("setup.theta", ZERO) }
      , temperature { p.template get<real_t>("setup.temperature", ZERO) }
      , density { p.template get<real_t>("setup.density", ONE) }
      , ext_current { epsilon, omega, k }
      , init_flds { B0, theta, epsilon, k,
          p.template get<real_t>("scales.larmor0", ONE) } {
      raise::ErrorIf(omega * dt >= ONE,
        "omega*dt >= 1: cannot resolve axion oscillation", HERE);
    }

    void InitPrtls(Domain<S, M>& domain) {
      if (density <= ZERO) {
        return;
      }
      const auto maxw = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(), temperature);
      arch::InjectUniform<S, M, decltype(maxw), decltype(maxw)>(
        params, domain, { 1, 2 }, { maxw, maxw }, density);
    }

  };

} // namespace user

#endif
