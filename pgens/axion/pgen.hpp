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

#include <string>

namespace user {
  using namespace ntt;

  /// InitFields — B + E satisfying ∇·E = ρ_a at t=0.
  ///
  /// Axion always propagates along x₁:  a(t, x₁) = cos(k·x₁ - ω·t)
  ///
  /// B field (2D Minkowski, x₁-x₂ plane), uniform:
  ///   Bx1 = B0        (along axion propagation)
  ///   Bx2 = B0_perp   (in-plane transverse, constant)
  ///   Bx3 = 0         (out-of-plane)
  ///
  /// Gauss law: ρ_a = -ε·B·∇a = ε·B0·k·sin(k·x₁)
  ///   (Bx2 does not contribute since ∂_x₂ a = 0)
  /// → Ex1 = -ε·B0·cos(k·x₁),  Ex2 = 0
  ///
  /// Ex3 = E0_x3: optional uniform seed (default 0), used by the vacuum
  /// ∇a×E forced-wave test.
  ///
  template <Dimension D>
  struct InitFields {
    const real_t B0;
    const real_t B0_perp;
    const real_t epsilon, k;
    const real_t E0_x3;

    InitFields(real_t Bmag, real_t Bperp, real_t eps, real_t k_in, real_t e3)
      : B0 { Bmag }
      , B0_perp { Bperp }
      , epsilon { eps }
      , k { k_in }
      , E0_x3 { e3 } {}

    // B field
    Inline auto bx1(const coord_t<D>&) const -> real_t { return B0; }
    Inline auto bx2(const coord_t<D>&) const -> real_t { return B0_perp; }

    // E field — axion-driven, matching ext_current normalization
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      if (cmp::AlmostZero(k)) {
        return ZERO;
      }
      return -epsilon * B0 * math::cos(k * x_Ph[0]);
    }

    // uniform Ex3 seed (vacuum ∇a×E test)
    Inline auto ex3(const coord_t<D>&) const -> real_t { return E0_x3; }
  };

  /// Axion external current: J_a = ε · (∂_t a · B + ∇a × E), reading B and E
  /// from the actual evolved EM field at (i1, i2).
  ///
  /// With φ = k·x₁ - ω·t:  ∂_t a = +ω·sinφ,  ∂_x₁ a = -k·sinφ
  ///   (∇a×E)₂ = -∂₁a·E₃ = +k·sinφ·E₃
  ///   (∇a×E)₃ = +∂₁a·E₂ = -k·sinφ·E₂
  ///
  /// 2D Minkowski basis: stored[ex1, ex2, bx1, bx2] = physical/dx,
  /// stored[ex3, bx3] = physical. The ȧB term keeps matching indices (basis
  /// factors cancel); ∇a×E crosses indices, so jx2 divides by ctx.dx and
  /// jx3 multiplies by ctx.dx (2D-specific; rederive before any 3D use).
  ///
  /// `grad_ae = false` disables ∇a×E for ON/OFF discrimination experiments.
  ///
  template <Dimension D>
  struct AxionExternalCurrent {
    const real_t epsilon, omega, k;
    const real_t coef; // skindepth0^2 / larmor0 — compensates Ampere normalization
    const bool   grad_ae;

    AxionExternalCurrent(real_t eps, real_t omega_in, real_t k_in, real_t c, bool g)
      : epsilon { eps }
      , omega { omega_in }
      , k { k_in }
      , coef { c }
      , grad_ae { g } {}

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
      const auto phase = k * ctx.x_Ph[0] - omega * ctx.time;
      auto       j     = coef * epsilon * omega * math::sin(phase)
               * ctx.em(ctx.i1, ctx.i2, em::bx2);
      if (grad_ae) {
        j += coef * epsilon * k * math::sin(phase)
             * ctx.em(ctx.i1, ctx.i2, em::ex3) / ctx.dx;
      }
      return j;
    }

    template <class Context>
    Inline auto jx3(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      const auto phase = k * ctx.x_Ph[0] - omega * ctx.time;
      auto       j     = coef * epsilon * omega * math::sin(phase)
               * ctx.em(ctx.i1, ctx.i2, em::bx3);
      if (grad_ae) {
        j -= coef * epsilon * k * math::sin(phase)
             * ctx.em(ctx.i1, ctx.i2, em::ex2) * ctx.dx;
      }
      return j;
    }
  };

  template <Dimension D>
  struct UnitWeightDistribution {
    Inline auto operator()(const coord_t<D>&) const -> Kokkos::pair<real_t, real_t> {
      return { ONE, ONE };
    }
  };

  /// Number-density profile along x₁ for InitPrtls.
  ///
  /// Returns { n(x₁)/n_max, ONE }: injection probability follows the profile
  /// while the particle weight stays constant, so N_D = ppc·(λ_D/dx)² is
  /// uniform along the gradient (vary ppc, not weights).
  ///
  /// Profiles (n in units of setup.density):
  ///   uniform : n = n_max
  ///   ramp    : linear n_min → n_max between ramp_xa and ramp_xb, clamped
  ///   barrier : n_max plateau inside [ramp_xa, ramp_xb], n_min outside,
  ///             tanh shoulders of width ~ 1/ramp_W
  ///
  template <Dimension D>
  struct RampDistribution {
    const int    profile; // 0 = uniform, 1 = ramp, 2 = barrier
    const real_t n_min, n_max, W, xa, xb;

    Inline auto operator()(const coord_t<D>& x) const -> Kokkos::pair<real_t, real_t> {
      real_t n { n_max };
      if (profile == 1) {
        auto s = (x[0] - xa) / (xb - xa);
        s      = s < ZERO ? ZERO : (s > ONE ? ONE : s);
        n      = n_min + (n_max - n_min) * s;
      } else if (profile == 2) {
        const auto left  = HALF * (ONE + math::tanh(W * (x[0] - xa)));
        const auto right = HALF * (ONE + math::tanh(W * (xb - x[0])));
        n = n_min + (n_max - n_min) * left * right;
      }
      const real_t w { ONE }; // local copy: ONE must not be odr-used in device code
      return { n / n_max, w };
    }
  };

  inline auto ParseDensityProfile(const std::string& name) -> int {
    if (name == "uniform") {
      return 0;
    }
    if (name == "ramp") {
      return 1;
    }
    if (name == "barrier") {
      return 2;
    }
    return -1;
  }

  /// PGen — 2D Axion-PIC with uniform transverse B field.
  ///
  /// Initial E satisfies the modified Gauss law. Evolution uses ext_current
  /// with the full J_a = ε(ȧB + ∇a×E) (the ∇a×E term can be switched off
  /// via setup.use_grad_a_cross_e). Dynamic pair injection is disabled in
  /// this version; initial plasma is loaded when setup.density > 0, with an
  /// optional x1 density profile (setup.profile = uniform/ramp/barrier).
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
    const int    profile;
    const real_t n_min, n_max, ramp_W, ramp_xa, ramp_xb;

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
      , profile { ParseDensityProfile(
          p.template get<std::string>("setup.profile", "uniform")) }
      , n_min { p.template get<real_t>("setup.n_min", ONE) }
      , n_max { p.template get<real_t>("setup.n_max", ONE) }
      , ramp_W { p.template get<real_t>("setup.ramp_W", ONE) }
      , ramp_xa { p.template get<real_t>("setup.ramp_xa", ZERO) }
      , ramp_xb { p.template get<real_t>("setup.ramp_xb", -ONE) }
      , ext_current { epsilon, omega, k,
          SQR(p.template get<real_t>("scales.skindepth0", ONE))
            / p.template get<real_t>("scales.larmor0", ONE),
          p.template get<bool>("setup.use_grad_a_cross_e", true) }
      , init_flds {
          B0,
          p.template get<real_t>("setup.B0_perp", ZERO),
          epsilon,
          k,
          p.template get<real_t>("setup.E0_x3", ZERO) } {
      raise::ErrorIf(omega * dt >= ONE,
        "omega*dt >= 1: cannot resolve axion oscillation", HERE);
      raise::ErrorIf(
        not p.template get<bool>("particles.use_weights", false),
        "axion pgen requires particles.use_weights = true", HERE);
      raise::ErrorIf(profile < 0,
        "setup.profile must be one of: uniform, ramp, barrier", HERE);
      raise::ErrorIf(profile != 0 and n_max <= ZERO,
        "setup.n_max must be > 0 for density profiles", HERE);
      raise::ErrorIf(profile != 0 and ramp_xb <= ramp_xa,
        "setup.ramp_xb must be > setup.ramp_xa for density profiles", HERE);
    }

    void InitPrtls(Domain<S, M>& domain) {
      if (density <= ZERO) {
        return;
      }
      const auto maxw = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(), temperature);
      if (profile == 0) {
        // uniform: code path identical to the pre-profile version
        const auto uniform = UnitWeightDistribution<D> {};
        arch::InjectNonUniform<S, M, decltype(maxw), decltype(maxw), decltype(uniform)>(
          params, domain, { 1, 2 }, { maxw, maxw }, uniform, density, true);
      } else {
        // profile: injection probability follows n(x1)/n_max, weight stays 1
        const auto sdist = RampDistribution<D> { profile, n_min, n_max,
                                                 ramp_W,  ramp_xa, ramp_xb };
        arch::InjectNonUniform<S, M, decltype(maxw), decltype(maxw), decltype(sdist)>(
          params, domain, { 1, 2 }, { maxw, maxw }, sdist, density * n_max, true);
      }
    }

  };

} // namespace user

#endif
