#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "archetypes/utils.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;
  using prmvec_t = std::vector<real_t>;

  /// InitFields — B + E satisfying ∇·E = ρ_a at t=0.
  ///
  /// Axion always propagates along x₁:  a(t, x₁) = cos(k·x₁ - ω·t)
  ///
  /// B in x-z plane at angle θ from x₁:
  ///   B = B₀ (cosθ, 0, sinθ),   θ=0 → B ∥ x₁
  template <Dimension D>
  struct InitFields {
    const real_t B0;
    const real_t b_para;   // B₀ cosθ — along x₁
    const real_t b_perp;   // B₀ sinθ — along x₃
    const real_t epsilon, k;

    InitFields(real_t Bmag, real_t theta_deg,
               real_t eps, real_t k_in)
      : B0 { Bmag }
      , b_para { Bmag * math::cos(theta_deg * static_cast<real_t>(convert::deg2rad)) }
      , b_perp { Bmag * math::sin(theta_deg * static_cast<real_t>(convert::deg2rad)) }
      , epsilon { eps }
      , k { k_in } {}

    // B field — B = B₀ (cosθ, 0, sinθ) in xz plane
    Inline auto bx1(const coord_t<D>&) const -> real_t { return b_para; }
    Inline auto bx3(const coord_t<D>&) const -> real_t { return b_perp; }

    // E field — satisfy ∂ₓ Eₓ = ρ_a at t=0
    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      if (cmp::AlmostZero(k)) { return ZERO; }
      return -epsilon * b_para * math::cos(k * x_Ph[0]);
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
      const auto phase  = k * ctx.x_Ph[0] - omega * ctx.time;
      const auto sin_ph = math::sin(phase);
      const auto adot   = epsilon * omega * sin_ph;
      const auto grad_a = -epsilon * k * sin_ph;
      return adot * ctx.em(ctx.i1, em::bx2) - grad_a * ctx.em(ctx.i1, em::ex3);
    }

    template <class Context>
    Inline auto jx3(const Context& ctx) const
      -> decltype(ctx.em, static_cast<real_t>(ZERO)) {
      const auto phase  = k * ctx.x_Ph[0] - omega * ctx.time;
      const auto sin_ph = math::sin(phase);
      const auto adot   = epsilon * omega * sin_ph;
      const auto grad_a = -epsilon * k * sin_ph;
      return adot * ctx.em(ctx.i1, em::bx3) + grad_a * ctx.em(ctx.i1, em::ex2);
    }
  };

  /// PGen — Stage 2 Axion-PIC (plasma + axion background)
  ///
  /// Initial E satisfies the modified Gauss law. Evolution uses ext_current
  /// so the axion current is added through the same Ampere path as plasma
  /// current.
  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {

    static constexpr auto engines {
      traits::compatible_with<SimEngine::SRPIC>::value
    };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D>::value
    };

    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t B0, omega, k, epsilon, dt, theta;
    const real_t temperature, density;

    AxionExternalCurrent<D> ext_current;
    InitFields<D>           init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , B0 { p.template get<real_t>("setup.B0", ONE) }
      , omega { p.template get<real_t>("setup.omega") }
      , k { p.template get<real_t>("setup.k", ZERO) }
      , epsilon { p.template get<real_t>("setup.epsilon") }
      , dt { p.template get<real_t>("algorithms.timestep.dt") }
      , theta { p.template get<real_t>("setup.theta", ZERO) }
      , temperature { p.template get<real_t>("setup.temperature", ZERO) }
      , density { p.template get<real_t>("setup.density", ONE) }
      , ext_current { epsilon, omega, k }
      , init_flds { B0, theta, epsilon, k } {
      (void)global_domain;
      raise::ErrorIf(omega * dt >= ONE,
        "omega*dt >= 1: cannot resolve axion oscillation", HERE);
    }

    void InitPrtls(Domain<S, M>& domain) {
      if (density <= ZERO) {
        return;
      }
      arch::InjectUniformMaxwellians<S, M>(
        params, domain, density,
        { temperature, temperature },
        { 1, 2 });
    }
  };

} // namespace user

#endif
