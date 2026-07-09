#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace user {
  using namespace ntt;

  /// All fields zero.
  template <Dimension D>
  struct InitFields {
    Inline auto bx1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto bx3(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto dx1(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto dx2(const coord_t<D>&) const -> real_t { return ZERO; }
    Inline auto dx3(const coord_t<D>&) const -> real_t { return ZERO; }
  };

  /// Spatially-varying drift: u_x ∝ x, u_y ∝ y.
  template <Dimension D>
  struct LinearDrift2D {
    const real_t sx, sy;
    LinearDrift2D(real_t a, real_t b) : sx { a }, sy { b } {}
    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& d) const {
      d[0] = sx * x_Ph[0];
      d[1] = sy * x_Ph[1];
      d[2] = ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines  { ::traits::pgen::compatible_with<SimEngine::SRPIC> {} };
    static constexpr auto metrics  { ::traits::pgen::compatible_with<Metric::Minkowski> {} };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

    const SimulationParams& params;
    const real_t temperature, density, drift_ux, drift_uy, drift_uz;
    const bool   use_spatial;
    InitFields<D> init_flds;

    PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : params { p }
      , temperature { params.template get<real_t>("setup.temperature", static_cast<real_t>(0.01)) }
      , density     { params.template get<real_t>("setup.density", ONE) }
      , drift_ux    { params.template get<real_t>("setup.drift_ux", ZERO) }
      , drift_uy    { params.template get<real_t>("setup.drift_uy", ZERO) }
      , drift_uz    { params.template get<real_t>("setup.drift_uz", ZERO) }
      , use_spatial { params.template get<bool>("setup.use_spatial", false) }
      , init_flds {} {}

    void InitPrtls(Domain<S, M>& domain) {
      if (use_spatial) {
        LinearDrift2D<M::Dim> lindrift { drift_ux, drift_uy };
        auto ed1 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType, LinearDrift2D<M::Dim>>(
          domain.random_pool(), temperature, lindrift);
        auto ed2 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType, LinearDrift2D<M::Dim>>(
          domain.random_pool(), temperature, lindrift);
        arch::InjectUniform<S, M, decltype(ed1), decltype(ed2)>(
          params, domain, { 1, 2 }, { ed1, ed2 }, density, false);
      } else {
        const std::vector<real_t> dv { drift_ux, drift_uy, drift_uz };
        auto ed1 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
          domain.random_pool(), temperature, dv);
        auto ed2 = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
          domain.random_pool(), temperature, dv);
        arch::InjectUniform<S, M, decltype(ed1), decltype(ed2)>(
          params, domain, { 1, 2 }, { ed1, ed2 }, density, false);
      }
    }
  };
} // namespace user
#endif
