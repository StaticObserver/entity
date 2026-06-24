#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/field_setter.h"
#include "archetypes/utils.h"

#include "framework/domain/metadomain.h"

#include <string>

namespace user {
  using namespace ntt;

  enum class FieldGeometry {
    dipole,
    monopole
  };

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar, const std::string& field_geometry)
      : Bsurf { bsurf }
      , Rstar { rstar }
      , field_geom { field_geometry == "monopole" ? FieldGeometry::monopole
                                                  : FieldGeometry::dipole } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return Bsurf / SQR(x_Ph[0] / Rstar);
      } else {
        return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return ZERO;
      } else {
        return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

  private:
    const real_t        Bsurf, Rstar;
    const FieldGeometry field_geom;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t bsurf, real_t rstar, real_t omega,
                real_t dOmega, real_t time, real_t t_pert_start,
                real_t t_stop, int num_cycles,
                real_t theta1, real_t theta2, const std::string& field_geometry)
      : InitFields<D> { bsurf, rstar, field_geometry }
      , Omega { omega }
      , dOmega { dOmega }
      , time { time }
      , t_pert_start { t_pert_start }
      , t_stop { t_stop }
      , num_cycles { num_cycles }
      , theta1 { theta1 }
      , theta2 { theta2 } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto get_dOmega(const coord_t<D>& x_Ph) const -> real_t {
      if (cmp::AlmostZero(dOmega) or time < t_pert_start
          or time > t_pert_start + t_stop) {
        return ZERO;
      }
      const auto theta = x_Ph[1];
      if (theta > constant::PI / TWO) {
        return ZERO;
      }
      const auto u = (TWO * theta - (theta1 + theta2)) / (theta1 - theta2);
      return dOmega * math::exp(-SQR(THREE) / TWO * SQR(u)) *
             math::sin(TWO * constant::PI * time *
                       static_cast<real_t>(num_cycles) / t_stop);
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      const auto dOm = get_dOmega(x_Ph);
      return (Omega + dOm) * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      const auto dOm = get_dOmega(x_Ph);
      return -(Omega + dOm) * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Omega, dOmega, time, t_pert_start, t_stop, theta1, theta2;
    const int    num_cycles;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Spherical, Metric::QSpherical> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D> {}
    };

    static constexpr auto D { M::Dim };

    const SimulationParams& params;

    const real_t      Bsurf, Rstar, Omega, Temperature, N0;
    const real_t      dOmega, t_pert_start, t_stop, theta1, theta2;
    const int         num_cycles;
    const std::string field_geom;
    InitFields<D>     init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { p.template get<real_t>("setup.Omega") }
      , Temperature { p.template get<real_t>("setup.temperature") }
      , N0 { p.template get<real_t>("setup.N0") }
      , dOmega { p.template get<real_t>("setup.dOmega", ZERO) }
      , t_pert_start { p.template get<real_t>("setup.t_pert_start", 50.0) }
      , t_stop { p.template get<real_t>("setup.t_stop", 10.0) }
      , theta1 { p.template get<real_t>("setup.theta1", 0.3) *
                 static_cast<real_t>(constant::PI / 180.0) }
      , theta2 { p.template get<real_t>("setup.theta2", 1.2) *
                 static_cast<real_t>(constant::PI / 180.0) }
      , num_cycles { p.template get<int>("setup.num_cycles", 20) }
      , field_geom { p.template get<std::string>("setup.field_geometry", "dipole") }
      , init_flds { Bsurf, Rstar, field_geom } {}

    struct BDist {
      real_t Rstar;
      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        return math::pow(Rstar / x_Ph[0], 3);
      }
    };

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::energy_dist::Maxwellian<D, M::CoordType>(
        local_domain.random_pool(), Temperature);
      const auto sdist = BDist { Rstar };
      const auto use_weights = params.template get<bool>("particles.use_weights");
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist),
                             BDist>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        sdist,
        N0,
        use_weights);
    }

    struct BTargetDensity {
      real_t floor0, Rstar, r_inj_min, r_inj_max;
      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < r_inj_min or x_Ph[0] > r_inj_max) {
          return ZERO;
        }
        return floor0 * math::pow(Rstar / x_Ph[0], 3);
      }
    };

    inline void CustomPostStep(timestep_t, simtime_t, Domain<S, M>& domain) {
      const auto floor0  = params.template get<real_t>("setup.floor0", 0.1);
      const auto r_inj_max = params.template get<real_t>("setup.r_inj_max", 8.0);
      const auto r_inj_min = Rstar + params.template get<real_t>(
        "grid.boundaries.atmosphere.ds", 0.36);
      const auto energy_dist = arch::energy_dist::Maxwellian<D, M::CoordType>(
        domain.random_pool(), Temperature);
      const auto use_weights = params.template get<bool>("particles.use_weights");
      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(
        params, domain, { 1, 2 }, domain.fields.buff, {}, 0);
      const auto sdist = arch::spatial_dist::Replenish<M, 3, BTargetDensity>(
        domain.mesh.metric, domain.fields.buff, 0,
        BTargetDensity { floor0, Rstar, r_inj_min, r_inj_max }, floor0);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist),
                             decltype(sdist)>(
        params, domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        sdist,
        ONE,
        use_weights);
    }

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { Bsurf, Rstar, Omega, dOmega, time, t_pert_start,
                              t_stop, num_cycles, theta1, theta2, field_geom };
    }

    auto MatchFields(real_t) const -> InitFields<D> {
      return InitFields<D> { Bsurf, Rstar, field_geom };
    }
  };

} // namespace user

#endif
