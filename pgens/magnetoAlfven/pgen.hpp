#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <string>

namespace user {
  using namespace ntt;

  enum class FieldGeometry {
    dipole,
    monopole
  };

  template <SimEngine::type S, class M>
  struct DensityDistribution : public arch::SpatialDistribution<S, M> {
    DensityDistribution(const M& metric, real_t n0)  
      : arch::SpatialDistribution<S, M> { metric }
      , N0 { n0 } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      // return N0 / CUBE(x_Ph[0]);
      return N0;
    }

    real_t N0;
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
    DriveFields(const real_t             bsurf,
                const real_t             rstar,
                const real_t             omega,
                const real_t             omega_0,
                const real_t             time,
                const real_t             t_stop,
                const real_t             theta1,
                const real_t             theta2,
                const int                num_cycles,
                const std::string& field_geometry)
      : InitFields<D> { bsurf, rstar, field_geometry }
      , Omega { omega } 
      , Omega_0 { omega_0 }
      , time { time }
      , t_stop { t_stop }
      , num_cycles { num_cycles }
      , theta1 { theta1 }
      , theta2 { theta2 }
      , Bsurf { bsurf } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      // real_t modified_Omega = Omega;
      // if ((time < t_stop + 10.0) && (time > 10.0)) {
      //   if (x_Ph[1] < constant::PI / 2.0) {
      //     modified_Omega = Omega + Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * x_Ph[1] - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   } else {
      //     modified_Omega = Omega - Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * (constant::PI - x_Ph[1]) - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   }
      //   return -modified_Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
      // } 
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      real_t modified_Omega = Omega;
      // if ((time < t_stop + 50.0) && (time > 50.0)) {
      //   if (x_Ph[1] < constant::PI / 2.0) {
      //     modified_Omega = Omega + Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * x_Ph[1] - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   } else {
      //     modified_Omega = Omega + Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * (constant::PI - x_Ph[1]) - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   }
      // } 
      return modified_Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      real_t modified_Omega = Omega;
      // if ((time < t_stop + 50.0) && (time > 50.0)){
      //   if (x_Ph[1] < constant::PI / 2.0) {
      //     modified_Omega = Omega + Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * x_Ph[1] - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   } else {
      //     modified_Omega = Omega + Omega_0
      //          * math::exp(-SQR(THREE)/ TWO * SQR((TWO * (constant::PI - x_Ph[1]) - (theta1 + theta2))/(theta1 - theta2))) 
      //          * math::sin(TWO * constant::PI * time * static_cast<real_t>(num_cycles) / t_stop);
      //   }
      // } 
      return -modified_Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Omega, Bsurf, time, Omega_0, t_stop, theta1, theta2;
    const int    num_cycles;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t      Bsurf, Rstar, Temperature, N0, t_stop, Omega, Omega_0, theta1, theta2;
    const int         num_cycles;
    const std::string field_geom;
    InitFields<D>     init_flds;


    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Temperature { p.template get<real_t>("setup.temperature") }
      , N0 { p.template get<real_t>("setup.N0") }
      , t_stop { p.template get<real_t>("setup.t_stop") }
      , Omega { p.template get<real_t>("setup.Omega") }
      , Omega_0 { p.template get<real_t>("setup.Omega_0") }
      , theta1 { p.template get<real_t>("setup.theta1") * static_cast<real_t>(constant::PI / 180.0)}
      , theta2 { p.template get<real_t>("setup.theta2") * static_cast<real_t>(constant::PI / 180.0)}
      , num_cycles { p.template get<int>("setup.num_cycles") }
      , field_geom { p.template get<std::string>("setup.field_geometry", "dipole") }
      , init_flds { Bsurf, Rstar, field_geom }{}

    inline PGen() {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto density_dist = DensityDistribution<S, M>(local_domain.mesh.metric, 
                                                          N0);
      const auto uniform_dist = arch::Uniform<S, M>(local_domain.mesh.metric);
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool,
                                                      Temperature);
      const auto injector = arch::experimental::Injector_with_weights<S, M, arch::Maxwellian, arch::Uniform, DensityDistribution>(
          energy_dist,
          uniform_dist,
          density_dist,
          { 1, 2 });
     arch::experimental::InjectWithWeights<S, M, decltype(injector)>(params,
                                                         local_domain,
                                                         injector,
                                                         1.0);
      
    }

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { Bsurf, Rstar, Omega, Omega_0, time, t_stop, theta1, theta2, num_cycles, field_geom };
    }

    auto MatchFields(real_t) const -> InitFields<D> {
      return InitFields<D> { Bsurf, Rstar, field_geom };
    }
  };

} // namespace user

#endif
