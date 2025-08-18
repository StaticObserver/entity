#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t b0_, real_t coeff_, real_t xsurf_, real_t ds_) 
          : b0 { b0_ }
          , coeff { coeff_ } 
          , xsurf { xsurf_ }
          , ds { ds_ }{}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
          if (x_Ph[0] <  xsurf + ds ){
              return ZERO;
          }else if (x_Ph[0] > xsurf + 1.33 * ds){
               return -coeff * (x_Ph[0] - xsurf - ds * (ONE + 0.03 * math::log(1.01) - 0.33 * math::log(0.01 + math::exp(-11.0 / ds))));
               
          }else{
               return -coeff * (x_Ph[0] - xsurf - ds
                                     + 0.03 * ds * math::log(0.01 + math::exp((xsurf + 1.0 * ds - x_Ph[0]) / 0.03 / ds))
                                     - 0.03 * ds * math::log(0.01 + ONE));
              //return -coeff * (x_Ph[0] - xsurf - ds);
          }
    }

  private:
    const real_t b0, coeff, xsurf, ds;
  };

  template <Dimension D>
  struct MFields {
    MFields(real_t b0_) 
         : b0 { b0_ } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  private:
    const real_t b0;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t b0_, real_t xsurf_, real_t ds_)
      : InitFields<D> { b0_, ZERO, xsurf_, ds_ } {}

    using InitFields<D>::bx1;

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  };


  template <SimEngine::type S, class M>
    struct TargetDensityProfile : public arch::SpatialDistribution<S, M> {
        const real_t nmax, height, xsurf, ds;

        TargetDensityProfile(const M& metric, real_t nmax, real_t height, real_t xsurf, real_t ds)
          : arch::SpatialDistribution<S, M>(metric)
          , nmax { nmax }
          , height { height }
          , xsurf { xsurf }
          , ds { ds } {}

        Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
             if (x_Ph[0] < xsurf){
                  return ZERO;
             }else{
                  return nmax * math::exp(-(x_Ph[0] - xsurf) / height);
             }
           }
      }; // TargetDensityProfile
    
    template <SimEngine::type S, class M>
    struct ExtraCharge : public arch::SpatialDistribution<S, M> {
        const real_t xsurf, ds;

        ExtraCharge(const M& metric, real_t xsurf_, real_t ds_)
          : arch::SpatialDistribution<S, M>(metric)
          , xsurf { xsurf_ }
          , ds { ds_ } {}  

        Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
               if ( x_Ph[0] < xsurf or x_Ph[0] > 1.33 * ds + xsurf){
                  return ZERO;
               }else{
                   if (x_Ph[0] < ds + xsurf){
                       return ONE;
                   }else{
                       return ONE - 0.01 / (0.01 + math::exp(-(x_Ph[0] - xsurf - 1.0 * ds) / 0.03 / ds));
                   }
               }
          }
      }; // ExtraCharge
  

  template <Dimension D>
  struct MagnetosphericCurrent { 
    MagnetosphericCurrent(const real_t J0_)
      : J0 { J0_ } {};

    Inline auto jx1(const coord_t<D>& x_Ph) const -> real_t {
        return J0;
      }

    Inline auto jx2(const coord_t<D>& x_Ph) const -> real_t {
        return ZERO;
      }

    Inline auto jx3(const coord_t<D>& x_Ph) const -> real_t {
        return ZERO;
      }

    private:
      const real_t J0; 
  };
   
  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_1D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t  b0, skindepth0, larmor0;
    const real_t  temp;
    const real_t  j0;
    const real_t  xsurf, ds;
    InitFields<D> init_flds;
    MagnetosphericCurrent<D> ext_current;
    

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , b0 { p.template get<real_t>("setup.B0", ONE) } 
      , skindepth0 { p.template get<real_t>("scales.skindepth0") }
      , larmor0 { p.template get<real_t>("scales.larmor0") }
      , temp { p.template get<real_t>("setup.temp") }
      , j0 { p.template get<real_t>("setup.j0") / p.template get<real_t>("scales.V0")}
      , ds { p.template get<real_t>("grid.boundaries.atmosphere.ds") }
      , xsurf([&m, this]() {
          const auto min_buff = params.template get<unsigned short>("algorithms.current_filters") + 2;
          const auto buffer_ncells = min_buff > 5 ? min_buff : 5;
          return m.mesh().metric.template convert<1, Crd::Cd, Crd::Ph>(static_cast<real_t>(buffer_ncells));
        }())
      //, l_atm { ZERO } 
      // , init_flds(b0, TWO * FOUR * constant::PI * b0 * Omega * SQR(skindepth0) / larmor0, l_atm, ds)
      , init_flds(b0, larmor0 / SQR(skindepth0), xsurf, ds)
      , ext_current(j0)
    {}

    inline PGen() {}


    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, b0, xsurf, ds};
    }

    auto MatchFields(real_t) const -> MFields<D> {
      return MFields<D> { b0 };
    }


    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp);
      // const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
      //   energy_dist,
      //   { 1, 2 }
      // );

      // arch::InjectUniform<S, M, decltype(injector)>(
      //   params,
      //   local_domain,
      //   injector,
      //   ONE
      // );
      const auto spatial_dist = TargetDensityProfile<S, M>(
          local_domain.mesh.metric,
          params.template get<real_t>("grid.boundaries.atmosphere.density"),
          params.template get<real_t>("grid.boundaries.atmosphere.height"),
          xsurf,
          ds);
      const auto injector = arch::NonUniformInjector<S, M, arch::Maxwellian, TargetDensityProfile>(
        energy_dist,
        spatial_dist,
        { 1, 2 }
      );
      arch::InjectNonUniform<S, M, decltype(injector)>(
        params,
        local_domain,
        injector,
        ONE);

      const auto extra_charge = ExtraCharge<S, M>(
        local_domain.mesh.metric,
        xsurf,
        ds
      );
      const auto injector_extra_charge = arch::NonUniformInjector<S, M, arch::Maxwellian, ExtraCharge>(
        energy_dist,
        extra_charge,
        { 2, 2 }
      );
      arch::InjectNonUniform<S, M, decltype(injector_extra_charge)>(
        params,
        local_domain,
        injector_extra_charge,
        TWO);
    }
  }; // PGen  


} // namespace user

#endif
