#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t b0_, real_t rho_GJ_, real_t l_atm_) 
          : b0 { b0_ }
          , rho_GJ { rho_GJ_ } 
          , l_atm { l_atm_ }{}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] > l_atm){
            return rho_GJ * (x_Ph[0] - l_atm);
        }else{
            return ZERO;
        }
    }

  private:
    const real_t b0, rho_GJ, l_atm;
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
    DriveFields(real_t time, real_t b0_, real_t l_atm_)
      : InitFields<D> { b0_, ZERO, l_atm_ } {}

    using InitFields<D>::bx1;

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

  };


  template <SimEngine::type S, class M>
    struct TargetDensityProfile : public arch::SpatialDistribution<S, M> {
        const real_t nmax, height, xsurf;

        TargetDensityProfile(const M& metric, real_t nmax, real_t height, real_t xsurf)
          : arch::SpatialDistribution<S, M>(metric)
          , nmax { nmax }
          , height { height }
          , xsurf { xsurf } {}

        Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
              return nmax * math::exp(-(x_Ph[0] - xsurf) / height);
          }
      }; // TargetDensityProfile
  

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

    const real_t  b0, Omega, skindepth0, larmor0;
    const real_t  temp;
    const real_t  j0;
    const real_t  l_atm;
    InitFields<D> init_flds;
    

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , b0 { p.template get<real_t>("setup.B0", ONE) } 
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , skindepth0 { p.template get<real_t>("scales.skindepth0") }
      , larmor0 { p.template get<real_t>("scales.larmor0") }
      , temp { p.template get<real_t>("setup.temp") }
      , j0 { p.template get<real_t>("setup.j0") }
      , l_atm([&m, this]() {
          const auto min_buff = params.template get<unsigned short>("algorithms.current_filters") + 2;
          const auto buffer_ncells = min_buff > 5 ? min_buff : 5;
          return m.mesh().metric.template convert<1, Crd::Cd, Crd::Ph>(static_cast<real_t>(buffer_ncells));
        }())
      , init_flds(b0, TWO * FOUR * constant::PI * b0 * Omega * SQR(skindepth0) / larmor0, l_atm)
    {}

    inline PGen() {}


    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, b0, l_atm };
    }

    auto MatchFields(real_t) const -> MFields<D> {
      return MFields<D> { b0 };
    }


    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp);
      const auto spatial_dist = TargetDensityProfile<S, M>(
          local_domain.mesh.metric,
          params.template get<real_t>("grid.boundaries.atmosphere.density"),
          params.template get<real_t>("grid.boundaries.atmosphere.height"),
          l_atm);
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
    }
  }; // PGen  


} // namespace user

#endif
