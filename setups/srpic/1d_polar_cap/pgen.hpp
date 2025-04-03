#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include "kernels/QED_process.hpp"

namespace user {
  using namespace ntt;
  using namespace kernel::QED;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t b0_, real_t rho_GJ_) 
          : b0 { b0_ }
          , rho_GJ { rho_GJ_ } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return rho_GJ * x_Ph[0];
      //return ZERO;
    }

  private:
    const real_t b0, rho_GJ;
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
    DriveFields(real_t time, real_t b0_)
      : InitFields<D> { b0_, ZERO} {}

    using InitFields<D>::bx1;

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

    const real_t  b0, Omega, skindepth0, larmor0;
    const real_t  temp;
    const real_t  drift_u_1, drift_u_2;
    const real_t  j0;
    InitFields<D> init_flds;
    cdfTable cdf;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , b0 { p.template get<real_t>("setup.B0", ONE) } 
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , skindepth0 { p.template get<real_t>("scales.skindepth0") }
      , larmor0 { p.template get<real_t>("scales.larmor0") }
      , temp { p.template get<real_t>("setup.temp") }
      , drift_u_1 { p.template get<real_t>("setup.drift_u_1") }
      , drift_u_2 { p.template get<real_t>("setup.drift_u_2") }
      , j0 { p.template get<real_t>("setup.j0") }
      , init_flds { b0, TWO * FOUR * constant::PI * b0 * Omega * SQR(skindepth0) / larmor0 }
      , cdf { "cdf_table.txt", "inverse_cdf_table.txt" } {}

    inline PGen() {}

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, b0 };
    }

    auto MatchFields(real_t) const -> MFields<D> {
      return MFields<D> { b0 };
    }

  //   inline void InitPrtls(Domain<S, M>& local_domain) {
  //     const auto energy_dist_1 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
  //                                                       local_domain.random_pool,
  //                                                       temp,
  //                                                       -drift_u_1,
  //                                                       in::x1);
  //     const auto energy_dist_2 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
  //                                                       local_domain.random_pool,
  //                                                       temp,
  //                                                       drift_u_2,
  //                                                       in::x1);
  //     const auto injector_1 = arch::UniformInjector<S, M, arch::Maxwellian>(
  //       energy_dist_1,
  //       { 1, 1 });
  //     const auto injector_2 = arch::UniformInjector<S, M, arch::Maxwellian>(
  //       energy_dist_2,
  //       { 2, 2 });
  //     arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
  //       params,
  //       local_domain,
  //       injector_1,
  //       ONE);
  //     arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
  //       params,
  //       local_domain,
  //       injector_2,
  //       ONE + j0);
  //     }

    void CustomPostStep(std::size_t, long double, Domain<S, M>& domain) {

    }
  };

} // namespace user

#endif
