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

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t b0, real_t angle, real_t Omega, real_t d0, real_t rho0, real_t r, real_t bp) 
          : b0 { b0 }
          , angle { angle }
          , Omega { Omega }
          , skindepth0 { d0 }
          , larmor0 { rho0 }
          , R { r } 
          , bp { bp } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::cos(angle);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return bp;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::sin(angle);
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      // return Omega * b0 * (-math::sin(angle) * R + TWO * SQR(skindepth0) * x_Ph[0] / larmor0);
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * R;
    }

  private:
    const real_t b0, angle, R, Omega, skindepth0, larmor0, bp;
  };

  template <Dimension D>
  struct MFields {
    MFields(real_t b0, real_t angle, real_t omega, real_t r, real_t bp) 
         : b0 { b0 }
         , angle { angle }
         , Omega { omega }
         , R { r }
         , bp { bp } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::cos(angle);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return bp;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::sin(angle);
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * R;
    }


  private:
    const real_t b0, angle, Omega, R, bp;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t b0, real_t angle, real_t omega, real_t r, real_t d0, real_t rho0, real_t bp)
      : InitFields<D> { b0, angle, omega, d0, rho0, r, bp}
      , time { time }
      , Omega { omega }
      , R { r }{}

    using InitFields<D>::bx1;
    using InitFields<D>::bx3;

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx3(x_Ph) * R;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return Omega * bx1(x_Ph) * R;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t time, Omega, R;
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

    const real_t  B0, angle, R, Omega, skindepth0, larmor0, bp;
    const real_t  temp;
    const real_t  drift_u_1, drift_u_2;
    const real_t  j0;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , B0 { p.template get<real_t>("setup.B0", ONE) }
      , R { p.template get<real_t>("setup.R") }
      , Omega { static_cast<real_t>(constant::TWO_PI) /
                p.template get<real_t>("setup.period", ONE) }
      , angle { p.template get<real_t>("setup.angle", ZERO) }
      , skindepth0 { p.template get<real_t>("scales.skindepth0") }
      , larmor0 { p.template get<real_t>("scales.larmor0") }
      , bp { p.template get<real_t>("setup.Bp") }
      , temp { p.template get<real_t>("setup.temp") }
      , drift_u_1 { p.template get<real_t>("setup.drift_u_1") }
      , drift_u_2 { p.template get<real_t>("setup.drift_u_2") }
      , j0 { p.template get<real_t>("setup.j0") }
      , init_flds { B0, angle, Omega, skindepth0, larmor0, R, bp } {}

    inline PGen() {}

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, B0, angle, Omega, R, skindepth0, larmor0, bp };
    }

    auto MatchFields(real_t) const -> MFields<D> {
      return MFields<D> { B0, angle, Omega, R, bp };
    }

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist_1 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp,
                                                        -drift_u_1,
                                                        in::x1);
      const auto energy_dist_2 = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temp,
                                                        drift_u_2,
                                                        in::x1);
      const auto injector_1 = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist_1,
        { 1, 1 });
      const auto injector_2 = arch::UniformInjector<S, M, arch::Maxwellian>(
        energy_dist_2,
        { 2, 2 });
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector_1,
        ONE);
      arch::InjectUniform<S, M, arch::UniformInjector<S, M, arch::Maxwellian>>(
        params,
        local_domain,
        injector_2,
        ONE + j0);
      }
  };

} // namespace user

#endif
