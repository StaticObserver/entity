#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t b0, real_t angle, real_t Omega, real_t d0, real_t rho0) 
          : b0 { b0 }
          , angle { angle }
          , Omega { Omega }
          , skindepth0 { d0 }
          , larmor0 { rho0 } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::cos(angle);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::sin(angle);
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return Omega * b0 * (math::sin(angle) * R + TWO * SQR(skindepth0) * x_Ph[0] / larmor0);
    }

  private:
    const real_t b0, angle, R, Omega, skindepth0, larmor0;
  };

  template <Dimension D>
  struct MFields {
    MFields(real_t b0, real_t angle) : b0 { b0 }, angle { angle } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::cos(angle);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return b0 * math::sin(angle);
    }


  private:
    const real_t b0, angle;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t time, real_t b0, real_t angle, real_t omega, real_t r)
      : InitFields<D> { b0, angle}
      , time { time }
      , Omega { omega }
      , R { r }{}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return Omega * bx2(x_Ph) * R;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * R;
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

    const real_t  B0, angle, R, Omega, skindepth0, larmor0;
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
      , init_flds { B0, angle} {}

    inline PGen() {}

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { time, B0, angle, Omega, R, skindepth0, larmor0 };
    }

    auto MatchFields(real_t) const -> MFields<D> {
      return MFields<D> { B0, angle };
    }
  };

} // namespace user

#endif
