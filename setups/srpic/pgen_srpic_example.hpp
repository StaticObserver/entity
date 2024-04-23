#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "archetypes/problem_generator.h"

namespace user {
  using namespace ntt;

  template <SimeEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::SRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Minkowski, Metric::Spherical, Metric::QSpherical>::value
    };
    static constexpr auto dimensions {
      traits::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D>::value
    };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;
    using arch::ProblemGenerator<S, M>::domain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : arch::ProblemGenerator<S, M>(p) {}

    inline PGen() {}
  };

} // namespace user

#endif
