/**
 * @file archetypes/spatial_dist.hpp
 * @brief Spatial distribution class passed to injectors
 * @implements
 *   - arch::SpatialDistribution<>
 *   - arch::UniformDist<> : arch::SpatialDistribution<>
 *   - arch::ReplenishDist<> : arch::SpatialDistribution<>
 * @namespace
 *   - arch::
 * @note
 * Instances of these functors take coordinate position in code units
 * and return a number between 0 and 1 that represents the spatial distribution
 */

#ifndef ARCHETYPES_SPATIAL_DIST_HPP
#define ARCHETYPES_SPATIAL_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct SpatialDistribution {
    static constexpr bool is_spatial_dist { true };
    static_assert(M::is_metric, "M must be a metric class");

    SpatialDistribution(const M& metric) : metric { metric } {}

    Inline virtual auto operator()(const coord_t<M::Dim>&) const -> real_t {
      return ONE;
    }

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct UniformDist : public SpatialDistribution<S, M> {
    UniformDist(const M& metric) : SpatialDistribution<S, M> { metric } {}

    Inline auto operator()(const coord_t<M::Dim>&) const -> real_t override {
      return ONE;
    }
  };

  template <SimEngine::type S, class M, class T>
  struct ReplenishDist : public SpatialDistribution<S, M> {
    using SpatialDistribution<S, M>::metric;
    const ndfield_t<M::Dim, 6> density;
    const unsigned short       idx;

    const T      target_density;
    const real_t target_max_density;

    ReplenishDist(const M&                    metric,
                  const ndfield_t<M::Dim, 6>& density,
                  unsigned short              idx,
                  const T&                    target_density,
                  real_t                      target_max_density)
      : SpatialDistribution<S, M> { metric }
      , density { density }
      , idx { idx }
      , target_density { target_density }
      , target_max_density { target_max_density } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t override {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_1D) {
        dens = density(static_cast<std::size_t>(x_Cd[0]) + N_GHOSTS, idx);
      } else if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<std::size_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<std::size_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else if constexpr (M::Dim == Dim::_3D) {
        dens = density(static_cast<std::size_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<std::size_t>(x_Cd[1]) + N_GHOSTS,
                       static_cast<std::size_t>(x_Cd[2]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "Invalid dimension");
      }
      return math::max((target_density(x_Ph) - dens) / target_max_density,
                       static_cast<real_t>(0.0));
    }
  };

} // namespace arch

#endif // ARCHETYPES_SPATIAL_DIST_HPP
