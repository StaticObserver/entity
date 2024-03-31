/**
 * @file metric_base.h
 * @brief Base class for all the metrics
 * @implements
 *   - ntt::MetricBase
 * @depends:
 *   - enums.h
 *   - global.h
 * @namespaces:
 *   - ntt::
 * @note
 * Other metrics inherit from this class using the CRTP pattern
 * see: https://en.cppreference.com/w/cpp/language/crtp
 * @note
 * Coordinate transformations
 *
 *   +---> Cart
 *   |
 *   v
 * Code
 *   ^
 *   |
 *   +---> Sph
 *
 * Code: coordinates in code units
 * Cart: coordinates in global Cartesian basis
 * Sph: coordinates in spherical basis
 * @note
 * Vector transformations
 *
 * Cntrv (A^mu)
 *   ^  ^
 *   |  |
 *   |  v
 *   | Hat <---> Cart (A_xyz)
 *   |  ^
 *   |  |
 *   v  v
 *   Cov (A_mu)
 *
 * Cntrv: contravariant vector
 * Cov: covariant vector
 * Hat: hatted (orthonormal) basis vector
 * Cart: global Cartesian basis vector (defined for diagonal only)
 */

#ifndef METRICS_METRIC_BASE_H
#define METRICS_METRIC_BASE_H

#include "enums.h"
#include "global.h"

#include "utils/numeric.h"

#include <string>
#include <utility>
#include <vector>

namespace ntt {

  /**
   * Virtual parent metric class template: h_ij
   * Coordinates vary from `0` to `nx1` ... (code units)
   */
  template <Dimension D, class M>
  struct MetricBase {
    static constexpr Dimension Dim { D };

    // text label of the metric
    const std::string label;
    // coordinate system to use
    const Coord::type coord;
    // max of coordinates in code units
    const real_t      nx1, nx2, nx3;
    // extent in `x1` in physical units
    const real_t      x1_min, x1_max;
    // extent in `x2` in physical units
    const real_t      x2_min, x2_max;
    // extent in `x3` in physical units
    const real_t      x3_min, x3_max;

    MetricBase(const std::string&                     label,
               const Coord::type&                     coord,
               std::vector<unsigned int>              res,
               std::vector<std::pair<real_t, real_t>> ext) :
      label { label },
      coord { coord },
      nx1 { res.size() > 0 ? (real_t)(res[0]) : ONE },
      nx2 { res.size() > 1 ? (real_t)(res[1]) : ONE },
      nx3 { res.size() > 2 ? (real_t)(res[2]) : ONE },
      x1_min { res.size() > 0 ? ext[0].first : ZERO },
      x1_max { res.size() > 0 ? ext[0].second : ZERO },
      x2_min { res.size() > 1 ? ext[1].first : ZERO },
      x2_max { res.size() > 1 ? ext[1].second : ZERO },
      x3_min { res.size() > 2 ? ext[2].first : ZERO },
      x3_max { res.size() > 2 ? ext[2].second : ZERO } {}

    ~MetricBase() = default;

    [[nodiscard]]
    virtual auto find_dxMin() const -> real_t = 0;

    [[nodiscard]]
    auto dxMin() const -> real_t {
      return dx_min;
    }

    auto set_dxMin(real_t dxmin) -> void {
      dx_min = dxmin;
    }

  protected:
    real_t dx_min;
  };

} // namespace ntt

#endif // METRICS_METRIC_BASE_H