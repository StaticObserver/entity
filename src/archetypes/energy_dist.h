/**
 * @file archetypes/energy_dist.h
 * @brief Defines an archetype for energy distributions
 * @implements
 *   - arch::energy_dist::Cold<>
 *   - arch::energy_dist::Powerlaw<>
 *   - arch::energy_dist::Maxwellian<>
 * @namespaces:
 *   - arch::energy_dist::
 */

#ifndef ARCHETYPES_ENERGY_DIST_HPP
#define ARCHETYPES_ENERGY_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace arch::energy_dist {
  using namespace ntt;

  template <Dimension D>
  struct Cold {
    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {

      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <Dimension D>
  struct Powerlaw {

    Powerlaw(random_number_pool_t& pool, real_t g_min, real_t g_max, real_t pl_ind)
      : g_min { g_min }
      , g_max { g_max }
      , pl_ind { pl_ind }
      , pool { pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      auto rand_gen = pool.get_state();
      auto rand_X1  = Random<real_t>(rand_gen);
      auto rand_gam = ONE;

      // Power-law distribution from uniform (see https://mathworld.wolfram.com/RandomNumber.html)
      if (pl_ind != -ONE) {
        rand_gam += math::pow(
          math::pow(g_min, ONE + pl_ind) +
            (-math::pow(g_min, ONE + pl_ind) + math::pow(g_max, ONE + pl_ind)) *
              rand_X1,
          ONE / (ONE + pl_ind));
      } else {
        rand_gam += math::pow(g_min, ONE - rand_X1) * math::pow(g_max, rand_X1);
      }
      auto rand_u  = math::sqrt(SQR(rand_gam) - ONE);
      auto rand_X2 = Random<real_t>(rand_gen);
      auto rand_X3 = Random<real_t>(rand_gen);
      v[0]         = rand_u * (TWO * rand_X2 - ONE);
      v[2]         = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
      v[1] = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * rand_X3);
      v[2] = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * rand_X3);

      pool.free_state(rand_gen);
    }

  private:
    const real_t         g_min, g_max, pl_ind;
    random_number_pool_t pool;
  };

  Inline void JuttnerSinge(vec_t<Dim::_3D>&            v,
                           real_t                      temp,
                           const random_number_pool_t& pool) {
    auto   rand_gen = pool.get_state();
    real_t randX1, randX2;
    if (temp < static_cast<real_t>(0.5)) {
      // Juttner-Synge distribution using the Box-Muller method - non-relativistic
      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[0]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);
    } else {
      // Juttner-Synge distribution using the Sobol method - relativistic
      auto randu   = ONE;
      auto randeta = Random<real_t>(rand_gen);
      while (SQR(randeta) <= SQR(randu) + ONE) {
        randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                 Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                   Random<real_t>(rand_gen);
        }
        randu  = -temp * math::log(randX1);
        randX2 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX2)) {
          randX2 = Random<real_t>(rand_gen);
        }
        randeta = -temp * math::log(randX1 * randX2);
      }
      randX1 = Random<real_t>(rand_gen);
      randX2 = Random<real_t>(rand_gen);
      v[0]   = randu * (TWO * randX1 - ONE);
      v[2]   = TWO * randu * math::sqrt(randX1 * (ONE - randX1));
      v[1]   = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * randX2);
      v[2]   = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * randX2);
    }
    pool.free_state(rand_gen);
  }

  template <bool CanBoost>
  Inline void SampleFromMaxwellian(vec_t<Dim::_3D>&            v,
                                   const random_number_pool_t& pool,
                                   real_t                      temperature,
                                   real_t boost_velocity = static_cast<real_t>(0),
                                   in   boost_direction = in::x1,
                                   bool flip_velocity   = false) {
    if (cmp::AlmostZero(temperature)) {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    } else {
      JuttnerSinge(v, temperature, pool);
    }
    if constexpr (CanBoost) {
      // Boost a symmetric distribution to a relativistic speed using flipping
      // method https://arxiv.org/pdf/1504.03910.pdf
      // @note: boost only when using cartesian coordinates
      if (not cmp::AlmostZero(boost_velocity)) {
        const auto boost_dir = static_cast<dim_t>(boost_direction);
        const auto boost_beta { boost_velocity /
                                math::sqrt(ONE + SQR(boost_velocity)) };
        const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
        auto       rand_gen = pool.get_state();
        if (-boost_beta * v[boost_dir] > gamma * Random<real_t>(rand_gen)) {
          v[boost_dir] = -v[boost_dir];
        }
        pool.free_state(rand_gen);
        v[boost_dir] = math::sqrt(ONE + SQR(boost_velocity)) *
                       (v[boost_dir] + boost_beta * gamma);
        if (flip_velocity) {
          v[0] = -v[0];
          v[1] = -v[1];
          v[2] = -v[2];
        }
      }
    }
  }

  /// Position-independent constant drift (default DriftF for Maxwellian).
  template <Dimension D>
  struct ConstantDrift {
    vec_t<Dim::_3D> drift;

    ConstantDrift() { drift[0] = ZERO; drift[1] = ZERO; drift[2] = ZERO; }
    explicit ConstantDrift(const std::vector<real_t>& v) {
      drift[0] = v[0]; drift[1] = v[1]; drift[2] = v[2];
    }

    Inline void operator()(const coord_t<D>& /*x_Ph*/, vec_t<Dim::_3D>& d) const {
      d[0] = drift[0]; d[1] = drift[1]; d[2] = drift[2];
    }
  };

  template <Dimension D, Coord::type C, class DriftF = ConstantDrift<D>>
  struct Maxwellian {

    /// Backward-compatible: constant drift as std::vector.
    template <class DF = DriftF, std::enable_if_t<std::is_same_v<DF, ConstantDrift<D>>, int> = 0>
    Maxwellian(random_number_pool_t&      pool,
               real_t                     temperature,
               const std::vector<real_t>& drift_four_vel = { ZERO, ZERO, ZERO })
      : pool { pool }, temperature { temperature }, drift_f { drift_four_vel } {
      raise::ErrorIf(drift_four_vel.size() != 3,
                     "Maxwellian: Drift velocity must be a 3D vector", HERE);
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative", HERE);
    }

    /// Custom (e.g. spatially-dependent) drift profile.
    template <class DF = DriftF, std::enable_if_t<!std::is_same_v<DF, ConstantDrift<D>>, int> = 0>
    Maxwellian(random_number_pool_t& pool, real_t temperature, const DF& df)
      : pool { pool }, temperature { temperature }, drift_f { df } {
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative", HERE);
    }

    Inline void operator()(const coord_t<D>& x_Ph, vec_t<Dim::_3D>& v) const {
      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO; v[1] = ZERO; v[2] = ZERO;
      } else {
        JuttnerSinge(v, temperature, pool);
      }
      // @note: boost only when using cartesian coordinates
      if constexpr (C == Coord::Cartesian) {
        // Get drift 4-velocity from drift profile (constant or position-dependent)
        vec_t<Dim::_3D> drift_4v;
        drift_f(x_Ph, drift_4v);
        const auto drift_mag = NORM(drift_4v[0], drift_4v[1], drift_4v[2]);
        if (not cmp::AlmostZero(drift_mag)) {
          const auto drift_3v = drift_mag / math::sqrt(ONE + SQR(drift_mag));
          // determine drift direction
          short dir = 4;
          for (auto d { 0u }; d < 3u; ++d) {
            const auto dprev = (d + 2) % 3;
            const auto dnext = (d + 1) % 3;
            if (cmp::AlmostZero(drift_4v[dprev]) and
                cmp::AlmostZero(drift_4v[dnext])) {
              dir = static_cast<short>(
                SIGN(drift_4v[d]) * static_cast<real_t>(d + 1));
              break;
            }
          }
          // Boost an isotropic Maxwellian with a drift velocity using
          // flipping method https://arxiv.org/pdf/1504.03910.pdf
          // 1. apply drift in X1 direction
          const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
          auto       rand_gen = pool.get_state();
          if (-drift_3v * v[0] > gamma * Random<real_t>(rand_gen)) {
            v[0] = -v[0];
          }
          pool.free_state(rand_gen);
          v[0] = math::sqrt(ONE + SQR(drift_mag)) * (v[0] + drift_3v * gamma);
          // 2. rotate to desired orientation
          if (dir == -1) {
            v[0] = -v[0];
          } else if (dir == 2 || dir == -2) {
            const auto tmp = v[1];
            v[1]           = dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (dir == 3 || dir == -3) {
            const auto tmp = v[2];
            v[2]           = dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (dir == 4) {
            const auto d1 = drift_4v[0] / drift_mag;
            const auto d2 = drift_4v[1] / drift_mag;
            const auto d3 = drift_4v[2] / drift_mag;
            vec_t<Dim::_3D> v_old;
            v_old[0] = v[0]; v_old[1] = v[1]; v_old[2] = v[2];

            v[0] = v_old[0] * d1 - v_old[1] * d2 - v_old[2] * d3;
            v[1] = (v_old[0] * d2 * (d1 + ONE) +
                    v_old[1] * (SQR(d1) + d1 + SQR(d3)) -
                    v_old[2] * d2 * d3) / (d1 + ONE);
            v[2] = (v_old[0] * d3 * (d1 + ONE) -
                    v_old[1] * d2 * d3 -
                    v_old[2] * (-d1 + SQR(d3) - ONE)) / (d1 + ONE);
          }
        }
      }
    }

  private:
    random_number_pool_t pool;
    const real_t         temperature;
    DriftF               drift_f;
  };

} // namespace arch::energy_dist

#endif // ARCHETYPES_ENERGY_DIST_HPP
