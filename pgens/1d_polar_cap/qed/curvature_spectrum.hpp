#ifndef POLAR_CAP_CURVATURE_SPECTRUM_HPP
#define POLAR_CAP_CURVATURE_SPECTRUM_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace user::polar_cap {
  using namespace ntt;

  class CurvatureSpectrum {
    array_t<real_t*> m_x;
    array_t<real_t*> m_ccdf;
    std::size_t      m_size { 0 };

  public:
    CurvatureSpectrum() = default;

    explicit CurvatureSpectrum(const std::string& filename) {
      // Allow both launch-directory paths and paths relative to this PGen.
      auto table_path = std::filesystem::path { filename };
      std::ifstream input(table_path);
      if (not input.is_open() and table_path.is_relative()) {
        table_path = std::filesystem::path { __FILE__ }.parent_path().parent_path() /
                     table_path;
        input.clear();
        input.open(table_path);
      }
      raise::ErrorIf(not input.is_open(),
                     "Could not open curvature spectrum table: " +
                       table_path.string(),
                     HERE);

      std::vector<real_t> x_host;
      std::vector<real_t> ccdf_host;
      real_t              x, ccdf;
      while (input >> x >> ccdf) {
        x_host.push_back(x);
        ccdf_host.push_back(ccdf);
      }
      raise::ErrorIf(not input.eof(),
                     "Malformed curvature spectrum table: " +
                       table_path.string(),
                     HERE);

      raise::ErrorIf(x_host.size() < 2,
                     "Curvature spectrum table must contain at least two rows",
                     HERE);
      for (std::size_t i = 0; i < x_host.size(); ++i) {
        // Strict monotonicity is required by both binary searches below.
        raise::ErrorIf(x_host[i] <= ZERO,
                       "Curvature spectrum x values must be positive",
                       HERE);
        raise::ErrorIf(ccdf_host[i] <= ZERO or ccdf_host[i] > ONE,
                       "Curvature spectrum CCDF values must be in (0, 1]",
                       HERE);
        if (i > 0) {
          raise::ErrorIf(x_host[i] <= x_host[i - 1],
                         "Curvature spectrum x values must increase",
                         HERE);
          raise::ErrorIf(ccdf_host[i] >= ccdf_host[i - 1],
                         "Curvature spectrum CCDF values must decrease",
                         HERE);
        }
      }

      m_size = x_host.size();
      m_x    = array_t<real_t*> { "curvature_spectrum_x", m_size };
      m_ccdf = array_t<real_t*> { "curvature_spectrum_ccdf", m_size };

      auto x_mirror    = Kokkos::create_mirror_view(m_x);
      auto ccdf_mirror = Kokkos::create_mirror_view(m_ccdf);
      for (std::size_t i = 0; i < m_size; ++i) {
        x_mirror(i)    = x_host[i];
        ccdf_mirror(i) = ccdf_host[i];
      }
      Kokkos::deep_copy(m_x, x_mirror);
      Kokkos::deep_copy(m_ccdf, ccdf_mirror);
    }

    Inline auto upper_tail_slope() const -> real_t {
      // The final two samples define a log(CCDF)-linear exponential tail.
      return (math::log(m_ccdf(m_size - 1)) -
              math::log(m_ccdf(m_size - 2))) /
             (m_x(m_size - 1) - m_x(m_size - 2));
    }

    Inline auto ccdf(real_t value) const -> real_t {
      if (m_size == 0) {
        return ONE;
      }
      if (value <= ZERO) {
        return ONE;
      }
      if (value <= m_x(0)) {
        // Small-x asymptote of the normalized curvature number spectrum.
        const auto approx = ONE + static_cast<real_t>(0.346) * value -
                            math::pow(value, ONE / THREE) *
                              (static_cast<real_t>(1.232) +
                               static_cast<real_t>(0.033) * SQR(value));
        return approx > ZERO ? approx : m_ccdf(0);
      }
      if (value > m_x(m_size - 1)) {
        return math::exp(math::log(m_ccdf(m_size - 1)) +
                         upper_tail_slope() * (value - m_x(m_size - 1)));
      }
      if (value == m_x(m_size - 1)) {
        return m_ccdf(m_size - 1);
      }

      std::size_t lo = 0;
      std::size_t hi = m_size - 1;
      // Interpolate log(CCDF) against log(x) after locating the bracket.
      while (hi - lo > 1) {
        const auto mid = (lo + hi) / 2;
        if (m_x(mid) <= value) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const auto t = (math::log(value) - math::log(m_x(lo))) /
                     (math::log(m_x(hi)) - math::log(m_x(lo)));
      return math::exp(math::log(m_ccdf(lo)) * (ONE - t) +
                       math::log(m_ccdf(hi)) * t);
    }

    Inline auto inverse_ccdf(real_t probability) const -> real_t {
      if (m_size == 0) {
        return ZERO;
      }
      if (probability <= ZERO) {
        probability = std::numeric_limits<real_t>::min();
      }
      if (probability >= m_ccdf(0)) {
        return CUBE((ONE - probability) / static_cast<real_t>(1.232));
      }
      if (probability <= m_ccdf(m_size - 1)) {
        return m_x(m_size - 1) +
               (math::log(probability) - math::log(m_ccdf(m_size - 1))) /
                 upper_tail_slope();
      }

      std::size_t lo = 0;
      std::size_t hi = m_size - 1;
      // CCDF is decreasing, so the comparison is reversed from ccdf().
      while (hi - lo > 1) {
        const auto mid = (lo + hi) / 2;
        if (m_ccdf(mid) >= probability) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const auto t = (math::log(probability) - math::log(m_ccdf(lo))) /
                     (math::log(m_ccdf(hi)) - math::log(m_ccdf(lo)));
      return math::exp(math::log(m_x(lo)) * (ONE - t) + math::log(m_x(hi)) * t);
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_CURVATURE_SPECTRUM_HPP
