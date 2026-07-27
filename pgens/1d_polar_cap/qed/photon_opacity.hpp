#ifndef POLAR_CAP_PHOTON_OPACITY_HPP
#define POLAR_CAP_PHOTON_OPACITY_HPP

#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/pushers/context.h"

namespace user::polar_cap {
  using namespace ntt;

  template <MetricClass M>
  struct PhotonOpacityUpdate {
    static_assert(M::Dim == Dim::_1D,
                  "Photon opacity is implemented only for 1D polar-cap runs");
    static_assert(M::CoordType == Coord::Cartesian,
                  "Photon opacity requires Cartesian coordinates");

    const bool    enabled;
    const spidx_t photon_species_index;
    const real_t  rho_c;
    const real_t  opacity_prefactor;
    const real_t  exponent_coefficient;
    const int     substeps;

    PhotonOpacityUpdate(bool    enabled,
                        spidx_t photon_species_index,
                        real_t  rho_c,
                        real_t  opacity_prefactor,
                        real_t  b_over_bq,
                        int     substeps)
      : enabled { enabled }
      , photon_species_index { photon_species_index }
      , rho_c { rho_c }
      , opacity_prefactor { opacity_prefactor }
      , exponent_coefficient { static_cast<real_t>(8.0) /
                               (static_cast<real_t>(3.0) * b_over_bq) }
      , substeps { substeps } {}

    Inline auto attenuation(real_t photon_energy, real_t theta) const -> real_t {
      // One-photon magnetic conversion is kinematically allowed only when
      // epsilon_gamma * |sin(theta_B)| >= 2 in electron-rest-mass units.
      const auto sin_theta = math::abs(math::sin(theta));
      if (sin_theta <= ZERO or photon_energy * sin_theta < TWO) {
        return ZERO;
      }
      return opacity_prefactor * sin_theta *
             math::exp(-exponent_coefficient / (photon_energy * sin_theta));
    }

    Inline void operator()(prtlidx_t                              p,
                           const kernel::sr::PusherContext&       context,
                           const kernel::sr::PusherBoundaries<M::Dim>&,
                           const ParticleArrays&                  particles,
                           const M&) const {
      if (not enabled or particles.sp != photon_species_index) {
        return;
      }
      if (particles.pld_r.extent(1) < 3) {
        raise::KernelError(HERE, "Photon species requires three real payloads");
      }

      const auto photon_energy = particles.pld_r(p, 0);
      const auto u_norm = NORM(particles.ux1(p), particles.ux2(p), particles.ux3(p));
      if (photon_energy <= ZERO or u_norm <= ZERO) {
        return;
      }

      // A massless particle travels a physical distance dt. Its x1 displacement
      // is shorter for oblique propagation and only that displacement changes
      // the prescribed field-line orientation.
      const auto trajectory_length = context.dt;
      const auto field_line_distance = context.dt * particles.ux1(p) / u_norm;
      const auto theta0 = particles.pld_r(p, 2);
      const auto theta1 = theta0 + field_line_distance / rho_c;
      const auto dtheta      = (theta1 - theta0) / static_cast<real_t>(substeps);
      const auto ds = trajectory_length / static_cast<real_t>(substeps);

      // Composite Simpson integration along the photon trajectory. PGen
      // validation guarantees an even, positive number of substeps.
      real_t integral = attenuation(photon_energy, theta0) +
                        attenuation(photon_energy, theta1);
      for (int i = 1; i < substeps; ++i) {
        const auto coefficient = (i % 2 == 0) ? TWO : FOUR;
        integral += coefficient *
                    attenuation(photon_energy,
                                theta0 + static_cast<real_t>(i) * dtheta);
      }
      particles.pld_r(p, 1) += ds * integral / THREE;
      // Payload 1 is cumulative optical depth; payload 2 is the updated angle.
      particles.pld_r(p, 2) = theta1;
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_PHOTON_OPACITY_HPP
