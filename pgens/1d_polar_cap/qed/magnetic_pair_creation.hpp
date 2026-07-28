#ifndef POLAR_CAP_MAGNETIC_PAIR_CREATION_HPP
#define POLAR_CAP_MAGNETIC_PAIR_CREATION_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

namespace user::polar_cap {
  using namespace ntt;

  template <Dimension D, Coord::type C>
  struct MagneticPairCreation {
    static_assert(D == Dim::_1D,
                  "Magnetic pair creation is implemented only for 1D polar-cap runs");
    static_assert(C == Coord::Cartesian,
                  "Magnetic pair creation requires Cartesian coordinates");

    const real_t conversion_optical_depth;
    const npart_t domain_index;

    array_t<int*>      photon_i1;
    array_t<prtldx_t*> photon_dx1;
    array_t<real_t*>   photon_ux1, photon_weight;
    array_t<short*>    photon_tag;
    array_t<real_t**>  photon_pld_r;

    array_t<int*>      electron_i1, electron_i2, electron_i3;
    array_t<int*>      positron_i1, positron_i2, positron_i3;
    array_t<prtldx_t*> electron_dx1, electron_dx2, electron_dx3;
    array_t<prtldx_t*> positron_dx1, positron_dx2, positron_dx3;
    array_t<real_t*>   electron_ux1, electron_ux2, electron_ux3;
    array_t<real_t*>   positron_ux1, positron_ux2, positron_ux3;
    array_t<real_t*>   electron_phi, positron_phi;
    array_t<real_t*>   electron_weight, positron_weight;
    array_t<short*>    electron_tag, positron_tag;
    array_t<npart_t**> electron_pld_i, positron_pld_i;

    const npart_t electron_offset, positron_offset;
    const npart_t electron_counter, positron_counter;
    const bool    electron_tracking, positron_tracking;

    array_t<npart_t> converted { "polar_cap_pairs_converted" };

    MagneticPairCreation(Particles<D, C>& photons,
                         Particles<D, C>& electrons,
                         Particles<D, C>& positrons,
                         npart_t          domain_index,
                         real_t           conversion_optical_depth)
      : conversion_optical_depth { conversion_optical_depth }
      , domain_index { domain_index }
      , photon_i1 { photons.i1 }
      , photon_dx1 { photons.dx1 }
      , photon_ux1 { photons.ux1 }
      , photon_weight { photons.weight }
      , photon_tag { photons.tag }
      , photon_pld_r { photons.pld_r }
      , electron_i1 { electrons.i1 }
      , electron_i2 { electrons.i2 }
      , electron_i3 { electrons.i3 }
      , positron_i1 { positrons.i1 }
      , positron_i2 { positrons.i2 }
      , positron_i3 { positrons.i3 }
      , electron_dx1 { electrons.dx1 }
      , electron_dx2 { electrons.dx2 }
      , electron_dx3 { electrons.dx3 }
      , positron_dx1 { positrons.dx1 }
      , positron_dx2 { positrons.dx2 }
      , positron_dx3 { positrons.dx3 }
      , electron_ux1 { electrons.ux1 }
      , electron_ux2 { electrons.ux2 }
      , electron_ux3 { electrons.ux3 }
      , positron_ux1 { positrons.ux1 }
      , positron_ux2 { positrons.ux2 }
      , positron_ux3 { positrons.ux3 }
      , electron_phi { electrons.phi }
      , positron_phi { positrons.phi }
      , electron_weight { electrons.weight }
      , positron_weight { positrons.weight }
      , electron_tag { electrons.tag }
      , positron_tag { positrons.tag }
      , electron_pld_i { electrons.pld_i }
      , positron_pld_i { positrons.pld_i }
      , electron_offset { electrons.npart() }
      , positron_offset { positrons.npart() }
      , electron_counter { electrons.counter() }
      , positron_counter { positrons.counter() }
      , electron_tracking { electrons.use_tracking() }
      , positron_tracking { positrons.use_tracking() } {
      Kokkos::deep_copy(converted, 0);
    }

    Inline void operator()(prtlidx_t p) const {
      if (photon_tag(p) != ParticleTag::alive) {
        return;
      }
      const auto photon_energy = photon_pld_r(p, 0);
      const auto optical_depth = photon_pld_r(p, 1);
      const auto sin_theta     = math::abs(math::sin(photon_pld_r(p, 2)));
      // Keep sub-threshold photons alive: continued propagation can increase
      // theta_B and optical depth on a later timestep.
      if (photon_energy * sin_theta < TWO or
          optical_depth < conversion_optical_depth) {
        return;
      }

      // One atomic conversion index reserves matching electron and positron
      // slots, preserving one-to-one pair bookkeeping.
      const auto relative = Kokkos::atomic_fetch_add(&converted(), 1);
      const auto electron_index = electron_offset + relative;
      const auto positron_index = positron_offset + relative;
      if (electron_index >= electron_ux1.extent(0) or
          positron_index >= positron_ux1.extent(0)) {
        raise::KernelError(HERE, "Magnetic pair creation exceeds maxnpart");
      }

      // Reduced 1D closure: split photon energy equally and place both
      // particles along the photon's x1 propagation direction.
      const auto gamma_pair = HALF * photon_energy;
      const auto u_magnitude = math::sqrt(
        math::max(static_cast<real_t>(0.0), SQR(gamma_pair) - ONE));
      const auto direction   = photon_ux1(p) >= ZERO ? ONE : -ONE;
      const vec_t<Dim::_3D> pair_u { direction * u_magnitude, ZERO, ZERO };
      const tuple_t<int, D> pair_i { photon_i1(p) };
      const tuple_t<prtldx_t, D> pair_dx { photon_dx1(p) };

      if (electron_tracking) {
        // Tracking counters are species-local, so each child uses its own
        // pre-kernel counter plus the shared conversion index.
        kernel::InjectParticle<D, C, true>(electron_index,
                                            electron_i1,
                                            electron_i2,
                                            electron_i3,
                                            electron_dx1,
                                            electron_dx2,
                                            electron_dx3,
                                            electron_ux1,
                                            electron_ux2,
                                            electron_ux3,
                                            electron_phi,
                                            electron_weight,
                                            electron_tag,
                                            electron_pld_i,
                                            pair_i,
                                            pair_dx,
                                            pair_u,
                                            photon_weight(p),
                                            ZERO,
                                            domain_index,
                                            electron_counter + relative);
      } else {
        kernel::InjectParticle<D, C, false>(electron_index,
                                             electron_i1,
                                             electron_i2,
                                             electron_i3,
                                             electron_dx1,
                                             electron_dx2,
                                             electron_dx3,
                                             electron_ux1,
                                             electron_ux2,
                                             electron_ux3,
                                             electron_phi,
                                             electron_weight,
                                             electron_tag,
                                             electron_pld_i,
                                             pair_i,
                                             pair_dx,
                                             pair_u,
                                             photon_weight(p));
      }

      if (positron_tracking) {
        kernel::InjectParticle<D, C, true>(positron_index,
                                            positron_i1,
                                            positron_i2,
                                            positron_i3,
                                            positron_dx1,
                                            positron_dx2,
                                            positron_dx3,
                                            positron_ux1,
                                            positron_ux2,
                                            positron_ux3,
                                            positron_phi,
                                            positron_weight,
                                            positron_tag,
                                            positron_pld_i,
                                            pair_i,
                                            pair_dx,
                                            pair_u,
                                            photon_weight(p),
                                            ZERO,
                                            domain_index,
                                            positron_counter + relative);
      } else {
        kernel::InjectParticle<D, C, false>(positron_index,
                                             positron_i1,
                                             positron_i2,
                                             positron_i3,
                                             positron_dx1,
                                             positron_dx2,
                                             positron_dx3,
                                             positron_ux1,
                                             positron_ux2,
                                             positron_ux3,
                                             positron_phi,
                                             positron_weight,
                                             positron_tag,
                                             positron_pld_i,
                                             pair_i,
                                             pair_dx,
                                             pair_u,
                                             photon_weight(p));
      }

      // Retire the photon only after both child particles have been written.
      photon_tag(p) = ParticleTag::dead;
    }

    auto number_converted() const -> npart_t {
      // The deep copy fences the kernel before PGen publishes new npart values.
      auto host = Kokkos::create_mirror_view(converted);
      Kokkos::deep_copy(host, converted);
      return host();
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_MAGNETIC_PAIR_CREATION_HPP
