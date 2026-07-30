/**
 * @file kernels/boundary_absorb.hpp
 * @brief Kill pass for flux-conserving absorption at global x1 boundaries
 * @implements
 *   - kernel::sr::BoundaryAbsorb_kernel<>
 * @namespaces:
 *   - kernel::sr::
 */

#ifndef KERNELS_BOUNDARY_ABSORB_HPP
#define KERNELS_BOUNDARY_ABSORB_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

namespace kernel::sr {
  using namespace ntt;

  // Second half of the flux-conserving absorption: the pusher clamps
  // particles crossing a global absorbing x1 boundary onto the boundary face
  // (kernels/pushers/sr.hpp boundaryConditions) so CurrentsDeposit still
  // counts their final displacement; this kernel tags the clamped particles
  // dead. It runs after CurrentsDeposit and before CommunicateParticles. The
  // criteria mirror the clamp markers: dx1 == 1 at i1 == ni1 - 1 on the right
  // (a normal push normalizes dx1 to [0, 1), so this state is unique to
  // clamped particles) and dx1 == 0 at i1 == 0 on the left (a genuine
  // particle landing exactly on x1min is absorbed anyway, so the rare
  // collision is benign). Out-of-range i1 is kept as a defensive fallback.
  // The boundary classification (ABSORB or ATMOSPHERE) matches
  // kernel::sr::PusherBoundaries.
  template <Dimension D>
  struct BoundaryAbsorb_kernel {
    array_t<int*>      i1;
    array_t<prtldx_t*> dx1;
    array_t<short*>    tag;
    const int          ni1;
    const bool         absorb_min, absorb_max;

    Inline void operator()(prtlidx_t p) const {
      if (tag(p) != ParticleTag::alive) {
        return;
      }
      if (absorb_max and
          (i1(p) >= ni1 or
           (i1(p) == ni1 - 1 and dx1(p) >= static_cast<prtldx_t>(ONE)))) {
        tag(p) = ParticleTag::dead;
        return;
      }
      if (absorb_min and
          (i1(p) < 0 or
           (i1(p) == 0 and dx1(p) <= static_cast<prtldx_t>(ZERO)))) {
        tag(p) = ParticleTag::dead;
      }
    }
  };

} // namespace kernel::sr

#endif // KERNELS_BOUNDARY_ABSORB_HPP
