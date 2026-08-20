#ifndef POLAR_CAP_BOUNDARY_DAMPING_HPP
#define POLAR_CAP_BOUNDARY_DAMPING_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>

namespace user::polar_cap {
  using namespace ntt;

  // Post-step multiplicative Ex damping inside the right MATCH layer, in the
  // spirit of Aperture4's damping boundary and TRISTAN-MP's absorb_ampere:
  // once per step, after the engine's Ampere/MATCH passes, scale Ex1 on the
  // last `cells` active faces of every transverse column by a cubic profile
  // that is strongest at the edge. With d the distance in active faces from
  // the last active face (d = 0), factor(d) = 1 - strength * ((cells - d) /
  // cells)^3, so factor(0) = 1 - strength and factor(cells) = 1; faces at
  // d >= cells are untouched. Only the longitudinal component is damped; the
  // engine's MATCH boundary keeps handling Bx1 alone.
  //
  // 2D generalization of the 1D damper: the parallel range covers every
  // active transverse cell of the local domain times the `cells` damping
  // faces. The i2 argument is already the field-array index (the host-side
  // range starts at i_min(x2) = N_GHOSTS), so no additional ghost offset is
  // applied here.
  template <Dimension D>
  struct BoundaryExDamper {
    static_assert(D == Dim::_2D,
                  "Boundary Ex damping is implemented only for 2D");
    // Cartesian coordinates are additionally required; that check lives in
    // the host-side PGen validation, which owns the metric type.

    ndfield_t<Dim::_2D, 6> EB;
    // Field-array index of the last active face: ni1 - 1 + N_GHOSTS.
    const cellidx_t last_active_face;
    const int       cells;
    const real_t    strength;

    Inline void operator()(cellidx_t i2, cellidx_t d) const {
      const auto s = static_cast<real_t>(cells - static_cast<int>(d)) /
                     static_cast<real_t>(cells);
      EB(last_active_face - static_cast<cellidx_t>(d), i2, em::ex1) *=
        ONE - strength * s * s * s;
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_BOUNDARY_DAMPING_HPP
