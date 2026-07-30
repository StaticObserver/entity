#ifndef POLAR_CAP_BOUNDARY_FLUX_HPP
#define POLAR_CAP_BOUNDARY_FLUX_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h"

#include "qed/photon_opacity.hpp"

#include <Kokkos_Core.hpp>

namespace user::polar_cap {
  using namespace ntt;

  // Number of face slots in the missing-flux accumulator. Slot k holds the
  // current that boundary-absorbed particles would have deposited on the
  // active face (ni1 - 1 - k). The stencil of an order-O deposit reaches at
  // most O + 1 faces below the last active one; slot 0 alone is used by the
  // order-0 zigzag deposit.
  inline constexpr std::size_t BoundaryFluxAccSize = SHAPE_ORDER + 2;

  // Composite CustomParticleUpdate policy: forwards the photon opacity update
  // and, for charged species, accumulates the current that the current-deposit
  // kernel drops when the right ABSORB boundary kills a particle. The engine
  // pushes particles before depositing currents and the deposit kernel skips
  // dead particles, so the final displacement of an absorbed particle never
  // contributes to J. The pusher calls this policy after the position update
  // but before the boundary condition tags the particle dead, which is the
  // only point where the kill is predictable and i1_prev/dx1_prev are valid.
  template <MetricClass M>
  struct BoundaryFluxCompensationUpdate {
    static_assert(M::Dim == Dim::_1D,
                  "Boundary-flux compensation is implemented only for 1D");
    static_assert(M::CoordType == Coord::Cartesian,
                  "Boundary-flux compensation requires Cartesian coordinates");

    // Set only for charged species when the feature is enabled. The photon
    // species is excluded explicitly even though its zero charge would give a
    // zero contribution.
    const bool       compensate;
    array_t<real_t*> missing_flux;
    PhotonOpacityUpdate<M> photon_opacity;

    Inline void operator()(prtlidx_t                              p,
                           const kernel::sr::PusherContext&       context,
                           const kernel::sr::PusherBoundaries<M::Dim>& bc,
                           const ParticleArrays&                  particles,
                           const M&                               metric) const {
      photon_opacity(p, context, bc, particles, metric);
      if (not compensate) {
        return;
      }
      // Mirror the pusher's right-edge kill condition exactly
      // (sr.hpp boundaryConditions: is_absorb_i1max && i1 >= ni1 marks the
      // particle dead). Every particle killed there is accumulated here and
      // no other particle is.
      if (not bc.is_absorb_i1max or particles.i1(p) < context.ni1) {
        return;
      }
      // coeff * inv_dt from the deposit kernel, with coeff = weight * charge.
      const auto q_over_dt = particles.weight(p) *
                             static_cast<real_t>(context.charge) / context.dt;
      if constexpr (SHAPE_ORDER == 0) {
        // Zigzag deposit: the active-face share of the crossing step is
        // Fx1_1 = (1 - dx1_prev) * coeff * inv_dt (currents_deposit.hpp).
        Kokkos::atomic_add(&missing_flux(0),
                           (ONE - static_cast<real_t>(particles.dx1_prev(p))) *
                             q_over_dt);
      } else {
        // Esirkepov deposit: replay the cumulative 1D stencil current and keep
        // only the contributions landing on active faces (cell index <=
        // ni1 - 1). Faces past the global edge are ghost faces whose current
        // the simulation discards anyway.
        real_t iS[SHAPE_ORDER + 2], fS[SHAPE_ORDER + 2];
        int    i_min, i_max;
        prtl_shape::for_deposit<SHAPE_ORDER>(
          particles.i1_prev(p),
          static_cast<real_t>(particles.dx1_prev(p)),
          particles.i1(p),
          static_cast<real_t>(particles.dx1(p)),
          i_min,
          i_max,
          iS,
          fS);
        real_t jx1 = ZERO;
        for (int i = 0; i < i_max - i_min; ++i) {
          jx1 -= q_over_dt * (fS[i] - iS[i]);
          const auto slot = (context.ni1 - 1) - (i_min + i);
          if (slot >= 0 and slot < static_cast<int>(missing_flux.extent(0))) {
            Kokkos::atomic_add(&missing_flux(slot), jx1);
          }
        }
      }
    }
  };

  // Post-step counterpart: adds the accumulated missing current to the last
  // active faces with the same normalization CurrentsAmpere_kernel uses, i.e.
  // E_x += J_macro * coeff and J_stored += J_macro / ppc0, then the PGen
  // zeroes the accumulator.
  template <Dimension D>
  struct BoundaryFluxApplier {
    static_assert(D == Dim::_1D,
                  "Boundary-flux compensation is implemented only for 1D");

    ndfield_t<D, 6>  EB;
    ndfield_t<D, 3>  J;
    array_t<real_t*> missing_flux;
    // Field-array index of the last active face: ni1 - 1 + N_GHOSTS.
    const cellidx_t  last_active_face;
    // -dt * q0 / (B0 * V0), identical to CurrentsAmpere.
    const real_t     ampere_coeff;
    const real_t     inv_ppc0;

    Inline void operator()(ncells_t slot) const {
      const auto flux = missing_flux(slot);
      if (flux == ZERO) {
        return;
      }
      const auto face = last_active_face - slot;
      EB(face, em::ex1) += flux * ampere_coeff;
      J(face, cur::jx1) += flux * inv_ppc0;
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_BOUNDARY_FLUX_HPP
