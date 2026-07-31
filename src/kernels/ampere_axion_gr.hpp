/**
 * @file kernels/ampere_axion_gr.hpp
 * @brief Axion background current contribution to Ampere's law in GR
 * @implements
 *   - kernel::gr::CurrentsAmpereAxion_kernel<>
 * @namespaces:
 *   - kernel::gr::
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_AMPERE_AXION_GR_HPP
#define KERNELS_AMPERE_AXION_GR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::gr {
  using namespace ntt;

  /**
   * @brief Adds the effective axion current
   * @brief   J_a = eps * ( dot_a * B + (grad a) x E )
   * @brief to the D field, `Df += coeff * J_a`, mimicking the staggering, the
   * @brief sqrt(det h) weighting and the AXIS treatment of CurrentsAmpere_kernel.
   * @brief The current is assembled as `sqrt(det h) * J_a` (same convention as the
   * @brief deposited particle current), so the Levi-Civita 1/sqrt(det h) inside the
   * @brief cross product cancels: all terms are plain interpolations of B, E and
   * @brief the analytically evaluated (at staggered points) axion derivatives.
   * @tparam M Metric.
   * @tparam A Axion background functor with `dot_a(x_Ph, t)`, `grad_a(x_Ph, t, g)`
   * @tparam (physical coordinates, physical covariant gradient) and member `eps`.
   * @tparam AuxMode true: B <- (Bf1 + Bf2) / 2 (J_a centered at n);
   * @tparam false: B <- Bf2 (J_a centered at n + 1/2).
   */
  template <GRMetricClass M, class A, bool AuxMode>
  class CurrentsAmpereAxion_kernel {
    static constexpr auto D = M::Dim;

    ndfield_t<D, 6>       Df;
    const ndfield_t<D, 6> Bf1;
    const ndfield_t<D, 6> Bf2;
    const ndfield_t<D, 6> Ef;
    const M               metric;
    const ncells_t        i2max;
    const real_t          coeff;
    const A               axion;
    const simtime_t       time;
    bool                  is_axis_i2min { false };
    bool                  is_axis_i2max { false };

  public:
    CurrentsAmpereAxion_kernel(const ndfield_t<D, 6>&      Df,
                               const ndfield_t<D, 6>&      Bf1,
                               const ndfield_t<D, 6>&      Bf2,
                               const ndfield_t<D, 6>&      Ef,
                               const M&                    metric,
                               real_t                      coeff,
                               const A&                    axion,
                               simtime_t                   time,
                               ncells_t                    ni2,
                               const boundaries_t<FldsBC>& boundaries)
      : Df { Df }
      , Bf1 { Bf1 }
      , Bf2 { Bf2 }
      , Ef { Ef }
      , metric { metric }
      , i2max { ni2 + N_GHOSTS }
      , coeff { coeff }
      , axion { axion }
      , time { time } {
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    /**
     * @brief B field component at the buffer location, optionally time-averaged.
     */
    Inline auto Bcomp(cellidx_t i1, cellidx_t i2, ntt::em c) const -> real_t {
      if constexpr (AuxMode) {
        return HALF * (Bf1(i1, i2, c) + Bf2(i1, i2, c));
      } else {
        return Bf2(i1, i2, c);
      }
    }

    /**
     * @brief axion derivatives at a code-coordinate point: dot(a) and the
     * @brief covariant gradient w.r.t. code coordinates.
     */
    Inline void axionDerivs(const coord_t<D>&      x_Cd,
                            real_t&                dot_a,
                            vec_t<Dim::_3D>&       grad_Cd) const {
      coord_t<D>      x_Ph { ZERO };
      vec_t<Dim::_3D> grad_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      dot_a = axion.dot_a(x_Ph, time);
      axion.grad_a(x_Ph, time, grad_Ph);
      metric.template transform<Idx::PD, Idx::D>(x_Cd, grad_Ph, grad_Cd);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        constexpr ncells_t i2min { N_GHOSTS };
        const real_t       i1_ { COORD(i1) };
        const real_t       i2_ { COORD(i2) };

        const real_t inv_sqrt_detH_0pH { ONE /
                                         metric.sqrt_det_h({ i1_, i2_ + HALF }) };

        // --- J_a^1 at (i1_ + 1/2, i2_)
        real_t          dot_a_pH0;
        vec_t<Dim::_3D> grad_pH0 { ZERO };
        axionDerivs({ i1_ + HALF, i2_ }, dot_a_pH0, grad_pH0);
        // B^1 at (i, j+1/2) -> 4-point average to (i+1/2, j)
        const real_t B1_pH0 { INV_4 *
                              (Bcomp(i1, i2 - 1, em::bx1) + Bcomp(i1, i2, em::bx1) +
                               Bcomp(i1 + 1, i2 - 1, em::bx1) +
                               Bcomp(i1 + 1, i2, em::bx1)) };
        // E_3 at (i, j) -> 2-point average to (i+1/2, j)
        const real_t E3_pH0 { HALF * (Ef(i1, i2, em::ex3) + Ef(i1 + 1, i2, em::ex3)) };
        // sqrt(h) * J_a^1 = sqrt(h) * dot_a * B^1 + d2(a) * E_3 - d3(a) * E_2
        const real_t Ja1 { metric.sqrt_det_h({ i1_ + HALF, i2_ }) * dot_a_pH0 *
                             B1_pH0 +
                           grad_pH0[1] * E3_pH0 };

        // --- J_a^2 at (i1_, i2_ + 1/2)
        real_t          dot_a_0pH;
        vec_t<Dim::_3D> grad_0pH { ZERO };
        axionDerivs({ i1_, i2_ + HALF }, dot_a_0pH, grad_0pH);
        // B^2 at (i+1/2, j) -> 4-point average to (i, j+1/2)
        const real_t B2_0pH { INV_4 *
                              (Bcomp(i1 - 1, i2, em::bx2) + Bcomp(i1, i2, em::bx2) +
                               Bcomp(i1 - 1, i2 + 1, em::bx2) +
                               Bcomp(i1, i2 + 1, em::bx2)) };
        // E_3 at (i, j) -> 2-point average to (i, j+1/2)
        const real_t E3_0pH { HALF * (Ef(i1, i2, em::ex3) + Ef(i1, i2 + 1, em::ex3)) };
        // sqrt(h) * J_a^2 = sqrt(h) * dot_a * B^2 + d3(a) * E_1 - d1(a) * E_3
        const real_t Ja2 { metric.sqrt_det_h({ i1_, i2_ + HALF }) * dot_a_0pH *
                             B2_0pH -
                           grad_0pH[0] * E3_0pH };

        if ((i2 == i2min) && is_axis_i2min) {
          // theta = 0 (first active cell)
          Df(i1, i2, em::dx1) += Ja1 * HALF * coeff / metric.polar_area(i1_ + HALF);
          Df(i1, i2, em::dx2) += Ja2 * coeff * inv_sqrt_detH_0pH;
        } else if ((i2 == i2max) && is_axis_i2max) {
          // theta = pi (first ghost cell from end)
          Df(i1, i2, em::dx1) += Ja1 * HALF * coeff / metric.polar_area(i1_ + HALF);
        } else {
          // 0 < theta < pi
          const real_t inv_sqrt_detH_00 { ONE / metric.sqrt_det_h({ i1_, i2_ }) };
          const real_t inv_sqrt_detH_pH0 { ONE / metric.sqrt_det_h(
                                                   { i1_ + HALF, i2_ }) };

          // --- J_a^3 at (i1_, i2_)
          real_t          dot_a_00;
          vec_t<Dim::_3D> grad_00 { ZERO };
          axionDerivs({ i1_, i2_ }, dot_a_00, grad_00);
          // B^3 at (i+1/2, j+1/2) -> 4-point average to (i, j)
          const real_t B3_00 { INV_4 *
                               (Bcomp(i1 - 1, i2 - 1, em::bx3) +
                                Bcomp(i1, i2 - 1, em::bx3) +
                                Bcomp(i1 - 1, i2, em::bx3) +
                                Bcomp(i1, i2, em::bx3)) };
          // E_1 at (i+1/2, j) -> 2-point average to (i, j)
          const real_t E1_00 { HALF *
                               (Ef(i1 - 1, i2, em::ex1) + Ef(i1, i2, em::ex1)) };
          // E_2 at (i, j+1/2) -> 2-point average to (i, j)
          const real_t E2_00 { HALF *
                               (Ef(i1, i2 - 1, em::ex2) + Ef(i1, i2, em::ex2)) };
          // sqrt(h) * J_a^3 = sqrt(h) * dot_a * B^3 + d1(a) * E_2 - d2(a) * E_1
          const real_t Ja3 { metric.sqrt_det_h({ i1_, i2_ }) * dot_a_00 * B3_00 +
                             grad_00[0] * E2_00 - grad_00[1] * E1_00 };

          Df(i1, i2, em::dx1) += Ja1 * coeff * inv_sqrt_detH_pH0;
          Df(i1, i2, em::dx2) += Ja2 * coeff * inv_sqrt_detH_0pH;
          Df(i1, i2, em::dx3) += Ja3 * coeff * inv_sqrt_detH_00;
        }
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpereAxion_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t, cellidx_t, cellidx_t) const {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(
          HERE,
          "CurrentsAmpereAxion_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::gr

#endif // KERNELS_AMPERE_AXION_GR_HPP
