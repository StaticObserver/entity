#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "archetypes/spatial_dist.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

#include <string>
#include <vector>

namespace user {
  using namespace ntt;

  enum CustomField : uint8_t {
    DB           = 0,
    Gamma        = 1,
    V            = 2,
    Ut           = 3,
    N            = 4,
    StressEnergy = 5,
    EckartFlux   = 6,
  };

  // Particle-to-grid deposition of custom moments (Lorentz factor / number).
  // Mirrors kernel::ParticleMoments_kernel but allows custom per-particle weights.
  template <class M, CustomField F>
  class CustomMoments_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const unsigned short     comp;
    scatter_ndfield_t<D, 6>  Buff;
    const idx_t              buff_idx;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const float              charge;
    const M                  metric;
    const int                ni2;
    const unsigned short     window;
    const real_t             smooth;

    bool is_axis_i2min { false }, is_axis_i2max { false };

  public:
    CustomMoments_kernel(const unsigned short            comp,
                         const scatter_ndfield_t<D, 6>&  scatter_buff,
                         idx_t                           buff_idx,
                         const array_t<int*>&            i1,
                         const array_t<int*>&            i2,
                         const array_t<int*>&            i3,
                         const array_t<prtldx_t*>&       dx1,
                         const array_t<prtldx_t*>&       dx2,
                         const array_t<prtldx_t*>&       dx3,
                         const array_t<real_t*>&         ux1,
                         const array_t<real_t*>&         ux2,
                         const array_t<real_t*>&         ux3,
                         const array_t<real_t*>&         phi,
                         const array_t<real_t*>&         weight,
                         const array_t<short*>&          tag,
                         float                           mass,
                         float                           charge,
                         const M&                        metric,
                         const boundaries_t<FldsBC>&     boundaries,
                         ncells_t                        ni2,
                         real_t                          inv_n0,
                         unsigned short                  window)
      : comp { comp }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , charge { charge }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , window { window }
      , smooth { inv_n0 / (real_t)(math::pow(TWO * (real_t)window + ONE,
                                             static_cast<int>(D))) } {
      raise::ErrorIf(buff_idx >= 6, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      raise::ErrorIf(comp > 2, "Invalid component index", HERE);
      raise::ErrorIf(D != Dim::_2D, "CustomMoments_kernel only supports 2D", HERE);
      raise::ErrorIf(M::CoordType != Coord::Qsph,
                     "CustomMoments_kernel only supports Qspherical coordinates",
                     HERE);
      raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
      is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
    }

    Inline void operator()(index_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == CustomField::Gamma) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        vec_t<Dim::_3D> u_Cntrv { ZERO };

        metric.template transform<Idx::D, Idx::U>(x_Code,
                                                  { ux1(p), ux2(p), ux3(p) },
                                                  u_Cntrv);
        coeff = math::sqrt(ONE + u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) +
                           u_Cntrv[2] * ux3(p));
      }

      if constexpr (F == CustomField::N) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));

        coeff = ONE / metric.alpha(x_Code);
      }

      coeff *= weight(p);
      coeff *= smooth / metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                            static_cast<real_t>(i2(p)) + HALF });

      auto buff_access = Buff.access();

      if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            // reflect contribution at axes
            if (is_axis_i2min && (i2(p) + di2 < 0)) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          N_GHOSTS - (i2(p) + di2),
                          buff_idx) += coeff;
            } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                          buff_idx) += coeff;
            } else {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff;
            }
          }
        }
      }
    } // operator()

  }; // class CustomMoments_kernel

  template <class M, CustomField F>
  class CorrectedGRMoments_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(F == CustomField::StressEnergy || F == CustomField::EckartFlux,
                  "CorrectedGRMoments_kernel only supports GR T and Eckart flux");
    static constexpr auto D = M::Dim;

    const unsigned short     c1, c2;
    scatter_ndfield_t<D, 6>  Buff;
    const idx_t              buff_idx;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const bool               use_weights;
    const M                  metric;
    const int                ni2;
    const unsigned short     window;
    const real_t             smooth;

    bool is_axis_i2min { false }, is_axis_i2max { false };

  public:
    CorrectedGRMoments_kernel(const unsigned short            c1,
                              const unsigned short            c2,
                              const scatter_ndfield_t<D, 6>&  scatter_buff,
                              idx_t                           buff_idx,
                              const array_t<int*>&            i1,
                              const array_t<int*>&            i2,
                              const array_t<int*>&            i3,
                              const array_t<prtldx_t*>&       dx1,
                              const array_t<prtldx_t*>&       dx2,
                              const array_t<prtldx_t*>&       dx3,
                              const array_t<real_t*>&         ux1,
                              const array_t<real_t*>&         ux2,
                              const array_t<real_t*>&         ux3,
                              const array_t<real_t*>&         phi,
                              const array_t<real_t*>&         weight,
                              const array_t<short*>&          tag,
                              float                           mass,
                              bool                            use_weights,
                              const M&                        metric,
                              const boundaries_t<FldsBC>&     boundaries,
                              ncells_t                        ni2,
                              real_t                          inv_n0,
                              unsigned short                  window)
      : c1 { c1 }
      , c2 { c2 }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , use_weights { use_weights }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , window { window }
      , smooth { inv_n0 / (real_t)(math::pow(TWO * (real_t)window + ONE,
                                             static_cast<int>(D))) } {
      raise::ErrorIf(buff_idx >= 6, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      raise::ErrorIf(c1 > 3 || c2 > 3, "Invalid 4-vector component index", HERE);
      raise::ErrorIf(D != Dim::_2D, "CorrectedGRMoments_kernel only supports 2D", HERE);
      raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
      is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
    }

    Inline void corrected_4velocity(index_t          p,
                                    const coord_t<D>& x_Code,
                                    real_t&          u0,
                                    vec_t<Dim::_3D>& u_cntrv,
                                    vec_t<Dim::_3D>& u_phys) const {
      vec_t<Dim::_3D> u_space { ZERO };
      metric.template transform<Idx::D, Idx::U>(x_Code,
                                                { ux1(p), ux2(p), ux3(p) },
                                                u_space);

      const real_t gamma_sq { ((mass == ZERO) ? ZERO : ONE) +
                              u_space[0] * ux1(p) + u_space[1] * ux2(p) +
                              u_space[2] * ux3(p) };
      const real_t gamma { math::sqrt(gamma_sq) };
      u0          = gamma / metric.alpha(x_Code);
      u_cntrv[0]  = u_space[0] - metric.beta1(x_Code) * u0;
      u_cntrv[1]  = u_space[1];
      u_cntrv[2]  = u_space[2];
      metric.template transform<Idx::U, Idx::PU>(x_Code, u_cntrv, u_phys);
    }

    Inline auto stress_energy_component(index_t p) const -> real_t {
      coord_t<D> x_Code { ZERO };
      x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
      x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));

      real_t          u0 { ZERO };
      vec_t<Dim::_3D> u_cntrv { ZERO };
      vec_t<Dim::_3D> u_phys { ZERO };
      corrected_4velocity(p, x_Code, u0, u_cntrv, u_phys);

      const real_t m { (mass == ZERO) ? ONE : static_cast<real_t>(mass) };
      const real_t u_c1 { (c1 == 0) ? u0 : u_phys[c1 - 1] };
      const real_t u_c2 { (c2 == 0) ? u0 : u_phys[c2 - 1] };
      return m * u_c1 * u_c2 / u0;
    }

    Inline auto eckart_flux_component(index_t p) const -> real_t {
      coord_t<D> x_Code { ZERO };
      x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
      x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));

      real_t          u0 { ZERO };
      vec_t<Dim::_3D> u_cntrv { ZERO };
      vec_t<Dim::_3D> u_phys { ZERO };
      corrected_4velocity(p, x_Code, u0, u_cntrv, u_phys);

      const real_t m { (mass == ZERO) ? ONE : static_cast<real_t>(mass) };
      if (c1 == 0) {
        return m;
      }
      return m * u_cntrv[c1 - 1] / u0;
    }

    Inline void operator()(index_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }

      real_t coeff { ZERO };
      if constexpr (F == CustomField::StressEnergy) {
        coeff = stress_energy_component(p);
      } else {
        coeff = eckart_flux_component(p);
      }

      coeff *= smooth / metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                            static_cast<real_t>(i2(p)) + HALF });
      if (use_weights) {
        coeff *= weight(p);
      }

      auto buff_access = Buff.access();
      if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            if (is_axis_i2min && (i2(p) + di2 < 0)) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          N_GHOSTS - (i2(p) + di2),
                          buff_idx) += coeff;
            } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                          buff_idx) += coeff;
            } else {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff;
            }
          }
        }
      }
    }
  };

  template <class M>
  class NormalizeEckartVelocity_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D = M::Dim;

    const ndfield_t<D, 6> Flux;
    ndfield_t<D, 6>       Vector;
    const idx_t            vector_idx;
    const unsigned short   comp;
    const M                metric;

  public:
    NormalizeEckartVelocity_kernel(const ndfield_t<D, 6>& flux,
                                   const ndfield_t<D, 6>& vector,
                                   idx_t                  vector_idx,
                                   unsigned short         comp,
                                   const M&               metric)
      : Flux { flux }
      , Vector { vector }
      , vector_idx { vector_idx }
      , comp { comp }
      , metric { metric } {
      raise::ErrorIf(vector_idx >= 6, "Invalid buffer index", HERE);
      raise::ErrorIf(comp > 3, "Invalid Eckart velocity component", HERE);
      raise::ErrorIf(D != Dim::_2D, "NormalizeEckartVelocity_kernel only supports 2D", HERE);
    }

    Inline auto norm_sq(const coord_t<D>& x_Code,
                        real_t           n0,
                        real_t           n1,
                        real_t           n2,
                        real_t           n3) const -> real_t {
      const real_t beta1 { metric.beta1(x_Code) };
      const real_t g00 { -SQR(metric.alpha(x_Code)) +
                         metric.template h_<1, 1>(x_Code) * SQR(beta1) };
      const real_t g01 { metric.template h_<1, 1>(x_Code) * beta1 };
      const real_t g03 { metric.template h_<1, 3>(x_Code) * beta1 };

      return g00 * SQR(n0) + TWO * g01 * n0 * n1 + TWO * g03 * n0 * n3 +
             metric.template h_<1, 1>(x_Code) * SQR(n1) +
             metric.template h_<2, 2>(x_Code) * SQR(n2) +
             metric.template h_<3, 3>(x_Code) * SQR(n3) +
             TWO * metric.template h_<1, 3>(x_Code) * n1 * n3;
    }

    Inline void zamo_fallback(index_t i1, index_t i2, const coord_t<D>& x_Code) const {
      const real_t al { metric.alpha(x_Code) };
      if (comp == 0) {
        Vector(i1, i2, vector_idx) = ONE / al;
      } else if (comp == 1) {
        Vector(i1, i2, vector_idx) = -metric.beta1(x_Code) / al;
      } else {
        Vector(i1, i2, vector_idx) = ZERO;
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x_Code { COORD(i1) + HALF, COORD(i2) + HALF };

        const real_t n0 { Flux(i1, i2, 0) };
        const real_t n1 { Flux(i1, i2, 1) };
        const real_t n2 { Flux(i1, i2, 2) };
        const real_t n3 { Flux(i1, i2, 3) };

        if (cmp::AlmostZero(n0) || !math::isfinite(n0)) {
          zamo_fallback(i1, i2, x_Code);
          return;
        }

        const real_t nsq { norm_sq(x_Code, n0, n1, n2, n3) };
        if (!(nsq < ZERO) || !math::isfinite(nsq)) {
          zamo_fallback(i1, i2, x_Code);
          return;
        }

        const real_t norm { math::sqrt(-nsq) };
        if (cmp::AlmostZero(norm) || !math::isfinite(norm)) {
          zamo_fallback(i1, i2, x_Code);
          return;
        }

        if (comp == 0) {
          Vector(i1, i2, vector_idx) = n0 / norm;
        } else if (comp == 1) {
          Vector(i1, i2, vector_idx) = n1 / norm;
        } else if (comp == 2) {
          Vector(i1, i2, vector_idx) = n2 / norm;
        } else {
          Vector(i1, i2, vector_idx) = n3 / norm;
        }
      }
    }
  };

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_, real_t m_eps_) : metric { metric_ }, m_eps { m_eps_ } {}

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_1(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<1, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 1>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    Inline auto A_0(const coord_t<D>& x_Cd) const -> real_t {
      real_t g_00 { -metric.alpha(x_Cd) * metric.alpha(x_Cd) +
                    metric.template h_<1, 1>(x_Cd) * metric.beta1(x_Cd) *
                      metric.beta1(x_Cd) };
      return HALF * (metric.template h_<1, 3>(x_Cd) * metric.beta1(x_Cd) +
                     TWO * metric.spin() * g_00);
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (cmp::AlmostZero(x_Ph[1])) {
        return ONE;
      } else {
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (cmp::AlmostZero(x_Ph[1])) {
        return ZERO;
      } else {
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP / m_eps;
      }
    }

    Inline auto bx3(
      const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;

      real_t inv_sqrt_detH_iPjP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      return -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_iPjP / m_eps;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      real_t alpha_iPj { metric.alpha({ xi[0], xi[1] }) };
      real_t beta_ij { metric.beta1({ xi[0] - HALF, xi[1] }) };
      real_t alpha_ij { metric.alpha({ xi[0] - HALF, xi[1] }) };

      // D1 at ( i + HALF , j )
      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];
      real_t E1d { (A_0(x0p) - A_0(x0m)) / m_eps };
      real_t D1d { E1d / alpha_iPj };

      // D3 at ( i , j )
      x0m[0] = xi[0] - HALF - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] - HALF + HALF * m_eps;
      x0p[1] = xi[1];
      real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij / m_eps };

      real_t D1u { metric.template h<1, 1>({ xi[0], xi[1] }) * D1d +
                   metric.template h<1, 3>({ xi[0], xi[1] }) * D3d };

      return D1u;
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;
      real_t alpha_ijP { metric.alpha({ xi[0], xi[1] }) };
      real_t beta_ijP { metric.beta1({ xi[0], xi[1] }) };

      real_t E2d { (A_0(x0p) - A_0(x0m)) / m_eps };
      real_t D2d { E2d / alpha_ijP -
                   (A_1(x0p) - A_1(x0m)) * beta_ijP / alpha_ijP / m_eps };
      real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

      return D2u;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
      real_t beta_ij { metric.beta1({ xi[0], xi[1] }) };
      real_t alpha_ij { metric.alpha({ xi[0], xi[1] }) };
      real_t alpha_iPj { metric.alpha({ xi[0] + HALF, xi[1] }) };

      // D3 at ( i , j )
      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];
      real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij / m_eps };

      // D1 at ( i + HALF , j )
      x0m[0] = xi[0] + HALF - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF + HALF * m_eps;
      x0p[1] = xi[1];
      real_t E1d { (A_0(x0p) - A_0(x0m)) / m_eps };
      real_t D1d { E1d / alpha_iPj };

      if (cmp::AlmostZero(x_Ph[1])) {
        return metric.template h<1, 3>({ xi[0], xi[1] }) * D1d;
      } else {
        return metric.template h<3, 3>({ xi[0], xi[1] }) * D3d +
               metric.template h<1, 3>({ xi[0], xi[1] }) * D1d;
      }
    }

  private:
    const M      metric;
    const real_t m_eps;
  };

  // Spatial distribution for sigma-driven pair injection. When `Weighted` is
  // true the functor returns a {fill, inj_weight} pair so that the injector can
  // assign a position-dependent particle weight.
  template <SimEngine::type S, class M, bool Weighted>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    static_assert(M::is_metric, "M must be a metric class");
    using arch::SpatialDistribution<S, M>::metric;

    PointDistribution(const std::vector<real_t>& xi_min,
                      const std::vector<real_t>& xi_max,
                      const real_t               sigma_thr,
                      const real_t               inj_coeff,
                      const real_t               db_thr,
                      const SimulationParams&    params,
                      Domain<S, M>*              domain_ptr)
      : arch::SpatialDistribution<S, M> { domain_ptr->mesh.metric }
      , EM { domain_ptr->fields.em }
      , density { domain_ptr->fields.buff }
      , sigma_thr { sigma_thr }
      , db_thr { db_thr }
      , inj_coeff { inj_coeff }
      , inv_n0 { ONE / params.template get<real_t>("scales.n0") }
      , d0 { params.template get<real_t>("scales.skindepth0") }
      , rho0 { params.template get<real_t>("scales.larmor0") } {
      std::copy(xi_min.begin(), xi_min.end(), x_min);
      std::copy(xi_max.begin(), xi_max.end(), x_max);

      std::vector<unsigned short> specs {};
      for (auto& sp : domain_ptr->species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }

      Kokkos::deep_copy(density, ZERO);
      auto       scatter_buff = Kokkos::Experimental::create_scatter_view(density);
      auto&      mesh         = domain_ptr->mesh;
      const auto use_weights  = params.template get<bool>("particles.use_weights");
      const auto ni2          = mesh.n_active(in::x2);

      for (const auto& sp : specs) {
        auto& prtl_spec = domain_ptr->species[sp - 1];
        // clang-format off
        Kokkos::parallel_for(
          "ComputeMoments",
          prtl_spec.rangeActiveParticles(),
          kernel::ParticleMoments_kernel<S, M, FldsID::Rho, 3>({}, scatter_buff, 0u,
                                                               prtl_spec.i1, prtl_spec.i2, prtl_spec.i3,
                                                               prtl_spec.dx1, prtl_spec.dx2, prtl_spec.dx3,
                                                               prtl_spec.ux1, prtl_spec.ux2, prtl_spec.ux3,
                                                               prtl_spec.phi, prtl_spec.weight, prtl_spec.tag,
                                                               prtl_spec.mass(), prtl_spec.charge(),
                                                               use_weights,
                                                               metric, mesh.flds_bc(),
                                                               ni2, inv_n0, TWO));
        // clang-format on
      }
      Kokkos::Experimental::contribute(density, scatter_buff);
    }

    Inline auto sigma_crit(const coord_t<M::Dim>& x_Ph) const -> bool {
      coord_t<M::Dim> xi { ZERO };
      if constexpr (M::Dim == Dim::_2D) {
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        const auto            i1 = static_cast<int>(xi[0]) + static_cast<int>(N_GHOSTS);
        const auto            i2 = static_cast<int>(xi[1]) + static_cast<int>(N_GHOSTS);
        const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                        EM(i1, i2, em::bx2),
                                        EM(i1, i2, em::bx3) };
        vec_t<Dim::_3D>       B_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
        const auto bsqr =
          DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto dens = density(i1, i2, 0);
        return (bsqr > sigma_thr * dens) || (dens < db_thr);
      }
      return false;
    }

    Inline auto weighted(const coord_t<M::Dim>& x_Ph) const
      -> Kokkos::pair<real_t, real_t> {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d] and sigma_crit(x_Ph);
      }
      const auto inj_n = fill ? inj_coeff * SQR(d0) / rho0 / x_Ph[0] *
                                  math::sqrt(x_Ph[0])
                              : ZERO;

      return { fill ? ONE : ZERO, inj_n };
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const {
      if constexpr (Weighted) {
        return weighted(x_Ph);
      } else {
        return density_fraction(x_Ph);
      }
    }

  private:
    Inline auto density_fraction(const coord_t<M::Dim>& x_Ph) const -> real_t {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d] and sigma_crit(x_Ph);
      }
      return fill ? ONE : ZERO;
    }

    tuple_t<real_t, M::Dim> x_min;
    tuple_t<real_t, M::Dim> x_max;
    const real_t            sigma_thr;
    const real_t            db_thr;
    const real_t            inj_coeff;
    const real_t            inv_n0;
    const real_t            d0;
    const real_t            rho0;
    ndfield_t<M::Dim, 3>    density;
    ndfield_t<M::Dim, 6>    EM;
  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines { traits::compatible_with<SimEngine::GRPIC>::value };
    static constexpr auto metrics {
      traits::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0>::value
    };
    static constexpr auto dimensions { traits::compatible_with<Dim::_2D>::value };

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const real_t sigma0, sigma_max, inj_coeff, db_thr, temperature, m_eps, inv_n0;

    InitFields<M, D>        init_flds;
    const Metadomain<S, M>* metadomain;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , xi_min { p.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { p.template get<std::vector<real_t>>("setup.xi_max") }
      , sigma0 { p.template get<real_t>("scales.sigma0") }
      , sigma_max { p.template get<real_t>("setup.sigma_max") }
      , inj_coeff { p.template get<real_t>("setup.inj_coeff") }
      , db_thr { p.template get<real_t>("setup.db_thr") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , m_eps { p.template get<real_t>("setup.m_eps") }
      , inv_n0 { ONE / p.template get<real_t>("scales.n0") }
      , init_flds { m.mesh().metric, m_eps }
      , metadomain { &m } {}

    void CustomPostStep(std::size_t, long double, Domain<S, M>& local_domain) {
      const auto energy_dist  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                      local_domain.random_pool(),
                                                      temperature);
      const auto spatial_dist = PointDistribution<S, M, true>(xi_min,
                                                              xi_max,
                                                              sigma_max / sigma0,
                                                              inj_coeff,
                                                              db_thr,
                                                              params,
                                                              &local_domain);

      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(spatial_dist)>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        spatial_dist,
        ONE,
        true);
    }

    void CustomFieldOutput(const std::string&    name,
                           ndfield_t<M::Dim, 6>& buffer,
                           index_t               index,
                           timestep_t,
                           simtime_t,
                           const Domain<S, M>&   domain) {
      if (name == "DB") {
        if constexpr (M::Dim == Dim::_2D) {
          const auto& EM     = domain.fields.em;
          const auto& metric = domain.mesh.metric;
          Kokkos::parallel_for(
            "DB",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2) {
              coord_t<M::Dim>       xi { static_cast<real_t>(i1 - N_GHOSTS),
                                   static_cast<real_t>(i2 - N_GHOSTS) };
              const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                              EM(i1, i2, em::bx2),
                                              EM(i1, i2, em::bx3) };
              vec_t<Dim::_3D>       B_cov { ZERO };
              const vec_t<Dim::_3D> D_cntrv { EM(i1, i2, em::dx1),
                                              EM(i1, i2, em::dx2),
                                              EM(i1, i2, em::dx3) };
              metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
              buffer(i1, i2, index) =
                DOT(B_cov[0], B_cov[1], B_cov[2], D_cntrv[0], D_cntrv[1], D_cntrv[2]) /
                DOT(B_cov[0], B_cov[1], B_cov[2], B_cntrv[0], B_cntrv[1], B_cntrv[2]);
            });
        }
      } else if (name == "Gamma_1" || name == "Gamma_2") {
        const auto sp_idx = (name == "Gamma_1") ? 0 : 1;
        auto&      sp     = domain.species[sp_idx];
        if constexpr (M::Dim == Dim::_2D) {
          auto        scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
          const auto& metric       = domain.mesh.metric;
          auto&       mesh         = domain.mesh;
          const auto  ni2          = mesh.n_active(in::x2);
          // clang-format off
          Kokkos::parallel_for(
            name,
            sp.rangeActiveParticles(),
            CustomMoments_kernel<M, CustomField::Gamma>(0, scatter_buff, index,
                                                        sp.i1, sp.i2, sp.i3,
                                                        sp.dx1, sp.dx2, sp.dx3,
                                                        sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                        sp.weight, sp.tag, sp.mass(), sp.charge(),
                                                        metric, mesh.flds_bc(), ni2, inv_n0, ZERO));
          Kokkos::Experimental::contribute(buffer, scatter_buff);

          auto n_buffer       = domain.fields.buff;
          Kokkos::deep_copy(n_buffer, ZERO);
          auto scatter_buff_n = Kokkos::Experimental::create_scatter_view(n_buffer);
          Kokkos::parallel_for(
            "ComputeMoments",
            sp.rangeActiveParticles(),
            kernel::ParticleMoments_kernel<S, M, FldsID::N, 3>({}, scatter_buff_n, 0u,
                                                                 sp.i1, sp.i2, sp.i3,
                                                                 sp.dx1, sp.dx2, sp.dx3,
                                                                 sp.ux1, sp.ux2, sp.ux3,
                                                                 sp.phi, sp.weight, sp.tag,
                                                                 sp.mass(), sp.charge(),
                                                                 true,
                                                                 metric, mesh.flds_bc(),
                                                                 ni2, inv_n0, ZERO));
          Kokkos::Experimental::contribute(n_buffer, scatter_buff_n);

          Kokkos::parallel_for(
            "NormalizeGamma",
            mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2) {
              if (cmp::AlmostZero(n_buffer(i1, i2, 0))) {
                buffer(i1, i2, index) = ZERO;
              } else {
                buffer(i1, i2, index) /= n_buffer(i1, i2, 0);
              }
            });
          // clang-format on
        }
      } else if (name.rfind("GR_T", 0) == 0) {
        const auto sep = name.find_last_of('_');
        raise::ErrorIf(sep == std::string::npos || sep + 1 >= name.size(),
                       "Invalid corrected stress-energy output name",
                       HERE);
        raise::ErrorIf(name.size() < 8,
                       "Invalid corrected stress-energy output name",
                       HERE);
        const auto c1     = static_cast<unsigned short>(name[4] - '0');
        const auto c2     = static_cast<unsigned short>(name[5] - '0');
        const auto sp_idx = static_cast<spidx_t>(std::stoi(name.substr(sep + 1))) - 1;
        raise::ErrorIf(sp_idx >= domain.species.size(),
                       "Invalid species in corrected stress-energy output",
                       HERE);
        auto& sp = domain.species[sp_idx];
        if constexpr (M::Dim == Dim::_2D) {
          auto        scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
          const auto& metric       = domain.mesh.metric;
          auto&       mesh         = domain.mesh;
          const auto  ni2          = mesh.n_active(in::x2);
          const auto  use_weights  = params.template get<bool>("particles.use_weights");
          // clang-format off
          Kokkos::parallel_for(
            name,
            sp.rangeActiveParticles(),
            CorrectedGRMoments_kernel<M, CustomField::StressEnergy>(c1, c2, scatter_buff, index,
                                                                     sp.i1, sp.i2, sp.i3,
                                                                     sp.dx1, sp.dx2, sp.dx3,
                                                                     sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                                     sp.weight, sp.tag, sp.mass(),
                                                                     use_weights,
                                                                     metric, mesh.flds_bc(), ni2,
                                                                     inv_n0, ZERO));
          Kokkos::Experimental::contribute(buffer, scatter_buff);
          // clang-format on
        }
      } else if (name.rfind("GR_U", 0) == 0) {
        const auto sep = name.find_last_of('_');
        raise::ErrorIf(sep == std::string::npos || sep + 1 >= name.size(),
                       "Invalid Eckart velocity output name",
                       HERE);
        raise::ErrorIf(name.size() < 7,
                       "Invalid Eckart velocity output name",
                       HERE);
        const auto comp   = static_cast<unsigned short>(name[4] - '0');
        const auto sp_idx = static_cast<spidx_t>(std::stoi(name.substr(sep + 1))) - 1;
        raise::ErrorIf(comp > 3, "Invalid Eckart velocity component", HERE);
        raise::ErrorIf(sp_idx >= domain.species.size(),
                       "Invalid species in Eckart velocity output",
                       HERE);
        auto& sp = domain.species[sp_idx];
        if constexpr (M::Dim == Dim::_2D) {
          const auto& metric      = domain.mesh.metric;
          auto&       mesh        = domain.mesh;
          const auto  ni2         = mesh.n_active(in::x2);
          const auto  use_weights = params.template get<bool>("particles.use_weights");

          Kokkos::deep_copy(buffer, ZERO);
          auto scatter_flux = Kokkos::Experimental::create_scatter_view(buffer);
          // clang-format off
          Kokkos::parallel_for(
            name + "_flux0",
            sp.rangeActiveParticles(),
            CorrectedGRMoments_kernel<M, CustomField::EckartFlux>(0, 0, scatter_flux, 0,
                                                                  sp.i1, sp.i2, sp.i3,
                                                                  sp.dx1, sp.dx2, sp.dx3,
                                                                  sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                                  sp.weight, sp.tag, sp.mass(),
                                                                  use_weights,
                                                                  metric, mesh.flds_bc(), ni2,
                                                                  inv_n0, ZERO));
          Kokkos::parallel_for(
            name + "_flux1",
            sp.rangeActiveParticles(),
            CorrectedGRMoments_kernel<M, CustomField::EckartFlux>(1, 0, scatter_flux, 1,
                                                                  sp.i1, sp.i2, sp.i3,
                                                                  sp.dx1, sp.dx2, sp.dx3,
                                                                  sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                                  sp.weight, sp.tag, sp.mass(),
                                                                  use_weights,
                                                                  metric, mesh.flds_bc(), ni2,
                                                                  inv_n0, ZERO));
          Kokkos::parallel_for(
            name + "_flux2",
            sp.rangeActiveParticles(),
            CorrectedGRMoments_kernel<M, CustomField::EckartFlux>(2, 0, scatter_flux, 2,
                                                                  sp.i1, sp.i2, sp.i3,
                                                                  sp.dx1, sp.dx2, sp.dx3,
                                                                  sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                                  sp.weight, sp.tag, sp.mass(),
                                                                  use_weights,
                                                                  metric, mesh.flds_bc(), ni2,
                                                                  inv_n0, ZERO));
          Kokkos::parallel_for(
            name + "_flux3",
            sp.rangeActiveParticles(),
            CorrectedGRMoments_kernel<M, CustomField::EckartFlux>(3, 0, scatter_flux, 3,
                                                                  sp.i1, sp.i2, sp.i3,
                                                                  sp.dx1, sp.dx2, sp.dx3,
                                                                  sp.ux1, sp.ux2, sp.ux3, sp.phi,
                                                                  sp.weight, sp.tag, sp.mass(),
                                                                  use_weights,
                                                                  metric, mesh.flds_bc(), ni2,
                                                                  inv_n0, ZERO));
          Kokkos::Experimental::contribute(buffer, scatter_flux);

          Kokkos::parallel_for(
            name,
            mesh.rangeActiveCells(),
            NormalizeEckartVelocity_kernel<M>(buffer, buffer, index, comp, metric));
          // clang-format on
        }
      } else {
        raise::Error("Custom output not provided", HERE);
      }
    }
  };

} // namespace user

#endif
