#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/comparators.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

#include "kernels/particle_moments.hpp"

#include <vector>

namespace user {
  using namespace ntt;

  enum CustomField : uint8_t {
    DB = 0,
    Gamma = 1,
    V = 2,
    Ut = 3,
  };

  template <class M, CustomField F>
  class CustomMoments_kernel{
  static_assert(M::is_metric, "M must be a metric class");
  static constexpr auto D = M::Dim;

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
  const unsigned short     comp;

  const real_t smooth;
  bool  is_axis_i2min { false }, is_axis_i2max { false };

  public:
  CustomMoments_kernel( const unsigned short comp,
                        const scatter_ndfield_t<D, 6>&     scatter_buff,
                        idx_t                              buff_idx,
                        const array_t<int*>&               i1,
                        const array_t<int*>&               i2,
                        const array_t<int*>&               i3,
                        const array_t<prtldx_t*>&          dx1,
                        const array_t<prtldx_t*>&          dx2,
                        const array_t<prtldx_t*>&          dx3,
                        const array_t<real_t*>&            ux1,
                        const array_t<real_t*>&            ux2,
                        const array_t<real_t*>&            ux3,
                        const array_t<real_t*>&            phi,
                        const array_t<real_t*>&            weight,
                        const array_t<short*>&             tag,
                        float                              mass,
                        float                              charge,
                        const M&                           metric,
                        const boundaries_t<FldsBC>&        boundaries,
                        ncells_t                           ni2,
                        real_t                             inv_n0,
                        unsigned short                     window)
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

      raise::ErrorIf(comp > 2 || comp < 0, "Invalid component index", HERE);

      raise::ErrorIf(D != Dim::_2D, "CustomMoments_kernel only supports 2D", HERE);
      raise::ErrorIf(M::CoordType != Coord::Qsph, "CustomMoments_kernel only supports Qspherical coordinates", HERE);

      raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
      is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
      is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
    }

    Inline void operator()(index_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == CustomField::Gamma){

        coord_t<D> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        vec_t<Dim::_3D> u_Cntrv { ZERO };

        metric.template transform<Idx::D, Idx::U>(x_Code,
                                                  { ux1(p), ux2(p), ux3(p) },
                                                  u_Cntrv);
        coeff = math::sqrt(ONE + u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) + u_Cntrv[2] * ux3(p));
      }

      if constexpr (F == CustomField::V){
        coord_t<D> x_Code { ZERO };
        real_t gamma { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        vec_t<Dim::_3D> u_Cntrv { ZERO };
        vec_t<Dim::_3D> u_Phys { ZERO };
        metric.template transform<Idx::D, Idx::U>(x_Code,
                                                  { ux1(p), ux2(p), ux3(p) },
                                                  u_Cntrv);
        gamma = math::sqrt(ONE + u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) + u_Cntrv[2] * ux3(p));
        
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_Cntrv, u_Phys);
        vec_t<Dim::_3D> beta_Phys { ZERO };
        metric.template transform<Idx::U, Idx::PU>(x_Code, 
                                                   { metric.beta1(x_Code), 0, 0 }, 
                                                   beta_Phys);
        coeff = u_Phys[comp] / gamma * metric.alpha(x_Code) 
                - beta_Phys[0] * static_cast<real_t>(comp == 0);
      }

      if constexpr (F == CustomField::Ut){
        coord_t<D> x_Code { ZERO };
        real_t gamma { ZERO };
        x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
        x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
        vec_t<Dim::_3D> u_Cntrv { ZERO };
        metric.template transform<Idx::D, Idx::U>(x_Code,
                                                  { ux1(p), ux2(p), ux3(p) },
                                                  u_Cntrv);
        gamma = math::sqrt(ONE + u_Cntrv[0] * ux1(p) + u_Cntrv[1] * ux2(p) + u_Cntrv[2] * ux3(p));
        coeff = -metric.alpha(x_Code) * gamma + metric.beta1(x_Code) * ux1(p);
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
      real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
      real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) }; 
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
      real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij / m_eps};

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
      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      real_t sqrt_detH_ijP { metric.sqrt_det_h({ xi[0], xi[1] }) };
      real_t alpha_ijP { metric.alpha({ xi[0], xi[1] }) };
      real_t beta_ijP { metric.beta1({ xi[0], xi[1] }) };

      real_t E2d { (A_0(x0p) - A_0(x0m)) / m_eps };
      real_t D2d { E2d / alpha_ijP - (A_1(x0p) - A_1(x0m)) * beta_ijP / alpha_ijP / m_eps};
      real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

      return D2u;
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
      real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0], xi[1] }) };
      real_t beta_ij { metric.beta1({ xi[0], xi[1] }) };
      real_t alpha_ij { metric.alpha({ xi[0], xi[1] }) };
      real_t alpha_iPj { metric.alpha({ xi[0] + HALF, xi[1] }) };

      // D3 at ( i , j )
      x0m[0] = xi[0] - HALF * m_eps;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF * m_eps;
      x0p[1] = xi[1];
      real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij / m_eps};

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
    const M metric;
    const real_t m_eps;
  };

  template <SimEngine::type S, class M>
  struct PointDistribution : public arch::SpatialDistribution<S, M> {
    PointDistribution(const std::vector<real_t>& xi_min,
                      const std::vector<real_t>& xi_max,
                      const real_t               sigma_thr,
                      const real_t               inj_coeff,
                      const real_t               db_thr,
                      const bool                 is_weight, 
                      const SimulationParams&    params,
                      Domain<S, M>*              domain_ptr)
      : arch::SpatialDistribution<S, M> { domain_ptr->mesh.metric }
      , metric { domain_ptr->mesh.metric }
      , EM { domain_ptr->fields.em }
      , density { domain_ptr->fields.buff }
      , sigma_thr { sigma_thr }
      , db_thr { db_thr }
      , d0 { params.template get<real_t>("scales.skindepth0") }
      , rho0 { params.template get<real_t>("scales.larmor0") }
      , inv_n0 { ONE / params.template get<real_t>("scales.n0") }
      , ppc0 { params.template get<real_t>("particles.ppc0") }
      , inj_coeff { inj_coeff }
      , is_weight { is_weight } {
      std::copy(xi_min.begin(), xi_min.end(), x_min);
      std::copy(xi_max.begin(), xi_max.end(), x_max);

      std::vector<unsigned short> specs {};
      for (auto& sp : domain_ptr->species) {
        if (sp.mass() > 0) {
          specs.push_back(sp.index());
        }
      }

      Kokkos::deep_copy(density, ZERO);
      auto  scatter_buff = Kokkos::Experimental::create_scatter_view(density);
      // some parameters
      auto& mesh         = domain_ptr->mesh;
      const auto use_weights = params.template get<bool>(
        "particles.use_weights");
      const auto ni2 = mesh.n_active(in::x2);

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
                                                               ni2, inv_n0, ZERO));
        // clang-format on
      }
      Kokkos::Experimental::contribute(density, scatter_buff);
    }

    Inline auto sigma_crit(const coord_t<M::Dim>& x_Ph) const -> bool {
      coord_t<M::Dim> xi { ZERO };
      if constexpr (M::Dim == Dim::_2D) {
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        const auto i1 = static_cast<int>(xi[0]) + static_cast<int>(N_GHOSTS);
        const auto i2 = static_cast<int>(xi[1]) + static_cast<int>(N_GHOSTS);
        const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                        EM(i1, i2, em::bx2),
                                        EM(i1, i2, em::bx3) };
        const vec_t<Dim::_3D> D_cntrv { EM(i1, i2, em::dx1),
                                        EM(i1, i2, em::dx2),
                                        EM(i1, i2, em::dx3) };
        vec_t<Dim::_3D>       B_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
        const auto bsqr =
          DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto db = DOT(D_cntrv[0], D_cntrv[1], D_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto dens = density(i1, i2, 0);
        return (bsqr > sigma_thr * dens); // && (db * SIGN(db) > db_thr * bsqr);
      }
      return false;
    }

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
      auto fill = true;
      for (auto d = 0u; d < M::Dim; ++d) {
        fill &= x_Ph[d] > x_min[d] and x_Ph[d] < x_max[d] and sigma_crit(x_Ph);
      }
      if (is_weight) {
        coord_t<M::Dim> xi { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        // const auto i1 = static_cast<int>(xi[0]) + static_cast<int>(N_GHOSTS);
        // const auto i2 = static_cast<int>(xi[1]) + static_cast<int>(N_GHOSTS);
        // const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
        //                                 EM(i1, i2, em::bx2),
        //                                 EM(i1, i2, em::bx3) };
        // const vec_t<Dim::_3D> D_cntrv { EM(i1, i2, em::dx1),
        //                                 EM(i1, i2, em::dx2),
        //                                 EM(i1, i2, em::dx3) };
        // vec_t<Dim::_3D>       B_cov { ZERO };
        // metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
        // const auto bsqr =
        //   DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        // const auto db = DOT(D_cntrv[0], D_cntrv[1], D_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        // const real_t inj_n = inj_coeff * db * SIGN(db) / math::sqrt(bsqr) * SQR(d0) / rho0;
        vec_t<Dim::_3D> x_cntrv { x_Ph[0], x_Ph[1], ZERO };
        vec_t<Dim::_3D> x_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(xi, x_cntrv, x_cov);
        const auto rsqr = DOT(x_cntrv[0], x_cntrv[1], x_cntrv[2], x_cov[0], x_cov[1], x_cov[2]);
        const auto inj_n = inj_coeff * SQR(d0) / rho0 / rsqr * math::sqrt(math::sqrt(rsqr));
      
        return fill ? inj_n * ppc0 : ZERO;
      } else {
        return fill ? TWO / ppc0 * 1.01 : ZERO;
      }
    }

  private:
    tuple_t<real_t, M::Dim> x_min;
    tuple_t<real_t, M::Dim> x_max;
    const real_t            sigma_thr;
    const real_t            db_thr;
    const real_t            inj_coeff;
    const real_t            inv_n0;
    const real_t            d0;
    const real_t            rho0;
    const real_t            ppc0;
    Domain<S, M>*           domain_ptr;
    ndfield_t<M::Dim, 3>    density;
    ndfield_t<M::Dim, 6>    EM;
    const M                 metric;
    const bool              is_weight;
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

    inline PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : arch::ProblemGenerator<S, M>(p)
      , xi_min { p.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { p.template get<std::vector<real_t>>("setup.xi_max") }
      , sigma_max { p.template get<real_t>("setup.sigma_max") }
      , sigma0 { p.template get<real_t>("scales.sigma0") }
      , inj_coeff { p.template get<real_t>("setup.inj_coeff") }
      , db_thr { p.template get<real_t>("setup.db_thr") }
      , temperature { p.template get<real_t>("setup.temperature") }
      , m_eps { p.template get<real_t>("setup.m_eps") }
      , inv_n0 { ONE / p.template get<real_t>("scales.n0") }
      , init_flds { m.mesh().metric, m_eps }
      , metadomain { &m } {}

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& local_domain) {
        const auto energy_dist  = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temperature);
        const auto spatial_dist1 = PointDistribution<S, M>(xi_min,
                                                          xi_max,
                                                          sigma_max / sigma0,
                                                          inj_coeff,
                                                          db_thr,
                                                          false,
                                                          params,
                                                          &local_domain);
        const auto spatial_dist2 = PointDistribution<S, M>(xi_min,
                                                          xi_max,
                                                          sigma_max / sigma0,
                                                          inj_coeff,
                                                          db_thr,
                                                          true,
                                                          params,
                                                          &local_domain);
  
        const auto injector =
          arch::experimental::Injector_with_weights<S, M, arch::Maxwellian, PointDistribution, PointDistribution>(
            energy_dist,
            spatial_dist1,
            spatial_dist2,
            { 1, 2 });
        arch::experimental::InjectWithWeights<S, M, decltype(injector)>(params,
                                                         local_domain,
                                                         injector,
                                                         1.0);
    }

    void CustomFieldOutput(const std::string&   name,
                           ndfield_t<M::Dim, 6> buffer,
                           index_t              index,
                           timestep_t,
                           simtime_t,
                           const Domain<S, M>&  domain) {

       
        if (name == "DB") {
          if constexpr (M::Dim == Dim::_2D) {
            const auto& EM = domain.fields.em;
            const auto& metric = domain.mesh.metric;
            Kokkos::parallel_for(
            "DB",
            domain.mesh.rangeActiveCells(),
            Lambda(index_t i1, index_t i2) {
              coord_t<M::Dim> xi { static_cast<real_t>(i1 - N_GHOSTS), static_cast<real_t>(i2 - N_GHOSTS) };
              const vec_t<Dim::_3D> B_cntrv { EM(i1, i2, em::bx1),
                                              EM(i1, i2, em::bx2),
                                              EM(i1, i2, em::bx3) };
              vec_t<Dim::_3D> B_cov { ZERO };
              const vec_t<Dim::_3D> D_cntrv { EM(i1, i2, em::dx1),  
                                              EM(i1, i2, em::dx2),
                                              EM(i1, i2, em::dx3) };
              metric.template transform<Idx::U, Idx::D>(xi, B_cntrv, B_cov);
              buffer(i1, i2, index) = DOT(B_cov[0], B_cov[1], B_cov[2], D_cntrv[0], D_cntrv[1], D_cntrv[2])
                                      / DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cntrv[0], B_cntrv[1], B_cntrv[2]);
            });
          }
        } else if (name == "Gamma_1" || name == "Gamma_2"){
          const auto sp_idx = (name == "Gamma_1") ? 0 : 1;
          auto& sp = domain.species[sp_idx];
          if constexpr (M::Dim == Dim::_2D){
            auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
            const auto& metric = domain.mesh.metric;
            auto& mesh = domain.mesh;
            const auto ni2 = mesh.n_active(in::x2);
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

            auto n_buffer = domain.fields.buff;
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
              "ComputeMoments",
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
        } else if (name == "Vr_1" || name == "Vth_1" || name == "Vph_1"
                   || name == "Vr_2" || name == "Vth_2" || name == "Vph_2"){
          const auto comp = (name == "Vr_1" || name == "Vr_2") ? 0 : (name == "Vth_1" || name == "Vth_2") ? 1 : 2;
          const auto sp_idx = (name == "Vr_1" || name == "Vth_1" || name == "Vph_1") ? 0 : 1;
          auto& sp = domain.species[sp_idx];

          
          if constexpr (M::Dim == Dim::_2D){
            auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
            const auto& metric = domain.mesh.metric;
            auto& mesh = domain.mesh;
            const auto ni2 = mesh.n_active(in::x2);
            // clang-format off
            Kokkos::parallel_for(
              name,
              sp.rangeActiveParticles(),
              CustomMoments_kernel<M, CustomField::V>(comp, scatter_buff, index,
                                                          sp.i1, sp.i2, sp.i3,
                                                          sp.dx1, sp.dx2, sp.dx3,
                                                          sp.ux1, sp.ux2, sp.ux3, sp.phi, 
                                                          sp.weight, sp.tag, sp.mass(), sp.charge(),
                                                          metric, mesh.flds_bc(), ni2, inv_n0, ZERO));

            Kokkos::Experimental::contribute(buffer, scatter_buff);

            auto n_buffer = domain.fields.buff;
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
              "ComputeMoments",
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
        } else if (name == "Ut_1"  || name == "Ut_2"){
          const auto sp_idx = (name == "Ut_1") ? 0 : 1;
          auto& sp = domain.species[sp_idx];
          auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);
          const auto& metric = domain.mesh.metric;
          auto& mesh = domain.mesh;
          const auto ni2 = mesh.n_active(in::x2);
          // clang-format off
          Kokkos::parallel_for(
            name,
            sp.rangeActiveParticles(),
            CustomMoments_kernel<M, CustomField::Ut>(0, scatter_buff, index,
                                                          sp.i1, sp.i2, sp.i3,
                                                          sp.dx1, sp.dx2, sp.dx3,
                                                          sp.ux1, sp.ux2, sp.ux3, sp.phi, 
                                                          sp.weight, sp.tag, sp.mass(), sp.charge(),
                                                          metric, mesh.flds_bc(), ni2, inv_n0, ZERO));
          Kokkos::Experimental::contribute(buffer, scatter_buff);

          auto n_buffer = domain.fields.buff;
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
              "ComputeMoments",
              mesh.rangeActiveCells(),
              Lambda(index_t i1, index_t i2) {
                if (cmp::AlmostZero(n_buffer(i1, i2, 0))) {
                  buffer(i1, i2, index) = ZERO;
                } else {
                  buffer(i1, i2, index) /= n_buffer(i1, i2, 0);
                }
              });
            // clang-format on
        } else {
          raise::Error("Custom output not provided", HERE);
        }
      }
  };



} // namespace user

#endif
