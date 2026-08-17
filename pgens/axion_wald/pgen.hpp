#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#ifndef AXION_ENABLED
  #error "pgen `axion_wald` requires building with -D axion=ON"
#endif

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include <Kokkos_Pair.hpp>

#include <string>
#include <vector>

enum class InitFieldGeometry : uint8_t {
  Wald,
  Vertical,
  Bhac,
};

namespace user {
  using namespace ntt;

  /**
   * @brief Fixed axion background, two modes (see docs/design.md):
   * @brief - "sinusoid": a = ampl * sin(omega * t - k1 * r)   [unit tests]
   * @brief - "cloud":    m=0 gravitational-atom eigenstate (Detweiler, n=0)
   * @brief   a = ampl * f_l(r) * P_l(cos th) * cos(omega_c * t + phase)
   * @brief   f_0 = exp(-lam r), f_1 = lam * r * exp(-lam r), lam = alpha^2/(l+1)
   * @brief   omega_c = alpha * (1 - alpha^2 / (2 * (l+1)^2))    [M = 1 units]
   * @brief `a` returns the field value, `dot_a` returns da/dt, `grad_a`
   * @brief returns the physical covariant spatial gradient (d_r a, d_th a, 0);
   * @brief `eps` is the coupling g_{a gamma} * a_0 in code units (multiplied
   * @brief into J_a by the engine).
   */
  enum class AxionMode : uint8_t {
    Sinusoid,
    Cloud,
  };

  template <Dimension D>
  struct AxionField {
    AxionField(const std::string& mode,
               real_t             eps_,
               real_t             ampl_,
               real_t             omega_,
               real_t             k1_,
               real_t             alpha_,
               unsigned short     l_,
               real_t             phase_)
      : eps { eps_ }
      , ampl { ampl_ }
      , omega { omega_ }
      , k1 { k1_ }
      , alpha { alpha_ }
      , l { l_ }
      , phase { phase_ } {
      if (mode == "sinusoid") {
        axion_mode = AxionMode::Sinusoid;
      } else if (mode == "cloud") {
        raise::ErrorIf(l > 1, "axion cloud mode supports l = 0 or 1 only", HERE);
        axion_mode = AxionMode::Cloud;
      } else {
        raise::Error(fmt::format("Unrecognized axion_mode: %s", mode.c_str()),
                     HERE);
      }
    }

    Inline auto a(const coord_t<D>& x_Ph, real_t t) const -> real_t {
      if (axion_mode == AxionMode::Sinusoid) {
        return ampl * math::sin(omega * t - k1 * x_Ph[0]);
      } else {
        real_t f, g;
        cloud_fg(x_Ph[0], x_Ph[1], f, g);
        return ampl * f * g * math::cos(cloud_omega() * t + phase);
      }
    }

    Inline auto dot_a(const coord_t<D>& x_Ph, real_t t) const -> real_t {
      if (axion_mode == AxionMode::Sinusoid) {
        return ampl * omega * math::cos(omega * t - k1 * x_Ph[0]);
      } else {
        // cloud: -omega_c * ampl * f(r) * g(th) * sin(omega_c * t + phase)
        real_t f, g;
        cloud_fg(x_Ph[0], x_Ph[1], f, g);
        return -cloud_omega() * ampl * f * g *
               math::sin(cloud_omega() * t + phase);
      }
    }

    Inline void grad_a(const coord_t<D>& x_Ph,
                       real_t            t,
                       vec_t<Dim::_3D>&  g) const {
      if (axion_mode == AxionMode::Sinusoid) {
        g[0] = -ampl * k1 * math::cos(omega * t - k1 * x_Ph[0]);
        g[1] = ZERO;
        g[2] = ZERO;
      } else {
        const real_t lam = cloud_lambda();
        real_t       f, gth;
        cloud_fg(x_Ph[0], x_Ph[1], f, gth);
        const real_t coswt { math::cos(cloud_omega() * t + phase) };
        // d_r a = ampl * f * (l/r - lam) * g(th) * cos(...)
        g[0] = ampl * f * ((real_t)l / x_Ph[0] - lam) * gth * coswt;
        // d_th a = ampl * f * g'(th) * cos(...), l=0: g'=0, l=1: g'=-sin(th)
        g[1] = (l == 0) ? ZERO
                        : -ampl * f * math::sin(x_Ph[1]) * coswt;
        g[2] = ZERO;
      }
    }

    const real_t eps, ampl, omega, k1, alpha, phase;
    const unsigned short l;
    AxionMode            axion_mode;

  private:
    Inline auto cloud_lambda() const -> real_t {
      return SQR(alpha) / (real_t)(l + 1);
    }

    Inline auto cloud_omega() const -> real_t {
      return alpha * (ONE - SQR(alpha) / (TWO * SQR((real_t)(l + 1))));
    }

    Inline void cloud_fg(real_t r, real_t th, real_t& f, real_t& g) const {
      const real_t lam = cloud_lambda();
      if (l == 0) {
        f = math::exp(-lam * r);
        g = ONE;
      } else {
        f = lam * r * math::exp(-lam * r);
        g = math::cos(th);
      }
    }
  };

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_,
               const std::string&   init_field_geometry,
               const AxionField<D>& axion_,
               real_t               eps_tilde_,
               bool                 gauss_init_,
               real_t               field_B0_   = ONE,
               real_t               r_decay_    = 200.0)
      : metric { metric_ }
      , axion { axion_ }
      , eps_tilde { eps_tilde_ }
      , gauss_init { gauss_init_ }
      , field_B0 { field_B0_ }
      , r_decay { r_decay_ } {
      if (init_field_geometry == "wald") {
        field_geometry = InitFieldGeometry::Wald;
      } else if (init_field_geometry == "vertical") {
        field_geometry = InitFieldGeometry::Vertical;
      } else if (init_field_geometry == "bhac") {
        field_geometry = InitFieldGeometry::Bhac;
      } else {
        raise::Error(fmt::format("Unrecognized field geometry: %s",
                                 init_field_geometry.c_str()),
                     HERE);
      }
    }

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      return HALF * (metric.template h_<3, 3>(x_Cd) +
                     TWO * metric.spin() * metric.template h_<1, 3>(x_Cd) *
                       metric.beta1(x_Cd));
    }

    // bhac-style azimuthal potential (pgens/bhac):
    //   A_phi = B0 * r * exp(-r / r_decay) * sin^2(theta)
    Inline auto A_3_bhac(const coord_t<D>& x_Cd) const -> real_t {
      coord_t<D> x_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
      return field_B0 * x_Ph[0] * math::exp(-x_Ph[0] / r_decay) *
             SQR(math::sin(x_Ph[1]));
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

    Inline auto bx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF;

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };

      if (field_geometry == InitFieldGeometry::Bhac) {
        if (cmp::AlmostZero(x_Ph[1])) {
          // Regularized limit at pole:
          // bx1 -> 2 * B0 * r * exp(-r/r_decay) / sqrt[(r^2+a^2)(r^2+a^2+2r)]
          const real_t r_ph  = x_Ph[0];
          const real_t a     = metric.spin();
          const real_t r2pa2 = SQR(r_ph) + SQR(a);
          return TWO * field_B0 * r_ph * math::exp(-r_ph / r_decay) /
                 math::sqrt(r2pa2 * (r2pa2 + TWO * r_ph));
        }
        return (A_3_bhac(x0p) - A_3_bhac(x0m)) * inv_sqrt_detH_ijP;
      }
      if (cmp::AlmostZero(x_Ph[1])) {
        return ONE;
      } else {
        return (A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0] - HALF;
      x0m[1] = xi[1];
      x0p[0] = xi[0] + HALF;
      x0p[1] = xi[1];

      real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
      if (field_geometry == InitFieldGeometry::Bhac) {
        if (cmp::AlmostZero(x_Ph[1])) {
          return ZERO;
        }
        return -(A_3_bhac(x0p) - A_3_bhac(x0m)) * inv_sqrt_detH_ijP;
      }
      if (cmp::AlmostZero(x_Ph[1])) {
        return ZERO;
      } else {
        return -(A_3(x0p) - A_3(x0m)) * inv_sqrt_detH_ijP;
      }
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j + HALF )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

        x0m[0] = xi[0];
        x0m[1] = xi[1] - HALF;
        x0p[0] = xi[0];
        x0p[1] = xi[1] + HALF;

        real_t inv_sqrt_detH_iPjP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        return -(A_1(x0p) - A_1(x0m)) * inv_sqrt_detH_iPjP;
      } else if (field_geometry == InitFieldGeometry::Vertical ||
                 field_geometry == InitFieldGeometry::Bhac) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    // Gauss-consistent initialization of the D field:
    //   D(x, 0)  +=  -eps_tilde * a(x, 0) * B(x, 0),
    // the exact particular solution of the generalized Gauss law
    // div D = rho_a = -eps_tilde * B.grad a  (docs/design.md;
    // axion-Komissarov notes, "D of a pure axion background" section).
    Inline auto d_axion(const coord_t<D>& x_Ph, real_t bx) const -> real_t {
      return gauss_init ? eps_tilde * axion.a(x_Ph, ZERO) * bx : ZERO;
    }

    Inline auto dx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i + HALF , j )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

        real_t alpha_iPj { metric.alpha({ xi[0], xi[1] }) };
        real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
        real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0] - HALF, xi[1] }) };
        real_t beta_ij { metric.beta1({ xi[0] - HALF, xi[1] }) };
        real_t alpha_ij { metric.alpha({ xi[0] - HALF, xi[1] }) };

        // D1 at ( i + HALF , j )
        x0m[0] = xi[0] - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF;
        x0p[1] = xi[1];
        real_t E1d { (A_0(x0p) - A_0(x0m)) };
        real_t D1d { E1d / alpha_iPj };

        // D3 at ( i , j )
        x0m[0] = xi[0] - HALF - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] - HALF + HALF;
        x0p[1] = xi[1];
        real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij };

        real_t D1u { metric.template h<1, 1>({ xi[0], xi[1] }) * D1d +
                     metric.template h<1, 3>({ xi[0], xi[1] }) * D3d };

        return D1u - d_axion(x_Ph, bx1(x_Ph));
      } else if (field_geometry == InitFieldGeometry::Vertical ||
                 field_geometry == InitFieldGeometry::Bhac) {
        return -d_axion(x_Ph, bx1(x_Ph));
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    Inline auto dx2(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        x0m[0] = xi[0];
        x0m[1] = xi[1] - HALF;
        x0p[0] = xi[0];
        x0p[1] = xi[1] + HALF;
        real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t sqrt_detH_ijP { metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t alpha_ijP { metric.alpha({ xi[0], xi[1] }) };
        real_t beta_ijP { metric.beta1({ xi[0], xi[1] }) };

        real_t E2d { (A_0(x0p) - A_0(x0m)) };
        real_t D2d { E2d / alpha_ijP -
                     (A_1(x0p) - A_1(x0m)) * beta_ijP / alpha_ijP };
        real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

        return D2u - d_axion(x_Ph, bx2(x_Ph));
      } else if (field_geometry == InitFieldGeometry::Vertical ||
                 field_geometry == InitFieldGeometry::Bhac) {
        return -d_axion(x_Ph, bx2(x_Ph));
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    Inline auto dx3(const coord_t<D>& x_Ph) const -> real_t { // at ( i , j )
      if (field_geometry == InitFieldGeometry::Wald) {
        coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);
        real_t inv_sqrt_detH_ij { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t sqrt_detH_ij { metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t beta_ij { metric.beta1({ xi[0], xi[1] }) };
        real_t alpha_ij { metric.alpha({ xi[0], xi[1] }) };
        real_t alpha_iPj { metric.alpha({ xi[0] + HALF, xi[1] }) };

        // D3 at ( i , j )
        x0m[0] = xi[0] - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF;
        x0p[1] = xi[1];
        real_t D3d { (A_3(x0p) - A_3(x0m)) * beta_ij / alpha_ij };

        // D1 at ( i + HALF , j )
        x0m[0] = xi[0] + HALF - HALF;
        x0m[1] = xi[1];
        x0p[0] = xi[0] + HALF + HALF;
        x0p[1] = xi[1];
        real_t E1d { (A_0(x0p) - A_0(x0m)) };
        real_t D1d { E1d / alpha_iPj };

        if (cmp::AlmostZero(x_Ph[1])) {
          return metric.template h<1, 3>({ xi[0], xi[1] }) * D1d -
                 d_axion(x_Ph, bx3(x_Ph));
        } else {
          return metric.template h<3, 3>({ xi[0], xi[1] }) * D3d +
                 metric.template h<1, 3>({ xi[0], xi[1] }) * D1d -
                 d_axion(x_Ph, bx3(x_Ph));
        }
      } else if (field_geometry == InitFieldGeometry::Vertical ||
                 field_geometry == InitFieldGeometry::Bhac) {
        return -d_axion(x_Ph, bx3(x_Ph));
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

  private:
    const M            metric;
    const AxionField<D> axion;
    const real_t       eps_tilde;
    const bool         gauss_init;
    const real_t       field_B0;
    const real_t       r_decay;
    InitFieldGeometry  field_geometry;
  };

  /**
   * @brief Local D.B-triggered pair injection, ported from the
   * @brief `bh-reconnection` pgen (Parfrey et al. 2019 prescription).
   * @brief Each cell (inside the injection box passed to InjectNonUniform)
   * @brief where
   * @brief   |D.B|/B^2 > ddotb_threshold  and  B^2 > sigma_min_fraction * rho
   * @brief receives exactly one electron-positron pair per call; the pair
   * @brief members carry a weight encoding the normalized density
   * @brief   delta_n = pair_creation_rate * nGJ * |D.B| / sqrt(B^2),
   * @brief with nGJ = B0 * skindepth0^2. Deliberately uses the local
   * @brief staggered-grid EM values without interpolation. The density
   * @brief (FldsID::Rho of both species) is recomputed from the current
   * @brief particles before each call, so injection self-regulates.
   */
  template <GRMetricClass M>
  struct DdotBWeightedPairs {
    const M                  metric;
    const ndfield_t<M::Dim, 6> em;
    const ndfield_t<M::Dim, 3> density;
    const real_t pair_creation_rate;
    const real_t ddotb_threshold;
    const real_t sigma_min_fraction;
    const real_t nGJ;
    const real_t ppc0;

    DdotBWeightedPairs(const M&                    metric_,
                       const ndfield_t<M::Dim, 6>& em_,
                       const ndfield_t<M::Dim, 3>& density_,
                       real_t pair_creation_rate_,
                       real_t ddotb_threshold_,
                       real_t sigma_min_fraction_,
                       real_t nGJ_,
                       real_t ppc0_)
      : metric { metric_ }
      , em { em_ }
      , density { density_ }
      , pair_creation_rate { pair_creation_rate_ }
      , ddotb_threshold { ddotb_threshold_ }
      , sigma_min_fraction { sigma_min_fraction_ }
      , nGJ { nGJ_ }
      , ppc0 { ppc0_ } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const
      -> Kokkos::pair<real_t, real_t> {
      if constexpr (M::Dim == Dim::_2D) {
        coord_t<M::Dim> x_Cd { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);

        const auto i1 = static_cast<int>(x_Cd[0]) + static_cast<int>(N_GHOSTS);
        const auto i2 = static_cast<int>(x_Cd[1]) + static_cast<int>(N_GHOSTS);

        // Deliberately use the local staggered-grid values without interpolation.
        const vec_t<Dim::_3D> B_cntrv { em(i1, i2, em::bx1),
                                        em(i1, i2, em::bx2),
                                        em(i1, i2, em::bx3) };
        const vec_t<Dim::_3D> D_cntrv { em(i1, i2, em::dx1),
                                        em(i1, i2, em::dx2),
                                        em(i1, i2, em::dx3) };
        vec_t<Dim::_3D> B_cov { ZERO };
        metric.template transform<Idx::U, Idx::D>(x_Cd, B_cntrv, B_cov);

        const auto bsqr =
          DOT(B_cntrv[0], B_cntrv[1], B_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        if (not math::isfinite(bsqr) || bsqr <= ZERO || cmp::AlmostZero(bsqr)) {
          return { ZERO, ZERO };
        }

        const auto ddotb =
          DOT(D_cntrv[0], D_cntrv[1], D_cntrv[2], B_cov[0], B_cov[1], B_cov[2]);
        const auto abs_ddotb = math::abs(ddotb);
        const auto chi       = abs_ddotb / bsqr;
        const auto dens      = density(i1, i2, 0);
        if (not math::isfinite(chi) || chi <= ddotb_threshold ||
            bsqr <= sigma_min_fraction * dens) {
          return { ZERO, ZERO };
        }

        // delta_n is the normalized density represented by each member of the
        // injected pair.  InjectNonUniform later applies sqrt(det(h))/V0 to the
        // stored particle weight, which cancels in the normalized moment.
        const auto delta_n = pair_creation_rate * nGJ * abs_ddotb / math::sqrt(bsqr);
        const auto weight  = ppc0 * delta_n;
        if (not math::isfinite(weight) || weight <= ZERO) {
          return { ZERO, ZERO };
        }
        return { static_cast<real_t>(1.0), weight };
      } else {
        raise::KernelError(HERE, "DdotBWeightedPairs: only 2D supported");
        return { ZERO, ZERO };
      }
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::GRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Kerr_Schild, Metric::QKerr_Schild, Metric::Kerr_Schild_0> {}
    };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

    AxionField<D>    axion;
    InitFields<M, D> init_flds;

    const SimulationParams&   params;
    // local D.B-triggered pair injection (Parfrey-style, ported from the
    // bh-reconnection pgen); an empty/invalid box disables it
    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const real_t              pair_creation_rate, ddotb_threshold,
      sigma_min_fraction, nGJ, ppc0, temperature;

    PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : axion { p.template get<std::string>("setup.axion_mode", "sinusoid"),
                p.template get<real_t>("setup.axion_eps", ZERO),
                p.template get<real_t>("setup.axion_amplitude", ONE),
                p.template get<real_t>("setup.axion_omega", ONE),
                p.template get<real_t>("setup.axion_k1", ZERO),
                p.template get<real_t>("setup.axion_alpha", HALF),
                (unsigned short)(p.template get<int>("setup.axion_l", 1)),
                p.template get<real_t>("setup.axion_phase", ZERO) }
      , init_flds { m.mesh().metric,
                    p.template get<std::string>("setup.init_field", "wald"),
                    axion,
                    p.template get<real_t>("scales.q0") *
                      p.template get<real_t>("setup.axion_eps", ZERO) /
                      p.template get<real_t>("scales.B0"),
                    p.template get<bool>("setup.axion_gauss_init", true),
                    p.template get<real_t>("setup.field_B0", ONE),
                    p.template get<real_t>("setup.r_decay", 200.0) }
      , params { p }
      , xi_min { p.template get<std::vector<real_t>>(
          "setup.xi_min",
          std::vector<real_t> { ZERO, ZERO }) }
      , xi_max { p.template get<std::vector<real_t>>(
          "setup.xi_max",
          std::vector<real_t> { ZERO, ZERO }) }
      , pair_creation_rate {
          p.template get<real_t>("setup.pair_creation_rate", 0.5) }
      , ddotb_threshold { p.template get<real_t>("setup.ddotb_threshold", 1e-2) }
      , sigma_min_fraction {
          p.template get<real_t>("setup.sigma_min_fraction", 0.05) }
      , nGJ { p.template get<real_t>("scales.B0") *
              SQR(p.template get<real_t>("scales.skindepth0")) }
      , ppc0 { p.template get<real_t>("particles.ppc0") }
      , temperature { p.template get<real_t>("setup.temperature", 0.01) } {
      raise::ErrorIf(pair_creation_rate <= ZERO,
                     "setup.pair_creation_rate must be positive",
                     HERE);
      raise::ErrorIf(ddotb_threshold <= ZERO,
                     "setup.ddotb_threshold must be positive",
                     HERE);
      raise::ErrorIf(sigma_min_fraction < ZERO,
                     "setup.sigma_min_fraction must be non-negative",
                     HERE);
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      InjectPlasma(local_domain);
    }

    void CustomPostStep(timestep_t /*step*/,
                        simtime_t /*time*/,
                        Domain<S, M>& local_domain) {
      InjectPlasma(local_domain);
    }

  private:
    void InjectPlasma(Domain<S, M>& local_domain) {
      // no injection box configured -> vacuum run, skip entirely
      auto no_box = true;
      for (auto d = 0u; d < D; ++d) {
        no_box &= xi_min[d] >= xi_max[d];
      }
      if (no_box) {
        return;
      }
      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(
        params,
        local_domain,
        { 1u, 2u },
        local_domain.fields.buff);
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        temperature);
      const auto spatial_dist = DdotBWeightedPairs<M>(local_domain.mesh.metric,
                                                      local_domain.fields.em,
                                                      local_domain.fields.buff,
                                                      pair_creation_rate,
                                                      ddotb_threshold,
                                                      sigma_min_fraction,
                                                      nGJ,
                                                      ppc0);
      boundaries_t<real_t> injection_box;
      for (auto d = 0u; d < D; ++d) {
        injection_box.emplace_back(xi_min[d], xi_max[d]);
      }
      // number_density = 2/ppc0 makes the injector place exactly one pair
      // per accepted cell; the physical density is carried by the weights
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(spatial_dist)>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        spatial_dist,
        TWO / ppc0,
        true,
        injection_box);
    }
  };

} // namespace user

#endif
