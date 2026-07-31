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

#include "framework/domain/metadomain.h"

#include <string>

enum class InitFieldGeometry : uint8_t {
  Wald,
  Vertical,
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
               real_t               screen_coeff_,
               bool                 screen_init_)
      : metric { metric_ }
      , axion { axion_ }
      , screen_coeff { screen_coeff_ }
      , screen_init { screen_init_ } {
      if (init_field_geometry == "wald") {
        field_geometry = InitFieldGeometry::Wald;
      } else if (init_field_geometry == "vertical") {
        field_geometry = InitFieldGeometry::Vertical;
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
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return ZERO;
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

    // Gauss-consistent "screened" initialization of the D field:
    //   D(x, 0)  +=  -eps_tilde * a(x, 0) * B(x, 0),
    // an exact particular solution of div D = rho_a = -eps_tilde * B.grad a
    // (docs/design.md; axion-Komissarov notes, "screening field" section).
    Inline auto screen(const coord_t<D>& x_Ph, real_t bx) const -> real_t {
      return screen_init ? screen_coeff * axion.a(x_Ph, ZERO) * bx : ZERO;
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

        return D1u - screen(x_Ph, bx1(x_Ph));
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return -screen(x_Ph, bx1(x_Ph));
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
        x0p[0] = xi[0] + HALF;
        x0p[1] = xi[1];
        real_t inv_sqrt_detH_ijP { ONE / metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t sqrt_detH_ijP { metric.sqrt_det_h({ xi[0], xi[1] }) };
        real_t alpha_ijP { metric.alpha({ xi[0], xi[1] }) };
        real_t beta_ijP { metric.beta1({ xi[0], xi[1] }) };

        real_t E2d { (A_0(x0p) - A_0(x0m)) };
        real_t D2d { E2d / alpha_ijP -
                     (A_1(x0p) - A_1(x0m)) * beta_ijP / alpha_ijP };
        real_t D2u { metric.template h<2, 2>({ xi[0], xi[1] }) * D2d };

        return D2u - screen(x_Ph, bx2(x_Ph));
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return -screen(x_Ph, bx2(x_Ph));
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
                 screen(x_Ph, bx3(x_Ph));
        } else {
          return metric.template h<3, 3>({ xi[0], xi[1] }) * D3d +
                 metric.template h<1, 3>({ xi[0], xi[1] }) * D1d -
                 screen(x_Ph, bx3(x_Ph));
        }
      } else if (field_geometry == InitFieldGeometry::Vertical) {
        return -screen(x_Ph, bx3(x_Ph));
      } else {
        raise::KernelError(HERE, "Unrecognized field geometry");
        return ZERO;
      }
    }

  private:
    const M            metric;
    const AxionField<D> axion;
    const real_t       screen_coeff;
    const bool         screen_init;
    InitFieldGeometry  field_geometry;
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
                    p.template get<bool>("setup.axion_screen_init", true) } {}
  };

} // namespace user

#endif
