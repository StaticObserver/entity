#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

namespace user {
  using namespace ntt;

  template <class M, Dimension D>
  struct InitFields {
    InitFields(M metric_, real_t flux0_, real_t m_eps_)
      : metric { metric_ }, flux0 { flux0_ }, m_eps { m_eps_ } {}

    Inline auto A_3(const coord_t<D>& x_Cd) const -> real_t {
      const auto theta = metric.template convert<2, Crd::Cd, Crd::Ph>(x_Cd[1]);
      return flux0 * (ONE - math::abs(math::cos(theta)));
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const
      -> real_t { // at ( i , j + HALF )
      coord_t<D> xi { ZERO }, x0m { ZERO }, x0p { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, xi);

      x0m[0] = xi[0];
      x0m[1] = xi[1] - HALF * m_eps;
      x0p[0] = xi[0];
      x0p[1] = xi[1] + HALF * m_eps;

      if (cmp::AlmostZero(math::sin(x_Ph[1]))) {
        const auto dtheta_dxi =
          (metric.template convert<2, Crd::Cd, Crd::Ph>(x0p[1]) -
           metric.template convert<2, Crd::Cd, Crd::Ph>(x0m[1])) /
          m_eps;
        const auto polarity = x_Ph[1] < HALF * static_cast<real_t>(constant::PI)
                                ? ONE
                                : -ONE;
        return polarity * flux0 * dtheta_dxi /
               metric.sqrt_det_h_tilde({ xi[0], xi[1] });
      }

      return (A_3(x0p) - A_3(x0m)) /
             (m_eps * metric.sqrt_det_h({ xi[0], xi[1] }));
    }

    Inline auto bx2(const coord_t<D>& /*x_Ph*/) const
      -> real_t { // at ( i + HALF , j )
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx1(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx2(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

    Inline auto dx3(const coord_t<D>& /*x_Ph*/) const -> real_t {
      return ZERO;
    }

  private:
    const M      metric;
    const real_t flux0;
    const real_t m_eps;
  };

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
        return { ONE, weight };
      } else {
        raise::KernelError(HERE, "DdotBWeightedPairs: only 2D supported");
        return { ZERO, ZERO };
      }
    }
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::GRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::QKerr_Schild> {}
    };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

    const SimulationParams& params;

    const std::vector<real_t> xi_min;
    const std::vector<real_t> xi_max;
    const real_t pair_creation_rate, ddotb_threshold, sigma_min_fraction;
    const real_t nGJ, ppc0, temperature, flux0, m_eps;

    InitFields<M, D>        init_flds;
    const Metadomain<S, M>* metadomain;

    PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , xi_min { params.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { params.template get<std::vector<real_t>>("setup.xi_max") }
      , pair_creation_rate {
          params.template get<real_t>("setup.pair_creation_rate") }
      , ddotb_threshold { params.template get<real_t>("setup.ddotb_threshold") }
      , sigma_min_fraction {
          params.template get<real_t>("setup.sigma_min_fraction") }
      , nGJ { params.template get<real_t>("scales.B0") *
              SQR(params.template get<real_t>("scales.skindepth0")) }
      , ppc0 { params.template get<real_t>("particles.ppc0") }
      , temperature { params.template get<real_t>("setup.temperature") }
      , flux0 { params.template get<real_t>("setup.flux0") }
      , m_eps { params.template get<real_t>("setup.m_eps") }
      , init_flds { m.mesh().metric, flux0, m_eps }
      , metadomain { &m } {
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

    void InjectDdotBPairs(
      Domain<S, M>& local_domain,
      const boundaries_t<real_t>& injection_box = {}) {
      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(
        params,
        local_domain,
        { 1u, 2u },
        local_domain.fields.buff);

      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        temperature);

      const auto spatial_dist =
        DdotBWeightedPairs<M>(
          local_domain.mesh.metric,
          local_domain.fields.em,
          local_domain.fields.buff,
          pair_creation_rate,
          ddotb_threshold,
          sigma_min_fraction,
          nGJ,
          ppc0);

      arch::InjectNonUniform<S,
                             M,
                             decltype(energy_dist),
                             decltype(energy_dist),
                             decltype(spatial_dist)>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        spatial_dist,
        TWO / ppc0,
        true,
        injection_box);
    }

    void InitPrtls(Domain<S, M>& /*local_domain*/) {}

    void CustomPostStep(timestep_t /*step*/,
                        simtime_t /*time*/,
                        Domain<S, M>& local_domain) {
      boundaries_t<real_t> injection_box;
      injection_box.emplace_back(xi_min[0], xi_max[0]);
      injection_box.emplace_back(xi_min[1], xi_max[1]);
      InjectDdotBPairs(local_domain, injection_box);
    }
  };

} // namespace user

#endif
