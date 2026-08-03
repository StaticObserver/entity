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

  template <Dimension D>
  struct TargetDensityProfile {
    const real_t density_floor_ref;
    const real_t r_ref_pow;

    TargetDensityProfile(real_t density_floor_ref_, real_t r_ref_)
      : density_floor_ref { density_floor_ref_ }
      , r_ref_pow { math::pow(r_ref_, static_cast<real_t>(1.5)) } {}

    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      const auto r = x_Ph[0];
      return density_floor_ref * r_ref_pow / (r * math::sqrt(r));
    }
  };

  template <MetricClass M, int N, class T>
  struct ReplenishFixedPairs {
    const M                    metric;
    const ndfield_t<M::Dim, N> density;
    const idx_t                idx;
    const T                    target_density;
    const real_t               injection_density;

    ReplenishFixedPairs(const M&                    metric_,
                        const ndfield_t<M::Dim, N>& density_,
                        idx_t                       idx_,
                        const T&                    target_density_,
                        real_t                      injection_density_)
      : metric { metric_ }
      , density { density_ }
      , idx { idx_ }
      , target_density { target_density_ }
      , injection_density { injection_density_ } {}

    Inline auto operator()(const coord_t<M::Dim>& x_Ph) const
      -> Kokkos::pair<real_t, real_t> {
      coord_t<M::Dim> x_Cd { ZERO };
      metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);

      real_t dens { ZERO };
      if constexpr (M::Dim == Dim::_2D) {
        dens = density(static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS,
                       static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS,
                       idx);
      } else {
        raise::KernelError(HERE, "ReplenishFixedPairs: only 2D supported");
      }

      const auto target = target_density(x_Ph);
      if (static_cast<real_t>(0.9) * target > dens) {
        return { static_cast<real_t>(1.0), (target - dens) / injection_density };
      }
      return { ZERO, ZERO };
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
    const int injection_pairs_per_cell;
    const real_t density_mult, r_ref, temperature, flux0, m_eps;
    const real_t density_floor_ref, injection_density_per_cell;

    InitFields<M, D>        init_flds;
    const Metadomain<S, M>* metadomain;

    PGen(SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , xi_min { params.template get<std::vector<real_t>>("setup.xi_min") }
      , xi_max { params.template get<std::vector<real_t>>("setup.xi_max") }
      , injection_pairs_per_cell {
          params.template get<int>("setup.injection_pairs_per_cell") }
      , density_mult { params.template get<real_t>("setup.density_mult") }
      , r_ref { params.template get<real_t>("setup.r_ref") }
      , temperature { params.template get<real_t>("setup.temperature") }
      , flux0 { params.template get<real_t>("setup.flux0") }
      , m_eps { params.template get<real_t>("setup.m_eps") }
      , density_floor_ref {
          density_mult * params.template get<real_t>("scales.B0") *
          SQR(params.template get<real_t>("scales.skindepth0")) }
      , injection_density_per_cell {
          TWO * static_cast<real_t>(injection_pairs_per_cell) /
          params.template get<real_t>("particles.ppc0") }
      , init_flds { m.mesh().metric, flux0, m_eps }
      , metadomain { &m } {
      raise::ErrorIf(injection_pairs_per_cell <= 0,
                     "setup.injection_pairs_per_cell must be positive",
                     HERE);
    }

    void InjectDensityFloor(
      Domain<S, M>& local_domain,
      const boundaries_t<real_t>& injection_box = {}) {
      arch::ComputeMomentWithSpecies<S, M, FldsID::N, 3>(
        params,
        local_domain,
        { 1u, 2u },
        local_domain.fields.buff);

      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        temperature);

      const auto target_profile =
        TargetDensityProfile<D> { density_floor_ref, r_ref };
      const auto spatial_dist =
        ReplenishFixedPairs<M, 3, TargetDensityProfile<D>>(
          local_domain.mesh.metric,
          local_domain.fields.buff,
          0u,
          target_profile,
          injection_density_per_cell);

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
        injection_density_per_cell,
        true,
        injection_box);
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      InjectDensityFloor(local_domain);
    }

    void CustomPostStep(timestep_t /*step*/,
                        simtime_t /*time*/,
                        Domain<S, M>& local_domain) {
      boundaries_t<real_t> replenishment_box;
      replenishment_box.emplace_back(xi_min[0], xi_max[0]);
      replenishment_box.emplace_back(xi_min[1], xi_max[1]);
      InjectDensityFloor(local_domain, replenishment_box);
    }
  };

} // namespace user

#endif
