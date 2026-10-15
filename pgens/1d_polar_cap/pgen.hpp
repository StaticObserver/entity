#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "traits/policies.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "framework/domain/metadomain.h"

#include "initial_injection.hpp"
#include "qed/curvature_emission.hpp"
#include "qed/curvature_spectrum.hpp"
#include "qed/magnetic_pair_creation.hpp"
#include "qed/photon_opacity.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <string>

namespace user {
  using namespace ntt;

  namespace polar_cap {

    inline auto PhotonRecycleInterval(const SimulationParams& params,
                                      bool                    pair_creation_enabled)
      -> timestep_t {
      if (not pair_creation_enabled) {
        return 1u;
      }

      constexpr auto key = "setup.polar_cap.qed.photon_recycle_interval";
      if (params.contains(key)) {
        // [setup] integers are stored as int by SimulationParams. Validate
        // before converting to the unsigned timestep type.
        const auto configured_interval = params.template get<int>(key);
        raise::ErrorIf(configured_interval <= 0,
                       "photon_recycle_interval must be positive",
                       HERE);
        return static_cast<timestep_t>(configured_interval);
      }

      const auto default_interval =
        params.template get<timestep_t>("particles.clear_interval");
      raise::ErrorIf(default_interval == 0u,
                     "photon_recycle_interval must be positive",
                     HERE);
      return default_interval;
    }

    // Filled start (CPT2024-style): the deposited net charge equals the
    // prescribed GJ background everywhere, so the Gauss-consistent initial
    // field vanishes. B is uniform.
    template <Dimension D>
    struct InitialFields {
      const real_t b0;

      explicit InitialFields(real_t b0) : b0 { b0 } {}

      Inline auto bx1(const coord_t<D>&) const -> real_t {
        return b0;
      }

      Inline auto ex1(const coord_t<D>&) const -> real_t {
        return ZERO;
      }
    };

    template <Dimension D>
    struct BoundaryFields {
      const real_t b0;

      explicit BoundaryFields(real_t b0) : b0 { b0 } {}

      Inline auto bx1(const coord_t<D>&) const -> real_t {
        return b0;
      }

      Inline auto ex1(const coord_t<D>&) const -> real_t {
        return ZERO;
      }
    };

    template <Dimension D>
    struct MatchBoundaryFields {
      const real_t b0;

      explicit MatchBoundaryFields(real_t b0) : b0 { b0 } {}

      Inline auto bx1(const coord_t<D>&) const -> real_t {
        return b0;
      }

      // Relax Ex to zero through the tanh match layer so no sheath field can
      // accumulate on the absorbing edge. Without this target the MATCH
      // boundary leaves Ex unevolved and the edge pumps up a spurious field
      // that reflects incoming particles back into the physical region.
      Inline auto ex1(const coord_t<D>&) const -> real_t {
        return ZERO;
      }
    };

    template <Dimension D>
    struct AtmosphereFields : public BoundaryFields<D> {
      explicit AtmosphereFields(real_t b0) : BoundaryFields<D> { b0 } {}
    };

    template <Dimension D>
    struct AtmosphereDensity {
      const real_t peak_density, scale_height, x_surface;

      AtmosphereDensity(real_t peak_density, real_t scale_height, real_t x_surface)
        : peak_density { peak_density }
        , scale_height { scale_height }
        , x_surface { x_surface } {}

      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < x_surface) {
          return ZERO;
        }
        return peak_density * math::exp(-(x_Ph[0] - x_surface) / scale_height);
      }
    };

    // Uniform bulk neutral fill: constant density from the surface through the
    // active domain. The near-star atmosphere is injected separately and adds
    // on top of this floor.
    template <Dimension D>
    struct FillDensity {
      const real_t fill_density, x_surface;

      FillDensity(real_t fill_density, real_t x_surface)
        : fill_density { fill_density }
        , x_surface { x_surface } {}

      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < x_surface) {
          return ZERO;
        }
        return fill_density;
      }
    };

    // GJ charge imbalance: extends through the whole domain and follows the
    // prescribed gradient, rho(x) = 1 + rho_gj_gradient * (x - x_surface) / L,
    // so that the total initial charge equals the GJ background pointwise and
    // E_x = 0.
    template <Dimension D>
    struct ExtraPositronDensity {
      const real_t x_surface;
      const real_t gradient, inv_length;

      ExtraPositronDensity(real_t x_surface, real_t gradient, real_t inv_length)
        : x_surface { x_surface }
        , gradient { gradient }
        , inv_length { inv_length } {}

      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < x_surface) {
          return ZERO;
        }
        return ONE + gradient * (x_Ph[0] - x_surface) * inv_length;
      }
    };

    template <Dimension D>
    struct MagnetosphericCurrent {
      const real_t current_x1_contravariant;

      MagnetosphericCurrent(real_t current_x1_tetrad, real_t dx)
        : current_x1_contravariant { current_x1_tetrad / dx } {}

      Inline auto jx1(const coord_t<D>&) const -> real_t {
        // Ampere adds this value directly to the deposited contravariant
        // current. The configured current is a tetrad component, so the
        // 1D Minkowski conversion is J^1 = J^(hat 1) / dx. Entity separately
        // handles the ppc0 normalization in CurrentsAmpere_kernel.
        return current_x1_contravariant;
      }
      Inline auto jx2(const coord_t<D>&) const -> real_t {
        return ZERO;
      }
      Inline auto jx3(const coord_t<D>&) const -> real_t {
        return ZERO;
      }
    };

  } // namespace polar_cap

  template <SimEngine::type S, class M>
  struct PGen {
    // The species ordering is part of the PGen-TOML contract and is also used
    // by the emission and pair-conversion policies.
    static constexpr auto D { M::Dim };
    static constexpr auto electron_index { static_cast<spidx_t>(1) };
    static constexpr auto positron_index { static_cast<spidx_t>(2) };
    static constexpr auto photon_index { static_cast<spidx_t>(3) };

    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D> {}
    };

    const SimulationParams& params;

    const real_t b0, temperature;
    const real_t x_surface, atmosphere_width;
    const real_t extra_positron_density;
    const real_t fill_density, rho_gj_gradient;

    const bool           qed_enabled, curvature_drag_enabled;
    const bool           curvature_emission_enabled, pair_creation_enabled;
    const bool           filter_nonconverting_photons;
    const real_t         rho_c, gamma_emit, photon_energy_min;
    const real_t         global_x_min, global_x_max;
    const real_t         gamma_rad, reference_electric_field;
    const real_t         emission_step_coefficient, drag_step_coefficient;
    const real_t         opacity_prefactor, b_over_bq;
    const real_t         max_drag_fraction, conversion_optical_depth;
    const int max_photons_per_particle;
    const int opacity_substeps;
    const timestep_t photon_recycle_interval;
    const bool       debug_edge_fields;
    const timestep_t debug_edge_fields_interval;

    polar_cap::InitialFields<D>         init_flds;
    polar_cap::MagnetosphericCurrent<D> ext_current;
    polar_cap::CurvatureSpectrum        curvature_spectrum;

    PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : params { p }
      , b0 { params.template get<real_t>("setup.polar_cap.B0") }
      , temperature { params.template get<real_t>("setup.polar_cap.temperature") }
      , x_surface([&metadomain, this]() {
          const auto minimum_buffer = params.template get<unsigned short>(
                                        "algorithms.current_filters") +
                                      2;
          const auto buffer_cells = minimum_buffer > 5 ? minimum_buffer : 5;
          return metadomain.mesh().metric.template convert<1, Crd::Cd, Crd::Ph>(
            static_cast<real_t>(buffer_cells));
        }())
      , atmosphere_width { params.template get<real_t>(
          "grid.boundaries.atmosphere.ds") }
      , extra_positron_density { params.template get<real_t>(
          "setup.polar_cap.extra_positron_density",
          ONE) }
      , fill_density { params.template get<real_t>(
          "setup.polar_cap.initial_injection.fill_density",
          ZERO) }
      , rho_gj_gradient { params.template get<real_t>(
          "setup.polar_cap.rho_gj_gradient",
          ZERO) }
      , qed_enabled { params.template get<bool>("setup.polar_cap.qed.enable", false) }
      , curvature_drag_enabled { params.template get<bool>(
          "setup.polar_cap.qed.curvature_drag",
          true) }
      , curvature_emission_enabled { params.template get<bool>(
          "setup.polar_cap.qed.curvature_emission",
          true) }
      , pair_creation_enabled { params.template get<bool>(
          "setup.polar_cap.qed.magnetic_pair_creation",
          true) }
      , filter_nonconverting_photons {
          qed_enabled and pair_creation_enabled and
          params.template get<bool>(
            "setup.polar_cap.qed.filter_nonconverting_photons",
            false)
        }
      , rho_c {
          qed_enabled
            ? params.template get<real_t>("setup.polar_cap.qed.rho_c")
            : ONE
        }
      , gamma_emit {
          qed_enabled
            ? params.template get<real_t>("setup.polar_cap.qed.gamma_emit")
            : static_cast<real_t>(2.0)
        }
      , photon_energy_min {
          qed_enabled
            ? params.template get<real_t>(
                "setup.polar_cap.qed.photon_energy_min")
            : ONE
        }
      , global_x_min { metadomain.mesh().extent(in::x1).first }
      , global_x_max { metadomain.mesh().extent(in::x1).second }
      , gamma_rad {
          qed_enabled and curvature_drag_enabled
            ? params.template get<real_t>("setup.polar_cap.qed.gamma_rad")
            : static_cast<real_t>(2.0)
        }
      , reference_electric_field {
          qed_enabled and curvature_drag_enabled
            ? params.template get<real_t>(
                "setup.polar_cap.qed.reference_electric_field")
            : ONE
        }
      , emission_step_coefficient {
          // Convert the configured photon-number coefficient to a per-step
          // coefficient. 5*pi/3 normalizes the curvature number spectrum.
          (qed_enabled
             ? params.template get<real_t>(
                 "setup.polar_cap.qed.emission_coefficient")
             : ZERO) *
          params.template get<real_t>("algorithms.timestep.dt") /
          params.template get<real_t>("scales.skindepth0") *
          static_cast<real_t>(5.0 * constant::PI / 3.0)
        }
      , drag_step_coefficient {
          // The QED curvature recoil normalization: K*gamma^4 balances
          // omegaB0*E_ref at gamma_rad. This removes any dependence on
          // macro-particle charge or ppc0.
          (qed_enabled and curvature_drag_enabled)
            ? params.template get<real_t>("algorithms.timestep.dt") *
                params.template get<real_t>("scales.omegaB0") *
                math::abs(reference_electric_field) /
                SQR(SQR(gamma_rad))
            : ZERO
        }
      , opacity_prefactor {
          // Erber-like magnetic conversion coefficient in Entity length units.
          (qed_enabled
             ? params.template get<real_t>(
                 "setup.polar_cap.qed.pair_coefficient")
             : ZERO) *
          static_cast<real_t>(0.23 * constant::PI * constant::SQRT3) *
          (qed_enabled
             ? params.template get<real_t>("setup.polar_cap.qed.b_over_bq")
             : ONE) /
          params.template get<real_t>("scales.skindepth0")
        }
      , b_over_bq {
          qed_enabled
            ? params.template get<real_t>("setup.polar_cap.qed.b_over_bq")
            : ONE
        }
      , max_drag_fraction { params.template get<real_t>(
          "setup.polar_cap.qed.max_drag_fraction",
          static_cast<real_t>(0.2)) }
      , conversion_optical_depth { params.template get<real_t>(
          "setup.polar_cap.qed.conversion_optical_depth",
          ONE) }
      , max_photons_per_particle {
          qed_enabled
            ? params.template get<int>(
                "setup.polar_cap.qed.max_photons_per_particle_step")
            : 1
        }
      , opacity_substeps { params.template get<int>(
          "setup.polar_cap.qed.opacity_substeps",
          8) }
      , photon_recycle_interval { polar_cap::PhotonRecycleInterval(
          params,
          qed_enabled and pair_creation_enabled) }
      , debug_edge_fields { params.template get<bool>(
          "setup.polar_cap.debug_edge_fields",
          false) }
      , debug_edge_fields_interval { static_cast<timestep_t>(
          params.template get<int>(
            "setup.polar_cap.debug_edge_fields_interval",
            1000)) }
      , init_flds { b0 }
      , ext_current { params.template get<real_t>(
                        "setup.polar_cap.external_current"),
                      params.template get<real_t>("scales.dx0") }
      , curvature_spectrum {
          // A QED-off run has no custom emission policy and therefore needs
          // neither the host table read nor the two device spectrum arrays.
          qed_enabled
            ? polar_cap::CurvatureSpectrum { params.template get<std::string>(
                "setup.polar_cap.qed.spectrum_table") }
            : polar_cap::CurvatureSpectrum {}
        }
    {
      // Validate signed host-side values before passing compact values into
      // device policies. In particular, negative integers must not wrap.
      raise::ErrorIf(atmosphere_width <= ZERO,
                     "Atmosphere width must be positive",
                     HERE);
      raise::ErrorIf(temperature < ZERO,
                     "temperature must be non-negative",
                     HERE);
      raise::ErrorIf(extra_positron_density <= ZERO,
                     "extra_positron_density must be positive as the GJ "
                     "net-charge scale",
                     HERE);
      raise::ErrorIf(fill_density < ZERO,
                     "initial_injection.fill_density must be non-negative",
                     HERE);
      raise::ErrorIf(rho_gj_gradient < ZERO,
                     "rho_gj_gradient must be non-negative",
                     HERE);
      raise::ErrorIf(
        params
            .template get<boundaries_t<real_t>>(
              "grid.boundaries.match.ds")[0]
            .second <= ZERO,
        "right MATCH width must be positive",
        HERE);
      if (qed_enabled) {
        raise::ErrorIf(
          params.template get<bool>(
            "setup.polar_cap.qed.filter_nonconverting_photons",
            false) and
            not pair_creation_enabled,
          "filter_nonconverting_photons requires magnetic pair creation",
          HERE);
        raise::ErrorIf(rho_c <= ZERO, "rho_c must be positive", HERE);
        raise::ErrorIf(gamma_emit <= ONE,
                       "gamma_emit must be greater than one",
                       HERE);
        raise::ErrorIf(photon_energy_min <= ZERO,
                       "photon_energy_min must be positive",
                       HERE);
        raise::ErrorIf(b_over_bq <= ZERO, "b_over_bq must be positive", HERE);
        raise::ErrorIf(
          params.template get<real_t>("setup.polar_cap.qed.emission_coefficient",
                                      ZERO) < ZERO,
          "emission_coefficient must be non-negative",
          HERE);
        raise::ErrorIf(
          params.template get<real_t>("setup.polar_cap.qed.pair_coefficient",
                                      ZERO) < ZERO,
          "pair_coefficient must be non-negative",
          HERE);
        raise::ErrorIf(max_photons_per_particle <= 0,
                       "max_photons_per_particle_step must be positive",
                       HERE);
        raise::ErrorIf(opacity_substeps <= 0 or opacity_substeps % 2 != 0,
                       "opacity_substeps must be a positive even number",
                       HERE);
        raise::ErrorIf(conversion_optical_depth <= ZERO,
                       "conversion_optical_depth must be positive",
                       HERE);
        if (filter_nonconverting_photons) {
          const auto particle_boundaries = metadomain.mesh().prtl_bc();
          const auto lower_bc = particle_boundaries[0].first;
          const auto upper_bc = particle_boundaries[0].second;
          raise::ErrorIf(
            (lower_bc != PrtlBC::ATMOSPHERE and lower_bc != PrtlBC::ABSORB) or
              (upper_bc != PrtlBC::ATMOSPHERE and upper_bc != PrtlBC::ABSORB),
            "filter_nonconverting_photons requires absorbing x1 boundaries",
            HERE);
        }
      }
      if (qed_enabled and curvature_drag_enabled) {
        raise::ErrorIf(gamma_rad <= ONE,
                       "curvature-drag gamma_rad must be greater than one",
                       HERE);
        raise::ErrorIf(reference_electric_field <= ZERO,
                       "reference_electric_field must be positive",
                       HERE);
        raise::ErrorIf(max_drag_fraction <= ZERO or max_drag_fraction >= ONE,
                       "max_drag_fraction must be in (0, 1)",
                       HERE);
      }
      raise::ErrorIf(not std::isfinite(emission_step_coefficient),
                     "emission_step_coefficient must be finite",
                     HERE);
      raise::ErrorIf(not std::isfinite(drag_step_coefficient),
                     "drag_step_coefficient must be finite",
                     HERE);
      raise::ErrorIf(not std::isfinite(opacity_prefactor),
                     "opacity_prefactor must be finite",
                     HERE);
      static_assert(EmissionPolicyClass<polar_cap::CurvatureEmission<M>, M>,
                    "CurvatureEmission does not satisfy EmissionPolicyClass");
      static_assert(
        CustomParticleUpdatePolicyClass<polar_cap::PhotonOpacityUpdate<M>, M>,
        "PhotonOpacityUpdate does not satisfy CustomParticleUpdatePolicyClass");
    }

    auto AtmFields(simtime_t) const -> polar_cap::AtmosphereFields<D> {
      return polar_cap::AtmosphereFields<D> { b0 };
    }

    auto MatchFields(simtime_t) const -> polar_cap::MatchBoundaryFields<D> {
      return polar_cap::MatchBoundaryFields<D> { b0 };
    }

    void InitPrtls(Domain<S, M>& domain) {
      // Electrons and positrons are always present. The photon species is a
      // real storage allocation, so require it only for a QED-enabled run.
      raise::ErrorIf(domain.species.size() < 2,
                     "Polar-cap requires electron and positron species",
                     HERE);
      const auto& electrons = domain.species[electron_index - 1];
      const auto& positrons = domain.species[positron_index - 1];
      raise::ErrorIf(not cmp::AlmostEqual_host(electrons.mass(), 1.0f) or
                       not cmp::AlmostEqual_host(electrons.charge(), -1.0f),
                     "Species 1 must be an electron with mass 1 and charge -1",
                     HERE);
      raise::ErrorIf(not cmp::AlmostEqual_host(positrons.mass(), 1.0f) or
                       not cmp::AlmostEqual_host(positrons.charge(), 1.0f),
                     "Species 2 must be a positron with mass 1 and charge +1",
                     HERE);
      raise::ErrorIf(electrons.radiative_drag_flags() != RadiativeDrag::NONE or
                       positrons.radiative_drag_flags() != RadiativeDrag::NONE,
                     "Built-in radiative drag must be disabled for polar-cap species",
                     HERE);

      if (qed_enabled) {
        // Curvature emission and magnetic conversion need one unambiguous
        // destination species with the three migrated QED payloads.
        raise::ErrorIf(
          domain.species.size() != 3,
          "QED-enabled polar-cap requires exactly three species",
          HERE);
        raise::ErrorIf(
          electrons.emission_policy_flag() != EmissionType::CUSTOM or
            positrons.emission_policy_flag() != EmissionType::CUSTOM,
          "QED-enabled electron and positron species must use custom emission",
          HERE);
        const auto& photons = domain.species[photon_index - 1];
        raise::ErrorIf(not cmp::AlmostZero_host(photons.mass()) or
                         not cmp::AlmostZero_host(photons.charge()) or
                         photons.pusher() != ParticlePusher::PHOTON,
                       "Species 3 must be a neutral massless photon species",
                       HERE);
        raise::ErrorIf(photons.npld_r() < 3,
                       "Photon species requires three real payloads",
                       HERE);
      } else {
        raise::ErrorIf(
          domain.species.size() != 2,
          "QED-off polar-cap requires exactly two species and no photon allocation",
          HERE);
        raise::ErrorIf(
          electrons.emission_policy_flag() != EmissionType::NONE or
            positrons.emission_policy_flag() != EmissionType::NONE,
          "QED-off electron and positron species must use emission=none",
          HERE);
      }

      const auto maxwellian = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        temperature);
      // First create the neutral atmosphere: InjectNonUniform divides the
      // requested total density equally between the two species.
      const auto atmosphere = polar_cap::AtmosphereDensity<D> {
        params.template get<real_t>("grid.boundaries.atmosphere.density"),
        params.template get<real_t>("grid.boundaries.atmosphere.height"),
        x_surface
      };
      arch::InjectNonUniform<S,
                             M,
                             decltype(maxwellian),
                             decltype(maxwellian),
                             decltype(atmosphere)>(params,
                                                   domain,
                                                   { electron_index, positron_index },
                                                   { maxwellian, maxwellian },
                                                   atmosphere,
                                                   ONE,
                                                   true);

      if (fill_density > ZERO) {
        // Bulk fill: uniform neutral pairs across the active domain, mimicking
        // the plasma left behind by a previous discharge.
        const auto fill = polar_cap::FillDensity<D> { fill_density, x_surface };
        arch::InjectNonUniform<S,
                               M,
                               decltype(maxwellian),
                               decltype(maxwellian),
                               decltype(fill)>(params,
                                               domain,
                                               { electron_index, positron_index },
                                               { maxwellian, maxwellian },
                                               fill,
                                               ONE,
                                               true);
      }

      // Add only the charge imbalance. A dedicated one-species injector avoids
      // the duplicate-species offset bug in the old {2, 2} pair injection.
      // The imbalance extends through the whole domain with the prescribed GJ
      // gradient, giving rho_net(x) = rho_GJ(x) and E_x = 0.
      const auto domain_length = global_x_max - x_surface;
      raise::ErrorIf(domain_length <= ZERO,
                     "Filled start requires x_max > x_surface",
                     HERE);
      const auto extra_charge = polar_cap::ExtraPositronDensity<D> {
        x_surface,
        rho_gj_gradient,
        ONE / domain_length
      };
      // SRPIC current deposition always consumes the stored particle weight.
      // The local injector therefore uses weight only in the dilute tail.
      polar_cap::InjectSingleSpecies(domain,
                                     positron_index,
                                     maxwellian,
                                     extra_charge,
                                     params.template get<real_t>("particles.ppc0"),
                                     extra_positron_density,
                                     params.template get<real_t>(
                                       "setup.polar_cap.initial_injection."
                                       "minimum_density"),
                                     params.template get<int>(
                                       "setup.polar_cap.initial_injection."
                                       "minimum_ppc"));
    }

    auto EmissionPolicy(simtime_t, spidx_t species, Domain<S, M>& domain) const
      -> polar_cap::CurvatureEmission<M> {
      // Entity asks for a policy for each species. Only the two charged species
      // activate this custom curvature process.
      if (not qed_enabled) {
        return polar_cap::CurvatureEmission<M> { domain.random_pool() };
      }
      const auto charged = species == electron_index or species == positron_index;
      return polar_cap::CurvatureEmission<M> {
        qed_enabled and curvature_emission_enabled and charged,
        qed_enabled and curvature_drag_enabled and charged,
        filter_nonconverting_photons,
        photon_index,
        domain.species[photon_index - 1],
        domain.index(),
        photon_energy_min,
        gamma_emit,
        rho_c,
        global_x_min,
        global_x_max,
        emission_step_coefficient,
        drag_step_coefficient,
        max_drag_fraction,
        static_cast<npart_t>(max_photons_per_particle),
        domain.random_pool(),
        curvature_spectrum
      };
    }

    auto CustomParticleUpdate(simtime_t, spidx_t species, Domain<S, M>&) const
      -> polar_cap::PhotonOpacityUpdate<M> {
      return polar_cap::PhotonOpacityUpdate<M> {
        qed_enabled and pair_creation_enabled and species == photon_index,
        photon_index,
        rho_c,
        opacity_prefactor,
        b_over_bq,
        opacity_substeps
      };
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      if (debug_edge_fields and (step % debug_edge_fields_interval == 0u)) {
        DebugPrintEdgeFields(step, time, domain);
      }
      if (not qed_enabled or not pair_creation_enabled) {
        return;
      }
      auto& electrons = domain.species[electron_index - 1];
      auto& positrons = domain.species[positron_index - 1];
      auto& photons   = domain.species[photon_index - 1];

      auto conversion = polar_cap::MagneticPairCreation<D, M::CoordType>(
        photons,
        electrons,
        positrons,
        domain.index(),
        conversion_optical_depth);
      Kokkos::parallel_for("PolarCapMagneticPairCreation",
                           photons.rangeActiveParticles(),
                           conversion);
      // number_converted() performs the device-to-host copy and therefore
      // completes the kernel before host-side species metadata is updated.
      const auto converted = conversion.number_converted();
      // The photon pusher and pair kernel can both create dead entries. Mark
      // the container unsorted before output and reclaim those entries after
      // each complete recycle interval, i.e. after pair conversion rather
      // than immediately before it in the next engine step.
      photons.set_unsorted();
      if (converted > 0) {
        electrons.set_npart(electrons.npart() + converted);
        electrons.set_counter(electrons.counter() + converted);
        electrons.set_unsorted();
        positrons.set_npart(positrons.npart() + converted);
        positrons.set_counter(positrons.counter() + converted);
        positrons.set_unsorted();
      }
      if ((step + 1u) % photon_recycle_interval == 0u) {
        photons.RemoveDead();
      }
    }

    void DebugPrintEdgeFields(timestep_t    step,
                              simtime_t     time,
                              Domain<S, M>& domain) const {
      const auto ni1  = domain.mesh.n_active(in::x1);
      auto       em_h = Kokkos::create_mirror_view(domain.fields.em);
      auto       cur_h = Kokkos::create_mirror_view(domain.fields.cur);
      Kokkos::deep_copy(em_h, domain.fields.em);
      Kokkos::deep_copy(cur_h, domain.fields.cur);
      const auto first = ni1 > 8 ? ni1 - 8 : 0;
      // Upstream reference 100 cells inside the right edge.
      const auto upstream = ni1 > 100 ? ni1 - 100 : 0;
      std::printf("EDGE step=%u t=%.5f Ex_up=%+.6e Ex=",
                  static_cast<unsigned>(step),
                  static_cast<double>(time),
                  static_cast<double>(em_h(upstream + N_GHOSTS, em::ex1)));
      for (auto i = first; i < ni1; ++i) {
        std::printf(" %+.4e",
                    static_cast<double>(em_h(i + N_GHOSTS, em::ex1)));
      }
      std::printf(" Jx=");
      for (auto i = first; i < ni1; ++i) {
        std::printf(" %+.4e",
                    static_cast<double>(cur_h(i + N_GHOSTS, cur::jx1)));
      }
      // Reflection diagnostic: inward-moving (ux1 < 0) weight vs total weight,
      // per charged species, in a 0.2-wide region just inside the right edge.
      // A clean absorbing edge shows the inward fraction at noise level.
      const auto region_start = global_x_max - static_cast<real_t>(0.2);
      for (auto s = 0u; s < 2; ++s) {
        const auto& species = domain.species[s];
        auto        i1      = species.i1;
        auto        dx1     = species.dx1;
        auto        ux1     = species.ux1;
        auto        w       = species.weight;
        auto        tag     = species.tag;
        const auto& metric  = domain.mesh.metric;
        real_t      total_w = ZERO, inward_w = ZERO;
        Kokkos::parallel_reduce(
          "PolarCapReflectTotal",
          species.rangeActiveParticles(),
          Lambda(prtlidx_t p, real_t& acc) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x = metric.template convert<1, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)));
            if (x >= region_start) {
              acc += w(p);
            }
          },
          total_w);
        Kokkos::parallel_reduce(
          "PolarCapReflectInward",
          species.rangeActiveParticles(),
          Lambda(prtlidx_t p, real_t& acc) {
            if (tag(p) == ParticleTag::dead) {
              return;
            }
            const auto x = metric.template convert<1, Crd::Cd, Crd::Ph>(
              static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p)));
            if (x >= region_start and ux1(p) < ZERO) {
              acc += w(p);
            }
          },
          inward_w);
        std::printf(" A_sp%u_inw=%.3e/%.3e",
                    s + 1,
                    static_cast<double>(inward_w),
                    static_cast<double>(total_w));
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
