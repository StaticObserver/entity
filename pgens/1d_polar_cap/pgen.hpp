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

#include "boundary_flux.hpp"
#include "initial_injection.hpp"
#include "qed/curvature_emission.hpp"
#include "qed/curvature_spectrum.hpp"
#include "qed/magnetic_pair_creation.hpp"
#include "qed/photon_opacity.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
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

    // Prescribed 1D polar-cap background. The electric field is the analytic
    // integral of the excess-positron charge minus the fixed
    // Goldreich-Julian background used by ExtraPositronDensity below.
    template <Dimension D>
    struct InitialFields {
      const real_t b0, e_coefficient, x_surface, atmosphere_width;

      InitialFields(real_t b0,
                    real_t e_coefficient,
                    real_t x_surface,
                    real_t atmosphere_width)
        : b0 { b0 }
        , e_coefficient { e_coefficient }
        , x_surface { x_surface }
        , atmosphere_width { atmosphere_width } {}

      Inline auto bx1(const coord_t<D>&) const -> real_t {
        return b0;
      }

      Inline auto charge_primitive(real_t value) const -> real_t {
        // Integral, in transition-length coordinates, of f(t) - 1 from
        // t = -inf to t = value for the centered logistic profile
        // f(t) = 1 / (1 + exp(t)).  At s = -inf the integral is zero,
        // giving E_x = 0 deep inside the atmosphere.
        return -math::log(ONE + math::exp(value));
      }

      Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < x_surface) {
          return ZERO;
        }

        // The excess positron density follows a logistic centered at the
        // atmosphere edge:
        //
        //   f(s) = 1 / (1 + exp(s)),
        //   s    = (x - atmosphere_edge) / transition.
        //
        // dE_x/dx = e_coefficient * (f - 1).  The primitive (charge_primitive)
        // is zero at s = -inf, giving E_x -> 0 deep inside the atmosphere.
        const auto atmosphere_edge = x_surface + atmosphere_width;
        const auto transition = static_cast<real_t>(0.03) * atmosphere_width;
        const auto cutoff_s = static_cast<real_t>(11.0);
        const auto s = (x_Ph[0] - atmosphere_edge) / transition;

        if (s < cutoff_s) {
          return e_coefficient * transition * charge_primitive(s);
        }

        // The profile is explicitly zero after 11 transition lengths
        // (x = x_surface + 1.33 * atmosphere_width). Continue E_x linearly
        // there because only the fixed negative background remains.
        const auto cutoff_x = atmosphere_edge + cutoff_s * transition;
        const auto field_at_cutoff =
          e_coefficient * transition * charge_primitive(cutoff_s);
        return field_at_cutoff -
               e_coefficient * (x_Ph[0] - cutoff_x);
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

    // The excess positrons fill the atmosphere at unit normalized density.
    // The S-shaped decline is centered at the atmosphere edge (f = 0.5) and
    // is truncated after 0.33 atmosphere widths. With peak density equal to
    // initial_e_coefficient, InitialFields closes the prescribed Gauss law.
    template <Dimension D>
    struct ExtraPositronDensity {
      const real_t x_surface, atmosphere_width;

      ExtraPositronDensity(real_t x_surface, real_t atmosphere_width)
        : x_surface { x_surface }
        , atmosphere_width { atmosphere_width } {}

      Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
        if (x_Ph[0] < x_surface) {
          return ZERO;
        }

        const auto atmosphere_edge = x_surface + atmosphere_width;
        const auto transition = static_cast<real_t>(0.03) * atmosphere_width;
        // Center the logistic descent at the atmosphere edge.
        // f(s) = 1 / (1 + exp(s)),   s = (x - atmosphere_edge) / transition.
        // f(-inf) = 1 (plateau inside), f(0) = 0.5 (center at edge),
        // f(+inf) = 0 (tail outside).
        const auto s = (x_Ph[0] - atmosphere_edge) / transition;
        if (s >= static_cast<real_t>(11.0)) {
          return ZERO;
        }

        return ONE / (ONE + math::exp(s));
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
    const real_t initial_e_coefficient, extra_positron_density;

    const bool           qed_enabled, curvature_drag_enabled;
    const bool           curvature_emission_enabled, pair_creation_enabled;
    const bool           filter_nonconverting_photons;
    const bool           radiation_reaction_enabled;
    const real_t         rho_c, gamma_emit, photon_energy_min;
    const real_t         global_x_min, global_x_max;
    const real_t         gamma_rad, reference_electric_field;
    const real_t         emission_step_coefficient, drag_step_coefficient;
    const real_t         opacity_prefactor, b_over_bq;
    const real_t         max_drag_fraction, conversion_optical_depth;
    const int max_photons_per_particle;
    const int opacity_substeps;
    const timestep_t photon_recycle_interval;
    const bool       boundary_flux_compensation_enabled;
    const real_t     boundary_flux_ampere_coeff;
    const real_t     boundary_flux_inv_ppc0;

    polar_cap::InitialFields<D>          init_flds;
    polar_cap::MagnetosphericCurrent<D> ext_current;
    polar_cap::CurvatureSpectrum         curvature_spectrum;
    array_t<real_t*>                     boundary_flux_missing;

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
      , initial_e_coefficient { params.template get<real_t>(
          "setup.polar_cap.initial_e_coefficient",
          params.template get<real_t>("scales.larmor0") /
            SQR(params.template get<real_t>("scales.skindepth0"))) }
      , extra_positron_density { params.template get<real_t>(
          "setup.polar_cap.extra_positron_density",
          initial_e_coefficient) }
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
      , radiation_reaction_enabled { params.template get<bool>(
          "setup.polar_cap.radiation_reaction.enable",
          false) }
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
            : radiation_reaction_enabled
                ? params.template get<real_t>(
                    "setup.polar_cap.radiation_reaction.gamma_rad")
                : static_cast<real_t>(2.0)
        }
      , reference_electric_field {
          qed_enabled and curvature_drag_enabled
            ? params.template get<real_t>(
                "setup.polar_cap.qed.reference_electric_field")
            : radiation_reaction_enabled
                ? params.template get<real_t>(
                    "setup.polar_cap.radiation_reaction."
                    "reference_electric_field")
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
          // Both the QED curvature recoil and the QED-off drag-only mode use
          // the same physical normalization: K*gamma^4 balances
          // omegaB0*E_ref at gamma_rad. This removes any dependence on
          // macro-particle charge or ppc0.
          (radiation_reaction_enabled or
           (qed_enabled and curvature_drag_enabled))
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
          "setup.polar_cap.radiation_reaction.max_drag_fraction",
          params.template get<real_t>("setup.polar_cap.qed.max_drag_fraction",
                                      static_cast<real_t>(0.2))) }
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
      , boundary_flux_compensation_enabled { params.template get<bool>(
          "setup.polar_cap.boundary_flux_compensation.enable",
          false) }
      , boundary_flux_ampere_coeff {
          // Same current-to-field coefficient as CurrentsAmpere:
          // coeff = -dt * q0 / (B0 * V0).
          boundary_flux_compensation_enabled
            ? -params.template get<real_t>("algorithms.timestep.dt") *
                params.template get<real_t>("scales.q0") /
                (params.template get<real_t>("scales.B0") *
                 params.template get<real_t>("scales.V0"))
            : ZERO
        }
      , boundary_flux_inv_ppc0 {
          boundary_flux_compensation_enabled
            ? ONE / params.template get<real_t>("particles.ppc0")
            : ZERO
        }
      , init_flds { b0, initial_e_coefficient, x_surface, atmosphere_width }
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
      , boundary_flux_missing { "polar_cap_boundary_flux_missing",
                                polar_cap::BoundaryFluxAccSize } {
      // Validate signed host-side values before passing compact values into
      // device policies. In particular, negative integers must not wrap.
      raise::ErrorIf(atmosphere_width <= ZERO,
                     "Atmosphere width must be positive",
                     HERE);
      raise::ErrorIf(temperature < ZERO,
                     "temperature must be non-negative",
                     HERE);
      raise::ErrorIf(extra_positron_density < ZERO,
                     "extra_positron_density must be non-negative",
                     HERE);
      raise::ErrorIf(
        params
            .template get<boundaries_t<real_t>>(
              "grid.boundaries.match.ds")[0]
            .second <= ZERO,
        "right MATCH width must be positive",
        HERE);
      raise::ErrorIf(qed_enabled and radiation_reaction_enabled,
                     "QED and explicit radiation reaction cannot both be enabled",
                     HERE);
      if (boundary_flux_compensation_enabled) {
        // The compensation predicts the right-edge ABSORB kill in the pusher;
        // any other right particle boundary makes the prediction wrong.
        const auto particle_boundaries = metadomain.mesh().prtl_bc();
        raise::ErrorIf(
          particle_boundaries[0].second != PrtlBC::ABSORB,
          "boundary_flux_compensation requires particles=ABSORB at x1 max",
          HERE);
      }
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
      if (radiation_reaction_enabled or
          (qed_enabled and curvature_drag_enabled)) {
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
        CustomParticleUpdatePolicyClass<
          polar_cap::BoundaryFluxCompensationUpdate<M>,
          M>,
        "BoundaryFluxCompensationUpdate does not satisfy "
        "CustomParticleUpdatePolicyClass");
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
        if (radiation_reaction_enabled) {
          raise::ErrorIf(
            electrons.emission_policy_flag() != EmissionType::CUSTOM or
              positrons.emission_policy_flag() != EmissionType::CUSTOM,
            "Radiation-reaction species must use emission=custom",
            HERE);
        } else {
          raise::ErrorIf(
            electrons.emission_policy_flag() != EmissionType::NONE or
              positrons.emission_policy_flag() != EmissionType::NONE,
            "QED-off electron and positron species must use emission=none",
            HERE);
        }
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

      // Add only the charge imbalance. A dedicated one-species injector avoids
      // the duplicate-species offset bug in the old {2, 2} pair injection.
      const auto extra_charge = polar_cap::ExtraPositronDensity<D> {
        x_surface,
        atmosphere_width
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
      const auto charged = species == electron_index or species == positron_index;
      if (not qed_enabled) {
        return polar_cap::CurvatureEmission<M> {
          radiation_reaction_enabled and charged,
          drag_step_coefficient,
          max_drag_fraction,
          domain.random_pool()
        };
      }
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
      -> polar_cap::BoundaryFluxCompensationUpdate<M> {
      // The same policy type is dispatched for every species. The opacity
      // update is restricted to photons by its own enabled flag; flux
      // compensation is restricted to the charged species.
      const auto charged = species == electron_index or species == positron_index;
      return polar_cap::BoundaryFluxCompensationUpdate<M> {
        boundary_flux_compensation_enabled and charged,
        boundary_flux_missing,
        polar_cap::PhotonOpacityUpdate<M> {
          qed_enabled and pair_creation_enabled and species == photon_index,
          photon_index,
          rho_c,
          opacity_prefactor,
          b_over_bq,
          opacity_substeps
        }
      };
    }

    void CustomPostStep(timestep_t step, simtime_t, Domain<S, M>& domain) {
      if (boundary_flux_compensation_enabled) {
        ApplyBoundaryFluxCompensation(domain);
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

    void ApplyBoundaryFluxCompensation(Domain<S, M>& domain) {
      // Only the domain owning the global right edge collects missing flux:
      // internal MPI boundaries are SYNC, never ABSORB, so this gate also
      // keeps the shared accumulator single-writer under domain
      // decomposition. The engine calls CustomPostStep after CurrentsAmpere,
      // so the correction uses the same J -> E coefficient the deposited
      // current would have seen this step.
      if (domain.mesh.prtl_bc()[0].second != PrtlBC::ABSORB) {
        return;
      }
      const auto ni1 = domain.mesh.n_active(in::x1);
      auto       nslots = static_cast<ncells_t>(polar_cap::BoundaryFluxAccSize);
      if (nslots > ni1) {
        nslots = ni1;
      }
      Kokkos::parallel_for(
        "PolarCapBoundaryFluxCompensation",
        nslots,
        polar_cap::BoundaryFluxApplier<M::Dim> {
          domain.fields.em,
          domain.fields.cur,
          boundary_flux_missing,
          static_cast<cellidx_t>(ni1 - 1 + N_GHOSTS),
          boundary_flux_ampere_coeff,
          boundary_flux_inv_ppc0
        });
      // deep_copy fences, so the applier finishes before the accumulator is
      // cleared for the next step.
      Kokkos::deep_copy(boundary_flux_missing, ZERO);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
