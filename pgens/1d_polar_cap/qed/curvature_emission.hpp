#ifndef POLAR_CAP_CURVATURE_EMISSION_HPP
#define POLAR_CAP_CURVATURE_EMISSION_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/injectors.hpp"

#include "curvature_spectrum.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include <limits>
#include <vector>

namespace user::polar_cap {
  using namespace ntt;

  Inline auto MaximumAbsSineOnInterval(real_t theta0, real_t theta1) -> real_t {
    const auto lower = theta0 < theta1 ? theta0 : theta1;
    const auto upper = theta0 < theta1 ? theta1 : theta0;
    const auto pi = static_cast<real_t>(3.14159265358979323846);
    const auto half_pi = static_cast<real_t>(1.57079632679489661923);
    const auto tolerance =
      static_cast<real_t>(64.0) * std::numeric_limits<real_t>::epsilon() *
      (static_cast<real_t>(1.0) + math::abs(lower) + math::abs(upper));
    const auto first_peak = half_pi +
                            pi * math::ceil((lower - tolerance - half_pi) / pi);
    if (first_peak <= upper + tolerance) {
      return static_cast<real_t>(1.0);
    }
    const auto sine0 = math::abs(math::sin(theta0));
    const auto sine1 = math::abs(math::sin(theta1));
    return sine0 > sine1 ? sine0 : sine1;
  }

  template <MetricClass M>
  struct CurvatureEmission {
    static_assert(M::Dim == Dim::_1D,
                  "Curvature emission is implemented only for 1D polar-cap runs");
    static_assert(M::CoordType == Coord::Cartesian,
                  "Curvature emission requires Cartesian coordinates");

    struct Payload {
      // Per-parent data passed from shouldEmit() to emit(). The probability
      // interval represents the curvature CCDF truncated by both energy bounds.
      npart_t n_photons { 0 };
      real_t  weight_scale { ONE };
      real_t  photon_energy_scale { ZERO };
      real_t  photon_energy_min { ZERO };
      real_t  photon_energy_max { ZERO };
      real_t  ccdf_at_max_energy { ZERO };
      real_t  ccdf_interval { ZERO };
    };

    const bool    emission_enabled;
    const bool    apply_drag;
    const bool    filter_nonconverting_photons;
    const spidx_t photon_species_index;
    const real_t  photon_energy_min;
    const real_t  gamma_emit;
    const real_t  rho_c;
    const real_t  global_x_min, global_x_max;
    const real_t  emission_step_coefficient;
    const real_t  drag_step_coefficient;
    const real_t  max_drag_fraction;
    const npart_t max_photons_per_particle;

    CurvatureSpectrum    spectrum;
    random_number_pool_t random_pool;

    array_t<npart_t> photon_injected { "polar_cap_photon_injected" };
    const npart_t    photon_offset, photon_counter, domain_index;
    const bool       photon_tracking;

    array_t<int*>      photon_i1, photon_i2, photon_i3;
    array_t<prtldx_t*> photon_dx1, photon_dx2, photon_dx3;
    array_t<real_t*>   photon_ux1, photon_ux2, photon_ux3;
    array_t<real_t*>   photon_phi, photon_weight;
    array_t<short*>    photon_tag;
    array_t<real_t**>  photon_pld_r;
    array_t<npart_t**> photon_pld_i;

    CurvatureEmission(bool                              emission_enabled,
                      bool                              apply_drag,
                      bool                              filter_nonconverting_photons,
                      spidx_t                           photon_species_index,
                      Particles<M::Dim, M::CoordType>& photon_species,
                      npart_t                           domain_idx,
                      real_t                            photon_energy_min,
                      real_t                            gamma_emit,
                      real_t                            rho_c,
                      real_t                            global_x_min,
                      real_t                            global_x_max,
                      real_t                            emission_step_coefficient,
                      real_t                            drag_step_coefficient,
                      real_t                            max_drag_fraction,
                      npart_t                           max_photons_per_particle,
                      random_number_pool_t&             pool,
                      const CurvatureSpectrum&          spectrum)
      : emission_enabled { emission_enabled }
      , apply_drag { apply_drag }
      , filter_nonconverting_photons { filter_nonconverting_photons }
      , photon_species_index { photon_species_index }
      , photon_energy_min { photon_energy_min }
      , gamma_emit { gamma_emit }
      , rho_c { rho_c }
      , global_x_min { global_x_min }
      , global_x_max { global_x_max }
      , emission_step_coefficient { emission_step_coefficient }
      , drag_step_coefficient { drag_step_coefficient }
      , max_drag_fraction { max_drag_fraction }
      , max_photons_per_particle { max_photons_per_particle }
      , spectrum { spectrum }
      , random_pool { pool }
      , photon_offset { photon_species.npart() }
      , photon_counter { photon_species.counter() }
      , domain_index { domain_idx }
      , photon_tracking { photon_species.use_tracking() }
      , photon_i1 { photon_species.i1 }
      , photon_i2 { photon_species.i2 }
      , photon_i3 { photon_species.i3 }
      , photon_dx1 { photon_species.dx1 }
      , photon_dx2 { photon_species.dx2 }
      , photon_dx3 { photon_species.dx3 }
      , photon_ux1 { photon_species.ux1 }
      , photon_ux2 { photon_species.ux2 }
      , photon_ux3 { photon_species.ux3 }
      , photon_phi { photon_species.phi }
      , photon_weight { photon_species.weight }
      , photon_tag { photon_species.tag }
      , photon_pld_r { photon_species.pld_r }
      , photon_pld_i { photon_species.pld_i } {
      Kokkos::deep_copy(photon_injected, 0);
    }

    // Drag-only construction for QED-off runs. The policy still uses Entity's
    // custom-emission hook so recoil is applied inside the pusher, but it owns
    // no photon destination and reports no injected species.
    CurvatureEmission(bool                  apply_drag,
                      real_t                drag_step_coefficient,
                      real_t                max_drag_fraction,
                      random_number_pool_t& pool)
      : emission_enabled { false }
      , apply_drag { apply_drag }
      , filter_nonconverting_photons { false }
      , photon_species_index { 0 }
      , photon_energy_min { ONE }
      , gamma_emit { static_cast<real_t>(2.0) }
      , rho_c { ONE }
      , global_x_min { ZERO }
      , global_x_max { ZERO }
      , emission_step_coefficient { ZERO }
      , drag_step_coefficient { drag_step_coefficient }
      , max_drag_fraction { max_drag_fraction }
      , max_photons_per_particle { 0 }
      , spectrum {}
      , random_pool { pool }
      , photon_offset { 0 }
      , photon_counter { 0 }
      , domain_index { 0 }
      , photon_tracking { false } {}

    Inline auto shouldEmit(const coord_t<M::PrtlDim>&,
                           const coord_t<M::PrtlDim>& x_Ph,
                           const vec_t<Dim::_3D>& u_Ph,
                           const vec_t<Dim::_3D>&,
                           const vec_t<Dim::_3D>&,
                           vec_t<Dim::_3D>& delta_u_Ph,
                           Payload&         payload) const -> Kokkos::pair<bool, bool> {
      if (not emission_enabled and not apply_drag) {
        return { false, false };
      }

      const auto u_mag = NORM(u_Ph[0], u_Ph[1], u_Ph[2]);
      if (u_mag <= ZERO) {
        return { false, false };
      }
      const auto gamma = math::sqrt(ONE + SQR(u_mag));
      const auto gamma_ratio = gamma / gamma_emit;

      auto retained_energy_min = photon_energy_min;
      auto retain_pair_capable_spectrum = true;
      if (emission_enabled and filter_nonconverting_photons) {
        const auto transverse_u = math::sqrt(SQR(u_Ph[1]) + SQR(u_Ph[2]));
        const auto initial_theta = math::atan2(transverse_u,
                                               math::abs(u_Ph[0]));
        auto exit_theta = initial_theta;
        if (math::abs(u_Ph[0]) > static_cast<real_t>(0.0)) {
          const auto exit_x = u_Ph[0] > static_cast<real_t>(0.0)
                                ? global_x_max
                                : global_x_min;
          exit_theta += (exit_x - x_Ph[0]) / rho_c;
        }
        const auto maximum_path_sine = MaximumAbsSineOnInterval(initial_theta,
                                                                 exit_theta);
        if (maximum_path_sine <= static_cast<real_t>(0.0)) {
          retain_pair_capable_spectrum = false;
        } else {
          const auto threshold_margin =
            static_cast<real_t>(1.0) -
            static_cast<real_t>(64.0) *
              std::numeric_limits<real_t>::epsilon();
          const auto pair_capable_energy = (TWO / maximum_path_sine) *
                                           threshold_margin;
          if (pair_capable_energy > retained_energy_min) {
            retained_energy_min = pair_capable_energy;
          }
        }
      }

      // No sampled photon may carry more than the parent's kinetic energy.
      const auto photon_energy_max = gamma - ONE;
      if (emission_enabled and retain_pair_capable_spectrum and
          photon_energy_max > retained_energy_min) {
        payload.photon_energy_scale = CUBE(gamma_ratio) / rho_c;
        payload.photon_energy_min   = retained_energy_min;
        payload.photon_energy_max   = photon_energy_max;

        const auto zeta = retained_energy_min / payload.photon_energy_scale;
        const auto max_normalized_energy = photon_energy_max /
                                           payload.photon_energy_scale;
        const auto ccdf_at_min_energy = spectrum.ccdf(zeta);
        payload.ccdf_at_max_energy = spectrum.ccdf(max_normalized_energy);
        const auto ccdf_interval = ccdf_at_min_energy -
                                   payload.ccdf_at_max_energy;
        // Kokkos::max takes its arguments by reference. Passing Entity's
        // namespace-scope ZERO constant would ODR-use a host object in device
        // code, so clamp with a typed literal instead.
        payload.ccdf_interval = ccdf_interval > static_cast<real_t>(0.0)
                                  ? ccdf_interval
                                  : static_cast<real_t>(0.0);

        // Integrating only over the retained CCDF interval removes the
        // kinematically forbidden high-energy tail from the photon count.
        const auto expected = emission_step_coefficient *
                              payload.ccdf_interval * gamma /
                              (CUBE(gamma_emit) * rho_c);
        if (expected >= static_cast<real_t>(max_photons_per_particle)) {
          // Cap macro-particle count without changing the represented photon
          // number: excess multiplicity is carried by macro-particle weight.
          payload.n_photons = max_photons_per_particle;
          payload.weight_scale = expected /
                                 static_cast<real_t>(max_photons_per_particle);
        } else {
          auto sampled = static_cast<npart_t>(expected);
          auto gen     = random_pool.get_state();
          if (Random<real_t>(gen) < expected - static_cast<real_t>(sampled)) {
            ++sampled;
          }
          random_pool.free_state(gen);
          payload.n_photons = sampled;
        }
      }

      // Continuous recoil is an ensemble loss term, independent of whether a
      // macro-photon happened to be sampled in this timestep.
      auto drag_fraction = drag_step_coefficient * CUBE(gamma);
      if (drag_fraction > max_drag_fraction) {
        drag_fraction = max_drag_fraction;
      }
      const auto do_drag = apply_drag and drag_fraction > ZERO;
      // Even in emission-only mode, delta_u supplies the parent direction to
      // Entity's emission dispatcher; it is applied only when do_drag is true.
      const auto direction_scale = do_drag ? drag_fraction : ONE / u_mag;
      delta_u_Ph[0] = -direction_scale * u_Ph[0];
      delta_u_Ph[1] = -direction_scale * u_Ph[1];
      delta_u_Ph[2] = -direction_scale * u_Ph[2];

      return { payload.n_photons > 0, do_drag };
    }

    Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                     const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                     const vec_t<Dim::_3D>&           direction,
                     real_t                           parent_weight,
                     real_t                           phi,
                     const Payload&                   payload) const {
      const auto relative_offset = Kokkos::atomic_fetch_add(&photon_injected(),
                                                            payload.n_photons);
      // Each parent atomically reserves one contiguous block in photon arrays.
      const auto first = photon_offset + relative_offset;
      if (first + payload.n_photons > photon_ux1.extent(0)) {
        raise::KernelError(HERE, "Curvature emission exceeds photon maxnpart");
      }
      // Entity's NORM macro is strictly three-dimensional.  Compute the
      // transverse magnitude explicitly so the pitch angle remains
      // atan2(sqrt(u_y^2 + u_z^2), |u_x|) without relying on a 2-D overload.
      const auto transverse_direction = math::sqrt(
        SQR(direction[1]) + SQR(direction[2]));
      const auto initial_theta = math::atan2(transverse_direction,
                                             math::abs(direction[0]));

      for (npart_t local = 0; local < payload.n_photons; ++local) {
        // Uniform probability inside the retained CCDF interval generates the
        // conditional curvature spectrum between the two energy bounds.
        auto gen = random_pool.get_state();
        auto probability = payload.ccdf_at_max_energy +
                           payload.ccdf_interval * Random<real_t>(gen);
        if (probability <= ZERO) {
          probability = payload.ccdf_interval * static_cast<real_t>(1.0e-7);
        }
        const auto sampled_energy = spectrum.inverse_ccdf(probability) *
                                    payload.photon_energy_scale;
        const auto photon_energy = math::max(
          payload.photon_energy_min,
          math::min(sampled_energy, payload.photon_energy_max));
        random_pool.free_state(gen);

        const auto index  = first + local;
        const auto weight = parent_weight * payload.weight_scale;
        const vec_t<Dim::_3D> photon_u { photon_energy * direction[0],
                                         photon_energy * direction[1],
                                         photon_energy * direction[2] };
        if (photon_tracking) {
          kernel::InjectParticle<M::Dim, M::CoordType, true>(index,
                                                              photon_i1,
                                                              photon_i2,
                                                              photon_i3,
                                                              photon_dx1,
                                                              photon_dx2,
                                                              photon_dx3,
                                                              photon_ux1,
                                                              photon_ux2,
                                                              photon_ux3,
                                                              photon_phi,
                                                              photon_weight,
                                                              photon_tag,
                                                              photon_pld_i,
                                                              xi_Cd,
                                                              dxi_Cd,
                                                              photon_u,
                                                              weight,
                                                              phi,
                                                              domain_index,
                                                              photon_counter + relative_offset + local);
        } else {
          kernel::InjectParticle<M::Dim, M::CoordType, false>(index,
                                                               photon_i1,
                                                               photon_i2,
                                                               photon_i3,
                                                               photon_dx1,
                                                               photon_dx2,
                                                               photon_dx3,
                                                               photon_ux1,
                                                               photon_ux2,
                                                               photon_ux3,
                                                               photon_phi,
                                                               photon_weight,
                                                               photon_tag,
                                                               photon_pld_i,
                                                               xi_Cd,
                                                               dxi_Cd,
                                                               photon_u,
                                                               weight,
                                                               phi);
        }
        photon_pld_r(index, 0) = photon_energy;
        photon_pld_r(index, 1) = ZERO;
        // Magnetic opacity depends on the unsigned angle to the local B axis.
        photon_pld_r(index, 2) = initial_theta;
      }
    }

    auto emitted_species_indices() const -> std::vector<spidx_t> {
      return emission_enabled ? std::vector<spidx_t> { photon_species_index }
                              : std::vector<spidx_t> {};
    }

    auto numbers_injected() const -> std::vector<npart_t> {
      if (not emission_enabled) {
        return {};
      }
      // Entity consumes this count after each charged-species pusher call and
      // advances photon npart/counter before the next species is processed.
      auto host = Kokkos::create_mirror_view(photon_injected);
      Kokkos::deep_copy(host, photon_injected);
      return { host() };
    }
  };

} // namespace user::polar_cap

#endif // POLAR_CAP_CURVATURE_EMISSION_HPP
