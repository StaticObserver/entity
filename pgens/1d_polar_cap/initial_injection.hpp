#ifndef POLAR_CAP_INITIAL_INJECTION_HPP
#define POLAR_CAP_INITIAL_INJECTION_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

namespace user::polar_cap {
  using namespace ntt;

  template <MetricClass M, class EnergyDistribution, class SpatialDistribution>
  struct SingleSpeciesInjector {
    static_assert(M::Dim == Dim::_1D,
                  "Polar-cap single-species injection is implemented only in 1D");
    static_assert(M::CoordType == Coord::Cartesian,
                  "Polar-cap single-species injection requires Cartesian coordinates");

    array_t<int*>      i1, i2, i3;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi, weight;
    array_t<short*>    tag;
    array_t<npart_t**> pld_i;

    const npart_t offset, counter, domain_index;
    const bool    use_tracking;
    const real_t  reference_ppc, density_scale, minimum_density;
    const npart_t minimum_ppc;

    const M                   metric;
    const EnergyDistribution  energy_distribution;
    const SpatialDistribution spatial_distribution;
    random_number_pool_t      random_pool;

    array_t<npart_t> injected { "polar_cap_single_species_injected" };

    SingleSpeciesInjector(Particles<M::Dim, M::CoordType>& species,
                          npart_t                          domain_idx,
                          const M&                         metric,
                          const EnergyDistribution&        energy_dist,
                          const SpatialDistribution&       spatial_dist,
                          real_t                           reference_ppc,
                          real_t                           density_scale,
                          real_t                           minimum_density,
                          npart_t                          minimum_ppc,
                          random_number_pool_t&            pool)
      : i1 { species.i1 }
      , i2 { species.i2 }
      , i3 { species.i3 }
      , dx1 { species.dx1 }
      , dx2 { species.dx2 }
      , dx3 { species.dx3 }
      , ux1 { species.ux1 }
      , ux2 { species.ux2 }
      , ux3 { species.ux3 }
      , phi { species.phi }
      , weight { species.weight }
      , tag { species.tag }
      , pld_i { species.pld_i }
      , offset { species.npart() }
      , counter { species.counter() }
      , domain_index { domain_idx }
      , use_tracking { species.use_tracking() }
      , reference_ppc { reference_ppc }
      , density_scale { density_scale }
      , minimum_density { minimum_density }
      , minimum_ppc { minimum_ppc }
      , metric { metric }
      , energy_distribution { energy_dist }
      , spatial_distribution { spatial_dist }
      , random_pool { pool } {
      Kokkos::deep_copy(injected, 0);
    }

    Inline void operator()(cellidx_t i1_) const {
      // Evaluate the target density at the cell center, then place accepted
      // macro-particles uniformly inside that cell.
      const auto i1_code = COORD(i1_);
      const coord_t<Dim::_1D> x_Cd { i1_code + HALF };
      coord_t<Dim::_1D>       x_Ph { ZERO };
      metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);

      const auto target_density = density_scale * spatial_distribution(x_Ph);
      if (target_density < minimum_density or target_density <= ZERO) {
        return;
      }

      const auto target_ppc = reference_ppc * target_density;
      npart_t   count;
      real_t    macro_weight { ONE };
      if (target_ppc < static_cast<real_t>(minimum_ppc)) {
        // Do not randomly create zero or one particle in the dilute tail.
        // Keep a fixed macro-particle sample and encode density in its weight.
        count        = minimum_ppc;
        macro_weight = target_ppc / static_cast<real_t>(minimum_ppc);
      } else {
        // Above the floor, stochastic rounding remains unbiased without
        // changing the unit-weight particle representation.
        count    = static_cast<npart_t>(target_ppc);
        auto gen = random_pool.get_state();
        if (Random<real_t>(gen) <
            target_ppc - static_cast<real_t>(count)) {
          ++count;
        }
        random_pool.free_state(gen);
      }

      for (npart_t local = 0; local < count; ++local) {
        // The atomic counter assigns non-overlapping slots across cells.
        const auto relative = Kokkos::atomic_fetch_add(&injected(), 1);
        const auto index    = offset + relative;
        if (index >= ux1.extent(0)) {
          raise::KernelError(HERE, "Single-species injector exceeds maxnpart");
        }

        auto random = random_pool.get_state();
        const auto dxi = Random<prtldx_t>(random);
        random_pool.free_state(random);

        vec_t<Dim::_3D> u_T { ZERO };
        vec_t<Dim::_3D> u_XYZ { ZERO };
        // Sample in the distribution basis, then store Cartesian momentum.
        energy_distribution(x_Ph, u_T);
        metric.template transform_xyz<Idx::T, Idx::XYZ>(x_Cd, u_T, u_XYZ);

        if (use_tracking) {
          kernel::InjectParticle<M::Dim, M::CoordType, true>(index,
                                                              i1,
                                                              i2,
                                                              i3,
                                                              dx1,
                                                              dx2,
                                                              dx3,
                                                              ux1,
                                                              ux2,
                                                              ux3,
                                                              phi,
                                                              weight,
                                                              tag,
                                                              pld_i,
                                                              { static_cast<int>(i1_code) },
                                                              { dxi },
                                                              u_XYZ,
                                                              macro_weight,
                                                              ZERO,
                                                              domain_index,
                                                              counter + relative);
        } else {
          kernel::InjectParticle<M::Dim, M::CoordType, false>(index,
                                                               i1,
                                                               i2,
                                                               i3,
                                                               dx1,
                                                               dx2,
                                                               dx3,
                                                               ux1,
                                                               ux2,
                                                               ux3,
                                                               phi,
                                                               weight,
                                                               tag,
                                                               pld_i,
                                                               { static_cast<int>(i1_code) },
                                                               { dxi },
                                                               u_XYZ,
                                                               macro_weight);
        }
      }
    }

    auto number_injected() const -> npart_t {
      // The deep copy synchronizes the injection kernel before npart/counter
      // are changed on the host.
      auto host = Kokkos::create_mirror_view(injected);
      Kokkos::deep_copy(host, injected);
      return host();
    }
  };

  template <SimEngine::type S,
            MetricClass M,
            class EnergyDistribution,
            class SpatialDistribution>
  void InjectSingleSpecies(Domain<S, M>&             domain,
                           spidx_t                    species_index,
                           const EnergyDistribution&  energy_distribution,
                           const SpatialDistribution& spatial_distribution,
                           real_t                     reference_ppc,
                           real_t                     density_scale,
                           real_t                     minimum_density,
                           int                        minimum_ppc) {
    raise::ErrorIf(species_index == 0 or species_index > domain.species.size(),
                   "Invalid species index for single-species injection",
                   HERE);
    raise::ErrorIf(reference_ppc <= ZERO,
                   "reference_ppc must be positive",
                   HERE);
    raise::ErrorIf(density_scale < ZERO,
                   "density_scale must be non-negative",
                   HERE);
    raise::ErrorIf(minimum_density < ZERO,
                   "minimum_density must be non-negative",
                   HERE);
    raise::ErrorIf(minimum_ppc <= 0,
                   "minimum_ppc must be positive",
                   HERE);
    auto& species = domain.species[species_index - 1];
    auto  kernel  = SingleSpeciesInjector<M, EnergyDistribution, SpatialDistribution>(
      species,
      domain.index(),
      domain.mesh.metric,
      energy_distribution,
      spatial_distribution,
      reference_ppc,
      density_scale,
      minimum_density,
      static_cast<npart_t>(minimum_ppc),
      domain.random_pool());
    Kokkos::parallel_for("PolarCapSingleSpeciesInjection",
                         domain.mesh.rangeActiveCells(),
                         kernel);
    const auto count = kernel.number_injected();
    // Particle arrays are already populated; publish the new active range and
    // tracking-counter range to the Entity container.
    species.set_npart(species.npart() + count);
    species.set_counter(species.counter() + count);
    species.set_unsorted();
  }

} // namespace user::polar_cap

#endif // POLAR_CAP_INITIAL_INJECTION_HPP
