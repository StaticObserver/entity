#include "kernels/QED_process.hpp"

#include "global.h"
#include "enums.h"
#include "utils/numeric.h"

#include "arch/kokkos_aliases.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>
#include <vector>

using namespace ntt;

struct init_particles {
    
    array_t<real_t*> ux1, ux2, ux3, weight;
    array_t<prtldx_t*> dx1;
    array_t<int*> i1;
    array_t<short*> tag;
    const real_t gamma;

    
    init_particles(Particles<Dim::_1D, Coord::Cart>& spec_, const real_t gamma_)
      : spec(spec_), gamma(gamma_) {}
    
    
    Inline void operator()(index_t p) const {
      const real_t u0 = math::sqrt(SQR(gamma) - ONE);
      spec.ux1(p) = u0;
      spec.ux2(p) = 0.0;
      spec.ux3(p) = 0.0;
      spec.weight(p) = 1.0;
      spec.i1(p) = 0;
      spec.dx1(p) = 0.0;
      spec.tag(p) = ParticleTag::alive;
    }

};

struct generate_bins{
    Particles<Dim::_1D, Coord::Cart> spec_ph;
    const real_t e_ph;
    const real_t min;
    const real_t max;
    const real_t dx;
    const size_t num_bins;
    array_t<int*> energy_bins;

    generate_bins(Particles<Dim::_1D, Coord::Cart>& spec_ph_,
                  const real_t e_ph_,
                  const real_t min_,
                  const real_t max_,
                  const real_t dx_,
                  const size_t num_bins_,
                  array_t<int*>& energy_bins_)
        : spec_ph(spec_ph_),
          e_ph(e_ph_),
          min(min_),
          max(max_),
          dx(dx_),
          num_bins(num_bins_),
          energy_bins(energy_bins_){}

    Inline void operator()(index_t i) const{
        if (spec_ph.tag(i) != ParticleTag::alive){
            return;
        }
        const real_t en = math::log10(spec_ph.pld(i, 0) / e_ph);
        const index_t bin = (en - min) / dx;
        if (bin < 0 || bin >= num_bins){
            printf("bin %lu out of range\n", bin);
            return;
        }
        energy_bins(bin) += 1;
    }
};


auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, 100, PrtlPusher::BORIS, false, Cooling::NONE, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, 1e8, PrtlPusher::PHOTON, false, Cooling::NONE, 1);
    using namespace kernel::QED;
    const real_t e_min { 2.0 };
    const real_t gamma_emit { 1e4 };
    const real_t gamma_rad  { 6.7e5 };
    const real_t gamma_pc { 7.2e7 };
    const real_t coeff { SQR(THREE / TWO) * constant::SQRT3 / constant::SQRT2 / constant::PI
                         * math::sqrt(gamma_pc) * SQR(SQR(gamma_emit / gamma_rad) * gamma_emit) };
    const real_t rho { 1.0 };
    random_number_pool_t random_pool;

    const real_t gamma { 1e6 };

    init_particles init(electron, gamma);
    Kokkos::parallel_for("InitParticles", electron.npart(), init);

    CurvatureEmission_kernel<Dim::_1D, Coord::Cart> curvature_emission(electron, photon, e_min, coeff, gamma_emit, rho, random_pool);

    Kokkos::parallel_for("CurvatureEmission", electron.npart(), curvature_emission);

    const size_t num_bins = 100;
    array_t<int*> energy_bins { "energy_bins", num_bins };
    const real_t e_ph = CUBE(gamma / gamma_emit) / rho;
    const real_t min = math::log10(e_min / e_ph);
    const real_t max = 1.0;
    const real_t dx = (max - min) / num_bins;

    generate_bins gen_bins(photon, e_ph, min, max, dx, num_bins, energy_bins);
    Kokkos::parallel_for("GenerateBins", photon.npart(), gen_bins);

    auto energy_bins_h = Kokkos::create_mirror_view(energy_bins);
    Kokkos::deep_copy(energy_bins_h, energy_bins);

    auto bin_centers = std::vector<real_t>(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
      bin_centers[i] = math::pow(10.0, min + (i + 0.5) * dx);
    }

    std::fstream file;
    file.open("energy_bins.dat", std::ios::out);
    for (size_t i = 0; i < num_bins; ++i) {
      file << bin_centers[i] << " " << energy_bins_h(i) << std::endl;
    }
    file.close();


  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  Kokkos::finalize();

  return 0;
}