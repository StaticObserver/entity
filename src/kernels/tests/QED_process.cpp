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

class init_particles {
  public:
    init_particles(Particles<Dim::_1D, Coord::Cart>& spec_, const real_t gamma_)
      : spec(spec_), gamma(gamma_) {}
    
    const real_t u0 = math::sqrt(SQR(gamma) - ONE);
    void operator()(index_t p) const {
      spec.ux1(p) = u0;
      spec.ux2(p) = 0.0;
      spec.ux3(p) = 0.0;
      spec.weight(p) = 1.0;
      spec.i1(p) = 0;
      spec.dx1(p) = 0.0;
      spec.tag(p) = ParticleTag::alive;
    }

  private:
    Particles<Dim::_1D, Coord::Cart>& spec;
    const real_t gamma;
};


auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, 100, PrtlPusher::Boris, false, Cooling::None, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, 1e8, PrtlPusher::Boris, false, Cooling::None, 1);
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


    Kokkos::parallel_for("GenerateEnergyBins", photon.npart(), KOKKOS_LAMBDA(index_t i) {
        if (photon.tag(i) != ParticleTag::alive) {
          return;
        }
        const real_t en = math::log10(spec.pld(i, 0) / e_ph);
        const index_t bin = (en - min) / dx;
        if (bin < 0 || bin >= num_bins) {
          printf("bin %d out of range\n", bin);
          return;
        }
        energy_bins(bin) += 1;
    });

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