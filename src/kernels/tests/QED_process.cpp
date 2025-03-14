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
using namespace kernel::QED;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    size_t N_e = 1e4;
    size_t N_ph = 1e8;
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, N_e, PrtlPusher::BORIS, false, Cooling::NONE, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, N_ph, PrtlPusher::PHOTON, false, Cooling::NONE, 1);
    

    auto& ux1 = electron.ux1;
    auto& ux2 = electron.ux2;
    auto& ux3 = electron.ux3;
    auto& weight = electron.weight;
    auto& tag = electron.tag;
    auto& i1 = electron.i1;
    auto& dx1 = electron.dx1;

    Kokkos::parallel_for("Init", N_e, KOKKOS_LAMBDA(index_t p) {
      ux1(p) = 0.0;
      ux2(p) = 0.0;
      ux3(p) = 0.0;
      weight(p) = 1.0;
      tag(p) = ParticleTag::alive;
      i1(p) = 0;
      dx1(p) = 0;
    });

    const real_t e_min { 2.0 };
    const real_t gamma_emit { 1e4 };
    const real_t gamma_rad  { 6.7e5 };
    const real_t gamma_pc { 7.2e7 };
    const real_t coeff { SQR(THREE / TWO) * constant::SQRT3 / constant::SQRT2 / constant::PI
                         * math::sqrt(gamma_pc) * SQR(SQR(gamma_emit / gamma_rad) * gamma_emit) };
    const real_t rho { 1.0 };
    
    random_number_pool_t random_pool;

    const real_t gamma { 1e6 };
    const real_t e_ph { CUBE(gamma / gamma_emit) / rho };

    CurvatureEmission_kernel<Dim::_1D, Coord::Cart> curvature_emission(electron, photon, e_min, coeff, gamma_emit, rho, 100, photons.npart(), random_pool);

    Kokkos::parallel_for("CurvatureEmission", N_e, curvature_emission);

    auto n_injected = curvature_emission.num_ph();
    auto total_ph = photon.npart() + n_injected;
    photons.set_npart(total_ph);

    size_t num_bins { 100 };
    real_t log_min { math::log(e_min / e_ph) };
    real_t log_max { ONE };
    auto dx = (log_max - log_min) / num_bins;

    auto scatter_ebins = Kokkos::Experimental::ScatterView<size_t>("scatter_ebins", num_bins);
    auto ebins = scatter_ebins.access();

    Kokkos::parallel_for("Scatter", photon.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      if (photon.tag(p) != ParticleTag::alive) {
        return;
      }
      auto log_e = math::log(photon.pld_ph(p, 0) / e_ph);
      auto bin = static_cast<index_t>((log_e - log_min) / dx);
      if (bin < 0) {
        ebins(0) += 1;
      }else if (bin >= num_bins) {
        ebins(num_bins - 1) += 1;
      }else {
        ebins(bin) += 1;
      }
    }); 
    Kokkos::Experimental::contribute(ebins, scatter_ebins);
    
    auto ebins_h = Kokkos::create_mirror_view(ebins);
    Kokkos::deep_copy(ebins_h, ebins);

    std::vector<real_t> bin_centers(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
      bin_centers[i] = math::exp(log_min + (i + 0.5) * dx);
    }

    std::ofstream file("pdf_ph.dat");
    for (size_t i = 0; i < num_bins; ++i) {
      file << bin_centers[i] << " " << ebins_h(i) / photon.npart() << std::endl;
    }
    file.close();


  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  Kokkos::finalize();

  return 0;
}