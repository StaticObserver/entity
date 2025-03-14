#include "kernels/QED_process.hpp"

#include "global.h"
#include "enums.h"
#include "utils/numeric.h"

#include "arch/kokkos_aliases.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <iostream>
#include <fstream>
#include <vector>

using namespace ntt;
using namespace kernel::QED;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    size_t N_e = 1e3;
    size_t N_ph = 1e6;
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, N_e, PrtlPusher::BORIS, false, Cooling::NONE, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, N_ph, PrtlPusher::PHOTON, false, Cooling::NONE, 1);
    
    const real_t gamma { 1e6 };

    auto& ux1 = electron.ux1;
    auto& ux2 = electron.ux2;
    auto& ux3 = electron.ux3;
    auto& weight = electron.weight;
    auto& tag = electron.tag;
    auto& i1 = electron.i1;
    auto& dx1 = electron.dx1;

    electron.set_npart(N_e);

    Kokkos::parallel_for("Init", electron.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      ux1(p) = math::sqrt(SQR(gamma) - ONE);
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
    
    random_number_pool_t random_pool(12345);

    const real_t e_ph { CUBE(gamma / gamma_emit) / rho };
    
    array_t<size_t*> N_phs("N_phs", electron.npart());

    // std::cout << "Begin curvature emission number." << std::endl;
    Curvature_Emission_Number<Dim::_1D, Coord::Cart> curvature_number(electron, 
                                                                      e_min, 
                                                                      coeff * 1e9, 
                                                                      gamma_emit, 
                                                                      rho, 
                                                                      N_phs);
    

    Kokkos::parallel_for("CurvatureEmissionNumber", electron.rangeActiveParticles(), curvature_number);

    // Kokkos::fence();
    // std::cout << "End curvature emission number." << std::endl;

    array_t<int*> offsets("offsets", electron.npart());
    Kokkos::deep_copy(offsets, 0);

    int n_injected { 0 };
    // std::cout << "Begin scan." << std::endl;
    Kokkos::parallel_scan("Scan", N_phs.extent(0), KOKKOS_LAMBDA(index_t p, int& update, const bool final) {
      if (final) {
        offsets(p) = update;
      }
      update += N_phs(p);
    }, n_injected);
    // Kokkos::fence();
    // std::cout << "End scan." << std::endl;

    //check values of offsets
    auto offsets_h = Kokkos::create_mirror_view(offsets);
    auto N_phs_h = Kokkos::create_mirror_view(N_phs);
    Kokkos::deep_copy(offsets_h, offsets);
    Kokkos::deep_copy(N_phs_h, N_phs);
    for (size_t i = 0; i < offsets.extent(0); ++i) {
      if (offsets_h(i) < 0 || offsets_h(i) + N_phs_h(i) - 1 >= n_injected) {
        std::cerr << "Invalid offset: " << offsets_h(i) << std::endl;
        return 1;
      }
    }

    auto total_ph = photon.npart() + n_injected;
    photon.set_npart(total_ph);
    std::cout << "Injecting..." << std::endl;
    CurvatureEmission_kernel<Dim::_1D, Coord::Cart> curvature_emission(electron, 
                                                                       photon, 
                                                                       e_min, 
                                                                       gamma_emit, 
                                                                       rho, 
                                                                       100,
                                                                       N_phs,
                                                                       offsets,
                                                                       random_pool);

    Kokkos::parallel_for("CurvatureEmission", electron.rangeActiveParticles(), curvature_emission);

    Kokkos::fence();
    std::cout << "Number of photons injected: " << n_injected << std::endl;

    size_t num_bins { 100 };
    real_t log_min { math::log10(e_min / e_ph) };
    real_t log_max { 2 };
    auto dx = (log_max - log_min) / num_bins;

    auto e_bins = Kokkos::View<size_t*>("e_bins", num_bins);
    Kokkos::deep_copy(e_bins, 0);
    auto scatter_ebins = Kokkos::Experimental::create_scatter_view(e_bins);

    auto& tag_ph = photon.tag;
    auto& pld_ph = photon.pld;
    Kokkos::parallel_for("Count ebins", photon.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      if (tag_ph(p) != ParticleTag::alive) {
        return;
      }
      auto log_e = math::log10(pld_ph(p, 0) / e_ph);
      auto bin = static_cast<index_t>((log_e - log_min) / dx);
      auto access = scatter_ebins.access();
      if (bin < 0) {
        access(0) += 1;
      }else if (bin >= num_bins) {
        access(num_bins - 1) += 1;
      }else {
        access(bin) += 1;
      }
    }); 
    Kokkos::Experimental::contribute(e_bins, scatter_ebins);
    
    auto ebins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(ebins_h, e_bins);

    std::vector<real_t> bin_centers(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
      bin_centers[i] = math::pow(10.0, log_min + (i + 0.5) * dx);
    }
    
    if (n_injected > 0) {
      std::ofstream file("pdf_ph.dat");
      for (size_t i = 0; i < num_bins; ++i) {
        file << bin_centers[i] << " " << ebins_h(i) << std::endl;
      }
      file.close();
    }


  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  Kokkos::finalize();

  return 0;
}