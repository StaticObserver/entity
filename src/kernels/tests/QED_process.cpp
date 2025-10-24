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
#include <chrono>

using namespace ntt;
using namespace kernel::QED;

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    size_t N_e = 1e6;
    size_t N_ph = 1e6;
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, N_e, PrtlPusher::BORIS, false, Cooling::NONE, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, N_ph, PrtlPusher::PHOTON, false, Cooling::NONE, 3);
    
    const real_t gamma { 1e6 };

    auto& ux1 = electron.ux1;
    auto& ux2 = electron.ux2;
    auto& ux3 = electron.ux3;
    auto& weight = electron.weight;
    auto& tag = electron.tag;
    auto& i1 = electron.i1;
    auto& dx1 = electron.dx1;

    electron.set_npart(1e3);

    Kokkos::parallel_for("Init", electron.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      ux1(p) = math::sqrt(SQR(gamma) - ONE);
      ux2(p) = ZERO;
      ux3(p) = ZERO;
      weight(p) = ONE;
      tag(p) = ParticleTag::alive;
      i1(p) = 0;
      dx1(p) = ZERO;
    });
    
    const real_t e_min { 2.0 };
    const real_t gamma_emit { 3e4 };
    const real_t gamma_rad  { 6.7e5 };
    const real_t gamma_pc { 7.2e7 };
    const real_t coeff { SQR(THREE / TWO) * constant::SQRT3 / constant::SQRT2 / constant::PI
                         * math::sqrt(gamma_pc) * SQR(SQR(gamma_emit / gamma_rad) * gamma_emit) };
    const real_t rho { 1.0 };
    
    random_number_pool_t random_pool(12345);

    const real_t e_ph { CUBE(gamma / gamma_emit) / rho };
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cdfTable cdf("cdf_table.txt", "inverse_cdf_table.txt" );
    // std::cout << "Begin curvature emission number." << std::endl;
    CurvatureEmission_kernel<Dim::_1D, Coord::Cart> curvature_emission(electron, 
                                                                      photon,
                                                                      e_min, 
                                                                      gamma_emit, 
                                                                      coeff, 
                                                                      rho, 
                                                                      100,
                                                                      random_pool,
                                                                      cdf);
    

    Kokkos::parallel_for("CurvatureEmission", electron.rangeActiveParticles(), curvature_emission);

    Kokkos::fence();

    auto n_injected = curvature_emission.num_injected();
    std::cout << "Number of photons injected: " << n_injected << std::endl;

    auto total_ph = photon.npart() + n_injected;
    photon.set_npart(total_ph);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " s" << std::endl;

    size_t num_bins { 300 };
    real_t min { e_min / e_ph };
    real_t max { 3.0 };
    auto dx = (max - min) / num_bins;

    auto e_bins = Kokkos::View<real_t*>("e_bins", num_bins);
    Kokkos::deep_copy(e_bins, 0);
    auto scatter_ebins = Kokkos::Experimental::create_scatter_view(e_bins);

    auto& tag_ph = photon.tag;
    auto& pld_ph = photon.pld;
    auto& weight_ph = photon.weight;
    Kokkos::parallel_for("Count ebins", photon.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      if (tag_ph(p) != ParticleTag::alive) {
        return;
      }
      auto e = pld_ph(p, 0) / e_ph;
      auto bin = static_cast<index_t>((e - min) / dx);
      auto access = scatter_ebins.access();
      if (bin < 0) {
        access(0) += weight_ph(p);
      }else if (bin >= num_bins) {
        access(num_bins - 1) += weight_ph(p);
      }else {
        access(bin) += weight_ph(p);
      }
    }); 
    Kokkos::Experimental::contribute(e_bins, scatter_ebins);
    
    auto ebins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(ebins_h, e_bins);

    std::vector<real_t> bin_centers(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
      bin_centers[i] = min + (i + 0.5) * dx;
    }
    
    if (n_injected > 0) {
      std::ofstream file("pdf_ph.dat");
      for (size_t i = 0; i < num_bins; ++i) {
        file << bin_centers[i] << " " << ebins_h(i) << std::endl;
      }
      file.close();
    }

    std::cout << "PairCreation" << std::endl;
    auto start_pair = std::chrono::high_resolution_clock::now();

    PayloadUpdate<Dim::_1D, Coord::Cart> payload_update(photon, 
                                                        1.0,
                                                        1.0,
                                                        1.0,
                                                        1.0,
                                                        10);
    Kokkos::parallel_for("PayloadUpdate", photon.rangeActiveParticles(), payload_update);

    Kokkos::fence();

    Particles<Dim::_1D, Coord::Cart> positron(1, "e+", 1.0, 1.0, N_e, PrtlPusher::BORIS, false, Cooling::NONE, 0);

    PairCreation_kernel<Dim::_1D, Coord::Cart> pair_creation(photon, 
                                                              electron, 
                                                              positron);
    Kokkos::parallel_for("PairCreation", photon.rangeActiveParticles(), pair_creation);

    Kokkos::fence();

    auto n_pairs = pair_creation.num_injected();
    std::cout << "Number of pairs injected: " << n_pairs << std::endl;

    auto end_pair = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_pair = end_pair - start_pair;
    std::cout << "Time: " << elapsed_pair.count() << " s" << std::endl;


    const real_t min_e { ZERO };
    const real_t max_e { 1e6 };
    const real_t dx_e = (max_e - min_e) / num_bins;
    std::vector<real_t> bin_centers_e(num_bins);
    for (size_t i = 0; i < num_bins; ++i) {
      bin_centers_e[i] = min_e + (i + 0.5) * dx_e;
    }
    

    array_t<real_t*> e_bins_e { "e_bins_e", num_bins };
    Kokkos::deep_copy(e_bins_e, 0);
    auto scatter_ebins_e = Kokkos::Experimental::create_scatter_view(e_bins_e);

    auto& tag_e = electron.tag;
    auto& ux1_e = electron.ux1;
    auto& ux2_e = electron.ux2;
    auto& ux3_e = electron.ux3;
    auto& weight_e = electron.weight;
    Kokkos::parallel_for("Count ebins_e", electron.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      if (tag_e(p) != ParticleTag::alive) {
        return;
      }
      auto e = math::sqrt(ONE + NORM_SQR(ux1_e(p), ux2_e(p), ux3_e(p))) - ONE;
      auto bin = static_cast<index_t>((e - min_e) / dx_e);
      // printf("bin: %lu\n", bin);
      auto access = scatter_ebins_e.access();
      if (bin < 0) {
        access(0) += weight_e(p);
      }else if (bin >= num_bins) {
        access(num_bins - 1) += weight_e(p);
      }else {
        access(bin) += weight_e(p);
      }
    });
    Kokkos::Experimental::contribute(e_bins_e, scatter_ebins_e);

    auto ebins_e_h = Kokkos::create_mirror_view(e_bins_e);
    Kokkos::deep_copy(ebins_e_h, e_bins_e);

    if (n_pairs > 0) {
      std::ofstream file("pdf_e.dat");
      for (size_t i = 0; i < num_bins; ++i) {
        file << bin_centers_e[i] << " " << ebins_e_h(i) << std::endl;
      }
      file.close();
    }

    array_t<real_t*> e_bins_p { "e_bins_p", num_bins };
    Kokkos::deep_copy(e_bins_p, 0);
    auto scatter_ebins_p = Kokkos::Experimental::create_scatter_view(e_bins_p);

    auto& tag_p = positron.tag;
    auto& ux1_p = positron.ux1;
    auto& ux2_p = positron.ux2;
    auto& ux3_p = positron.ux3;
    auto& weight_p = positron.weight;
    Kokkos::parallel_for("Count ebins_p", positron.rangeActiveParticles(), KOKKOS_LAMBDA(index_t p) {
      if (tag_p(p) != ParticleTag::alive) {
        return;
      }
      auto e = math::sqrt(ONE + NORM_SQR(ux1_p(p), ux2_p(p), ux3_p(p))) - ONE;
      auto bin = static_cast<index_t>((e - min_e) / dx_e);
      // printf("bin: %lu\n", bin);
      auto access = scatter_ebins_p.access();
      if (bin < 0) {
        access(0) += weight_p(p);
      }else if (bin >= num_bins) {
        access(num_bins - 1) += weight_p(p);
      }else {
        access(bin) += weight_p(p);
      }
    });
    Kokkos::Experimental::contribute(e_bins_p, scatter_ebins_p);

    auto ebins_p_h = Kokkos::create_mirror_view(e_bins_p);
    Kokkos::deep_copy(ebins_p_h, e_bins_p);

    if (n_pairs > 0) {
      std::ofstream file("pdf_p.dat");
      for (size_t i = 0; i < num_bins; ++i) {
        file << bin_centers_e[i] << " " << ebins_p_h(i) << std::endl;
      }
      file.close();
    }



  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  Kokkos::finalize();

  return 0;
}