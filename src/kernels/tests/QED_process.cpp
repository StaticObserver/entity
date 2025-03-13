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
    Particles<Dim::_1D, Coord::Cart> electron(1, "e-", 1.0, -1.0, 1e4, PrtlPusher::BORIS, false, Cooling::NONE, 0);
    Particles<Dim::_1D, Coord::Cart> photon(2, "photon", 0.0, 0.0, 1e8, PrtlPusher::PHOTON, false, Cooling::NONE, 1);
    
    array_t<real_t*>        ux1 { "ux1", 1e4 };
    array_t<real_t*>        ux2 { "ux2", 1e4 };
    array_t<real_t*>        ux3 { "ux3", 1e4 };
    array_t<real_t*>        weight { "weight", 1e4 };
    array_t<short*>         tag { "tag", 1e4 };
    array_t<int*>           i1 { "i1", 1e4 };
    array_t<prtldx_t*>      dx1 { "dx1", 1e4 };

    Kokkos::parallel_for("Init", 1e4, KOKKOS_LAMBDA(index_t p) {
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


  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  Kokkos::finalize();

  return 0;
}