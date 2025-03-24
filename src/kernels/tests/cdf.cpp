#include "kernels/QED_process.hpp"

#include "global.h"
#include "enums.h"
#include "utils/numeric.h"

#include "arch/kokkos_aliases.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>
#include <vector>


using namespace ntt;
using namespace kernel::QED;


auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  
  try{
    cdfTable cdf("cdf_table.txt", "inverse_cdf_table.txt" );

    size_t N { 201 };
    array_t<real_t*> x_cdf("x", N);
    array_t<real_t*> y_cdf("y", N);
    array_t<real_t*> x_inv_cdf("x", N);
    array_t<real_t*> y_inv_cdf("y", N);

    const real_t x_min { 1e-8 };
    const real_t x_max { 11.13 };
    const real_t y_min { 1e-6 };
    const real_t y_max { 0.99 };
    
    const real_t dx = (math::log10(x_max) - math::log10(x_min)) / (N - 1);
    const real_t dy = (math::log10(y_max) - math::log10(y_min)) / (N - 1);

    Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(index_t i) {
      x_cdf(i) = x_min * math::pow(10, i * dx);
      y_cdf(i) = cdf.CDF(x_cdf(i));
      x_inv_cdf(i) = y_min * math::pow(10, i * dy);
      y_inv_cdf(i) = cdf.Inverse_CDF(x_inv_cdf(i));
    });

    auto x_cdf_h = Kokkos::create_mirror_view(x_cdf);
    auto y_cdf_h = Kokkos::create_mirror_view(y_cdf);
    auto x_inv_cdf_h = Kokkos::create_mirror_view(x_inv_cdf);
    auto y_inv_cdf_h = Kokkos::create_mirror_view(y_inv_cdf);

    Kokkos::deep_copy(x_cdf_h, x_cdf);
    Kokkos::deep_copy(y_cdf_h, y_cdf);
    Kokkos::deep_copy(x_inv_cdf_h, x_inv_cdf);
    Kokkos::deep_copy(y_inv_cdf_h, y_inv_cdf);

    std::ofstream file("cdf.dat");
    for (size_t i = 0; i < N; ++i){
      file << x_cdf_h(i) << " " << y_cdf_h(i) << std::endl;
    }
    file.close();

    std::ofstream file2("inverse_cdf.dat");
    for (size_t i = 0; i < N; ++i){
      file2 << x_inv_cdf_h(i) << " " << y_inv_cdf_h(i) << std::endl;
    }
    file2.close();

  }catch (std::exception& e){
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }

  Kokkos::finalize();
  return 0;

}
