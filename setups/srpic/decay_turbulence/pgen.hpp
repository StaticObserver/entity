#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

namespace user {
  using namespace ntt;

  template <Dimension D>
  class InitFields {
    public:
    InitFields(const std::string& file_bx1, 
               const std::string& file_bx2,
               const std::pair<real_t, real_t>& extent_x1,
               const std::pair<real_t, real_t>& extent_x2);

    // 插值函数
    Inline real_t interpolate(const ndfield_t<D, 1>& data, const coord_t<D>& x_Ph) const {
      // 获取插值点的坐标
      real_t x = x_Ph[0];
      real_t y = x_Ph[1];

      // 计算网格间距
      real_t dx1 = (extent_x1.second - extent_x1.first) / (nx1 - 1);
      real_t dx2 = (extent_x2.second - extent_x2.first) / (nx2 - 1);

      // 计算网格索引
      real_t xi = (x - extent_x1.first) / dx1;
      real_t yi = (y - extent_x2.first) / dx2;
      size_t i = static_cast<size_t>(xi);
      size_t j = static_cast<size_t>(yi);

      // 确保索引在有效范围内
      if (i >= nx1 - 1) i = nx1 - 2;
      if (j >= nx2 - 1) j = nx2 - 2;

      // 计算权重
      real_t tx = xi - i;
      real_t ty = yi - j;

      // 获取周围节点的值
      real_t f00 = data(i, j);
      real_t f10 = data(i + 1, j);
      real_t f01 = data(i, j + 1);
      real_t f11 = data(i + 1, j + 1);

      // 双线性插值
      return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11;
    }

    // 磁场分量函数
    Inline auto bx1(const coord_t<D>& x_Ph) const {
      return interpolate(bx1_data, x_Ph);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const {
      return interpolate(bx2_data, x_Ph);
    }

  private:
    ndfield_t<D, 1> bx1_data; // 替代 std::vector<std::vector<real_t>>
    ndfield_t<D, 1> bx2_data; // 替代 std::vector<std::vector<real_t>>
    size_t nx1;
    size_t nx2;
    std::pair<real_t, real_t> extent_x1;
    std::pair<real_t, real_t> extent_x2;
  };

  template <Dimension D>
  InitFields<D>::InitFields(const std::string& file_bx1, 
                            const std::string& file_bx2,
                            const std::pair<real_t, real_t>& extent_x1,
                            const std::pair<real_t, real_t>& extent_x2)
    : extent_x1 { extent_x1 },
      extent_x2 { extent_x2 } {
    // 读取 bx1 数据并推断 nx1 和 nx2
    std::ifstream input_bx1(file_bx1);
    if (!input_bx1.is_open()) {
      throw std::runtime_error("无法打开文件：" + file_bx1);
    }

    std::vector<real_t> temp_bx1;
    std::string line;

    while (std::getline(input_bx1, line)) {
      std::istringstream iss(line);
      real_t value;
      while (iss >> value) {
        temp_bx1.push_back(value);
      }
    }
    input_bx1.close();

    nx1 = static_cast<size_t>(std::sqrt(temp_bx1.size()));
    nx2 = nx1;

    static_assert(D == Dim::_2D, "仅支持二维情况");

    bx1_data = ndfield_t<D, 1>("bx1_data", nx1, nx2);
    auto bx1_data_mirror = Kokkos::create_mirror_view(bx1_data);
    for (size_t i = 0; i < nx1; ++i) {
      for (size_t j = 0; j < nx2; ++j) {
        bx1_data_mirror(i, j) = temp_bx1[i * nx2 + j];
      }
    }
    Kokkos::deep_copy(bx1_data, bx1_data_mirror);

    // 对 bx2 进行类似的操作
    std::ifstream input_bx2(file_bx2);
    if (!input_bx2.is_open()) {
      throw std::runtime_error("无法打开文件：" + file_bx2);
    }

    std::vector<real_t> temp_bx2;
    while (std::getline(input_bx2, line)) {
      std::istringstream iss(line);
      real_t value;
      while (iss >> value) {
        temp_bx2.push_back(value);
      }
    }
    input_bx2.close();

    if (temp_bx2.size() != temp_bx1.size()) {
      throw std::runtime_error("文件 " + file_bx2 + " 的尺寸与 " + file_bx1 + " 不一致。");
    }

    bx2_data = ndfield_t<D, 1>("bx2_data", nx1, nx2);
    auto bx2_data_mirror = Kokkos::create_mirror_view(bx2_data);
    for (size_t i = 0; i < nx1; ++i) {
      for (size_t j = 0; j < nx2; ++j) {
        bx2_data_mirror(i, j) = temp_bx2[i * nx2 + j];
      }
    }
    Kokkos::deep_copy(bx2_data, bx2_data_mirror);
  }

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    // 定义 InitFields 对象
    InitFields<D> init_flds;
    const real_t  temperature;

    inline PGen(const SimulationParams& params, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { params },
        init_flds { params.template get<std::string>("setup.file_bx1"),
                    params.template get<std::string>("setup.file_bx2"),
                    global_domain.mesh().extent(in::x1),
                    global_domain.mesh().extent(in::x2) },
        temperature { params.template get<real_t>("setup.temperature", 1.0) } {}


    inline void InitPrtls(Domain<S, M>& local_domain) {
        const auto energy_dist = arch::Maxwellian<S, M>(local_domain.mesh.metric,
                                                        local_domain.random_pool,
                                                        temperature);
        const auto injector = arch::UniformInjector<S, M, arch::Maxwellian>(
          energy_dist,
          { 1, 2 });
        const real_t ndens = 1.0;
        arch::InjectUniform<S, M, decltype(injector)>(params,
                                                      local_domain,
                                                      injector,
                                                      ndens);
    }

  };

} // namespace user

#endif
