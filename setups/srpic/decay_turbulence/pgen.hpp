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

    // 修改 interpolate 函数，支持任意维度
    Inline real_t interpolate(const ndfield_t<D, 1>& data, const coord_t<D>& x_Ph) const {
      // 获取插值点的坐标
      real_t xi[D];
      real_t t[D];
      size_t idx[D];

      // 计算每个维度的网格间距和索引
      for (int d = 0; d < D; ++d) {
        real_t dx = (extent_x[d].second - extent_x[d].first) / (nx[d] - 1);
        xi[d] = (x_Ph[d] - extent_x[d].first) / dx;
        idx[d] = static_cast<size_t>(xi[d]);
        if (idx[d] >= nx[d] - 1) idx[d] = nx[d] - 2;
        t[d] = xi[d] - idx[d];
      }

      // 进行多维线性插值
      if constexpr (D == Dim::_1D) {
        // 一维插值
        real_t f0 = data(idx[0]);
        real_t f1 = data(idx[0] + 1);
        return (1 - t[0]) * f0 + t[0] * f1;
      } else if constexpr (D == Dim::_2D) {
        // 二维插值
        real_t f00 = data(idx[0], idx[1]);
        real_t f10 = data(idx[0] + 1, idx[1]);
        real_t f01 = data(idx[0], idx[1] + 1);
        real_t f11 = data(idx[0] + 1, idx[1] + 1);
        return (1 - t[0]) * (1 - t[1]) * f00
             + t[0] * (1 - t[1]) * f10
             + (1 - t[0]) * t[1] * f01
             + t[0] * t[1] * f11;
      } else if constexpr (D == Dim::_3D) {
        // 三维插值
        real_t f000 = data(idx[0], idx[1], idx[2]);
        real_t f100 = data(idx[0] + 1, idx[1], idx[2]);
        real_t f010 = data(idx[0], idx[1] + 1, idx[2]);
        real_t f110 = data(idx[0] + 1, idx[1] + 1, idx[2]);
        real_t f001 = data(idx[0], idx[1], idx[2] + 1);
        real_t f101 = data(idx[0] + 1, idx[1], idx[2] + 1);
        real_t f011 = data(idx[0], idx[1] + 1, idx[2] + 1);
        real_t f111 = data(idx[0] + 1, idx[1] + 1, idx[2] + 1);
        return ((1 - t[0]) * (1 - t[1]) * (1 - t[2]) * f000
              + t[0] * (1 - t[1]) * (1 - t[2]) * f100
              + (1 - t[0]) * t[1] * (1 - t[2]) * f010
              + t[0] * t[1] * (1 - t[2]) * f110
              + (1 - t[0]) * (1 - t[1]) * t[2] * f001
              + t[0] * (1 - t[1]) * t[2] * f101
              + (1 - t[0]) * t[1] * t[2] * f011
              + t[0] * t[1] * t[2] * f111);
      }
    }

    // 根据维度定义磁场分量函数
    Inline auto bx1(const coord_t<D>& x_Ph) const {
      return interpolate(bx1_data, x_Ph);
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const {
      if constexpr (D >= Dim::_2D) {
        return interpolate(bx2_data, x_Ph);
      } else {
        return 0.0;
      }
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const {
      if constexpr (D == Dim::_3D) {
        return interpolate(bx3_data, x_Ph);
      } else {
        return 0.0;
      }
    }

    // 修改构造函数，适应不同维度的数据读取
    InitFields(const std::vector<std::string>& files_bx,
               const boundaries_t<real_t>* extent_x)
      : extent_x { extent_x } {
      // 初始化 nx 和数据容器
      for (int d = 0; d < D; ++d) {
        nx[d] = 0;
      }

      // 读取 bx1 数据
      std::ifstream input_bx1(files_bx[0]);
      if (!input_bx1.is_open()) {
        throw std::runtime_error("无法打开文件：" + files_bx[0]);
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

      nx[0] = static_cast<size_t>(std::sqrt(temp_bx1.size()));
      if constexpr (D >= Dim::_2D) {
        nx[1] = nx[0];
      }
      if constexpr (D == Dim::_3D) {
        nx[2] = nx[0];
      }

      bx1_data = ndfield_t<D, 1>("bx1_data", nx[0], nx[1], nx[2]);
      auto bx1_data_mirror = Kokkos::create_mirror_view(bx1_data);
      for (size_t i = 0; i < nx[0]; ++i) {
        for (size_t j = 0; j < nx[1]; ++j) {
          for (size_t k = 0; k < nx[2]; ++k) {
            bx1_data_mirror(i, j, k) = temp_bx1[i * nx[1] * nx[2] + j * nx[2] + k];
          }
        }
      }
      Kokkos::deep_copy(bx1_data, bx1_data_mirror);

      // 根据维度 D，读取相应的 bx 数据
      if constexpr (D >= Dim::_2D) {
        // 读取 bx2 数据
        std::ifstream input_bx2(files_bx[1]);
        if (!input_bx2.is_open()) {
          throw std::runtime_error("无法打开文件：" + files_bx[1]);
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
          throw std::runtime_error("文件 " + files_bx[1] + " 的尺寸与 " + files_bx[0] + " 不一致。");
        }

        bx2_data = ndfield_t<D, 1>("bx2_data", nx[0], nx[1], nx[2]);
        auto bx2_data_mirror = Kokkos::create_mirror_view(bx2_data);
        for (size_t i = 0; i < nx[0]; ++i) {
          for (size_t j = 0; j < nx[1]; ++j) {
            for (size_t k = 0; k < nx[2]; ++k) {
              bx2_data_mirror(i, j, k) = temp_bx2[i * nx[1] * nx[2] + j * nx[2] + k];
            }
          }
        }
        Kokkos::deep_copy(bx2_data, bx2_data_mirror);
      }
      if constexpr (D == Dim::_3D) {
        // 读取 bx3 数据
        std::ifstream input_bx3(files_bx[2]);
        if (!input_bx3.is_open()) {
          throw std::runtime_error("无法打开文件：" + files_bx[2]);
        }

        std::vector<real_t> temp_bx3;
        while (std::getline(input_bx3, line)) {
          std::istringstream iss(line);
          real_t value;
          while (iss >> value) {
            temp_bx3.push_back(value);
          }
        }
        input_bx3.close();

        if (temp_bx3.size() != temp_bx1.size()) {
          throw std::runtime_error("文件 " + files_bx[2] + " 的尺寸与 " + files_bx[0] + " 不一致。");
        }

        bx3_data = ndfield_t<D, 1>("bx3_data", nx[0], nx[1], nx[2]);
        auto bx3_data_mirror = Kokkos::create_mirror_view(bx3_data);
        for (size_t i = 0; i < nx[0]; ++i) {
          for (size_t j = 0; j < nx[1]; ++j) {
            for (size_t k = 0; k < nx[2]; ++k) {
              bx3_data_mirror(i, j, k) = temp_bx3[i * nx[1] * nx[2] + j * nx[2] + k];
            }
          }
        }
        Kokkos::deep_copy(bx3_data, bx3_data_mirror);
      }
    }

  private:
    ndfield_t<D, 1> bx1_data;
    ndfield_t<D, 1> bx2_data;
    ndfield_t<D, 1> bx3_data;
    size_t nx[D];
    boundaries_t<real_t> extent_x[D];
  };

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
        init_flds { { params.template get<std::string>("setup.file_bx1"),
                      params.template get<std::string>("setup.file_bx2") },
                    { global_domain.mesh().extent(in::x1),
                      global_domain.mesh().extent(in::x2) } },
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
