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
      real_t dx = (extent_x_view(d, 1) - extent_x_view(d, 0)) / (nx[d] - 1);
      xi[d] = (x_Ph[d] - extent_x_view(d, 0)) / dx;
      idx[d] = static_cast<size_t>(xi[d]);
      if (idx[d] >= nx[d] - 1) idx[d] = nx[d] - 2;
      t[d] = xi[d] - idx[d];
    }

    // 进行多维线性插值
    return multidimensionalInterpolation(data, idx, t);
  }

  // 通用的多维线性插值函数
  Inline real_t multidimensionalInterpolation(const ndfield_t<D, 1>& data, const size_t idx[], const real_t t[]) const {
    if constexpr (D == Dim::_1D) {
      // 一维插值
      real_t f0 = data(idx[0], 0);
      real_t f1 = data(idx[0] + 1, 0);
      return (1 - t[0]) * f0 + t[0] * f1;
    } else if constexpr (D == Dim::_2D) {
      // 二维插值
      return bilinearInterpolation(data, idx, t);
    } else if constexpr (D == Dim::_3D) {
      // 三维插值
      return trilinearInterpolation(data, idx, t);
    } else {
      throw std::runtime_error("不支持的维度");
    }
  }

  // 二维线性插值
  Inline real_t bilinearInterpolation(const ndfield_t<D, 1>& data, const size_t idx[], const real_t t[]) const {
    real_t f00 = data(idx[0], idx[1], 0);
    real_t f10 = data(idx[0] + 1, idx[1], 0);
    real_t f01 = data(idx[0], idx[1] + 1, 0);
    real_t f11 = data(idx[0] + 1, idx[1] + 1, 0);
    return (1 - t[0]) * (1 - t[1]) * f00
        + t[0] * (1 - t[1]) * f10
        + (1 - t[0]) * t[1] * f01
        + t[0] * t[1] * f11;
  }

  // 三维线性插值
  Inline real_t trilinearInterpolation(const ndfield_t<D, 1>& data, const size_t idx[], const real_t t[]) const {
    real_t f000 = data(idx[0], idx[1], idx[2], 0);
    real_t f100 = data(idx[0] + 1, idx[1], idx[2], 0);
    real_t f010 = data(idx[0], idx[1] + 1, idx[2], 0);
    real_t f110 = data(idx[0] + 1, idx[1] + 1, idx[2], 0);
    real_t f001 = data(idx[0], idx[1], idx[2] + 1, 0);
    real_t f101 = data(idx[0] + 1, idx[1], idx[2] + 1, 0);
    real_t f011 = data(idx[0], idx[1] + 1, idx[2] + 1, 0);
    real_t f111 = data(idx[0] + 1, idx[1] + 1, idx[2] + 1, 0);
    return ((1 - t[0]) * (1 - t[1]) * (1 - t[2]) * f000
          + t[0] * (1 - t[1]) * (1 - t[2]) * f100
          + (1 - t[0]) * t[1] * (1 - t[2]) * f010
          + t[0] * t[1] * (1 - t[2]) * f110
          + (1 - t[0]) * (1 - t[1]) * t[2] * f001
          + t[0] * (1 - t[1]) * t[2] * f101
          + (1 - t[0]) * t[1] * t[2] * f011
          + t[0] * t[1] * t[2] * f111);
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

  // 构造函数声明
  InitFields(const std::vector<std::string>& files_bx,
            const boundaries_t<real_t>& extent_x);
  
    // 通用的初始化磁场数据函数
  void initializeFieldData(const std::string& file_name, ndfield_t<D, 1>& field_data, size_t nx[]);

  private:
    ndfield_t<D, 1> bx1_data;
    ndfield_t<D, 1> bx2_data;
    ndfield_t<D, 1> bx3_data;
    size_t nx[D] = {0};
    array_t<real_t* [2]> extent_x_view;
  };

  // 类外定义构造函数
  template <Dimension D>
  InitFields<D>::InitFields(const std::vector<std::string>& files_bx,
                            const boundaries_t<real_t>& extent_x)
    : extent_x_view("extent_x_view", D) { // 初始化 Kokkos View
    // 将 extent_x 的数据复制到 Kokkos View
    auto temp_extent_x = Kokkos::create_mirror_view(extent_x_view);
    for (int d = 0; d < D; ++d) {
      temp_extent_x(d, 0) = extent_x[d].first;
      temp_extent_x(d, 1) = extent_x[d].second;
    }
    Kokkos::deep_copy(extent_x_view, temp_extent_x);

    // 读取和初始化磁场分量数据
    initializeFieldData(files_bx[0], bx1_data, nx);
    if constexpr (D >= Dim::_2D) {
      initializeFieldData(files_bx[1], bx2_data, nx);
    }
    if constexpr (D == Dim::_3D) {
      initializeFieldData(files_bx[2], bx3_data, nx);
    }
  }

  // 通用的初始化磁场数据函数
  template <Dimension D>
  void InitFields<D>::initializeFieldData(const std::string& file_name, ndfield_t<D, 1>& field_data, size_t nx[]){
    std::ifstream input_file(file_name);
    if (!input_file.is_open()) {
      throw std::runtime_error("无法打开文件：" + file_name);
    }

    std::vector<real_t> temp_data;
    std::string line;

    while (std::getline(input_file, line)) {
      std::istringstream iss(line);
      real_t value;
      while (iss >> value) {
        temp_data.push_back(value);
      }
    }
    input_file.close();

    // 设置 nx 的值
    if constexpr (D == Dim::_1D) {
      nx[0] = temp_data.size();
      field_data = ndfield_t<D, 1>("field_data", nx[0]);
    } else if constexpr (D == Dim::_2D) {
      nx[0] = static_cast<size_t>(std::sqrt(temp_data.size()));
      nx[1] = nx[0];
      field_data = ndfield_t<D, 1>("field_data", nx[0], nx[1]);
    } else if constexpr (D == Dim::_3D) {
      nx[0] = static_cast<size_t>(std::cbrt(temp_data.size()));
      nx[1] = nx[0];
      nx[2] = nx[0];
      field_data = ndfield_t<D, 1>("field_data", nx[0], nx[1], nx[2]);
    }

    // 创建镜像视图并填充数据
    auto field_data_mirror = Kokkos::create_mirror_view(field_data);
    if constexpr (D == Dim::_1D) {
      for (size_t i = 0; i < nx[0]; ++i) {
        field_data_mirror(i, 0) = temp_data[i];
      }
    } else if constexpr (D == Dim::_2D) {
      for (size_t i = 0; i < nx[0]; ++i) {
        for (size_t j = 0; j < nx[1]; ++j) {
          field_data_mirror(i, j, 0) = temp_data[i * nx[1] + j];
        }
      }
    } else if constexpr (D == Dim::_3D) {
      for (size_t i = 0; i < nx[0]; ++i) {
        for (size_t j = 0; j < nx[1]; ++j) {
          for (size_t k = 0; k < nx[2]; ++k) {
            field_data_mirror(i, j, k, 0) = temp_data[i * nx[1] * nx[2] + j * nx[2] + k];
          }
        }
      }
    }
    Kokkos::deep_copy(field_data, field_data_mirror);
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
