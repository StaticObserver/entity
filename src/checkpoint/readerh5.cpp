#include "checkpoint/readerh5.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <Kokkos_Core.hpp>
#include <H5Cpp.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

template <Dimension D, int N>
void ReadFields(H5::H5File&                    file,
                const std::string&             field,
                const std::vector<hsize_t>&    start,
                const std::vector<hsize_t>&    count,
                ndfield_t<D, N>&               array) {
  logger::Checkpoint(fmt::format("Reading field: {}", field), HERE);

  H5::DataSet dataset;
  try {
    dataset = file.openDataSet(field);
  } catch (H5::Exception& e) {
    raise::Error(fmt::format("Field variable: {} not found", field.c_str()), HERE);
  }

  H5::DataSpace fspace = dataset.getSpace();
  // 检查维度匹配
  int ndims = fspace.getSimpleExtentNdims();
  if (ndims != D) {
    raise::Error("Dimension mismatch in ReadFields", HERE);
  }

  fspace.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

  H5::DataSpace mspace(D, count.data());
  size_t total_elems = 1;
  for (auto c : count) total_elems *= c;

  std::vector<real_t> buffer(total_elems);

  // 从文件中读取数据到buffer
  dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE, mspace, fspace);

  // 将buffer拷贝到Kokkos数组
  auto array_h = Kokkos::create_mirror_view(array);
  
  // 根据D和N的维度将数据从buffer拷贝到array_h中
  // 假设array为按C-order存储，buffer与array布局一致
  // 若布局不同，需要相应调整下标变换逻辑
  // 以下为简化处理（按LayoutRight存储假设）:
  
  if constexpr (D == Dim::_1D) {
    // array(i, comp) = buffer[i*N + comp] if N>1
    // 若N>1，说明array为ndfield_t<D,N>意味着多分量，这里需根据实际定义来循环
    for (size_t i = 0; i < count[0]; i++) {
      for (int c = 0; c < N; c++) {
        array_h(i, c) = buffer[i * N + c];
      }
    }
  } else if constexpr (D == Dim::_2D) {
    // array(i1, i2, comp)
    // total_elems = count[0]*count[1]*N
    size_t idx = 0;
    for (size_t i1 = 0; i1 < count[0]; i1++) {
      for (size_t i2 = 0; i2 < count[1]; i2++) {
        for (int c = 0; c < N; c++) {
          array_h(i1, i2, c) = buffer[idx++];
        }
      }
    }
  } else if constexpr (D == Dim::_3D) {
    // array(i1, i2, i3, comp)
    size_t idx = 0;
    for (size_t i1 = 0; i1 < count[0]; i1++) {
      for (size_t i2 = 0; i2 < count[1]; i2++) {
        for (size_t i3 = 0; i3 < count[2]; i3++) {
          for (int c = 0; c < N; c++) {
            array_h(i1, i2, i3, c) = buffer[idx++];
          }
        }
      }
    }
  }

  Kokkos::deep_copy(array, array_h);
}

std::pair<std::size_t, std::size_t> ReadParticleCount(H5::H5File&    file,
                                                      const std::string& varname,
                                                      unsigned short  s,
                                                      std::size_t     local_dom,
                                                      std::size_t     ndomains) {
  logger::Checkpoint(fmt::format("Reading particle count for: {}", s + 1), HERE);

  // varname应为 "s{S}_npart"
  auto npart_name = fmt::format("s{}_npart", s + 1);

  H5::DataSet dataset;
  try {
    dataset = file.openDataSet(npart_name);
  } catch (H5::Exception& e) {
    raise::Error("npart_var is not found", HERE);
  }

  H5::DataSpace fspace = dataset.getSpace();
  int ndims = fspace.getSimpleExtentNdims();
  if (ndims != 1) {
    raise::Error("npart_var.Shape().size() != 1", HERE);
  }

  hsize_t dims[1];
  fspace.getSimpleExtentDims(dims, nullptr);
  if ((std::size_t)dims[0] != ndomains) {
    raise::Error("npart_var.Shape()[0] != ndomains", HERE);
  }

  // 读取本地dom对应的粒子数
  hsize_t start[1] = { local_dom };
  hsize_t count[1] = { 1 };
  fspace.selectHyperslab(H5S_SELECT_SET, count, start);
  H5::DataSpace mspace(1, count);

  std::size_t npart;
  dataset.read(&npart, H5::PredType::NATIVE_ULLONG, mspace, fspace);
  const auto loc_npart = npart;

#if !defined(MPI_ENABLED)
  std::size_t offset_npart = 0;
#else
  std::vector<std::size_t> glob_nparts(ndomains);
  MPI_Allgather(&loc_npart,
                1,
                mpi::get_type<std::size_t>(),
                glob_nparts.data(),
                1,
                mpi::get_type<std::size_t>(),
                MPI_COMM_WORLD);
  std::size_t offset_npart = 0;
  for (std::size_t d = 0; d < local_dom; ++d) {
    offset_npart += glob_nparts[d];
  }
#endif

  return { loc_npart, offset_npart };
}

template <typename T>
void ReadParticleData(H5::H5File&    file,
                      const std::string& quantity,
                      unsigned short s,
                      array_t<T*>&  array,
                      std::size_t   count,
                      std::size_t   offset) {
  logger::Checkpoint(
    fmt::format("Reading quantity: s{}_{}", s + 1, quantity),
    HERE);

  auto var_name = fmt::format("s{}_{}", s + 1, quantity);

  H5::DataSet dataset;
  try {
    dataset = file.openDataSet(var_name);
  } catch (H5::Exception& e) {
    raise::Error(fmt::format("Variable: s{}_{} not found", s + 1, quantity), HERE);
  }

  H5::DataSpace fspace = dataset.getSpace();
  hsize_t start[1] = { offset };
  hsize_t c[1] = { count };
  fspace.selectHyperslab(H5S_SELECT_SET, c, start);

  H5::DataSpace mspace(1, c);
  std::vector<T> buffer(count);

  // 假设T为可直接对应的原生类型之一，如double, int等
  // 如果是非原生类型或需要to_string转换，需要额外处理
  // 简化：使用NATIVE_DOUBLE作为范例，对于int/float/short需要对应修改
  H5::PredType dtype;
  if constexpr (std::is_same_v<T, double>) dtype = H5::PredType::NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, float>) dtype = H5::PredType::NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, int>) dtype = H5::PredType::NATIVE_INT;
  else if constexpr (std::is_same_v<T, short>) dtype = H5::PredType::NATIVE_SHORT;
  else {
    // 若还有其他类型需要支持，请继续添加
    static_assert(!sizeof(T*), "Unsupported particle data type");
  }

  dataset.read(buffer.data(), dtype, mspace, fspace);

  auto array_h = Kokkos::create_mirror_view(array);
  for (size_t i = 0; i < count; i++) {
    array_h(i) = buffer[i];
  }
  Kokkos::deep_copy(array, array_h);
}

// 显式实例化
template void ReadFields<Dim::_1D, 3>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_1D, 3>&);
template void ReadFields<Dim::_2D, 3>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_2D, 3>&);
template void ReadFields<Dim::_3D, 3>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_3D, 3>&);
template void ReadFields<Dim::_1D, 6>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_1D, 6>&);
template void ReadFields<Dim::_2D, 6>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_2D, 6>&);
template void ReadFields<Dim::_3D, 6>(H5::H5File&,
                                      const std::string&,
                                      const std::vector<hsize_t>&,
                                      const std::vector<hsize_t>&,
                                      ndfield_t<Dim::_3D, 6>&);

template void ReadParticleData<int>(H5::H5File&,
                                    const std::string&,
                                    unsigned short,
                                    array_t<int*>&,
                                    std::size_t,
                                    std::size_t);
template void ReadParticleData<float>(H5::H5File&,
                                      const std::string&,
                                      unsigned short,
                                      array_t<float*>&,
                                      std::size_t,
                                      std::size_t);
template void ReadParticleData<double>(H5::H5File&,
                                       const std::string&,
                                       unsigned short,
                                       array_t<double*>&,
                                       std::size_t,
                                       std::size_t);
template void ReadParticleData<short>(H5::H5File&,
                                      const std::string&,
                                      unsigned short,
                                      array_t<short*>&,
                                      std::size_t,
                                      std::size_t);

} // namespace checkpoint
