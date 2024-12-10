#include "checkpoint/writer.h"

#include "global.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "framework/parameters.h"

#include <Kokkos_Core.hpp>
#include <H5Cpp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

void Writer::init(const std::string& filename,
                  std::size_t        interval,
                  long double        interval_time,
                  int                keep) {
  m_keep    = keep;
  m_enabled = keep != 0;
  if (!m_enabled) {
    return;
  }
  m_tracker.init("checkpoint", interval, interval_time);

  // 创建 checkponits 目录（若不存在）
  CallOnce([]() {
    const std::filesystem::path save_path { "checkpoints" };
    if (!std::filesystem::exists(save_path)) {
      std::filesystem::create_directory(save_path);
    }
  });

  // 不在此打开文件，而是在 beginSaving 时打开对应的文件
  // 这里仅完成初始化工作
}

void Writer::defineFieldVariables(const ntt::SimEngine&,
                                  const std::vector<std::size_t>&,
                                  const std::vector<std::size_t>&,
                                  const std::vector<std::size_t>&) {
  // 原代码中使用ADIOS2在此定义变量，对于HDF5可以延后到写入时创建数据集。
  // 若需要预先创建数据集，可在此记录形状信息并在beginSaving时创建数据集。
  // 在此示例中，不做任何操作，等写入时根据实际数据维度创建数据集。
}

void Writer::defineParticleVariables(const ntt::Coord&,
                                     Dimension,
                                     std::size_t,
                                     const std::vector<unsigned short>&) {
  // 同上，对于粒子数据集的定义，也可在写入时动态创建
  // 在此示例中不提前定义，写入时创建
}

auto Writer::shouldSave(std::size_t step, long double time) -> bool {
  return m_enabled && m_tracker.shouldWrite(step, time);
}

void Writer::beginSaving(std::size_t step, long double time) {
  raise::ErrorIf(!m_enabled, "Checkpoint is not enabled", HERE);
  if (m_writing_mode) {
    raise::Fatal("Already writing", HERE);
  }
  m_writing_mode = true;

  try {
    auto fname      = fmt::format("checkpoints/step-%08lu.h5", step);
    auto meta_fname = fmt::format("checkpoints/meta-%08lu.toml", step);
    m_written.push_back({ fname, meta_fname });

    logger::Checkpoint(fmt::format("Writing checkpoint to {} and {}",
                                   fname,
                                   meta_fname),
                       HERE);

    // 打开/创建HDF5文件
    // 若存在则覆盖
    m_file = new H5::H5File(fname, H5F_ACC_TRUNC);

    // 写入Step和Time为文件属性或单独创建Group "/"
    {
      H5::DataSpace scalar_space(H5S_SCALAR);

      // Step属性
      {
        auto attr = m_file->createAttribute("Step", H5::PredType::NATIVE_ULLONG, scalar_space);
        unsigned long long step_ull = static_cast<unsigned long long>(step);
        attr.write(H5::PredType::NATIVE_ULLONG, &step_ull);
      }

      // Time属性
      {
        auto attr = m_file->createAttribute("Time", H5::PredType::NATIVE_LDOUBLE, scalar_space);
        attr.write(H5::PredType::NATIVE_LDOUBLE, &time);
      }

      // NGhosts属性
      {
        auto attr = m_file->createAttribute("NGhosts", H5::PredType::NATIVE_INT, scalar_space);
        int nghosts = ntt::N_GHOSTS;
        attr.write(H5::PredType::NATIVE_INT, &nghosts);
      }
    }

  } catch (std::exception& e) {
    raise::Fatal(e.what(), HERE);
  }
}

void Writer::endSaving() {
  if (!m_writing_mode) {
    raise::Fatal("Not writing", HERE);
  }
  m_writing_mode = false;

  // flush并关闭文件
  if (m_file) {
    m_file->flush(H5F_SCOPE_GLOBAL);
    delete m_file;
    m_file = nullptr;
  }

  // 移除最老的checkpoint（若超出m_keep）
  CallOnce([&]() {
    if (m_keep > 0 && m_written.size() > (std::size_t)m_keep) {
      const auto oldest = m_written.front();
      if (std::filesystem::exists(oldest.first) &&
          std::filesystem::exists(oldest.second)) {
        std::filesystem::remove_all(oldest.first);
        std::filesystem::remove(oldest.second);
        m_written.erase(m_written.begin());
      } else {
        raise::Warning("Checkpoint file does not exist for some reason", HERE);
      }
    }
  });
}

void Writer::saveAttrs(const ntt::SimulationParams& params, long double time) {
  CallOnce([&]() {
    if (m_written.empty()) {
      raise::Fatal("No checkpoint file to save metadata", HERE);
    }
    std::ofstream metadata(m_written.back().second.c_str());
    metadata << "[metadata]\n"
             << "  time = " << time << "\n\n"
             << params.data() << std::endl;
    metadata.close();
  });
}

template <typename T>
void Writer::savePerDomainVariable(const std::string& varname,
                                   std::size_t        total,
                                   std::size_t        offset,
                                   T                  data) {
  raise::ErrorIf(m_file == nullptr, "File not open", HERE);

  // 创建或打开数据集 varname，一维大小为total
  // 若需支持多次写入同一数据集，可使用extend + hyperslab，此处简化为每步新文件重新创建数据集
  {
    hsize_t dim = total;
    H5::DataSpace space(1, &dim);
    auto dataset = m_file->createDataSet(varname, 
                                         detail::H5PredTypeSelector<T>::type(), 
                                         space);

    // 选择写入区域
    // 对于整个数据集只有一维，offset表示起始位置
    // 若需要分块写入可用selectHyperslab，这里简化为一次性全写
    // 因为只有1个域数据，就直接写data到[0]对应位置
    // 如果需要只写1个元素，可先读出来然后写入那部分
    // 简化：如果offset>0,说明域数据需要填到offset位置上，
    // 可用读写修改，但这里不考虑多域累加的数据集写入复杂情况
    //
    // 简化处理：如果多域写入，应该预先创建 dataset 并全部写入，否则需要 read-modify-write
    // 此处假设只有一个域或写入方式可覆盖整数组
    // 如果必须支持多域写入，请使用 extend 或先读取数据再写入。
    
    // 简单方式：如果只是单域写入，就直接写data到index=offset位置
    // 需要selectHyperslab
    hsize_t start[1] = { offset };
    hsize_t count[1] = { 1 };
    H5::DataSpace fspace = dataset.getSpace();
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);

    H5::DataSpace mspace(1, count);
    dataset.write(&data, detail::H5PredTypeSelector<T>::type(), mspace, fspace);
  }
}


template <Dimension D, int N>
void Writer::saveField(const std::string& fieldname,
                       const ndfield_t<D, N>& field) {
  raise::ErrorIf(m_file == nullptr, "File not open", HERE);

  auto field_h = Kokkos::create_mirror_view(field);
  Kokkos::deep_copy(field_h, field);

  // 假设 field维度大小为 field.extent(0),field.extent(1),... 并含有N分量 (最后一维)
  // 全局数据集大小: D维+1用于N分量
  hsize_t dims[D+1];
  for (int d = 0; d < D; d++) {
    dims[d] = field.extent(d);
  }
  dims[D] = N;

  H5::DataSpace space(D+1, dims);
  auto dataset = m_file->createDataSet(fieldname,
                                       H5::PredType::NATIVE_DOUBLE,
                                       space);

  // 将数据写入数据集
  dataset.write(field_h.data(), H5::PredType::NATIVE_DOUBLE);
}


template <typename T>
void Writer::saveParticleQuantity(const std::string& quantity,
                                  std::size_t species_id,
                                  std::size_t offset,
                                  std::size_t count,
                                  const array_t<T*>& data) {
  raise::ErrorIf(m_file == nullptr, "File not open", HERE);

  // 假设全局粒子数量 glob_total = offset+count，若需要全局数可作为参数传入
  // 简化：如果全局粒子数未知，至少本地count已知
  // 与ADIOS2不同，需要在创建数据集时指定大小，若需要扩展可用可扩展数据集
  hsize_t dim = offset + count; // 假设此为全局粒子数 (需要根据实际情况修改)
  H5::DataSpace space(1, &dim);
  auto dataset = m_file->createDataSet(
    fmt::format("s{}_{}", species_id + 1, quantity),
    detail::H5PredTypeSelector<T>::type(),
    space);

  auto data_h = Kokkos::create_mirror_view(data);
  Kokkos::deep_copy(data_h, data);

  // 写入 [offset, offset+count) 区域
  H5::DataSpace fspace = dataset.getSpace();
  hsize_t start[1] = { offset };
  hsize_t c[1] = { count };
  fspace.selectHyperslab(H5S_SELECT_SET, c, start);

  H5::DataSpace mspace(1, c);
  dataset.write(data_h.data(), detail::H5PredTypeSelector<T>::type(), mspace, fspace);
}


// 定义 detail::H5PredTypeSelector 用于类型映射
namespace detail {
  template<typename T>
  struct H5PredTypeSelector {
    static H5::PredType type() {
      if constexpr (std::is_same_v<T,int>) return H5::PredType::NATIVE_INT;
      else if constexpr (std::is_same_v<T,unsigned int>) return H5::PredType::NATIVE_UINT;
      else if constexpr (std::is_same_v<T,std::size_t>) return H5::PredType::NATIVE_ULLONG;
      else if constexpr (std::is_same_v<T,float>) return H5::PredType::NATIVE_FLOAT;
      else if constexpr (std::is_same_v<T,double>) return H5::PredType::NATIVE_DOUBLE;
      else if constexpr (std::is_same_v<T,short>) return H5::PredType::NATIVE_SHORT;
      else {
        static_assert(!sizeof(T*), "Unsupported type for H5PredTypeSelector");
      }
    }
  };
} // namespace detail

// 模板实例化
template void Writer::savePerDomainVariable<int>(const std::string&, std::size_t, std::size_t, int);
template void Writer::savePerDomainVariable<float>(const std::string&, std::size_t, std::size_t, float);
template void Writer::savePerDomainVariable<double>(const std::string&, std::size_t, std::size_t, double);
template void Writer::savePerDomainVariable<std::size_t>(const std::string&, std::size_t, std::size_t, std::size_t);

template void Writer::saveField<Dim::_1D, 3>(const std::string&, const ndfield_t<Dim::_1D, 3>&);
template void Writer::saveField<Dim::_1D, 6>(const std::string&, const ndfield_t<Dim::_1D, 6>&);
template void Writer::saveField<Dim::_2D, 3>(const std::string&, const ndfield_t<Dim::_2D, 3>&);
template void Writer::saveField<Dim::_2D, 6>(const std::string&, const ndfield_t<Dim::_2D, 6>&);
template void Writer::saveField<Dim::_3D, 3>(const std::string&, const ndfield_t<Dim::_3D, 3>&);
template void Writer::saveField<Dim::_3D, 6>(const std::string&, const ndfield_t<Dim::_3D, 6>&);

template void Writer::saveParticleQuantity<int>(const std::string&, std::size_t, std::size_t, std::size_t, const array_t<int*>&);
template void Writer::saveParticleQuantity<float>(const std::string&, std::size_t, std::size_t, std::size_t, const array_t<float*>&);
template void Writer::saveParticleQuantity<double>(const std::string&, std::size_t, std::size_t, std::size_t, const array_t<double*>&);
template void Writer::saveParticleQuantity<short>(const std::string&, std::size_t, std::size_t, std::size_t, const array_t<short*>&);

} // namespace checkpoint
