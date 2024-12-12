/**
 * @file checkpoint/writer.h
 * @brief Class that dumps checkpoints using HDF5
 * @implements
 *   - checkpoint::Writer
 * @cpp:
 *   - writer.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_WRITER_H
#define CHECKPOINT_WRITER_H

#include "enums.h"
#include "global.h"

#include "utils/tools.h"
#include "framework/parameters.h"

// 使用HDF5代替ADIOS2
#include <H5Cpp.h>

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  class Writer {
    H5::H5File* m_file { nullptr };

    tools::Tracker m_tracker {};

    bool m_writing_mode { false };

    // 存储已写出数据的信息（例如变量名与数据集名）
    std::vector<std::pair<std::string, std::string>> m_written;

    int  m_keep;
    bool m_enabled;

  public:
    Writer() : m_keep(0), m_enabled(true) {}
    ~Writer() {
      if (m_file) {
        delete m_file;
        m_file = nullptr;
      }
    }

    /**
     * @brief 初始化检查点写入器
     * @param filename 要写入的HDF5文件名
     * @param step 当前步数
     * @param time 当前时间
     * @param keep 保留的最多检查点数量（可能影响历史文件删除策略）
     */
    void init(const std::string& filename, std::size_t step, long double time, int keep);

    /**
     * @brief 判断在给定步骤和时间下是否应保存检查点
     * @param step 当前步数
     * @param time 当前时间
     * @return 是否应保存
     */
    auto shouldSave(std::size_t step, long double time) -> bool;

    /**
     * @brief 开始保存检查点
     * 创建类似 "/Step00000123" 的组，并在其中写入 Step 与 Time 属性
     * @param step 当前步数
     * @param time 当前时间
     */
    void beginSaving(std::size_t step, long double time);

    /**
     * @brief 结束保存检查点
     * 冲刷数据到磁盘
     */
    void endSaving();

    /**
     * @brief 将仿真参数写入HDF5文件作为属性
     * @param sim_params 仿真参数对象
     * @param time 当前时间
     */
    void saveAttrs(const ntt::SimulationParams& sim_params, long double time);

    /**
     * @brief 保存按域(per-domain)的标量变量（如粒子数等）
     * @tparam T 标量类型
     * @param name 数据集名称
     * @param local_dom 本地域索引
     * @param ndomains 总域数量
     * @param value 要保存的值
     */
    template <typename T>
    void savePerDomainVariable(const std::string& name, std::size_t local_dom, std::size_t ndomains, T value);

    /**
     * @brief 保存场数据（field arrays）
     * @tparam D 维度
     * @tparam N 分量数
     * @param name 数据集名称
     * @param field 要保存的场数据(ndfield_t)
     */
    template <Dimension D, int N>
    void saveField(const std::string& name, const ndfield_t<D, N>& field);

    /**
     * @brief 保存粒子属性数据
     * @tparam T 数据类型
     * @param quantity 属性名（如"X", "U", "W"）
     * @param offset 粒子数据起始偏移（全局）
     * @param count 本地粒子数量
     * @param species_id 种群ID
     * @param array 粒子属性数据数组
     */
    template <typename T>
    void saveParticleQuantity(const std::string& quantity,
                              std::size_t species_id,
                              std::size_t offset,
                              std::size_t count,
                              const array_t<T*>& array);

    /**
     * @brief 定义场数据集的元数据（如尺寸）以便后续保存field时使用
     * @param engine 仿真引擎（存有格点数目和相关信息）
     * @param glob_shape 全局形状
     * @param loc_corner 本地起始
     * @param loc_shape 本地形状
     */
    void defineFieldVariables(const ntt::SimEngine& engine,
                              const std::vector<std::size_t>& glob_shape,
                              const std::vector<std::size_t>& loc_corner,
                              const std::vector<std::size_t>& loc_shape);

    /**
     * @brief 定义粒子数据集的元数据（如坐标、速度分量名称等）以便后续保存粒子数据
     * @param coord 坐标系
     * @param dim 维度
     * @param ndomains 域数量
     * @param species_list 粒子种群列表
     */
    void defineParticleVariables(const ntt::Coord& coord,
                                 Dimension dim,
                                 std::size_t ndomains,
                                 const std::vector<unsigned short>& species_list);

    [[nodiscard]]
    auto enabled() const -> bool {
      return m_enabled;
    }
  };

} // namespace checkpoint

#endif // CHECKPOINT_WRITER_H
