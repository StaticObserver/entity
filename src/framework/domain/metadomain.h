/**
 * @file framework/domain/metadomain.h
 * @brief ...
 * @implements
 *   - ntt::Metadomain<>
 * @cpp:
 *   - metadomain.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 */

#ifndef FRAMEWORK_DOMAIN_METADOMAIN_H
#define FRAMEWORK_DOMAIN_METADOMAIN_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/timer.h"

#include "framework/containers/species.h"
#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/parameters.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#if defined(OUTPUT_ENABLED)
  // 移除adios2相关包含，使用HDF5版本的writer
  #include "checkpoint/writer.h"
  #include "output/writer.h"
#endif // OUTPUT_ENABLED

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <numeric> // for std::accumulate

namespace ntt {

  template <SimEngine::type S, class M>
  struct Metadomain {
    static_assert(M::is_metric,
                  "template arg for Metadomain class has to be a metric");
    static constexpr Dimension D { M::Dim };

    void initialValidityCheck() const;
    void finalValidityCheck() const;
    void metricCompatibilityCheck() const;

    void createEmptyDomains();
    void redefineNeighbors();
    void redefineBoundaries();

    template <typename Func, typename... Args>
    void runOnLocalDomains(Func func, Args&&... args) {
      for (auto& ldidx : l_subdomain_indices()) {
        func(g_subdomains[ldidx], std::forward<Args>(args)...);
      }
    }

    template <typename Func, typename... Args>
    void runOnLocalDomainsConst(Func func, Args&&... args) const {
      for (auto& ldidx : l_subdomain_indices()) {
        func(g_subdomains[ldidx], std::forward<Args>(args)...);
      }
    }

    void CommunicateFields(Domain<S, M>&, CommTags);
    void SynchronizeFields(Domain<S, M>&, CommTags, const range_tuple_t& = { 0, 0 });
    void CommunicateParticles(Domain<S, M>&, timer::Timers*);

    Metadomain(unsigned int,
               const std::vector<int>&,
               const std::vector<std::size_t>&,
               const boundaries_t<real_t>&,
               const boundaries_t<FldsBC>&,
               const boundaries_t<PrtlBC>&,
               const std::map<std::string, real_t>&,
               const std::vector<ParticleSpecies>&);

    Metadomain(const Metadomain&)            = delete;
    Metadomain& operator=(const Metadomain&) = delete;

    ~Metadomain() = default;

#if defined(OUTPUT_ENABLED)
    /**
     * @brief 初始化用于输出场数据的writer (HDF5版本)
     * @param params 仿真参数
     * @param is_resuming 若为true表示从已有数据继续运行
     */
    void InitWriter(const SimulationParams&, bool is_resuming);

    /**
     * @brief 将场数据和粒子数据写出到HDF5文件中
     * @param params 仿真参数
     * @param current_step 当前步骤数
     * @param finished_step 已完成步骤数
     * @param current_time 当前时间
     * @param finished_time 已完成时间
     * @param extra_save_func 可选的额外输出回调函数
     * @return 是否写出了数据
     */
    auto Write(const SimulationParams&,
               std::size_t,
               std::size_t,
               long double,
               long double,
               std::function<void(const std::string&,
                                  ndfield_t<M::Dim, 6>&,
                                  std::size_t,
                                  const Domain<S, M>&)> = {}) -> bool;

    /**
     * @brief 初始化检查点写入器 (HDF5版本)
     */
    void InitCheckpointWriter(const SimulationParams&);

    /**
     * @brief 写出检查点数据 (HDF5版本)
     */
    auto WriteCheckpoint(const SimulationParams&,
                         std::size_t,
                         std::size_t,
                         long double,
                         long double) -> bool;

    /**
     * @brief 从检查点数据继续仿真 (HDF5版本)
     */
    void ContinueFromCheckpoint(const SimulationParams&);
#endif

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto ndomains() const -> unsigned int {
      return g_ndomains;
    }

    [[nodiscard]]
    auto ndomains_per_dim() const -> std::vector<unsigned int> {
      return g_ndomains_per_dim;
    }

    [[nodiscard]]
    auto subdomain(unsigned int idx) const -> const Domain<S, M>& {
      raise::ErrorIf(idx >= g_subdomains.size(), "subdomain() failed", HERE);
      return g_subdomains[idx];
    }

    [[nodiscard]]
    auto subdomain_ptr(unsigned int idx) -> Domain<S, M>* {
      raise::ErrorIf(idx >= g_subdomains.size(), "subdomain_ptr() failed", HERE);
      return &g_subdomains[idx];
    }

    [[nodiscard]]
    auto mesh() const -> const Mesh<M>& {
      return g_mesh;
    }

    [[nodiscard]]
    auto species_params() const -> const std::vector<ParticleSpecies>& {
      return g_species_params;
    }

    [[nodiscard]]
    auto l_subdomain_indices() const -> std::vector<unsigned int> {
      return g_local_subdomain_indices;
    }

    [[nodiscard]]
    auto l_npart_perspec() const -> std::vector<std::size_t> {
      std::vector<std::size_t> npart(g_species_params.size(), 0);
      for (const auto& ldidx : l_subdomain_indices()) {
        for (std::size_t i = 0; i < g_species_params.size(); ++i) {
          npart[i] += g_subdomains[ldidx].species[i].npart();
        }
      }
      return npart;
    }

    [[nodiscard]]
    auto l_maxnpart_perspec() const -> std::vector<std::size_t> {
      std::vector<std::size_t> maxnpart(g_species_params.size(), 0);
      for (const auto& ldidx : l_subdomain_indices()) {
        for (std::size_t i = 0; i < g_species_params.size(); ++i) {
          maxnpart[i] += g_subdomains[ldidx].species[i].maxnpart();
        }
      }
      return maxnpart;
    }

    [[nodiscard]]
    auto l_npart() const -> std::size_t {
      const auto npart = l_npart_perspec();
      return std::accumulate(npart.begin(), npart.end(), 0ull);
    }

    [[nodiscard]]
    auto l_ncells() const -> std::size_t {
      std::size_t ncells_local = 0;
      for (const auto& ldidx : l_subdomain_indices()) {
        std::size_t ncells = 1;
        for (const auto& n : g_subdomains[ldidx].mesh.n_all()) {
          ncells *= n;
        }
        ncells_local += ncells;
      }
      return ncells_local;
    }

    [[nodiscard]]
    auto species_labels() const -> std::vector<std::string> {
      std::vector<std::string> labels;
      for (const auto& sp : g_species_params) {
        labels.push_back(sp.label());
      }
      return labels;
    }

  private:
    // domain information
    unsigned int g_ndomains;

    std::vector<int>                                  g_decomposition;
    std::vector<unsigned int>                         g_ndomains_per_dim;
    std::vector<std::vector<unsigned int>>            g_domain_offsets;
    std::map<std::vector<unsigned int>, unsigned int> g_domain_offset2index;

    std::vector<Domain<S, M>> g_subdomains;
    std::vector<unsigned int> g_local_subdomain_indices;

    Mesh<M>                             g_mesh;
    const std::map<std::string, real_t> g_metric_params;
    const std::vector<ParticleSpecies>  g_species_params;

#if defined(OUTPUT_ENABLED)
    // 使用HDF5版本的writer
    out::Writer        g_writer;           // 对应重构后的 output/writerh5.h
    checkpoint::Writer g_checkpoint_writer; // 对应重构后的 checkpoint/writerh5.h
#endif

#if defined(MPI_ENABLED)
    int g_mpi_rank, g_mpi_size;
#endif
  };

} // namespace ntt

#endif // FRAMEWORK_DOMAIN_METADOMAIN_H


