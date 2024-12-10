#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "checkpoint/readerh5.h"
#include "checkpoint/writerh5.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include <filesystem>
#include <mpi.h>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::InitCheckpointWriter(const SimulationParams& params) {
    raise::ErrorIf(l_subdomain_indices().size() != 1,
                   "Checkpoint writing only supported for one subdomain per rank",
                   HERE);
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);

    std::vector<std::size_t> glob_shape_with_ghosts, off_ncells_with_ghosts;
    for (auto d { 0u }; d < M::Dim; ++d) {
      off_ncells_with_ghosts.push_back(
        local_domain->offset_ncells()[d] +
        2 * N_GHOSTS * local_domain->offset_ndomains()[d]);
      glob_shape_with_ghosts.push_back(
        mesh().n_active()[d] + 2 * N_GHOSTS * ndomains_per_dim()[d]);
    }
    auto loc_shape_with_ghosts = local_domain->mesh.n_all();

    std::vector<unsigned short> nplds;
    for (auto s { 0u }; s < local_domain->species.size(); ++s) {
      nplds.push_back(local_domain->species[s].npld());
    }

    auto interval      = params.template get<std::size_t>("checkpoint.interval");
    auto interval_time = params.template get<long double>("checkpoint.interval_time");
    auto keep          = params.template get<int>("checkpoint.keep");

    // 使用HDF5版本的g_checkpoint_writer
    g_checkpoint_writer.init("checkpoints", interval, interval_time, keep);
    if (g_checkpoint_writer.enabled()) {
      g_checkpoint_writer.defineFieldVariables(S, glob_shape_with_ghosts, off_ncells_with_ghosts, loc_shape_with_ghosts);
      g_checkpoint_writer.defineParticleVariables(M::CoordType, M::Dim, local_domain->species.size(), nplds);
    }
  }

  template <SimEngine::type S, class M>
  auto Metadomain<S, M>::WriteCheckpoint(const SimulationParams& params,
                                         std::size_t current_step,
                                         std::size_t finished_step,
                                         long double current_time,
                                         long double finished_time) -> bool {
    raise::ErrorIf(l_subdomain_indices().size() != 1,
                   "Checkpointing only supported for one subdomain per rank",
                   HERE);
    if (!g_checkpoint_writer.shouldSave(finished_step, finished_time) || finished_step <= 1) {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing checkpoint", HERE);

    g_checkpoint_writer.beginSaving(current_step, current_time);
    {
      g_checkpoint_writer.saveAttrs(params, current_time);
      g_checkpoint_writer.saveField<M::Dim, 6>("em", local_domain->fields.em);
      if constexpr (S == SimEngine::GRPIC) {
        g_checkpoint_writer.saveField<M::Dim, 6>("em0", local_domain->fields.em0);
        g_checkpoint_writer.saveField<M::Dim, 3>("cur0", local_domain->fields.cur0);
      }
      std::size_t dom_offset = 0, dom_tot = 1;
#if defined(MPI_ENABLED)
      dom_offset = g_mpi_rank;
      dom_tot    = g_mpi_size;
#endif // MPI_ENABLED

      for (auto s { 0u }; s < local_domain->species.size(); ++s) {
        auto        npart    = local_domain->species[s].npart();
        std::size_t offset   = 0;
        auto        glob_tot = npart;
#if defined(MPI_ENABLED)
        auto glob_npart = std::vector<std::size_t>(g_ndomains);
        MPI_Allgather(&npart,
                      1,
                      mpi::get_type<std::size_t>(),
                      glob_npart.data(),
                      1,
                      mpi::get_type<std::size_t>(),
                      MPI_COMM_WORLD);
        glob_tot = 0;
        for (auto r = 0; r < g_mpi_size; ++r) {
          if (r < g_mpi_rank) {
            offset += glob_npart[r];
          }
          glob_tot += glob_npart[r];
        }
#endif // MPI_ENABLED
        g_checkpoint_writer.savePerDomainVariable<std::size_t>(
          fmt::format("s%d_npart", s + 1), dom_tot, dom_offset, npart);

        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i1", glob_tot, offset, npart, local_domain->species[s].i1);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx1", glob_tot, offset, npart, local_domain->species[s].dx1);
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i1_prev", glob_tot, offset, npart, local_domain->species[s].i1_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx1_prev", glob_tot, offset, npart, local_domain->species[s].dx1_prev);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i2", glob_tot, offset, npart, local_domain->species[s].i2);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx2", glob_tot, offset, npart, local_domain->species[s].dx2);
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i2_prev", glob_tot, offset, npart, local_domain->species[s].i2_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx2_prev", glob_tot, offset, npart, local_domain->species[s].dx2_prev);
        }
        if constexpr (M::Dim == Dim::_3D) {
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i3", glob_tot, offset, npart, local_domain->species[s].i3);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx3", glob_tot, offset, npart, local_domain->species[s].dx3);
          g_checkpoint_writer.saveParticleQuantity<int>(
            "s" + std::to_string(s+1) + "_i3_prev", glob_tot, offset, npart, local_domain->species[s].i3_prev);
          g_checkpoint_writer.saveParticleQuantity<prtldx_t>(
            "s" + std::to_string(s+1) + "_dx3_prev", glob_tot, offset, npart, local_domain->species[s].dx3_prev);
        }
        if constexpr (M::Dim == Dim::_2D and M::CoordType != Coord::Cart) {
          g_checkpoint_writer.saveParticleQuantity<real_t>(
            "s" + std::to_string(s+1) + "_phi", glob_tot, offset, npart, local_domain->species[s].phi);
        }
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          "s" + std::to_string(s+1) + "_ux1", glob_tot, offset, npart, local_domain->species[s].ux1);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          "s" + std::to_string(s+1) + "_ux2", glob_tot, offset, npart, local_domain->species[s].ux2);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          "s" + std::to_string(s+1) + "_ux3", glob_tot, offset, npart, local_domain->species[s].ux3);
        g_checkpoint_writer.saveParticleQuantity<short>(
          "s" + std::to_string(s+1) + "_tag", glob_tot, offset, npart, local_domain->species[s].tag);
        g_checkpoint_writer.saveParticleQuantity<real_t>(
          "s" + std::to_string(s+1) + "_weight", glob_tot, offset, npart, local_domain->species[s].weight);

        auto nplds = local_domain->species[s].npld();
        for (auto p { 0u }; p < nplds; ++p) {
          g_checkpoint_writer.saveParticleQuantity<real_t>(
            fmt::format("s{}_pld{}", s+1, p+1), glob_tot, offset, npart, local_domain->species[s].pld[p]);
        }
      }
    }
    g_checkpoint_writer.endSaving();
    logger::Checkpoint("Checkpoint written", HERE);
    return true;
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::ContinueFromCheckpoint(const SimulationParams& params) {
    auto fname = fmt::format(
      "checkpoints/step-%08lu.h5",
      params.template get<std::size_t>("checkpoint.start_step"));
    logger::Checkpoint(fmt::format("Reading checkpoint from {}", fname), HERE);

    // 使用HDF5打开文件
    H5::H5File file(fname, H5F_ACC_RDONLY);

    // 不需要BeginStep/EndStep逻辑，HDF5是直接读取数据集即可

    for (auto& ldidx : l_subdomain_indices()) {
      auto& domain = g_subdomains[ldidx];
      raise::ErrorIf(domain.is_placeholder(), "Domain is placeholder", HERE);

      std::vector<hsize_t> start(M::Dim), count(M::Dim);
      for (auto d {0u}; d < M::Dim; ++d) {
        start[d] = domain.offset_ncells()[d] + 2 * N_GHOSTS * domain.offset_ndomains()[d];
        count[d] = domain.mesh.n_all()[d];
      }

      // 读电磁场: "em"维度多加一维为6分量
      checkpoint::ReadFields<M::Dim, 6>(file, "em", start, count, domain.fields.em);
      if constexpr (S == ntt::SimEngine::GRPIC) {
        checkpoint::ReadFieldsH5<M::Dim, 6>(file, "em0", start, count, domain.fields.em0);

        std::vector<hsize_t> count3(M::Dim), start3(M::Dim);
        for (auto d {0u}; d < M::Dim; ++d) {
          start3[d] = start[d];
          count3[d] = count[d];
        }
        // 3分量场 cur0
        checkpoint::ReadFields<M::Dim, 3>(file, "cur0", start3, count3, domain.fields.cur0);
      }

      // 读粒子数据
      for (auto s {0u}; s < (unsigned short)domain.species.size(); ++s) {
        const auto [loc_npart, offset_npart] = checkpoint::ReadParticleCountH5(file, s, ldidx, ndomains());

        raise::ErrorIf(loc_npart > domain.species[s].maxnpart(),
                       "loc_npart > domain.species[s].maxnpart()", HERE);
        if (loc_npart == 0) {
          continue;
        }

        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(file, "i1", s, domain.species[s].i1, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx1", s, domain.species[s].dx1, loc_npart, offset_npart);
          checkpoint::ReadParticleData<int>(file, "i1_prev", s, domain.species[s].i1_prev, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx1_prev", s, domain.species[s].dx1_prev, loc_npart, offset_npart);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(file, "i2", s, domain.species[s].i2, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx2", s, domain.species[s].dx2, loc_npart, offset_npart);
          checkpoint::ReadParticleData<int>(file, "i2_prev", s, domain.species[s].i2_prev, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx2_prev", s, domain.species[s].dx2_prev, loc_npart, offset_npart);
        }
        if constexpr (M::Dim == Dim::_3D) {
          checkpoint::ReadParticleData<int>(file, "i3", s, domain.species[s].i3, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx3", s, domain.species[s].dx3, loc_npart, offset_npart);
          checkpoint::ReadParticleData<int>(file, "i3_prev", s, domain.species[s].i3_prev, loc_npart, offset_npart);
          checkpoint::ReadParticleData<prtldx_t>(file, "dx3_prev", s, domain.species[s].dx3_prev, loc_npart, offset_npart);
        }
        if constexpr (M::Dim == Dim::_2D and M::CoordType != Coord::Cart) {
          checkpoint::ReadParticleData<real_t>(file, "phi", s, domain.species[s].phi, loc_npart, offset_npart);
        }
        checkpoint::ReadParticleData<real_t>(file, "ux1", s, domain.species[s].ux1, loc_npart, offset_npart);
        checkpoint::ReadParticleData<real_t>(file, "ux2", s, domain.species[s].ux2, loc_npart, offset_npart);
        checkpoint::ReadParticleData<real_t>(file, "ux3", s, domain.species[s].ux3, loc_npart, offset_npart);
        checkpoint::ReadParticleData<short>(file, "tag", s, domain.species[s].tag, loc_npart, offset_npart);
        checkpoint::ReadParticleData<real_t>(file, "weight", s, domain.species[s].weight, loc_npart, offset_npart);
        for (auto p {0u}; p < domain.species[s].npld(); ++p) {
          checkpoint::ReadParticleData<real_t>(file, fmt::format("pld{}", p+1), s, domain.species[s].pld[p], loc_npart, offset_npart);
        }
        domain.species[s].set_npart(loc_npart);
      } // species loop
    } // local subdomain loop

    logger::Checkpoint(fmt::format("Checkpoint reading done from {}", fname), HERE);
  }

  // 模板实例化
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt

