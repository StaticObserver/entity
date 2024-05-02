#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/timer.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"

  #include "framework/domain/comm_mpi.hpp"
#else
  #include "framework/domain/comm_nompi.hpp"
#endif

#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::Communicate(Domain<S, M>&  domain,
                                     CommTags       tags,
                                     timer::Timers* timers) {
    const auto comm_fields = (tags & Comm::E) || (tags & Comm::B) ||
                             (tags & Comm::J) || (tags & Comm::D) ||
                             (tags & Comm::D0) || (tags & Comm::B0);
    const bool comm_em = (tags & Comm::E) || (tags & Comm::B) || (tags & Comm::D);
    const bool comm_em0 = (tags & Comm::B0) || (tags & Comm::D0);
    const bool comm_j   = (tags & Comm::J);
    const bool sync_j   = (tags & Comm::J_sync);
    raise::ErrorIf(comm_fields && sync_j,
                   "Cannot communicate fields and sync currents simultaneously",
                   HERE);

    std::string comms = "";
    if (tags & Comm::E) {
      comms += "E ";
    }
    if (tags & Comm::B) {
      comms += "B ";
    }
    if (tags & Comm::J) {
      comms += "J ";
    }
    if (tags & Comm::D) {
      comms += "D ";
    }
    if (tags & Comm::D0) {
      comms += "D0 ";
    }
    if (tags & Comm::B0) {
      comms += "B0 ";
    }
    if (tags & Comm::Prtl) {
      comms += "Prtl ";
    }
    if (tags & Comm::J_sync) {
      comms += "J_sync ";
    }
    logger::Checkpoint(fmt::format("Communicating %s\n", comms.c_str()), HERE);

    /**
     * @note this block is designed to support in the future multiple domains
     * on a single rank, however that is not yet implemented
     */
    if (comm_fields || sync_j) {
      // establish the last index ranges for fields (i.e., components)
      auto comp_range_fld = range_tuple_t {};
      auto comp_range_cur = range_tuple_t {};
      if constexpr (S == SimEngine::GRPIC) {
        if (((tags & Comm::D) && (tags & Comm::B)) ||
            ((tags & Comm::D0) && (tags & Comm::B0))) {
          comp_range_fld = range_tuple_t(em::dx1, em::bx3 + 1);
        } else if ((tags & Comm::D) || (tags & Comm::D0)) {
          comp_range_fld = range_tuple_t(em::dx1, em::dx3 + 1);
        } else if ((tags & Comm::B) || (tags & Comm::B0)) {
          comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
        }
      } else if constexpr (S == SimEngine::SRPIC) {
        if ((tags & Comm::E) && (tags & Comm::B)) {
          comp_range_fld = range_tuple_t(em::ex1, em::bx3 + 1);
        } else if (tags & Comm::E) {
          comp_range_fld = range_tuple_t(em::ex1, em::ex3 + 1);
        } else if (tags & Comm::B) {
          comp_range_fld = range_tuple_t(em::bx1, em::bx3 + 1);
        }
      } else {
        raise::Error("Unknown simulation engine", HERE);
      }
      if (comm_j || sync_j) {
        comp_range_cur = range_tuple_t(cur::jx1, cur::jx3 + 1);
      }
      // traverse in all directions and send/recv the fields
      for (auto& direction : dir::Directions<M::Dim>::all) {
        Domain<S, M>* send_to_nghbr_ptr   = nullptr;
        Domain<S, M>* recv_from_nghbr_ptr = nullptr;
        // set pointers to the correct send/recv domains
        // can coincide with the current domain if periodic
        if (domain.mesh.flds_bc_in(direction) == FldsBC::PERIODIC) {
          // sending / receiving from itself
          raise::ErrorIf(
            domain.neighbor_idx_in(direction) != domain.index(),
            "Periodic boundaries imply communication within the same domain",
            HERE);
          raise::ErrorIf(
            domain.mesh.flds_bc_in(-direction) != FldsBC::PERIODIC,
            "Periodic boundary conditions must be set in both directions",
            HERE);
          send_to_nghbr_ptr = recv_from_nghbr_ptr = &domain;
        } else if (domain.mesh.flds_bc_in(direction) == FldsBC::SYNC) {
          // sending to other domain
          raise::ErrorIf(
            domain.neighbor_idx_in(direction) == domain.index(),
            "Sync boundaries imply communication between separate domains",
            HERE);
          send_to_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(direction));
          if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
            // receiving from other domain
            raise::ErrorIf(
              domain.neighbor_idx_in(-direction) == domain.index(),
              "Sync boundaries imply communication between separate domains",
              HERE);
            recv_from_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(-direction));
          }
        } else if (domain.mesh.flds_bc_in(-direction) == FldsBC::SYNC) {
          // only receiving from other domain
          raise::ErrorIf(
            domain.neighbor_idx_in(-direction) == domain.index(),
            "Sync boundaries imply communication between separate domains",
            HERE);
          recv_from_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(-direction));
        } else {
          // no communication necessary
          continue;
        }
        const auto is_sending   = (send_to_nghbr_ptr != nullptr);
        const auto is_receiving = (recv_from_nghbr_ptr != nullptr);
        auto       send_slice   = std::vector<range_tuple_t> {};
        auto       recv_slice   = std::vector<range_tuple_t> {};
        const in   components[] = { in::x1, in::x2, in::x3 };
        // find the field components and indices to be sent/received
        for (std::size_t d { 0 }; d < direction.size(); ++d) {
          const auto c   = components[d];
          const auto dir = direction[d];
          if (comm_fields) {
            if (is_sending) {
              if (dir == 0) {
                send_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (dir == 1) {
                send_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                        domain.mesh.i_max(c));
              } else {
                send_slice.emplace_back(domain.mesh.i_min(c),
                                        domain.mesh.i_min(c) + N_GHOSTS);
              }
            }
            if (is_receiving) {
              if (-dir == 0) {
                recv_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (-dir == 1) {
                recv_slice.emplace_back(domain.mesh.i_max(c),
                                        domain.mesh.i_max(c) + N_GHOSTS);
              } else {
                recv_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                        domain.mesh.i_min(c));
              }
            }
          } else if (sync_j) {
            if (is_sending) {
              if (dir == 0) {
                send_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (dir == 1) {
                send_slice.emplace_back(domain.mesh.i_max(c),
                                        domain.mesh.i_max(c) + N_GHOSTS);
              } else {
                send_slice.emplace_back(domain.mesh.i_min(c) - N_GHOSTS,
                                        domain.mesh.i_min(c));
              }
            }
            if (is_receiving) {
              if (-dir == 0) {
                recv_slice.emplace_back(domain.mesh.i_min(c), domain.mesh.i_max(c));
              } else if (-dir == 1) {
                recv_slice.emplace_back(domain.mesh.i_max(c) - N_GHOSTS,
                                        domain.mesh.i_max(c));
              } else {
                recv_slice.emplace_back(domain.mesh.i_min(c),
                                        domain.mesh.i_min(c) + N_GHOSTS);
              }
            }
          }
        }
#if defined(MPI_ENABLED)
        const auto send_rank = (send_to_nghbr_ptr != nullptr)
                                 ? send_to_nghbr_ptr->mpi_rank()
                                 : -1;
        const auto recv_rank = (recv_from_nghbr_ptr != nullptr)
                                 ? recv_from_nghbr_ptr->mpi_rank()
                                 : -1;
#else
        const auto send_rank = (send_to_nghbr_ptr != nullptr) ? 0 : -1;
        const auto recv_rank = (recv_from_nghbr_ptr != nullptr) ? 0 : -1;
#endif
        const auto send_idx = (send_to_nghbr_ptr != nullptr)
                                ? send_to_nghbr_ptr->index()
                                : 0;
        const auto recv_idx = (recv_from_nghbr_ptr != nullptr)
                                ? recv_from_nghbr_ptr->index()
                                : 0;
        if (comm_em) {
          comm::CommunicateField<M::Dim, 6>(domain.index(),
                                            domain.fields.em,
                                            send_idx,
                                            recv_idx,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_fld,
                                            false);
        }
        if constexpr (S == SimEngine::GRPIC) {
          if (comm_em0) {
            comm::CommunicateField<M::Dim, 6>(domain.index(),
                                              domain.fields.em0,
                                              send_idx,
                                              recv_idx,
                                              send_rank,
                                              recv_rank,
                                              send_slice,
                                              recv_slice,
                                              comp_range_fld,
                                              false);
          }
        }
        if (comm_j || sync_j) {
          comm::CommunicateField<M::Dim, 3>(domain.index(),
                                            domain.fields.cur,
                                            send_idx,
                                            recv_idx,
                                            send_rank,
                                            recv_rank,
                                            send_slice,
                                            recv_slice,
                                            comp_range_cur,
                                            sync_j);
        }
      }
    }
    if (tags & Comm::Prtl) {
      raise::ErrorIf(timers == nullptr,
                     "Timers not passed when Comm::Prtl called",
                     HERE);
      for (auto& species : domain.species) {
        // at this point particles should already by tagged in the pusher
        timers->start("Sorting");
        const auto npart_per_tag = species.SortByTags();
        timers->stop("Sorting");
#if defined(MPI_ENABLED)
        timers.start("Communications");
        // only necessary when MPI is enabled
        /**
         *                                                        index_last
         *                                                            |
         *    alive      new dead         tag1       tag2             v     dead
         * [ 11111111   000000000    222222222    3333333 .... nnnnnnn  00000000 ... ]
         *                           ^        ^
         *                           |        |
         *     tag_offset[tag1] -----+        +----- tag_offset[tag1] + npart_per_tag[tag1]
         *          "send_pmin"                      "send_pmax" (after last element)
         */
        auto tag_offset { npart_per_tag };
        for (std::size_t i { 1 }; i < tag_offset.size(); ++i) {
          tag_offset[i] += tag_offset[i - 1];
        }
        for (std::size_t i { 0 }; i < tag_offset.size(); ++i) {
          tag_offset[i] -= npart_per_tag[i];
        }
        auto index_last = tag_offset[tag_offset.size() - 1] +
                          npart_per_tag[npart_per_tag.size() - 1];
        for (auto& direction : dir::Directions<D>::all) {
          Domain<S, M>* send_to_nghbr_ptr   = nullptr;
          Domain<S, M>* recv_from_nghbr_ptr = nullptr;
          // set pointers to the correct send/recv domains
          // can coincide with the current domain if periodic
          if (domain.mesh.prtl_bc_in(direction) == PrtlBC::PERIODIC) {
            // sending / receiving from itself
            raise::ErrorIf(
              domain.neighbor_idx_in(direction) != domain.index(),
              "Periodic boundaries imply communication within the same domain",
              HERE);
            raise::ErrorIf(
              domain.mesh.prtl_bc_in(-direction) != PrtlBC::PERIODIC,
              "Periodic boundary conditions must be set in both directions",
              HERE);
            send_to_nghbr_ptr = recv_from_nghbr_ptr = &domain;
          } else if (domain.mesh.prtl_bc_in(direction) == PrtlBC::SYNC) {
            // sending to other domain
            raise::ErrorIf(
              domain.neighbor_idx_in(direction) == domain.index(),
              "Sync boundaries imply communication between separate domains",
              HERE);
            send_to_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(direction));
            if (domain.mesh.prtl_bc_in(-direction) == PrtlBC::SYNC) {
              // receiving from other domain
              raise::ErrorIf(
                domain.neighbor_idx_in(-direction) == domain.index(),
                "Sync boundaries imply communication between separate domains",
                HERE);
              recv_from_nghbr_ptr = subdomain_ptr(
                domain.neighbor_idx_in(-direction));
            }
          } else if (domain.mesh.prtl_bc_in(-direction) == PrtlBC::SYNC) {
            // only receiving from other domain
            raise::ErrorIf(
              domain.neighbor_idx_in(-direction) == domain.index(),
              "Sync boundaries imply communication between separate domains",
              HERE);
            recv_from_nghbr_ptr = subdomain_ptr(domain.neighbor_idx_in(-direction));
          } else {
            // no communication necessary
            continue;
          }

          const auto send_dir_tag = mpi::PrtlSendTag<D>::dir2tag(direction);
          const auto nsend        = npart_per_tag[send_dir_tag];
          const auto send_pmin    = tag_offset[send_dir_tag];
          const auto send_pmax    = tag_offset[send_dir_tag] + nsend;
          const auto send_rank    = (send_to_nghbr_ptr != nullptr)
                                      ? send_to_nghbr_ptr->mpi_rank()
                                      : -1;
          const auto recv_rank    = (recv_from_nghbr_ptr != nullptr)
                                      ? recv_from_nghbr_ptr->mpi_rank()
                                      : -1;
          const auto recv_count = comm::CommunicateParticles<M::Dim, M::CoordType>(
            species,
            send_rank,
            recv_rank,
            { send_pmin, send_pmax },
            index_last);
          if (recv_count > 0) {
            if constexpr (D == Dim::_1D) {
              int shift_in_x1 { 0 };
              if ((-direction)[0] == -1) {
                shift_in_x1 = -recv_from_nghbr_ptr->mesh.n_active(in::x1);
              } else if ((-direction)[0] == 1) {
                shift_in_x1 = domain.mesh.n_active(in::x1);
              }
              auto& this_tag     = species.tag;
              auto& this_i1      = species.i1;
              auto& this_i1_prev = species.i1_prev;
              Kokkos::parallel_for(
                "CommunicateParticles",
                recv_count,
                Lambda(index_t p) {
                  this_tag(index_last + p)      = ParticleTag::alive;
                  this_i1(index_last + p)      += shift_in_x1;
                  this_i1_prev(index_last + p) += shift_in_x1;
                });
            } else if constexpr (D == Dim::_2D) {
              int shift_in_x1 { 0 }, shift_in_x2 { 0 };
              if ((-direction)[0] == -1) {
                shift_in_x1 = -recv_from_nghbr_ptr->mesh.n_active(in::x1);
              } else if ((-direction)[0] == 1) {
                shift_in_x1 = domain.mesh.n_active()[0];
              }
              if ((-direction)[1] == -1) {
                shift_in_x2 = -recv_from_nghbr_ptr->mesh.n_active(in::x2);
              } else if ((-direction)[1] == 1) {
                shift_in_x2 = domain.mesh.n_active(in::x2);
              }
              auto& this_tag     = species.tag;
              auto& this_i1      = species.i1;
              auto& this_i2      = species.i2;
              auto& this_i1_prev = species.i1_prev;
              auto& this_i2_prev = species.i2_prev;
              Kokkos::parallel_for(
                "CommunicateParticles",
                recv_count,
                Lambda(index_t p) {
                  this_tag(index_last + p)      = ParticleTag::alive;
                  this_i1(index_last + p)      += shift_in_x1;
                  this_i2(index_last + p)      += shift_in_x2;
                  this_i1_prev(index_last + p) += shift_in_x1;
                  this_i2_prev(index_last + p) += shift_in_x2;
                });
            } else if constexpr (D == Dim::_3D) {
              int shift_in_x1 { 0 }, shift_in_x2 { 0 }, shift_in_x3 { 0 };
              if ((-direction)[0] == -1) {
                shift_in_x1 = -recv_from_nghbr_ptr->mesh.n_active(in::x1);
              } else if ((-direction)[0] == 1) {
                shift_in_x1 = domain.mesh.n_active(in::x1);
              }
              if ((-direction)[1] == -1) {
                shift_in_x2 = -recv_from_nghbr_ptr->mesh.n_active(in::x2);
              } else if ((-direction)[1] == 1) {
                shift_in_x2 = domain.mesh.n_active(in::x2);
              }
              if ((-direction)[2] == -1) {
                shift_in_x3 = -recv_from_nghbr_ptr->mesh.n_active(in::x3);
              } else if ((-direction)[2] == 1) {
                shift_in_x3 = domain.mesh.n_active(in::x3);
              }
              auto& this_tag     = species.tag;
              auto& this_i1      = species.i1;
              auto& this_i2      = species.i2;
              auto& this_i3      = species.i3;
              auto& this_i1_prev = species.i1_prev;
              auto& this_i2_prev = species.i2_prev;
              auto& this_i3_prev = species.i3_prev;
              Kokkos::parallel_for(
                "CommunicateParticles",
                recv_count,
                Lambda(index_t p) {
                  this_tag(index_last + p)      = ParticleTag::alive;
                  this_i1(index_last + p)      += shift_in_x1;
                  this_i2(index_last + p)      += shift_in_x2;
                  this_i3(index_last + p)      += shift_in_x3;
                  this_i1_prev(index_last + p) += shift_in_x1;
                  this_i2_prev(index_last + p) += shift_in_x2;
                  this_i3_prev(index_last + p) += shift_in_x3;
                });
            }
            index_last += recv_count;
            species.set_npart(index_last);
          }

          Kokkos::deep_copy(
            Kokkos::subview(species.tag, std::make_pair(send_pmin, send_pmax)),
            ParticleTag::dead);
        }
        timers.stop("Communications");
        // !TODO: maybe there is a way to not sort twice
        timers->start("Sorting");
        species.SortByTags();
        timers->stop("Sorting");
#endif
      }
    }
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt