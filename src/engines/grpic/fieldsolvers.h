/**
 * @file engines/grpic/fieldsolvers.h
 * @brief Field solver routines (Faraday, Ampere, auxiliary fields) for the GRPIC engine
 * @implements
 *   - enum ntt::grpic::gr_getE
 *   - enum ntt::grpic::gr_getH
 *   - enum ntt::grpic::gr_faraday
 *   - enum ntt::grpic::gr_ampere
 *   - ntt::grpic::range_with_axis_BCs<> -> auto
 *   - ntt::grpic::ComputeAuxE<> -> void
 *   - ntt::grpic::ComputeAuxH<> -> void
 *   - ntt::grpic::Faraday<> -> void
 *   - ntt::grpic::Ampere<> -> void
 *   - ntt::grpic::AmpereCurrents<> -> void
 * @namespaces:
 *   - ntt::grpic::
 */

#ifndef ENGINES_GRPIC_FIELDSOLVERS_H
#define ENGINES_GRPIC_FIELDSOLVERS_H

#include "enums.h"

#include "utils/log.h"

#include "traits/pgen.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/ampere_axion_gr.hpp"
#include "kernels/ampere_gr.hpp"
#include "kernels/aux_fields_gr.hpp"
#include "kernels/faraday_gr.hpp"

#include <cstdint>

namespace ntt {
  namespace grpic {

    enum class gr_getE : uint8_t {
      D0_B,
      D_B0,
      D0_B0
    };
    enum class gr_getH : uint8_t {
      D_B0,
      D0_B0
    };
    enum class gr_faraday : uint8_t {
      aux,
      main
    };
    enum class gr_ampere : uint8_t {
      init,
      aux,
      main
    };

    template <GRMetricClass M>
    auto range_with_axis_BCs(const Domain<SimEngine::GRPIC, M>& domain)
      -> range_t<M::Dim> {
      auto range = domain.mesh.rangeActiveCells();
      /**
       * @brief taking one extra cell in the x1 and x2 directions if AXIS BCs
       */
      if constexpr (M::Dim == Dim::_2D) {
        if (domain.mesh.flds_bc_in({ 0, +1 }) == FldsBC::AXIS) {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1) - 1, domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        } else {
          range = CreateRangePolicy<Dim::_2D>(
            { domain.mesh.i_min(in::x1) - 1, domain.mesh.i_min(in::x2) },
            { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) });
        }
      } else if constexpr (M::Dim == Dim::_3D) {
        raise::Error("Invalid dimension", HERE);
      }
      return range;
    }

    template <GRMetricClass M>
    void ComputeAuxE(Domain<SimEngine::GRPIC, M>& domain, const gr_getE& g) {
      auto range = range_with_axis_BCs(domain);
      if (g == gr_getE::D0_B) {
        Kokkos::parallel_for(
          "ComputeAuxE",
          range,
          kernel::gr::ComputeAuxE_kernel<M>(domain.fields.em0, // D
                                            domain.fields.em,  // B
                                            domain.fields.aux, // E
                                            domain.mesh.metric));
      } else if (g == gr_getE::D_B0) {
        Kokkos::parallel_for("ComputeAuxE",
                             range,
                             kernel::gr::ComputeAuxE_kernel<M>(domain.fields.em,
                                                               domain.fields.em0,
                                                               domain.fields.aux,
                                                               domain.mesh.metric));
      } else if (g == gr_getE::D0_B0) {
        Kokkos::parallel_for("ComputeAuxE",
                             range,
                             kernel::gr::ComputeAuxE_kernel<M>(domain.fields.em0,
                                                               domain.fields.em0,
                                                               domain.fields.aux,
                                                               domain.mesh.metric));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    template <GRMetricClass M>
    void ComputeAuxH(Domain<SimEngine::GRPIC, M>& domain, const gr_getH& g) {
      auto range = range_with_axis_BCs(domain);
      if (g == gr_getH::D_B0) {
        Kokkos::parallel_for(
          "ComputeAuxH",
          range,
          kernel::gr::ComputeAuxH_kernel<M>(domain.fields.em,  // D
                                            domain.fields.em0, // B
                                            domain.fields.aux, // H
                                            domain.mesh.metric));
      } else if (g == gr_getH::D0_B0) {
        Kokkos::parallel_for("ComputeAuxH",
                             range,
                             kernel::gr::ComputeAuxH_kernel<M>(domain.fields.em0,
                                                               domain.fields.em0,
                                                               domain.fields.aux,
                                                               domain.mesh.metric));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    template <GRMetricClass M>
    void Faraday(Domain<SimEngine::GRPIC, M>& domain,
                 const SimulationParams&      params,
                 const prm::Parameters&       engine_params,
                 const gr_faraday&            g,
                 real_t                       fraction = ONE) {
      logger::Checkpoint("Launching Faraday kernel", HERE);
      const auto dt = engine_params.get<real_t>("dt");
      const auto dT = fraction *
                      params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      if (g == gr_faraday::aux) {
        Kokkos::parallel_for(
          "Faraday",
          domain.mesh.rangeActiveCells(),
          kernel::gr::Faraday_kernel<M>(domain.fields.em0, // Bin
                                        domain.fields.em0, // Bout
                                        domain.fields.aux, // E
                                        domain.mesh.metric,
                                        dT,
                                        domain.mesh.n_active(in::x2),
                                        domain.mesh.flds_bc()));
      } else if (g == gr_faraday::main) {
        Kokkos::parallel_for(
          "Faraday",
          domain.mesh.rangeActiveCells(),
          kernel::gr::Faraday_kernel<M>(domain.fields.em,
                                        domain.fields.em0,
                                        domain.fields.aux,
                                        domain.mesh.metric,
                                        dT,
                                        domain.mesh.n_active(in::x2),
                                        domain.mesh.flds_bc()));

      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    template <GRMetricClass M>
    void Ampere(Domain<SimEngine::GRPIC, M>& domain,
                const SimulationParams&      params,
                const prm::Parameters&       engine_params,
                const gr_ampere&             g,
                real_t                       fraction = ONE) {
      logger::Checkpoint("Launching Ampere kernel", HERE);

      const auto dt = engine_params.get<real_t>("dt");
      const auto dT = fraction *
                      params.template get<real_t>(
                        "algorithms.timestep.correction") *
                      dt;
      auto range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto ni2 = domain.mesh.n_active(in::x2);

      if (g == gr_ampere::aux) {
        // First push, updates D0 with J.
        Kokkos::parallel_for("Ampere-1",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em0, // Din
                                                          domain.fields.em0, // Dout
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else if (g == gr_ampere::main) {
        // Second push, updates D with J0 but assigns it to D0.
        Kokkos::parallel_for("Ampere-2",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.fields.em0,
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else if (g == gr_ampere::init) {
        // Second push, updates D with J0 and assigns it to D.
        Kokkos::parallel_for("Ampere-3",
                             range,
                             kernel::gr::Ampere_kernel<M>(domain.fields.em,
                                                          domain.fields.em,
                                                          domain.fields.aux,
                                                          domain.mesh.metric,
                                                          dT,
                                                          ni2,
                                                          domain.mesh.flds_bc()));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    template <GRMetricClass M>
    void AmpereCurrents(Domain<SimEngine::GRPIC, M>& domain,
                        const SimulationParams&      params,
                        const prm::Parameters&       engine_params,
                        const gr_ampere&             g) {
      logger::Checkpoint("Launching Ampere kernel for adding currents", HERE);

      const auto dt = engine_params.get<real_t>("dt");

      const auto q0    = params.template get<real_t>("scales.q0");
      const auto B0    = params.template get<real_t>("scales.B0");
      const auto coeff = -dt * q0 / B0;
      auto       range = CreateRangePolicy<Dim::_2D>(
        { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
        { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
      const auto ni2 = domain.mesh.n_active(in::x2);

      if (g == gr_ampere::aux) {
        // Updates D0 with J: D0(n-1/2) -> (J(n)) -> D0(n+1/2)
        Kokkos::parallel_for(
          "AmpereCurrentsAux",
          range,
          kernel::gr::CurrentsAmpere_kernel<M>(domain.fields.em0,
                                               domain.fields.cur,
                                               domain.mesh.metric,
                                               coeff,
                                               ni2,
                                               domain.mesh.flds_bc()));
      } else if (g == gr_ampere::main) {
        // Updates D0 with J0: D0(n) -> (J0(n+1/2)) -> D0(n+1)
        Kokkos::parallel_for(
          "AmpereCurrentsMain",
          range,
          kernel::gr::CurrentsAmpere_kernel<M>(domain.fields.em0,
                                               domain.fields.cur0,
                                               domain.mesh.metric,
                                               coeff,
                                               ni2,
                                               domain.mesh.flds_bc()));
      } else {
        raise::Error("Wrong option for `g`", HERE);
      }
    }

    /**
     * @brief Adds the axion background current to the D field, aligned with the
     * @brief two AmpereCurrents passes:
     * @brief - aux  : J_a evaluated at t = time (B = (em::B + em0::B)/2, E = aux::E at n)
     * @brief - main : J_a evaluated at t = time + dt/2 (B = em0::B, E = aux::E at n+1/2)
     * @brief No-op unless compiled with -D axion=ON and the pgen provides an
     * @brief `axion` functor (with `dot_a`/`grad_a` methods and member `eps`).
     */
#ifdef AXION_ENABLED
    template <GRMetricClass M, class PG>
    void AmpereAxion(Domain<SimEngine::GRPIC, M>& domain,
                     const SimulationParams&      params,
                     const prm::Parameters&       engine_params,
                     const PG&                    pgen,
                     const gr_ampere&             g,
                     simtime_t                    time) {
      if constexpr (traits::pgen::HasAxionField<PG>) {
        logger::Checkpoint("Launching Ampere kernel for adding axion current", HERE);

        const auto dt = engine_params.get<real_t>("dt");

        const auto q0    = params.template get<real_t>("scales.q0");
        const auto B0    = params.template get<real_t>("scales.B0");
        const auto coeff = -dt * q0 * pgen.axion.eps / B0;
        auto       range = CreateRangePolicy<Dim::_2D>(
          { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
          { domain.mesh.i_max(in::x1), domain.mesh.i_max(in::x2) + 1 });
        const auto ni2 = domain.mesh.n_active(in::x2);

        if (g == gr_ampere::aux) {
          // Updates D0 with J_a(n): D0(n-1/2) -> D0(n+1/2), B <- (em + em0)/2 at n
          Kokkos::parallel_for(
            "AmpereAxionAux",
            range,
            kernel::gr::CurrentsAmpereAxion_kernel<M, decltype(pgen.axion), true>(
              domain.fields.em0,
              domain.fields.em,
              domain.fields.em0,
              domain.fields.aux,
              domain.mesh.metric,
              coeff,
              pgen.axion,
              time,
              ni2,
              domain.mesh.flds_bc()));
        } else if (g == gr_ampere::main) {
          // Updates D0 with J_a(n+1/2): D0(n) -> D0(n+1), B <- em0 at n+1/2
          Kokkos::parallel_for(
            "AmpereAxionMain",
            range,
            kernel::gr::CurrentsAmpereAxion_kernel<M, decltype(pgen.axion), false>(
              domain.fields.em0,
              domain.fields.em0,
              domain.fields.em0,
              domain.fields.aux,
              domain.mesh.metric,
              coeff,
              pgen.axion,
              time,
              ni2,
              domain.mesh.flds_bc()));
        } else {
          raise::Error("Wrong option for `g`", HERE);
        }
      }
    }
#else  // AXION_ENABLED
    template <GRMetricClass M, class PG>
    Inline void AmpereAxion(Domain<SimEngine::GRPIC, M>&,
                            const SimulationParams&,
                            const prm::Parameters&,
                            const PG&,
                            const gr_ampere&,
                            simtime_t) {
      /* no-op without -D axion=ON */
    }
#endif // AXION_ENABLED

  } // namespace grpic
} // namespace ntt

#endif // ENGINES_GRPIC_FIELDSOLVERS_H