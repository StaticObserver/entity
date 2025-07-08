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

  InitFields(const unsigned int mode1,
             const unsigned int mode2,
             const real_t lx_,
             const real_t b_rms_,
             const real_t bz_)
             : n1 { mode1 }, 
               n2 { mode2 },
               lx { lx_ } ,
               b_rms { b_rms_ } ,
               bz { bz_ },
               phases { "Phases", n2 - n1 + 1, n2 - n1 + 1 } {
                raise::FatalIf(n2 <= n1, "Mode n2 must be greater than n1.", HERE);
                Kokkos::Random_XorShift1024_Pool<HostExeSpace> pool { constant::RandomSeed };
                Kokkos::Random_XorShift1024_Pool<HostExeSpace>::generator_type  rand_gen = pool.get_state();
                auto phases_h = Kokkos::create_mirror_view(phases);
                for (size_t i = 0; i < n2 - n1 + 1; ++i) {
                  for (size_t j = 0; j < n2 - n1 + 1; ++j) {
                    phases_h(i, j, 0) = rand_gen.drand() * constant::TWO_PI;
                    phases_h(i, j, 1) = rand_gen.drand() * constant::TWO_PI;
                  }
                }
                Kokkos::deep_copy(phases, phases_h);
                pool.free_state(rand_gen);
               }
  ~InitFields() = default;

  Inline auto bx1(const coord_t<D>& x_Ph) const {
    if(D != Dim::_2D) {
      return ZERO;
    }else{
      real_t value { ZERO };
      for (size_t i = 0; i < n2 - n1 + 1; ++i) {
        for (size_t j = 0; j < n2 - n1 + 1; ++j) {
          value += 2.0 * b_rms * (n1 + i) / (n2 - n1) / math::sqrt(SQR(n1 + i) + SQR(n1 + j))
                  * math::sin(constant::TWO_PI * (n1 + j) * x_Ph[0] / lx + phases(i, j, 0))
                  * math::cos(constant::TWO_PI * (n1 + i) * x_Ph[1] / lx + phases(i, j, 1));
        }
      }
      return value;
    }
  }

  Inline auto bx2(const coord_t<D>& x_Ph) const {
    if(D != Dim::_2D) {
      return ZERO;
    }else{
      real_t value { ZERO };
      for (size_t i = 0; i < n2 - n1 + 1; ++i) {
        for (size_t j = 0; j < n2 - n1 + 1; ++j) {
          value += - 2.0 * b_rms * (n1 + j) / (n2 - n1) / math::sqrt(SQR(n1 + i) + SQR(n1 + j))
                  * math::cos(constant::TWO_PI * (n1 + j) * x_Ph[0] / lx + phases(i, j, 0))
                  * math::sin(constant::TWO_PI * (n1 + i) * x_Ph[1] / lx + phases(i, j, 1));
        }
      }
      return value;
    }
  }

  Inline auto bx3(const coord_t<D>& x_Ph) const {
    return bz;
  }

  private:
    const unsigned int n1;
    const unsigned int n2;
    const real_t lx;
    const real_t b_rms;
    const real_t bz;
    array_t<real_t **[2]> phases;

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
        init_flds { static_cast<unsigned int>(params.template get<int>("setup.mode1")),
                    static_cast<unsigned int>(params.template get<int>("setup.mode2")),
                    global_domain.mesh().extent(in::x1).second - 
                    global_domain.mesh().extent(in::x1).first,
                    params.template get<real_t>("setup.b_rms", 1.0),
                    params.template get<real_t>("setup.bz", 1.0) },
        temperature { params.template get<real_t>("setup.temperature", 0.1) }{}


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
