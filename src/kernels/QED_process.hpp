#ifndef QED_PROCESS_HPP
#define QED_PROCESS_HPP

#include "global.h"
#include "enums.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"

namespace kernel::QED{
    using namespace ntt;
    
    template <Dimension D, Coord::type C>
    class CurvatureEmission_kernel{
        static_assert(D == Dim::_1D, "Curvature emission is only implemented in 1D");
        static_assert(C == Coord::Cart, "Curvature emission is only implemented in cartesian coordinates");

        Particles<D, C>               spec;
        Particles<D, C>               spec_ph;
        const real_t                  e_min;
        const real_t                  coeff;
        const real_t                  gamma_emit;
        const real_t                  rho;
        random_number_pool_t          random_pool;


        public:
            CurvatureEmission_kernel(Particles<D, C>&       spec_,
                                     Particles<D, C>&       spec_ph_,
                                     const real_t           e_min_,
                                     const real_t           coeff_,
                                     const real_t           gamma_emit_,
                                     const real_t           rho_,
                                     random_number_pool_t&  random_pool_)
                : spec(spec_),
                  spec_ph(spec_ph_),
                  e_min(e_min_),
                  coeff(coeff_),
                  gamma_emit(gamma_emit_),
                  rho(rho_),
                  random_pool(random_pool_){
                    if (spec.charge() == 0){
                        raise::KernelError(HERE, "The first species must be charged.");
                    }
                    if (spec_ph.mass()!= 0){
                        raise::KernelError(HERE, "The second species must be photon.");
                    }
                  }
            ~CurvatureEmission_kernel() = default;

            Inline auto CDF(real_t zeta_) const -> real_t{
                return math::exp(-zeta_); 
           }

            Inline auto inverseCDF(real_t u) const -> real_t{
                return -math::log(u);
            }

            void operator()(index_t p) const{
                if(sepc.tag(p) != ParticleTag::alive){
                    if(spec.tag(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                const real_t pp = math::sqrt(ONE + NORM_SQR(spec.ux1(p), spec.ux2(p), spec.ux3(p)));
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                auto N_ph = static_cast<size_t>(coeff * CDF(zeta) / SQR(pp));
                //if N_ph is less than 1, no emission.
                if (N_ph < 1){
                    return;
                }
                auto n_ph = spec_ph.npart();
                spec_ph.set_npart(n_ph + N_ph);
                for (short i = 0; i < N_ph; ++i){
                    auto rand_gen = random_pool.get_state();

                    spec_ph.ux1(n_ph + i) = SIGN(spec.ux1(p)) * ONE;
                    spec_ph.pld(n_ph + i, 0) = inverseCDF(CDF(zeta) * Random<real_t>(rand_gen))
                                             * CUBE(pp / gamma_emit) / rho;
                    spec_ph.i1(n_ph + i) = spec.i1(p);
                    spec_ph.dx1(n_ph + i) = spec.dx1(p);
                    spec_ph.weight(n_ph + i) = spec.weight(p);
                    spec_ph.tag(n_ph + i) = ParticleTag::alive;

                    random_pool.free_state(rand_gen);
                }
            }
    };
} // namespace QED


#endif // QED_PROCESS_HPP