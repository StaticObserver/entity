#ifndef QED_PROCESS_HPP
#define QED_PROCESS_HPP

#include "global.h"
#include "enums.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>

namespace kernel::QED{
    using namespace ntt;
    
    template <Dimension D, Coord::type C>
    class CurvatureEmission_kernel{
        static_assert(D == Dim::_1D, "Curvature emission is only implemented in 1D");
        static_assert(C == Coord::Cart, "Curvature emission is only implemented in cartesian coordinates");

        const array_t<real_t*>        ux1, ux2, ux3;
        const array_t<real_t*>        weight;
        const array_t<short*>         tag;
        const array_t<int*>           i1;
        const array_t<prtldx_t*>      dx1;
        
        array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
        array_t<real_t*>              weight_ph;
        array_t<short*>               tag_ph;
        array_t<real_t**>             pld_ph;
        array_t<int*>                 i1_ph;
        array_t<prtldx_t*>            dx1_ph;

        const array_t<size_t*>        N_phs;
        const array_t<int*>           offsets;

        const real_t                  e_min;
        const real_t                  gamma_emit;
        const real_t                  rho;
        const size_t                  N_max;
        random_number_pool_t          random_pool;



        public:
            CurvatureEmission_kernel(Particles<D, C>&          charges,
                                     Particles<D, C>&          photons,
                                     const real_t              e_min_,
                                     const real_t              gamma_emit_,
                                     const real_t              rho_,
                                     const size_t              N_max,
                                     const array_t<size_t*>&   N_phs_,
                                     const array_t<int*>&      offsets_,
                                     random_number_pool_t&     random_pool_)
                : ux1 { charges.ux1 }
                , ux2 { charges.ux2 }
                , ux3 { charges.ux3 }
                , weight { charges.weight }
                , tag { charges.tag }
                , i1 { charges.i1 }
                , dx1 { charges.dx1 }
                , ux1_ph { photons.ux1 }
                , ux2_ph { photons.ux2 }
                , ux3_ph { photons.ux3 }
                , weight_ph { photons.weight }
                , tag_ph { photons.tag }
                , pld_ph { photons.pld }
                , i1_ph { photons.i1 }
                , dx1_ph { photons.dx1 }
                , e_min { e_min_ }
                , gamma_emit { gamma_emit_ }
                , rho { rho_ }
                , N_max { N_max }
                , N_phs { N_phs_ }
                , offsets { offsets_ }
                , random_pool { random_pool_ } { }
            ~CurvatureEmission_kernel() = default;

            Inline auto CDF(real_t zeta_) const -> real_t{
                return math::exp(-zeta_); 
           }

            Inline auto inverseCDF(real_t u) const -> real_t{
                return -math::log(u);
            }

            Inline void operator()(index_t p) const{
                if(tag(p) != ParticleTag::alive){
                    if(tag(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                if (offsets(p) < 0){
                    return;
                }
                // weight correction
                auto w_crect = ONE;
                auto N_ph = N_phs(p);
                if (N_ph > N_max){
                    w_crect *= N_ph / N_max;
                    N_ph = N_max;
                }
            
                for (short i = 0; i < N_ph; ++i){
                    auto rand_gen = random_pool.get_state();

                    ux1_ph(offsets(p) + i) = SIGN(ux1(p)) * ONE;
                    pld_ph(offsets(p) + i, 0) = inverseCDF(CDF(zeta) * Random<real_t>(rand_gen))
                                             * CUBE(pp / gamma_emit) / rho;
                    i1_ph(offsets(p) + i) = i1(p);
                    dx1_ph(offsets(p) + i) = dx1(p);
                    weight_ph(offsets(p) + i) = w_crect * weight(p);
                    tag_ph(offsets(p) + i) = ParticleTag::alive;

                    random_pool.free_state(rand_gen);
                }
            }
    };
    
    template <Dimension D, Coord::type C>
    class Curvature_Emission_Number{
        static_assert(D == Dim::_1D, "Curvature emission is only implemented in 1D");
        static_assert(C == Coord::Cart, "Curvature emission is only implemented in cartesian coordinates");

        const array_t<real_t*>        ux1, ux2, ux3;
        const array_t<short*>         tag;
        
        array_t<size_t*>              N_phs;

        const real_t                  e_min;
        const real_t                  coeff;
        const real_t                  gamma_emit;
        const real_t                  rho;

       
        Inline auto CDF(real_t zeta_) const -> real_t{
            return math::exp(-zeta_); 
        }

        public:
            Curvature_Emission_Number(const Particles<D, C>& charge,
                                       const real_t e_min_,
                                       const real_t coeff_,
                                       const real_t gamma_emit_,
                                       const real_t rho_,
                                       array_t<size_t*>& N_phs_)
                : ux1 { charge.ux1 }
                , ux2 { charge.ux2 }
                , ux3 { charge.ux3 }
                , tag { charge.tag }
                , e_min { e_min_ }
                , coeff { coeff_ }
                , gamma_emit { gamma_emit_ }
                , rho { rho_ }
                , N_phs { N_phs_ }  {
                    Kokkos::deep_copy(N_phs, 0);
                    if (N_phs.extent(0) != charge.npart()){
                        raise::KernelError(HERE, "N_phs array size does not match the number of particles.");
                    }
                }
            ~Curvature_Emission_Number() = default;

            Inline void operator()(index_t p) const{
                if (tag(p) != ParticleTag::alive){
                    if (tag(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                const real_t pp = math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                auto N_ph = static_cast<size_t>(coeff * CDF(zeta) / SQR(pp));

                N_phs(p) = N_ph;
            }
    };
} // namespace QED


#endif // QED_PROCESS_HPP