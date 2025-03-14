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

        const real_t                  e_min;
        const real_t                  coeff;
        const real_t                  gamma_emit;
        const real_t                  rho;
        const size_t                  N_max;
        random_number_pool_t          random_pool;

        Kokkos::View<size_t>          num_ph { "num_ph" };


        public:
            // CurvatureEmission_kernel(const array_t<real_t*>        ux1_, 
            //                          const array_t<real_t*>        ux2_,
            //                          const array_t<real_t*>        ux3_,
            //                          const array_t<real_t*>        weight_,  
            //                          const array_t<short*>         tag_,
            //                          const array_t<int*>           i1_,
            //                          const array_t<prtldx_t*>      dx1_,
            //                          array_t<real_t*>              ux1_ph_,
            //                          array_t<real_t*>              ux2_ph_,
            //                          array_t<real_t*>              ux3_ph_,
            //                          array_t<real_t*>              weight_ph_,
            //                          array_t<short*>               tag_ph_,
            //                          array_t<real_t**>             pld_ph_,
            //                          array_t<int*>                 i1_ph_,
            //                          array_t<prtldx_t*>            dx1_ph_,
            //                          const real_t                  e_min_,
            //                          const real_t                  coeff_,
            //                          const real_t                  gamma_emit_,
            //                          const real_t                  rho_,
            //                          const real_t                  npart_ph,
            //                          const size_t                  N_max,
            //                          random_number_pool_t&         random_pool_)
            CurvatureEmission_kernel(Particles<D, C>&    charges,
                                     Particles<D, C>&    photons,
                                     const real_t        e_min_,
                                     const real_t        coeff_,
                                     const real_t        gamma_emit_,
                                     const real_t        rho_,
                                     const size_t        N_max,
                                     const size_t        npart_ph,
                                     random_number_pool_t& random_pool_)
                : ux1 { charges.ux1_ }
                , ux2 { charges.ux2_ }
                , ux3 { charges.ux3_ }
                , weight { charges.weight_ }
                , tag { charges.tag_ }
                , i1 { charges.i1_ }
                , dx1 { charges.dx1_ }
                , ux1_ph { photons.ux1_ph_ }
                , ux2_ph { photons.ux2_ph_ }
                , ux3_ph { photons.ux3_ph_ }
                , weight_ph { photons.weight_ph_ }
                , tag_ph { photons.tag_ph_ }
                , pld_ph { photons.pld_ph_ }
                , i1_ph { photons.i1_ph_ }
                , dx1_ph { photons.dx1_ph_ }
                , e_min { e_min_ }
                , coeff { coeff_ }
                , gamma_emit { gamma_emit_ }
                , rho { rho_ }
                , N_max { N_max }
                , npart_ph { photons.npart() }
                , random_pool { random_pool_ } {
                    Kokkos::deep_copy(num_ph, 0);
                }
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
                const real_t pp = math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                auto N_ph = static_cast<size_t>(coeff * CDF(zeta) / SQR(pp));
                //if N_ph is less than 1, no emission.
                if (N_ph < 1){
                    return;
                }
                // weight correction
                auto w_crect = ONE;
                if (N_ph > N_max){
                    N_ph = N_max;
                    w_crect *= N_ph / N_max;
                }
                 
                size_t offset = Kokkos::atomic_fetch_add(&num_ph(), N_ph);
                offset += npart_ph;

                if (offset + N_ph > ux1_ph.extent(0)){
                    raise::KernelError(HERE, "Exceeded maximum number of photons.");
                }
            
                for (short i = 0; i < N_ph; ++i){
                    auto rand_gen = random_pool.get_state();

                    ux1_ph(offset + i) = SIGN(ux1(p)) * ONE;
                    pld_ph(offset + i, 0) = inverseCDF(CDF(zeta) * Random<real_t>(rand_gen))
                                             * CUBE(pp / gamma_emit) / rho;
                    i1_ph(offset + i) = i1(p);
                    dx1_ph(offset + i) = dx1(p);
                    weight_ph(offset + i) = w_crect * weight(p);
                    tag_ph(offset + i) = ParticleTag::alive;

                    random_pool.free_state(rand_gen);
                }
            }
    };
} // namespace QED


#endif // QED_PROCESS_HPP