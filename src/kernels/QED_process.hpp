#ifndef QED_PROCESS_HPP
#define QED_PROCESS_HPP

#include "global.h"
#include "enums.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"


namespace kernel::QED{
    using namespace ntt;

    class CurvatureEmission_kernel{
        array_t<real_t*>              ux1, ux2, ux3;
        array_t<int*>                 i1;
        array_t<prtldx_t*>            dx1;
        array_t<real_t*>              weight;
        array_t<short*>               tag;
        array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
        arrat_t<int*>                 i1_ph;
        array_t<prtldx_t*>            dx1_ph;
        array_t<real_t*>              weight_ph;
        array_t<short*>               tag_ph;
        const real_t                  p_min;
        const real_t                  e_min;
        const real_t                  coeff_cdf;
        const short                   Nmax_ph;
        const short                   Nbin;
        const short                   res_cdf;
        const real_t                  dzeta;
        array_t<real_t*>              cdf_table;
        const real_t                  gamma_emit;
        const real_t                  gamma_rad;
        const real_t                  gamma_pc;
        const real_t                  rho;
        random_number_pool_t          random_pool;


        public:
            CurvatureEmission_kernel() = default;
            ~CurvatureEmission_kernel() = default;

            Inline auto CDF(real_t zeta_) const -> real_t{
                if (zeta_ < 0.01){
                    return ONE + 0.346 * zeta_ - (1.232 + 0.033 * SQR(zeta_)) * math::pow(zeta_, ONE / THREE);
                }else if (zeta_ < 10.0){
                    auto i = static_cast<int>((zeta_ - 0.01) / d_zeta);
                    return cdf_table(i) + (zeta_ - (0.01 + i * d_zeta)) * (cdf_table(i + 1) - cdf_table(i)) / d_zeta;
                }else{
                    return math:exp(-zeta_) * (1.25 + (-0.625 + 0.9375 / zeta_) / zeta_) / math::sqrt(zeta_);
                }
            }

            Inline auto rand_uniform(short mean) const -> short{
                auto rand_gen = random_pool.get_state();
                return static_cast<short>(mean * TWO * Random<real_t>(rand_gen));
            }

            Inline void sample_photon(short                 N, 
                                      real_t                zeta_,
                                      array_t<real_t*>      ux1_ph_,
                                      array_t<real_t*>      ux2_ph_,
                                      array_t<real_t*>      ux3_ph_,
                                      array_t<int*>         i1_ph_,
                                      array_t<prtldx_t*>    dx1_ph_,
                                      array_t<real_t*>      weight_ph_,
                                      array_t<short*>       tag_ph_) const;

            void operator()(index_t p) const{
                if(tag(p) != ParticleTag::alive){
                    if(tag(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                const real_t pp = math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
                //if particle energy is less than p_min, no emission
                if(pp < p_min){
                    return;
                }
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                auto N_ph = static_cast<short>(coeff_cdf * CDF(zeta) / SQR(pp));
                //if N_ph is less than Nmax_ph, generate ramdon N_ph from uniform distribution
                if (N_ph < Nmax_ph){
                    N_ph = rand_uniform(N_ph);
                }else{
                //if N_ph is greater than Nmax_ph, N_ph is set to Nbin
                    N_ph = Nbin;
                }
                //inject photons
                sample_photon(N_ph, zeta, ux1_ph, ux2_ph, ux3_ph, i1_ph, dx1_ph, weight_ph, tag_ph);
            }
    };
} // namespace QED


#endif // QED_PROCESS_HPP