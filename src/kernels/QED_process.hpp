#ifndef QED_PROCESS_HPP
#define QED_PROCESS_HPP

#include "global.h"
#include "enums.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>

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
        const array_t<size_t*>        offsets;

        const real_t                  e_min;
        const real_t                  gamma_emit;
        const real_t                  rho;
        const size_t                  N_max;
        random_number_pool_t          random_pool;

        const cdfTable&               cdf_table;



        public:
            CurvatureEmission_kernel(Particles<D, C>&          charges,
                                     Particles<D, C>&          photons,
                                     const real_t              e_min_,
                                     const real_t              gamma_emit_,
                                     const real_t              rho_,
                                     const size_t              N_max,
                                     const array_t<size_t*>&   N_phs_,
                                     const array_t<size_t*>&      offsets_,
                                     random_number_pool_t&     random_pool_,
                                     const cdfTable&           cdf_table_)
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
                , random_pool { random_pool_ } 
                , cdf_table { cdf_table_ } {
                    if (N_phs.extent(0) != charges.npart()){
                        raise::KernelError(HERE, "N_phs array size does not match the number of particles.");
                    }
                    if (offsets.extent(0) != charges.npart()){
                        raise::KernelError(HERE, "offsets array size does not match the number of particles.");
                    }
                }
            ~CurvatureEmission_kernel() = default;

            Inline void operator()(index_t p) const{
                if(tag(p) != ParticleTag::alive){
                    if(tag(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                if (N_phs(p) == 0){
                    return;
                }
                // weight correction
                auto w_crect = ONE;
                auto N_ph = N_phs(p);
                if (N_ph > N_max){
                    w_crect *= N_ph / N_max;
                    N_ph = N_max;
                }
                
                const real_t pp = math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                for (short i = 0; i < N_ph; ++i){
                    auto rand_gen = random_pool.get_state();

                    ux1_ph(offsets(p) + i) = SIGN(ux1(p)) * ONE;
                    pld_ph(offsets(p) + i, 0) = cdf_table.Inverse_CDF(cdf_table.CDF(zeta) * Random<real_t>(rand_gen))
                                             * CUBE(pp / gamma_emit) / rho;
                    pld_ph(offsets(p) + i, 1) = ZERO;
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
        const real_t                  gamma_min;
        const real_t                  rho;
        const cdfTable&               cdf_table;


        public:
            Curvature_Emission_Number(const Particles<D, C>& charge,
                                       const real_t e_min_,
                                       const real_t coeff_,
                                       const real_t gamma_emit_,
                                       const real_t gamma_min_,
                                       const real_t rho_,
                                       array_t<size_t*>& N_phs_,
                                       const cdfTable& cdf_table_)
                : ux1 { charge.ux1 }
                , ux2 { charge.ux2 }
                , ux3 { charge.ux3 }
                , tag { charge.tag }
                , e_min { e_min_ }
                , coeff { coeff_ }
                , gamma_emit { gamma_emit_ }
                , gamma_min { gamma_min_ }
                , rho { rho_ }
                , N_phs { N_phs_ } 
                , cdf_table { cdf_table_ } {
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
                if (pp < gamma_min){
                    N_phs(p) = 0;
                    return;
                }
                const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
                auto N_ph = static_cast<size_t>(coeff * cdf_table.CDF(zeta) / SQR(pp));
                
                N_phs(p) = N_ph;
            }
    };




    template <Dimension D, Coord::type C>
    class PairCreation_kernel{
        static_assert(D == Dim::_1D, "Pair creation is only implemented in 1D");
        static_assert(C == Coord::Cart, "Pair creation is only implemented in cartesian coordinates"); 

        const array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
        const array_t<real_t**>             pld_ph;
        const array_t<short*>               tag_ph;
        const array_t<int*>                 i1_ph;
        const array_t<prtldx_t*>            dx1_ph;

        array_t<bool*>                       should_inj;

        const real_t                              coeff1;
        const real_t                              coeff2;
        const real_t                              rho0;
        const real_t                              L;

        const real_t                              dt;
        const size_t                              n_steps;
        
        Inline auto Rho(const real_t x) const -> real_t{
            return rho0 * (ONE + 0.8 * x / L);
        }
    

        Inline auto integrate(const index_t p, const size_t n_steps) const -> real_t{
            const real_t dx { ux1_ph(p) * dt / n_steps };

            const real_t rho { Rho(static_cast<real_t>(i1_ph(p)) + static_cast<real_t>(dx1_ph(p))) };

            real_t sum { pld_ph(p, 2) * exp(-coeff2 / (pld_ph(p, 2) * pld_ph(p, 0)))
                         + (pld_ph(p, 2) + ux1_ph(p) * dt / rho) * exp(-coeff2 / ((pld_ph(p, 2) + ux1_ph(p) * dt / rho) * pld_ph(p, 0))) };
            
            for (size_t i = 1; i < n_steps; i += 2){
                sum += FOUR * (pld_ph(p, 2) + i * dx / rho) * exp(-coeff2 / ((pld_ph(p, 2)+ i * dx / rho) * pld_ph(p, 0)));
            }

            for (size_t i = 2; i < n_steps; i += 2){
                sum += TWO * (pld_ph(p, 2) + i * dx / rho) * exp(-coeff2 / ((pld_ph(p, 2) + i * dx / rho) * pld_ph(p, 0)));
            }

            return dx / THREE * sum;

        }

        public:
            PairCreation_kernel(Particles<D, C>&          photons,
                                const real_t              coeff1_,
                                const real_t              coeff2_,
                                const real_t              rho0_,
                                const real_t              L_,
                                const real_t              dt_,
                                const size_t              n_steps_,
                                array_t<bool*>&            should_inj_)
                : ux1_ph { photons.ux1 }
                , ux2_ph { photons.ux2 }
                , ux3_ph { photons.ux3 }
                , pld_ph { photons.pld }
                , tag_ph { photons.tag }
                , i1_ph { photons.i1 }
                , dx1_ph { photons.dx1 }
                , coeff1 { coeff1_ }
                , coeff2 { coeff2_ }
                , rho0 { rho0_ }
                , L { L_ }
                , dt { dt_ }
                , n_steps { n_steps_ }
                , should_inj { should_inj_ } {}
            ~PairCreation_kernel() = default;

            Inline void operator()(index_t p) const{
                if (tag_ph(p) != ParticleTag::alive){
                    if (tag_ph(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                pld_ph(p, 1) += coeff1 * integrate(p, n_steps);
                pld_ph(p, 2) += dt * ux1_ph(p) / Rho(static_cast<real_t>(i1_ph(p)) + static_cast<real_t>(dx1_ph(p)));
                if (pld_ph(p, 0) * math::sin(pld_ph(p, 1)) < TWO){
                    tag_ph(p) = ParticleTag::dead;
                    return;
                }
                if (pld_ph(p, 1) > ONE){
                    should_inj(p) = true;
                } 
            }

    }; // class PairCreation_kernel

    template <Dimension D, Coord::type C>
    class InjectPairs_kernel{
        static_assert(D == Dim::_1D, "Pair creation is only implemented in 1D");
        static_assert(C == Coord::Cart, "Pair creation is only implemented in cartesian coordinates"); 
        
        const array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
        const array_t<real_t**>             pld_ph;
        const array_t<int*>                 i1_ph;
        const array_t<prtldx_t*>            dx1_ph;
        const array_t<short*>               tag_ph;
        const array_t<real_t*>              weight_ph;

        array_t<real_t*>                   ux1_p, ux2_p, ux3_p;
        array_t<real_t*>                   weight_p;
        array_t<short*>                    tag_p;
        array_t<int*>                      i1_p;
        array_t<prtldx_t*>                 dx1_p;

        array_t<real_t*>                   ux1_e, ux2_e, ux3_e;
        array_t<real_t*>                   weight_e;
        array_t<short*>                    tag_e;
        array_t<int*>                      i1_e;
        array_t<prtldx_t*>                 dx1_e;

        array_t<bool*>                    should_inj;
        array_t<size_t*>                  offsets_e;
        array_t<size_t*>                  offsets_p;

        public:

            InjectPairs_kernel(Particles<D, C>&          photon,
                               Particles<D, C>&          electron,
                               Particles<D, C>&          positron,
                               array_t<bool*>&           should_inj_,
                               array_t<size_t*>&         offsets_e_,
                               array_t<size_t*>&         offsets_p_)
                : ux1_ph { photon.ux1 }
                , ux2_ph { photon.ux2 }
                , ux3_ph { photon.ux3 }
                , pld_ph { photon.pld }
                , i1_ph { photon.i1 }
                , dx1_ph { photon.dx1 }
                , tag_ph { photon.tag }
                , weight_ph { photon.weight }
                , ux1_p { positron.ux1 }
                , ux2_p { positron.ux2 }
                , ux3_p { positron.ux3 }
                , i1_p { positron.i1 }
                , dx1_p { positron.dx1 }
                , tag_p { positron.tag }
                , weight_p { positron.weight }
                , ux1_e { electron.ux1 }
                , ux2_e { electron.ux2 }
                , ux3_e { electron.ux3 }
                , i1_e { electron.i1 }
                , dx1_e { electron.dx1 }
                , tag_e { electron.tag }
                , weight_e { electron.weight }
                , should_inj { should_inj_ }
                , offsets_e { offsets_e_ }
                , offsets_p { offsets_p_ } {}

            ~InjectPairs_kernel() = default;

            Inline void operator()(index_t p) const{
                if (!should_inj(p)){
                    return;
                }
                const real_t u = math::sqrt((SQR(pld_ph(p, 0)) - FOUR) / (SQR(pld_ph(p, 0) * pld_ph(p, 2)) + FOUR));

                ux1_p(offsets_p(p)) = u;
                ux2_p(offsets_p(p)) = ZERO;
                ux3_p(offsets_p(p)) = ZERO;
                i1_p(offsets_p(p)) = i1_ph(p);
                dx1_p(offsets_p(p)) = dx1_ph(p);
                weight_p(offsets_p(p)) = weight_ph(p);
                tag_p(offsets_p(p)) = ParticleTag::alive;

                ux1_e(offsets_e(p)) = u;
                ux2_e(offsets_e(p)) = ZERO;
                ux3_e(offsets_e(p)) = ZERO;
                i1_e(offsets_e(p)) = i1_ph(p);
                dx1_e(offsets_e(p)) = dx1_ph(p);
                weight_e(offsets_e(p)) = weight_ph(p);
                tag_e(offsets_e(p)) = ParticleTag::alive;

                tag_ph(p) = ParticleTag::dead;
            }
                
    }; // class InjectPairs_kernel

    class cdfTable{
        array_t<real_t*> cdf;
        array_t<real_t*> inv_cdf;

        real_t x_min;
        real_t x_max;
        real_t y_min;
        real_t y_max;
        real_t dx;
        real_t dy;

        size_t Nx;
        size_t Ny;

        

        auto read_from_file(const std::string& filename) -> array_t<real_t**>{
            std::fstream file;
            file.open(filename, std::ios::in);
            if (!file.is_open()){
                raise::Fatal("Could not open file.", HERE);
            }
            std::vector<real_t> x_data;
            std::vector<real_t> y_data;
            while (!file.eof()){
                real_t x, y;
                file >> x >> y;
                x_data.push_back(x);
                y_data.push_back(y);
            }
            file.close();
            auto N = x_data.size();
            auto data = array_t<real_t**>("data", N);
            auto data_h = Kokkos::create_mirror_view(data);
            for (size_t i = 0; i < N; ++i){
                data_h(i, 0) = x_data[i];
                data_h(i, 1) = y_data[i];
            }
            Kokkos::deep_copy(data, data_h);
            return data;
        }

        public:
            cdfTable(const std::string& cdf_filename, const std::string& inverse_cdf_filename)
                 {
                    auto cdf_data = read_from_file(cdf_filename);
                    auto inv_cdf_data = read_from_file(inverse_cdf_filename);

                    Nx = cdf_data.extent(0);
                    Ny = inv_cdf_data.extent(0);

                    x_min = cdf_data(0, 0);
                    x_max = cdf_data(Nx - 1, 0);
                    dx = (math::log10(x_max) - math::log10(x_min)) / (Nx - 1);
                    y_min = inv_cdf_data(0, 0);
                    y_max = inv_cdf_data(Ny - 1, 0);
                    dy = (y_max - y_min) / (Ny - 1);

                    cdf = array_t<real_t*>("cdf", Nx);
                    inv_cdf = array_t<real_t*>("inv_cdf", Ny);
                    
                    auto cdf_subview = Kokkos::subview(cdf_data, Kokkos::ALL, 1);
                    auto inv_cdf_subview = Kokkos::subview(inv_cdf_data, Kokkos::ALL, 1);
                    Kokkos::deep_copy(cdf, cdf_subview);
                    Kokkos::deep_copy(inv_cdf, inv_cdf_subview);
                }

            Inline cdfTable(const cdfTable &rhs) 
                            : cdf(rhs.cdf)
                            , inv_cdf(rhs.inv_cdf)
                            , x_min(rhs.x_min)
                            , x_max(rhs.x_max)
                            , y_min(rhs.y_min)
                            , y_max(rhs.y_max)
                            , dx(rhs.dx)
                            , dy(rhs.dy)
                            , Nx(rhs.Nx)
                            , Ny(rhs.Ny) {}

            ~cdfTable() = default;

            Inline auto CDF(const real_t x) const -> real_t{
                if (x < x_min){
                    return ONE + 0.346 * x - math::pow(x, 1/3) * (1.232 + 0.033 * SQR(x));
                }
                if (x > x_max){
                    return cdf(Nx - 1) * math::exp(x - x_max);
                }
                const size_t i = static_cast<size_t>((math::log10(x) - math::log10(x_min)) / dx);
                return cdf(i) + (x - x_min - i * dx) / dx * (cdf(i + 1) - cdf(i));
            }

            Inline auto Inverse_CDF(const real_t y) const -> real_t{
                if (y < y_min){
                    return x_max + math::log(y_min / y);
                }
                if (y > y_max){
                    return CUBE((ONE - y) / 1.232);
                }
                const size_t i = static_cast<size_t>((y - y_min) / dy);
                return inverse_cdf(i) + (y - y_min - i * dy) / dy * (inverse_cdf(i + 1) - inverse_cdf(i));
            }

    }
} // namespace QED


#endif // QED_PROCESS_HPP

    // template <Dimension D, Coord::type C>
    // class AtomicCurvatureEmission_kernel{
    //     static_assert(D == Dim::_1D, "Curvature emission is only implemented in 1D");
    //     static_assert(C == Coord::Cart, "Curvature emission is only implemented in cartesian coordinates");

    //     const array_t<real_t*>        ux1, ux2, ux3;
    //     const array_t<real_t*>        weight;
    //     const array_t<short*>         tag;
    //     const array_t<int*>           i1;
    //     const array_t<prtldx_t*>      dx1;
        
    //     array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
    //     array_t<real_t*>              weight_ph;
    //     array_t<short*>               tag_ph;
    //     array_t<real_t**>             pld_ph;
    //     array_t<int*>                 i1_ph;
    //     array_t<prtldx_t*>            dx1_ph;

    //     const real_t                  e_min;
    //     const real_t                  gamma_emit;
    //     const real_t                  coeff;
    //     const real_t                  rho;
    //     const size_t                  N_max;
    //     const size_t                  npart_ph;
    //     random_number_pool_t          random_pool;

    //     array_t<size_t>               n_inj { "n_inj" };


    //     public:
    //         AtomicCurvatureEmission_kernel(Particles<D, C>&          charges,
    //                                         Particles<D, C>&          photons,
    //                                         const real_t              e_min_,
    //                                         const real_t              gamma_emit_,
    //                                         const real_t              coeff_,
    //                                         const real_t              rho_,
    //                                         const size_t              N_max,
    //                                         const size_t              npart_ph_,
    //                                         random_number_pool_t&     random_pool_)
    //             : ux1 { charges.ux1 }
    //             , ux2 { charges.ux2 }
    //             , ux3 { charges.ux3 }
    //             , weight { charges.weight }
    //             , tag { charges.tag }
    //             , i1 { charges.i1 }
    //             , dx1 { charges.dx1 }
    //             , ux1_ph { photons.ux1 }
    //             , ux2_ph { photons.ux2 }
    //             , ux3_ph { photons.ux3 }
    //             , weight_ph { photons.weight }
    //             , tag_ph { photons.tag }
    //             , pld_ph { photons.pld }
    //             , i1_ph { photons.i1 }
    //             , dx1_ph { photons.dx1 }
    //             , e_min { e_min_ }
    //             , gamma_emit { gamma_emit_ }
    //             , coeff { coeff_ }
    //             , rho { rho_ }
    //             , N_max { N_max }
    //             , npart_ph { npart_ph_ }
    //             , random_pool { random_pool_ } { 
    //                 Kokkos::deep_copy(n_inj, 0);
    //             }
    //         ~AtomicCurvatureEmission_kernel() = default;

    //         Inline auto CDF(real_t zeta_) const -> real_t{
    //             return math::exp(-zeta_); 
    //         }

    //         Inline auto inverseCDF(real_t u) const -> real_t{
    //             return -math::log(u);
    //         }
           
    //         auto num_injected() const -> size_t{
    //             auto n_inj_h = Kokkos::create_mirror_view(n_inj);
    //             Kokkos::deep_copy(n_inj_h, n_inj);
    //             return n_inj_h();
    //         }

    //         Inline void operator()(index_t p) const{
    //             if(tag(p) != ParticleTag::alive){
    //                 if(tag(p) != ParticleTag::dead){
    //                     raise::KernelError(HERE, "Invalid particle tag in pusher");
    //                 }
    //                 return;
    //             }
    //             const real_t pp = math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));
    //             const real_t zeta = e_min * rho * CUBE(gamma_emit / pp);
    //             auto N_ph = static_cast<size_t>(coeff * CDF(zeta) / SQR(pp));

    //             if (N_ph < 1){
    //                 return;
    //             }   
    //             // weight correction
    //             auto w_crect = ONE;
    //             if (N_ph > N_max){
    //                 w_crect *= N_ph / N_max;
    //                 N_ph = N_max;
    //             }
                
    //             auto offset = Kokkos::atomic_fetch_add(&n_inj(), N_ph);
    //             offset += npart_ph;
    //             if (offset + N_ph - 1 > ux1_ph.extent(0)){
    //                 raise::KernelError(HERE, "Number of photons exceeds maxnpart.");
    //             }

    //             for (short i = 0; i < N_ph; ++i){
    //                 auto rand_gen = random_pool.get_state();

    //                 ux1_ph(offset + i) = SIGN(ux1(p)) * ONE;
    //                 pld_ph(offset + i, 0) = inverseCDF(CDF(zeta) * Random<real_t>(rand_gen))
    //                                          * CUBE(pp / gamma_emit) / rho;
    //                 pld_ph(offset + i, 1) = ZERO;
    //                 i1_ph(offset + i) = i1(p);
    //                 dx1_ph(offset + i) = dx1(p);
    //                 weight_ph(offset + i) = w_crect * weight(p);
    //                 tag_ph(offset + i) = ParticleTag::alive;

    //                 random_pool.free_state(rand_gen);
    //             }
    //         }
    // };