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
    
    class cdfTable {
    private:
        array_t<real_t*> cdf;
        array_t<real_t*> inv_cdf;

        real_t x_min_lg;
        real_t x_max_lg;
        real_t y_min_lg;
        real_t y_max_lg;
        real_t dx;
        real_t dy;

        size_t Nx;
        size_t Ny;

        // 改进的文件读取方法
        auto read_from_file(const std::string& filename) -> std::pair<std::vector<real_t>, std::vector<real_t>> {
            std::fstream file;
            file.open(filename, std::ios::in);
            if (!file.is_open()) {
                raise::Fatal("Could not open file.", HERE);
            }
            
            std::vector<real_t> x_data;
            std::vector<real_t> y_data;
            
            while (!file.eof()) {
                real_t x, y;
                if (file >> x >> y) {  // 确保成功读取
                    if(x <= ZERO || y <= ZERO){
                        raise::Fatal("Invalid data value (must be positive).", HERE);
                    }
                    x_data.push_back(x);
                    y_data.push_back(y);
                }
            }
            file.close();
            
            if (x_data.empty()) {
                raise::Fatal("Empty data file.", HERE);
            }
            
            return {x_data, y_data};
        }

    public:
        cdfTable(const std::string& cdf_filename, const std::string& inverse_cdf_filename) {
            // 从文件读取数据，使用std::vector暂存
            auto [x_cdf_data, y_cdf_data] = read_from_file(cdf_filename);
            auto [x_inv_cdf_data, y_inv_cdf_data] = read_from_file(inverse_cdf_filename);
            
            // 设置尺寸
            Nx = x_cdf_data.size();
            Ny = x_inv_cdf_data.size();
            
            // 设置范围参数
            x_min_lg = math::log10(x_cdf_data[0]);
            x_max_lg = math::log10(x_cdf_data[Nx - 1]);
            dx = (x_max_lg - x_min_lg) / (Nx - 1);
            y_min_lg = math::log10(x_inv_cdf_data[0]);
            y_max_lg = math::log10(x_inv_cdf_data[Ny - 1]);
            dy = (y_max_lg - y_min_lg) / (Ny - 1);
            
            // 分配设备内存
            cdf = array_t<real_t*>("cdf", Nx);
            inv_cdf = array_t<real_t*>("inv_cdf", Ny);
            
            // 创建主机镜像
            auto cdf_h = Kokkos::create_mirror_view(cdf);
            auto inv_cdf_h = Kokkos::create_mirror_view(inv_cdf);
            
            // 填充数据
            for (size_t i = 0; i < Nx; ++i) {
                cdf_h(i) = y_cdf_data[i];
            }
            
            for (size_t i = 0; i < Ny; ++i) {
                inv_cdf_h(i) = y_inv_cdf_data[i];
            }
            
            // 复制到设备内存
            Kokkos::deep_copy(cdf, cdf_h);
            Kokkos::deep_copy(inv_cdf, inv_cdf_h);
        }

        // 拷贝构造函数保持不变，因为是浅拷贝
        Inline cdfTable(const cdfTable &rhs) 
            : cdf(rhs.cdf)
            , inv_cdf(rhs.inv_cdf)
            , x_min_lg(rhs.x_min_lg)
            , x_max_lg(rhs.x_max_lg)
            , y_min_lg(rhs.y_min_lg)
            , y_max_lg(rhs.y_max_lg)
            , dx(rhs.dx)
            , dy(rhs.dy)
            , Nx(rhs.Nx)
            , Ny(rhs.Ny) {}

        ~cdfTable() = default;

        // 保持原始的CDF超出范围处理逻辑
        Inline auto CDF(const real_t x) const -> real_t {
            if (x <= ZERO) {
                raise::KernelError(HERE, "Invalid argument for CDF.");
            }
            const int idx = static_cast<int>((math::log10(x) - x_min_lg) / dx);
            if (idx < 0) {
                return ONE + 0.346 * x - math::pow(x, ONE / THREE) * (1.232 + 0.033 * SQR(x));
            }
            auto i = static_cast<size_t>(idx);
            if (i >= Nx - 1) {
                return cdf(Nx - 1);
            }
            
            const real_t t = (x - math::pow(10.0, i * dx + x_min_lg))
                             / (math::pow(10.0, (i + 1) * dx + x_min_lg) - math::pow(10.0, i * dx + x_min_lg));
            
            // 插值公式
            return cdf(i) * (ONE - t) + cdf(i + 1) * t;
        }

        // 保持原始的Inverse_CDF超出范围处理逻辑
        Inline auto Inverse_CDF(const real_t y) const -> real_t {
            if (y <= ZERO || y >= ONE) {
                raise::KernelError(HERE, "Invalid argument for Inverse_CDF.");
            }
            const int idx = static_cast<int>((math::log10(y) - y_min_lg) / dy);
            if (idx < 0) {
                return math::pow(10.0, x_max_lg);
            }
            auto i = static_cast<size_t>(idx);
            if (i >= Ny - 1) {
                return CUBE((ONE - y) / 1.232);  
            }
            
            const real_t t = (y - math::pow(10.0, i * dy + y_min_lg))
                             / (math::pow(10.0, (i + 1) * dy + y_min_lg) - math::pow(10.0, i * dy + y_min_lg));

            // 原始插值公式
            return inv_cdf(i) * (ONE - t) + inv_cdf(i + 1) * t;
        }
    }; // class cdfTable


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

        const real_t                  e_min; // minimum ejection energy of the photon 
        const real_t                  gamma_emit; // energy scale of charges
        const real_t                  coeff;
        const real_t                  rho; //curvature radius of magnetic field
        const size_t                  N_max; // maximum number of photons to be injected
        const size_t                  npart_ph;
        random_number_pool_t          random_pool;
        const cdfTable                cdf;

        array_t<size_t>               n_inj { "n_inj" };


        public:
            CurvatureEmission_kernel(Particles<D, C>&          charges,
                                            Particles<D, C>&          photons,
                                            const real_t              e_min_,
                                            const real_t              gamma_emit_,
                                            const real_t              coeff_,
                                            const real_t              rho_,
                                            const size_t              N_max,
                                            random_number_pool_t&     random_pool_,
                                            const cdfTable&           cdf_)
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
                , coeff { coeff_ }
                , rho { rho_ }
                , N_max { N_max }
                , npart_ph { photons.npart() }
                , random_pool { random_pool_ }
                , cdf { cdf_ } { 
                    Kokkos::deep_copy(n_inj, 0);
                }
            ~CurvatureEmission_kernel() = default;

           
            auto num_injected() const -> size_t{
                auto n_inj_h = Kokkos::create_mirror_view(n_inj);
                Kokkos::deep_copy(n_inj_h, n_inj);
                return n_inj_h();
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
                auto N_ph = static_cast<size_t>(coeff * cdf.CDF(zeta) / SQR(pp));

                if (N_ph < 1){
                    return;
                }   
                // weight correction
                auto w_crect = ONE;
                if (N_ph > N_max){
                    w_crect *= N_ph / N_max;
                    N_ph = N_max;
                }
                
                auto offset = Kokkos::atomic_fetch_add(&n_inj(), N_ph);
                offset += npart_ph;
                if (offset + N_ph - 1 > ux1_ph.extent(0)){
                    raise::KernelError(HERE, "Number of photons exceeds maxnpart.");
                }

                for (short i = 0; i < N_ph; ++i){
                    auto rand_gen = random_pool.get_state();

                    ux1_ph(offset + i) = SIGN(ux1(p)) * ONE;
                    pld_ph(offset + i, 0) = cdf.Inverse_CDF(cdf.CDF(zeta) * Random<real_t>(rand_gen))
                                             * CUBE(pp / gamma_emit) / rho;
                    pld_ph(offset + i, 1) = ZERO;
                    i1_ph(offset + i) = i1(p);
                    dx1_ph(offset + i) = dx1(p);
                    weight_ph(offset + i) = w_crect * weight(p);
                    tag_ph(offset + i) = ParticleTag::alive;

                    random_pool.free_state(rand_gen);
                }
            }
    }; // class CurvatureEmission_kernel

    template <Dimension D, Coord::type C>
    struct PayloadUpdate {
      const real_t                        coeff1, coeff2, dt, rho;
      const array_t<real_t**>             pld_ph;
      const array_t<real_t*>              ux1_ph;
      const array_t<int*>                 i1_ph;
      const array_t<short*>               tag_ph;
      array_t<prtldx_t*>                  dx1_ph;

      const size_t                        n_steps;



    Inline auto integrate(const index_t p, const size_t n_steps) const -> real_t{
        const real_t dx { ux1_ph(p) * dt / n_steps };

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
        PayloadUpdate(Particles<D, C>& photon,
                      const real_t coeff1_,
                      const real_t coeff2_,
                      const real_t dt_,
                      const real_t rho_,
                      const size_t n_steps_)
          : coeff1 { coeff1_ }
          , coeff2 { coeff2_ }
          , dt { dt_ }
          , rho { rho_ }
          , pld_ph { photon.pld }
          , ux1_ph { photon.ux1 }
          , i1_ph { photon.i1 }
          , tag_ph { photon.tag }
          , dx1_ph { photon.dx1 }
          , n_steps { n_steps_ } {}    
        ~PayloadUpdate() = default;

        Inline void operator()(index_t p) const{
          if (tag_ph(p) != ParticleTag::alive){
            if (tag_ph(p) != ParticleTag::dead){
              raise::KernelError(HERE, "Invalid particle tag in pusher");
            }
            return;
          }
          pld_ph(p, 1) += coeff1 * integrate(p, n_steps);
          pld_ph(p, 2) += dt * ux1_ph(p) / rho;
        }
    };// class PayloadUpdate

    template <Dimension D, Coord::type C>
    class PairCreation_kernel{
        static_assert(D == Dim::_1D, "Pair creation is only implemented in 1D");
        static_assert(C == Coord::Cart, "Pair creation is only implemented in cartesian coordinates"); 

        const array_t<real_t*>              ux1_ph, ux2_ph, ux3_ph;
        const array_t<real_t**>             pld_ph;
        const array_t<real_t*>              weight_ph;
        const array_t<short*>               tag_ph;
        const array_t<int*>                 i1_ph;
        const array_t<prtldx_t*>            dx1_ph;

        array_t<real_t*>                   ux1_p, ux2_p, ux3_p;
        array_t<real_t*>                   weight_p;
        array_t<short*>                    tag_p;
        array_t<int*>                      i1_p; 
        array_t<prtldx_t*>                 dx1_p;
        const size_t                       npart_p;

        array_t<real_t*>                   ux1_e, ux2_e, ux3_e;
        array_t<real_t*>                   weight_e;
        array_t<short*>                    tag_e;
        array_t<int*>                      i1_e;
        array_t<prtldx_t*>                 dx1_e;
        const size_t                       npart_e;

        array_t<size_t>               n_inj { "n_inj" };

        public:
            PairCreation_kernel(Particles<D, C>&          photons,
                                Particles<D, C>&          positrons,
                                Particles<D, C>&          electrons)
                : ux1_ph { photons.ux1 }
                , ux2_ph { photons.ux2 }
                , ux3_ph { photons.ux3 }
                , pld_ph { photons.pld }
                , weight_ph { photons.weight }
                , tag_ph { photons.tag }
                , i1_ph { photons.i1 }
                , dx1_ph { photons.dx1 }
                , ux1_p { positrons.ux1 }
                , ux2_p { positrons.ux2 }
                , ux3_p { positrons.ux3 }
                , i1_p { positrons.i1 }
                , dx1_p { positrons.dx1 }
                , tag_p { positrons.tag }
                , weight_p { positrons.weight }
                , npart_p { positrons.npart() }
                , ux1_e { electrons.ux1 }
                , ux2_e { electrons.ux2 }
                , ux3_e { electrons.ux3 }
                , i1_e { electrons.i1 }
                , dx1_e { electrons.dx1 }
                , tag_e { electrons.tag }
                , weight_e { electrons.weight }
                , npart_e { electrons.npart() }
                , coeff1 { coeff1_ }
                , coeff2 { coeff2_ }
                , rho0 { rho0_ }
                , L { L_ }
                , dt { dt_ }
                , n_steps { n_steps_ }{
                    Kokkos::deep_copy(n_inj, 0);
                }
            ~PairCreation_kernel() = default;

            auto num_injected() const -> size_t{
                auto n_inj_h = Kokkos::create_mirror_view(n_inj);
                Kokkos::deep_copy(n_inj_h, n_inj);
                return n_inj_h();
            }
            

            Inline void operator()(index_t p) const{
                if (tag_ph(p) != ParticleTag::alive){
                    if (tag_ph(p) != ParticleTag::dead){
                        raise::KernelError(HERE, "Invalid particle tag in pusher");
                    }
                    return;
                }
                if (pld_ph(p, 0) * math::sin(pld_ph(p, 2)) < TWO){
                    tag_ph(p) = ParticleTag::dead;
                    return;
                }
                auto offset_p = Kokkos::atomic_fetch_add(&n_inj(), 1);
                auto offset_e = Kokkos::atomic_fetch_add(&n_inj(), 1);

                offset_p += npart_p;
                offset_e += npart_e;

                if (pld_ph(p, 1) > ONE){
                    const real_t u = math::sqrt((SQR(pld_ph(p, 0)) - FOUR) / (SQR(pld_ph(p, 0) * pld_ph(p, 2)) + FOUR));

                    ux1_p(offset_p) = u;
                    ux2_p(offset_p) = ZERO;
                    ux3_p(offset_p) = ZERO;
                    i1_p(offset_p) = i1_ph(p);
                    dx1_p(offset_p) = dx1_ph(p);
                    weight_p(offset_p) = weight_ph(p);
                    tag_p(offset_p) = ParticleTag::alive;
    
                    ux1_e(offset_e) = u;
                    ux2_e(offset_e) = ZERO;
                    ux3_e(offset_e) = ZERO;
                    i1_e(offset_e) = i1_ph(p);
                    dx1_e(offset_e) = dx1_ph(p);
                    weight_e(offset_e) = weight_ph(p);
                    tag_e(offset_e) = ParticleTag::alive;
    
                    tag_ph(p) = ParticleTag::dead;
                } 
            }

    }; // class PairCreation_kernel

} // namespace QED


#endif // QED_PROCESS_HPP

