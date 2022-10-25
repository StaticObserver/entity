#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "input.h"
#include "pgen.h"
#include "sim_params.h"
#include "meshblock.h"
#include "field_macros.h"

namespace ntt {

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) {
      spinup_time = readFromInput<real_t>(params.inputdata(), "problem", "spinup_time");
      omega_max   = readFromInput<real_t>(params.inputdata(), "problem", "omega_max");
    }

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&);
    inline void UserBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&);

  private:
    real_t spinup_time;
    real_t omega_max;
  };

  Inline auto UserTargetField_br_hat(const Meshblock<Dim2, TypePIC>& mblock,
                                     const coord_t<Dim2>&            x) -> real_t {
    coord_t<Dim2> rth_;
    rth_[0] = ZERO;
    real_t r_min {mblock.metric.x1_min};
    mblock.metric.x_Code2Sph(x, rth_);
    return ONE * SQR(r_min / rth_[0]);
  }

  Inline void monopoleField(const coord_t<Dim2>& x_ph,
                            vec_t<Dim3>&         e_out,
                            vec_t<Dim3>&         b_out,
                            real_t               rmin) {
    b_out[0] = SQR(rmin / x_ph[0]);
  }

  Inline void surfaceRotationField(const coord_t<Dim2>& x_ph,
                                   vec_t<Dim3>&         e_out,
                                   vec_t<Dim3>&         b_out,
                                   real_t               rmin,
                                   real_t               omega) {
    b_out[0] = SQR(rmin / x_ph[0]);
    e_out[1] = omega * math::sin(x_ph[1]);
    e_out[2] = 0.0;
  }

  template <>
  inline void
  ProblemGenerator<Dim2, TypePIC>::UserInitFields(const SimulationParams&,
                                                  Meshblock<Dim2, TypePIC>& mblock) {
    auto r_min = mblock.metric.x1_min;
    Kokkos::parallel_for(
      "UserInitFlds", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, monopoleField, r_min);
      });
  }

  template <>
  inline void ProblemGenerator<Dim2, TypePIC>::UserBCFields(const real_t& time,
                                                            const SimulationParams&,
                                                            Meshblock<Dim2, TypePIC>& mblock) {
    real_t omega;
    auto   r_min = mblock.metric.x1_min;
    if (time < spinup_time) {
      omega = omega_max * time / spinup_time;
    } else {
      omega = omega_max;
    }
    Kokkos::parallel_for(
      "UserBcFlds_rmin",
      CreateRangePolicy<Dim2>({mblock.i1_min(), mblock.i2_min()},
                              {mblock.i1_min() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        // set_em_fields_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_ex2_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_ex3_2d(mblock, i, j, surfaceRotationField, r_min, omega);
        set_bx1_2d(mblock, i, j, surfaceRotationField, r_min, omega);
      });

    Kokkos::parallel_for(
      "UserBcFlds_rmax",
      CreateRangePolicy<Dim2>({mblock.i1_max(), mblock.i2_min()},
                              {mblock.i1_max() + 1, mblock.i2_max()}),
      Lambda(index_t i, index_t j) {
        mblock.em(i, j, em::ex3) = 0.0;
        mblock.em(i, j, em::ex2) = 0.0;
        mblock.em(i, j, em::bx1) = 0.0;
      });
  }

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserInitFields(const SimulationParams&,
                                                              Meshblock<Dim1, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim1, TypePIC>::UserBCFields(const real_t&,
                                                            const SimulationParams&,
                                                            Meshblock<Dim1, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserInitFields(const SimulationParams&,
                                                              Meshblock<Dim3, TypePIC>&) {}

  template <>
  inline void ProblemGenerator<Dim3, TypePIC>::UserBCFields(const real_t&,
                                                            const SimulationParams&,
                                                            Meshblock<Dim3, TypePIC>&) {}

} // namespace ntt

#endif