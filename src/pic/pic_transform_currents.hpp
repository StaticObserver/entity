#ifndef PIC_TRANSFORM_CURRENTS_H
#define PIC_TRANSFORM_CURRENTS_H

#include "global.h"
#include "pic.h"

namespace ntt {
  /**
   * Transform currents to the coordinate basis.
   *
   * @tparam D Dimension.
   */
  template <Dimension D>
  class TransformCurrentsSubstep {
    using index_t = typename RealFieldND<D, 3>::size_type;
    Meshblock<D, SimulationType::PIC> m_mblock;

  public:
    TransformCurrentsSubstep(const Meshblock<D, SimulationType::PIC>& mblock)
      : m_mblock {mblock} {}
    Inline void operator()(const index_t) const;
    Inline void operator()(const index_t, const index_t) const;
    Inline void operator()(const index_t, const index_t, const index_t) const;
  };

  template <>
  Inline void TransformCurrentsSubstep<Dimension::ONE_D>::operator()(const index_t i) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t coeff_x1 {
      ONE / (m_mblock.metric.h_22({i_ + HALF}) * m_mblock.metric.h_33({i_ + HALF}))};
    real_t coeff_x2 {ONE / (m_mblock.metric.h_11({i_}) * m_mblock.metric.h_33({i_}))};
    real_t coeff_x3 {ONE / (m_mblock.metric.h_11({i_}) * m_mblock.metric.h_22({i_}))};
    m_mblock.cur(i, cur::jx1) *= coeff_x1;
    m_mblock.cur(i, cur::jx2) *= coeff_x2;
    m_mblock.cur(i, cur::jx3) *= coeff_x3;
  }

  template <>
  Inline void TransformCurrentsSubstep<Dimension::TWO_D>::operator()(const index_t i,
                                                                     const index_t j) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t coeff_x1 {
      ONE / (m_mblock.metric.h_22({i_ + HALF, j_}) * m_mblock.metric.h_33({i_ + HALF, j_}))};
    real_t coeff_x2 {
      ONE / (m_mblock.metric.h_11({i_, j_ + HALF}) * m_mblock.metric.h_33({i_, j_ + HALF}))};
    real_t coeff_x3 {ONE / (m_mblock.metric.h_11({i_, j_}) * m_mblock.metric.h_22({i_, j_}))};
    m_mblock.cur(i, j, cur::jx1) *= coeff_x1;
    m_mblock.cur(i, j, cur::jx2) *= coeff_x2;
    m_mblock.cur(i, j, cur::jx3) *= coeff_x3;
  }

  template <>
  Inline void TransformCurrentsSubstep<Dimension::THREE_D>::operator()(const index_t i,
                                                                       const index_t j,
                                                                       const index_t k) const {
    real_t i_ {static_cast<real_t>(static_cast<int>(i) - N_GHOSTS)};
    real_t j_ {static_cast<real_t>(static_cast<int>(j) - N_GHOSTS)};
    real_t k_ {static_cast<real_t>(static_cast<int>(k) - N_GHOSTS)};
    real_t coeff_x1 {ONE
                     / (m_mblock.metric.h_22({i_ + HALF, j_, k_})
                        * m_mblock.metric.h_33({i_ + HALF, j_, k_}))};
    real_t coeff_x2 {ONE
                     / (m_mblock.metric.h_11({i_, j_ + HALF, k_})
                        * m_mblock.metric.h_33({i_, j_ + HALF, k_}))};
    real_t coeff_x3 {ONE
                     / (m_mblock.metric.h_11({i_, j_, k_ + HALF})
                        * m_mblock.metric.h_22({i_, j_, k_ + HALF}))};
    m_mblock.cur(i, j, k, cur::jx1) *= coeff_x1;
    m_mblock.cur(i, j, k, cur::jx2) *= coeff_x2;
    m_mblock.cur(i, j, k, cur::jx3) *= coeff_x3;
  }
} // namespace ntt

#endif
