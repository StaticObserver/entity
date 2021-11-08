#ifndef PIC_FIELDSOLVER_AMPERE_H
#define PIC_FIELDSOLVER_AMPERE_H

#include "global.h"
#include "meshblock.h"
#include "fieldsolver.h"

namespace ntt {

// * * * * Ampere's law * * * * * * * * * * * * * * * *
template <Dimension D>
class Ampere : public FieldSolver<D> {
  using index_t = typename RealFieldND<D, 1>::size_type;
  real_t coeff_x1, coeff_x2, coeff_x3;

public:
  Ampere(
      const Meshblock<D>& m_mblock_,
      const real_t& coeff_x1_,
      const real_t& coeff_x2_,
      const real_t& coeff_x3_)
      : FieldSolver<D> {m_mblock_}, coeff_x1(coeff_x1_), coeff_x2(coeff_x2_), coeff_x3(coeff_x3_) {}
  Inline void operator()(const index_t) const;
  Inline void operator()(const index_t, const index_t) const;
  Inline void operator()(const index_t, const index_t, const index_t) const;
};

template <>
Inline void Ampere<ONE_D>::operator()(const index_t i) const {
  m_mblock.em_fields(i, fld::ex2)
      += coeff_x1 * (m_mblock.em_fields(i - 1, fld::bx3) - m_mblock.em_fields(i, fld::bx3));
  m_mblock.em_fields(i, fld::ex3)
      += coeff_x1 * (-m_mblock.em_fields(i - 1, fld::bx2) + m_mblock.em_fields(i, fld::bx2));
}

template <>
Inline void Ampere<TWO_D>::operator()(const index_t i, const index_t j) const {
  m_mblock.em_fields(i, j, fld::ex1)
      += coeff_x2 * (-m_mblock.em_fields(i, j - 1, fld::bx3) + m_mblock.em_fields(i, j, fld::bx3));
  m_mblock.em_fields(i, j, fld::ex2)
      += coeff_x1 * (m_mblock.em_fields(i - 1, j, fld::bx3) - m_mblock.em_fields(i, j, fld::bx3));
  m_mblock.em_fields(i, j, fld::ex3)
      += coeff_x2 * (m_mblock.em_fields(i, j - 1, fld::bx1) - m_mblock.em_fields(i, j, fld::bx1))
         + coeff_x1
               * (-m_mblock.em_fields(i - 1, j, fld::bx2) + m_mblock.em_fields(i, j, fld::bx2));
}

template <>
Inline void Ampere<THREE_D>::operator()(const index_t i, const index_t j, const index_t k) const {
  m_mblock.em_fields(i, j, k, fld::ex1)
      += coeff_x3
             * (m_mblock.em_fields(i, j, k - 1, fld::bx2) - m_mblock.em_fields(i, j, k, fld::bx2))
         + coeff_x2
               * (-m_mblock.em_fields(i, j - 1, k, fld::bx3)
                  + m_mblock.em_fields(i, j, k, fld::bx3));
  m_mblock.em_fields(i, j, k, fld::ex2)
      += coeff_x1
             * (m_mblock.em_fields(i - 1, j, k, fld::bx3) - m_mblock.em_fields(i, j, k, fld::bx3))
         + coeff_x3
               * (-m_mblock.em_fields(i, j, k - 1, fld::bx1)
                  + m_mblock.em_fields(i, j, k, fld::bx1));
  m_mblock.em_fields(i, j, k, fld::ex3)
      += coeff_x2
             * (m_mblock.em_fields(i, j - 1, k, fld::bx1) - m_mblock.em_fields(i, j, k, fld::bx1))
         + coeff_x1
               * (-m_mblock.em_fields(i - 1, j, k, fld::bx2)
                  + m_mblock.em_fields(i, j, k, fld::bx2));
}

} // namespace ntt

template class ntt::Ampere<ntt::ONE_D>;
template class ntt::Ampere<ntt::TWO_D>;
template class ntt::Ampere<ntt::THREE_D>;

#endif
