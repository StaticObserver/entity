#include "global.h"
#include "grpic.h"
#include "grpic_particles_pusher.hpp"

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::pushParticlesSubstep(const real_t&, const real_t& factor) {
    for (auto& species : this->m_mblock.particles) {
      const real_t dt {factor * this->m_mblock.timestep()};
      const real_t coeff {(species.charge() / species.mass()) * HALF * dt / this->sim_params().larmor0()};
      Pusher<D>    pusher(this->m_mblock, species, coeff, dt);
      pusher.pushParticles();
    }
  }

} // namespace ntt

template class ntt::GRPIC<ntt::Dimension::TWO_D>;
template class ntt::GRPIC<ntt::Dimension::THREE_D>;