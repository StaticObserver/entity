#include "global.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template <Dimension D>
Meshblock<D>::Meshblock(std::vector<std::size_t> res, std::vector<ParticleSpecies>& parts)
    : Fields<D>(res) {
  for (auto& part : parts) {
    particles.emplace_back(part);
  }
}

template <>
void Meshblock<ONE_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("# Error: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<TWO_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("# Error: undefined particle pusher.");
    }
  }
}

template <>
void Meshblock<THREE_D>::verify(const SimulationParams&) {
  for (auto& p : particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("# Error: undefined particle pusher.");
    }
  }
}

} // namespace ntt

template struct ntt::Meshblock<ntt::ONE_D>;
template struct ntt::Meshblock<ntt::TWO_D>;
template struct ntt::Meshblock<ntt::THREE_D>;
