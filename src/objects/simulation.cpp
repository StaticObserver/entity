#include "global.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <cassert>

namespace ntt {

template <template <typename T> class D>
Simulation<D>::Simulation(const toml::value &inputdata)
    : m_dim{}, m_sim_params{inputdata, m_dim.dim}, m_meshblock{m_sim_params.m_resolution}, m_pGen{m_sim_params} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

template <template <typename T> class D> void Simulation<D>::initialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  PLOGD << "Simulation initialized.";
}

template <template <typename T> class D> void Simulation<D>::verify() {
  assert(m_sim_params.m_simtype != UNDEFINED_SIM);
  assert(m_sim_params.m_coord_system != UNDEFINED_COORD);
  for (auto &b : m_sim_params.m_boundaries) {
    UNUSED(b);
    assert(b != UNDEFINED_BC);
  }
  // for now only cartesian
  assert(m_meshblock.m_coord_system == CARTESIAN_COORD);
  if (m_meshblock.m_coord_system == CARTESIAN_COORD) {
    // uniform cartesian grid
    if (m_dim.dim == 2) {
      assert(m_meshblock.get_dx1() == m_meshblock.get_dx2());
    }
    if (m_dim.dim == 3) {
      assert(m_meshblock.get_dx1() == m_meshblock.get_dx2());
      assert(m_meshblock.get_dx2() == m_meshblock.get_dx3());
    }
  }
  // TODO: maybe some other tests
  PLOGD << "Simulation prerun check passed.";
}
template <template <typename T> class D>
void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
  m_sim_params.m_outputpath = outdirname;
  m_sim_params.m_inputfilename = infname;
}

template <template <typename T> class D> void Simulation<D>::printDetails() {
  PLOGI << "[Simulation details]";
  PLOGI << "   title: " << m_sim_params.m_title;
  PLOGI << "   type: " << stringifySimulationType(m_sim_params.m_simtype);
  PLOGI << "   total runtime: " << m_sim_params.m_runtime;
  PLOGI << "   dt: " << m_sim_params.m_timestep << " ["
        << static_cast<int>(m_sim_params.m_runtime / m_sim_params.m_timestep) << " steps]";

  PLOGI << "[domain]";
  PLOGI << "   dimension: " << m_dim.dim << "D";
  PLOGI << "   coordinate system: " << stringifyCoordinateSystem(m_sim_params.m_coord_system, m_dim.dim);

  std::string bc{"   boundary conditions: { "};
  for (auto &b : m_sim_params.m_boundaries) {
    bc += stringifyBoundaryCondition(b) + " x ";
  }
  bc.erase(bc.size() - 3);
  bc += " }";
  PLOGI << bc;

  std::string res{"   resolution: { "};
  for (auto &r : m_sim_params.m_resolution) {
    res += std::to_string(r) + " x ";
  }
  res.erase(res.size() - 3);
  res += " }";
  PLOGI << res;

  std::string ext{"   extent: "};
  for (std::size_t i{0}; i < m_sim_params.m_extent.size(); i += 2) {
    ext += "{" + std::to_string(m_sim_params.m_extent[i]) + ", " + std::to_string(m_sim_params.m_extent[i + 1]) + "} ";
  }
  PLOGI << ext;
}

template <template <typename T> class D> void Simulation<D>::finalize() { PLOGD << "Simulation finalized."; }

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt
