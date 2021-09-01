#ifndef SIMULATION_SIM_H
#define SIMULATION_SIM_H

#include "global.h"
#include "arrays.h"

#include <toml/toml.hpp>

#include <vector>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace ntt {
class AbstractSimulation {
public:
  AbstractSimulation() = default;
  ~AbstractSimulation() = default;
  virtual void parseInput(int argc, char *argv[]) = 0;
  virtual void run() = 0;
  virtual void printDetails(std::ostream&) = 0;
  virtual void printDetails() = 0;
};


class Simulation : public AbstractSimulation {
protected:
  std::string m_title;

  const Dimension m_dimension;
  const CoordinateSystem m_coord_system;
  const SimulationType m_simulation_type;

  bool m_initialized { false };

  std::string_view m_inputfilename;
  std::string_view m_outputpath;
  toml::value m_inputdata;

  std::vector<int> m_resolution;
  std::vector<real_t> m_size;
  real_t m_runtime;
  real_t m_timestep;

public:
  Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type);
  ~Simulation() = default;
  [[nodiscard]] auto get_title() const -> std::string { return m_title; }
  [[nodiscard]] auto get_precision() const -> std::size_t { return sizeof(real_t); }
  [[nodiscard]] auto get_dimension() const -> Dimension { return m_dimension; }
  [[nodiscard]] auto get_coord_system() const -> CoordinateSystem { return m_coord_system; }
  [[nodiscard]] auto get_simulation_type() const -> SimulationType { return m_simulation_type; }

  void parseInput(int argc, char *argv[]) override;
  void run() override;
  void printDetails(std::ostream& os) override;
  void printDetails() override;

  template <typename T> auto readFromInput(const std::string &blockname, const std::string &variable) -> T;
  template <typename T> auto readFromInput(const std::string &blockname, const std::string &variable, const T &defval) -> T;

  void initialize();


  // virtual void restart() = 0;
  // virtual void mainloop() = 0;
  // virtual void finalize() = 0;
};

class PICSimulation : public Simulation {
protected:
  ParticlePusher m_pusher { UNDEFINED_PUSHER };
public:
  PICSimulation(Dimension dim, CoordinateSystem coord_sys, ParticlePusher pusher) : Simulation{dim, coord_sys, PIC_SIM}, m_pusher(pusher) {};
  PICSimulation(Dimension dim, CoordinateSystem coord_sys) : Simulation{dim, coord_sys, PIC_SIM} {};
  ~PICSimulation() = default;
  void printDetails(std::ostream& os);
};

class PICSimulation1D : public PICSimulation {
public:
  PICSimulation1D(CoordinateSystem coord_sys, ParticlePusher pusher) : PICSimulation{ONE_D, coord_sys, pusher} {};
  PICSimulation1D(CoordinateSystem coord_sys) : PICSimulation{ONE_D, coord_sys} {};
  ~PICSimulation1D() = default;
};

} // namespace ntt

#endif
