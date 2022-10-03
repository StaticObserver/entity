#include "vis.h"
#include "api.h"
#include "tools.h"

#include "global.h"
#include "cargs.h"
#include "input.h"

#ifdef PIC_SIMTYPE
#  include "pic.h"
#  define SIMULATION_CONTAINER PIC
#elif defined(GRPIC_SIMTYPE)
#  include "grpic.h"
#  include "init_fields.hpp"
#  define SIMULATION_CONTAINER GRPIC
#endif

#include <toml/toml.hpp>
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

using plog_t = plog::ColorConsoleAppender<plog::NTTFormatter>;
void initLogger(plog_t* console_appender);

struct NTTSimulationVis : public nttiny::SimulationAPI<real_t, 2> {
  int                                   sx1, sx2;
  ntt::SIMULATION_CONTAINER<ntt::Dim2>& m_sim;
  std::vector<std::string>              m_fields_to_plot;

  NTTSimulationVis(ntt::SIMULATION_CONTAINER<ntt::Dim2>& sim,
                   const std::vector<std::string>&       fields_to_plot)
#ifdef PIC_SIMTYPE
    : nttiny::SimulationAPI<real_t, 2> {sim.mblock()->metric.label == "minkowski"
                                          ? nttiny::Coord::Cartesian
                                          : nttiny::Coord::Spherical,
                                        {sim.mblock()->Ni1(), sim.mblock()->Ni2()},
                                        ntt::N_GHOSTS},
#elif defined(GRPIC_SIMTYPE)
    : nttiny::SimulationAPI<real_t, 2> {nttiny::Coord::Spherical,
                                        {sim.mblock()->Ni1(), sim.mblock()->Ni2()},
                                        ntt::N_GHOSTS},
#endif
      sx1 {sim.mblock()->Ni1()},
      sx2 {sim.mblock()->Ni2()},
      m_sim(sim),
      m_fields_to_plot(fields_to_plot) {
    this->m_timestep = 0;
    this->m_time     = 0.0;
    generateFields();
    generateGrid();
    generateParticles();

    nttiny::ScrollingBuffer flux_E;
    this->buffers.insert({"flux_Er", std::move(flux_E)});
    setData();
  }

  void setData() override {
    Kokkos::deep_copy(m_sim.mblock()->em_h, m_sim.mblock()->em);
    Kokkos::deep_copy(m_sim.mblock()->cur_h, m_sim.mblock()->cur);
#ifdef GRPIC_SIMTYPE
    Kokkos::deep_copy(m_sim.mblock()->aphi_h, m_sim.mblock()->aphi);
    // compute the vector potential
    m_sim.computeVectorPotential();
#endif
    const auto ngh = this->m_global_grid.m_ngh;
    real_t     flux_E {ZERO};
    for (int j {-ngh}; j < sx2 + ngh; ++j) {
      for (int i {-ngh}; i < sx1 + ngh; ++i) {
        const int  I = i + ngh, J = j + ngh;
        const auto i_ = (real_t)(i);
        const auto j_ = (real_t)(j);
        for (std::size_t f {0}; f < m_fields_to_plot.size(); ++f) {
#ifdef PIC_SIMTYPE
          if ((i >= 0) && (i < sx1) && (j >= 0) && (j < sx2)) {
            //! TODO: no interpolation to cell center
            ntt::vec_t<ntt::Dim3> e_hat {ZERO}, b_hat {ZERO}, j_hat {ZERO};
            if (m_fields_to_plot[f].at(0) == 'E') {
              m_sim.mblock()->metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF},
                                                 {m_sim.mblock()->em_h(I, J, ntt::em::ex1),
                                                  m_sim.mblock()->em_h(I, J, ntt::em::ex2),
                                                  m_sim.mblock()->em_h(I, J, ntt::em::ex3)},
                                                 e_hat);
            } else if (m_fields_to_plot[f].at(0) == 'B') {
              m_sim.mblock()->metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF},
                                                 {m_sim.mblock()->em_h(I, J, ntt::em::bx1),
                                                  m_sim.mblock()->em_h(I, J, ntt::em::bx2),
                                                  m_sim.mblock()->em_h(I, J, ntt::em::bx3)},
                                                 b_hat);
            } else if (m_fields_to_plot[f].at(0) == 'J') {
              m_sim.mblock()->metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF},
                                                 {m_sim.mblock()->cur_h(I, J, ntt::cur::jx1),
                                                  m_sim.mblock()->cur_h(I, J, ntt::cur::jx2),
                                                  m_sim.mblock()->cur_h(I, J, ntt::cur::jx3)},
                                                 j_hat);
            }
            real_t val {0.0};
            if (m_fields_to_plot[f] == "Er" || m_fields_to_plot[f] == "Ex") {
              if (i == 105) {
                ntt::coord_t<ntt::Dim2> rth_;
                m_sim.mblock()->metric.x_Code2Sph({i_ + HALF, j_ + HALF}, rth_);
                flux_E += e_hat[0] * math::sin(rth_[1]);
              }
              val = e_hat[0];
            } else if (m_fields_to_plot[f] == "Etheta" || m_fields_to_plot[f] == "Ey") {
              val = e_hat[1];
            } else if (m_fields_to_plot[f] == "Ephi" || m_fields_to_plot[f] == "Ez") {
              val = e_hat[2];
            } else if (m_fields_to_plot[f] == "Br" || m_fields_to_plot[f] == "Bx") {
              val = b_hat[0];
            } else if (m_fields_to_plot[f] == "Btheta" || m_fields_to_plot[f] == "By") {
              val = b_hat[1];
            } else if (m_fields_to_plot[f] == "Bphi" || m_fields_to_plot[f] == "Bz") {
              val = b_hat[2];
            } else if (m_fields_to_plot[f] == "Jr" || m_fields_to_plot[f] == "Jx") {
              val = j_hat[0];
            } else if (m_fields_to_plot[f] == "Jtheta" || m_fields_to_plot[f] == "Jy") {
              val = j_hat[1];
            } else if (m_fields_to_plot[f] == "Jphi" || m_fields_to_plot[f] == "Jz") {
              val = j_hat[2];
            }
            auto idx                                 = Index(i, j);
            (this->fields)[m_fields_to_plot[f]][idx] = val;
          } else {
            real_t val {ZERO};
            if (m_fields_to_plot[f] == "Er" || m_fields_to_plot[f] == "Ex") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::ex1);
            } else if (m_fields_to_plot[f] == "Etheta" || m_fields_to_plot[f] == "Ey") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::ex2);
            } else if (m_fields_to_plot[f] == "Ephi" || m_fields_to_plot[f] == "Ez") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::ex3);
            } else if (m_fields_to_plot[f] == "Br" || m_fields_to_plot[f] == "Bx") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::bx1);
            } else if (m_fields_to_plot[f] == "Btheta" || m_fields_to_plot[f] == "By") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::bx2);
            } else if (m_fields_to_plot[f] == "Bphi" || m_fields_to_plot[f] == "Bz") {
              val = m_sim.mblock()->em_h(I, J, ntt::em::bx3);
            } else if (m_fields_to_plot[f] == "Jr" || m_fields_to_plot[f] == "Jx") {
              val = m_sim.mblock()->cur_h(I, J, ntt::cur::jx1);
            } else if (m_fields_to_plot[f] == "Jtheta" || m_fields_to_plot[f] == "Jy") {
              val = m_sim.mblock()->cur_h(I, J, ntt::cur::jx2);
            } else if (m_fields_to_plot[f] == "Jphi" || m_fields_to_plot[f] == "Jz") {
              val = m_sim.mblock()->cur_h(I, J, ntt::cur::jx3);
            }
            auto idx                                 = Index(i, j);
            (this->fields)[m_fields_to_plot[f]][idx] = val;
          }
#elif defined(GRPIC_SIMTYPE)
          // interpolate and transform to spherical
          // @TODO: mirrors for em0, aux etc
          ntt::vec_t<ntt::Dim3> Dsph {ZERO}, Bsph {ZERO}, D0sph {ZERO}, B0sph {ZERO};
          if ((i >= 0) && (i < sx1) && (j >= 0) && (j < sx2)) {
            if (m_fields_to_plot[f].at(0) == 'D') {
              if (m_fields_to_plot[f].at(1) == '0') {
                real_t Dx1, Dx2, Dx3;
                // interpolate to cell center
                Dx1 = 0.5
                      * (m_sim.mblock()->em0(I, J, ntt::em::ex1)
                         + m_sim.mblock()->em0(I, J + 1, ntt::em::ex1));
                Dx2 = 0.5
                      * (m_sim.mblock()->em0(I, J, ntt::em::ex2)
                         + m_sim.mblock()->em0(I + 1, J, ntt::em::ex2));
                Dx3 = 0.25
                      * (m_sim.mblock()->em0(I, J, ntt::em::ex3)
                         + m_sim.mblock()->em0(I + 1, J, ntt::em::ex3)
                         + m_sim.mblock()->em0(I, J + 1, ntt::em::ex3)
                         + m_sim.mblock()->em0(I + 1, J + 1, ntt::em::ex3));
                m_sim.mblock()->metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, D0sph);
              } else {
                real_t Dx1, Dx2, Dx3;
                // interpolate to cell center
                Dx1 = 0.5
                      * (m_sim.mblock()->em(I, J, ntt::em::ex1)
                         + m_sim.mblock()->em(I, J + 1, ntt::em::ex1));
                Dx2 = 0.5
                      * (m_sim.mblock()->em(I, J, ntt::em::ex2)
                         + m_sim.mblock()->em(I + 1, J, ntt::em::ex2));
                Dx3 = 0.25
                      * (m_sim.mblock()->em(I, J, ntt::em::ex3)
                         + m_sim.mblock()->em(I + 1, J, ntt::em::ex3)
                         + m_sim.mblock()->em(I, J + 1, ntt::em::ex3)
                         + m_sim.mblock()->em(I + 1, J + 1, ntt::em::ex3));
                m_sim.mblock()->metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, Dsph);
              }
            } else if (m_fields_to_plot[f].at(0) == 'B') {
              if (m_fields_to_plot[f].at(1) == '0') {
                real_t Bx1, Bx2, Bx3;
                // interpolate to cell center
                Bx1 = 0.5
                      * (m_sim.mblock()->em0(I + 1, J, ntt::em::bx1)
                         + m_sim.mblock()->em0(I, J, ntt::em::bx1));
                Bx2 = 0.5
                      * (m_sim.mblock()->em0(I, J + 1, ntt::em::bx2)
                         + m_sim.mblock()->em0(I, J, ntt::em::bx2));
                Bx3 = m_sim.mblock()->em0(I, J, ntt::em::bx3);
                m_sim.mblock()->metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, B0sph);
              } else {
                real_t Bx1, Bx2, Bx3;
                // interpolate to cell center
                Bx1 = 0.5
                      * (m_sim.mblock()->em(I + 1, J, ntt::em::bx1)
                         + m_sim.mblock()->em(I, J, ntt::em::bx1));
                Bx2 = 0.5
                      * (m_sim.mblock()->em(I, J + 1, ntt::em::bx2)
                         + m_sim.mblock()->em(I, J, ntt::em::bx2));
                Bx3 = m_sim.mblock()->em(I, J, ntt::em::bx3);
                m_sim.mblock()->metric.v_Cntr2SphCntrv(
                  {i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, Bsph);
              }
            }
          } else {
            Dsph[0]  = m_sim.mblock()->em(I, J, ntt::em::ex1);
            Dsph[1]  = m_sim.mblock()->em(I, J, ntt::em::ex2);
            Dsph[2]  = m_sim.mblock()->em(I, J, ntt::em::ex3);
            Bsph[0]  = m_sim.mblock()->em(I, J, ntt::em::bx1);
            Bsph[1]  = m_sim.mblock()->em(I, J, ntt::em::bx2);
            Bsph[2]  = m_sim.mblock()->em(I, J, ntt::em::bx3);
            D0sph[0] = m_sim.mblock()->em0(I, J, ntt::em::ex1);
            D0sph[1] = m_sim.mblock()->em0(I, J, ntt::em::ex2);
            D0sph[2] = m_sim.mblock()->em0(I, J, ntt::em::ex3);
            B0sph[0] = m_sim.mblock()->em0(I, J, ntt::em::bx1);
            B0sph[1] = m_sim.mblock()->em0(I, J, ntt::em::bx2);
            B0sph[2] = m_sim.mblock()->em0(I, J, ntt::em::bx3);
          }
          real_t val {ZERO};
          if (m_fields_to_plot[f] == "Dr") {
            val = Dsph[0];
          } else if (m_fields_to_plot[f] == "Dtheta") {
            val = Dsph[1];
          } else if (m_fields_to_plot[f] == "Dphi") {
            val = Dsph[2];
          } else if (m_fields_to_plot[f] == "Br") {
            val = Bsph[0];
          } else if (m_fields_to_plot[f] == "Btheta") {
            val = Bsph[1];
          } else if (m_fields_to_plot[f] == "Bphi") {
            val = Bsph[2];
          } else if (m_fields_to_plot[f] == "Er") {
            val = m_sim.mblock()->aux(I, J, ntt::em::ex1);
          } else if (m_fields_to_plot[f] == "Etheta") {
            val = m_sim.mblock()->aux(I, J, ntt::em::ex2);
          } else if (m_fields_to_plot[f] == "Ephi") {
            val = m_sim.mblock()->aux(I, J, ntt::em::ex3);
          } else if (m_fields_to_plot[f] == "Hr") {
            val = m_sim.mblock()->aux(I, J, ntt::em::bx1);
          } else if (m_fields_to_plot[f] == "Htheta") {
            val = m_sim.mblock()->aux(I, J, ntt::em::bx2);
          } else if (m_fields_to_plot[f] == "Hphi") {
            val = m_sim.mblock()->aux(I, J, ntt::em::bx3);
          } else if (m_fields_to_plot[f] == "D0r") {
            val = D0sph[0];
          } else if (m_fields_to_plot[f] == "D0theta") {
            val = D0sph[1];
          } else if (m_fields_to_plot[f] == "D0phi") {
            val = D0sph[2];
          } else if (m_fields_to_plot[f] == "B0r") {
            val = B0sph[0];
          } else if (m_fields_to_plot[f] == "B0theta") {
            val = B0sph[1];
          } else if (m_fields_to_plot[f] == "B0phi") {
            val = B0sph[2];
          } else if (m_fields_to_plot[f] == "Aphi") {
            val = m_sim.mblock()->aphi(I, J, 0);
          }
          auto idx                                 = Index(i, j);
          (this->fields)[m_fields_to_plot[f]][idx] = val;
#endif
        }
      }
    }
    auto s {0};
    for (const auto& [lbl, species] : this->particles) {
      auto sim_species = m_sim.mblock()->particles[s];
      for (int p {0}; p < species.first; ++p) {
        real_t                  x1 {(real_t)(sim_species.i1(p)) + sim_species.dx1(p)};
        real_t                  x2 {(real_t)(sim_species.i2(p)) + sim_species.dx2(p)};
        ntt::coord_t<ntt::Dim2> xy {ZERO, ZERO};
        m_sim.mblock()->metric.x_Code2Cart({x1, x2}, xy);
        species.second[0][p] = xy[0];
        species.second[1][p] = xy[1];
      }
      ++s;
    }

    this->buffers["flux_Er"].AddPoint(m_time, (float)(-flux_E));
  }
  void stepFwd() override {
    m_sim.step_forward(m_time);
    ++m_timestep;
    m_time += m_sim.mblock()->timestep();
  }
  void restart() override {
    m_sim.resetCurrents(ZERO);
    m_sim.resetFields(ZERO);
    m_sim.resetParticles(ZERO);
    m_sim.initializeSetup();
    m_sim.initial_step(ZERO);
    setData();
    m_time     = 0.0;
    m_timestep = 0;
  }
  void stepBwd() override {
    // m_sim.step_backward(m_time);
    // --m_timestep;
    // m_time -= m_sim.mblock()->timestep();
  }

  void generateFields() {
    const auto nx1 {this->m_global_grid.m_size[0] + this->m_global_grid.m_ngh * 2};
    const auto nx2 {this->m_global_grid.m_size[1] + this->m_global_grid.m_ngh * 2};
    for (std::size_t i {0}; i < m_fields_to_plot.size(); ++i) {
      this->fields.insert({m_fields_to_plot[i], new real_t[nx1 * nx2]});
    }
  }

  void generateParticles() {
    int s {0};
    for (auto& species : m_sim.mblock()->particles) {
      auto nprtl {m_sim.mblock()->particles[s].npart()};
      this->particles.insert(
        {species.label(), {nprtl, {new real_t[nprtl], new real_t[nprtl]}}});
      ++s;
    }
  }

  void generateGrid() {
    if (this->m_global_grid.m_coord == nttiny::Coord::Spherical) {
      const auto sx1 {this->m_global_grid.m_size[0]};
      const auto sx2 {this->m_global_grid.m_size[1]};
      for (int i {0}; i <= sx1; ++i) {
        auto                    i_ {(real_t)(i)};
        auto                    j_ {ZERO};
        ntt::coord_t<ntt::Dim2> rth_;
        m_sim.mblock()->metric.x_Code2Sph({i_, j_}, rth_);
        this->m_global_grid.m_xi[0][i] = rth_[0];
      }
      for (int j {0}; j <= sx2; ++j) {
        auto                    i_ {ZERO};
        auto                    j_ {(real_t)(j)};
        ntt::coord_t<ntt::Dim2> rth_;
        m_sim.mblock()->metric.x_Code2Sph({i_, j_}, rth_);
        this->m_global_grid.m_xi[1][j] = rth_[1];
      }
      this->m_global_grid.ExtendGridWithGhosts();
    } else {
      const auto s1 {m_sim.mblock()->metric.x1_max - m_sim.mblock()->metric.x1_min};
      const auto s2 {m_sim.mblock()->metric.x2_max - m_sim.mblock()->metric.x2_min};
      const auto sx1 {this->m_global_grid.m_size[0]};
      const auto sx2 {this->m_global_grid.m_size[1]};
      for (int i {0}; i <= sx1; ++i) {
        this->m_global_grid.m_xi[0][i]
          = m_sim.mblock()->metric.x1_min + s1 * (real_t)(i) / (real_t)(sx1);
      }
      for (int j {0}; j <= sx2; ++j) {
        this->m_global_grid.m_xi[1][j]
          = m_sim.mblock()->metric.x2_min + s2 * (real_t)(j) / (real_t)(sx2);
      }
    }
  }

  void customAnnotatePcolor2d(const nttiny::UISettings& ui_settings) override {
#if GRPIC_SIMTYPE
    real_t a        = m_sim.sim_params()->metric_parameters()[4];
    real_t r_absorb = m_sim.sim_params()->metric_parameters()[2];
    real_t rh       = 1.0f + math::sqrt(1.0f - a * a);
    nttiny::tools::drawCircle(
      {0.0f, 0.0f}, rh, {0.0f, ntt::constant::PI}, 128, ui_settings.OutlineColor);
    nttiny::tools::drawCircle(
      {0.0f, 0.0f}, r_absorb, {0.0f, ntt::constant::PI}, 128, ui_settings.OutlineColor);
#elif defined(PIC_SIMTYPE)
    ntt::coord_t<ntt::Dim2> rth_;
    m_sim.mblock()->metric.x_Code2Sph({(float)105 + HALF, HALF}, rth_);
    nttiny::tools::drawCircle(
      {0.0f, 0.0f}, rth_[0], {0.0f, ntt::constant::PI}, 128, ui_settings.OutlineColor);
#endif
  }
};

auto main(int argc, char* argv[]) -> int {
  plog_t console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto  scale_str     = cl_args.getArgument("-scale", "1.0");
    auto  scale         = std::stof(std::string(scale_str));
    auto  inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto  inputdata     = toml::parse(static_cast<std::string>(inputfilename));
    auto& vis_data      = toml::find(inputdata, "visualization");
    std::vector<std::string> fields_to_plot
      = toml::find<std::vector<std::string>>(vis_data, "fields");

    ntt::SIMULATION_CONTAINER<ntt::Dim2> sim(inputdata);
    sim.initialize();
    sim.initializeSetup();
    sim.verify();
    sim.printDetails();
    sim.initial_step(ZERO);
    NTTSimulationVis visApi(sim, fields_to_plot);

    nttiny::Visualization<real_t, 2> vis {scale};
    vis.bindSimulation(&visApi);
    vis.loop();
  }
  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}

void initLogger(plog_t* console_appender) {
  plog::Severity max_severity;
#ifdef DEBUG
  max_severity = plog::verbose;
#else
  max_severity = plog::info;
#endif
  plog::init(max_severity, console_appender);
}