#include "writer.h"

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "simulation.h"
#include "utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <plog/Log.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <type_traits>

namespace ntt {

#ifdef OUTPUT_ENABLED
  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::Initialize(const SimulationParams& params, const Meshblock<D, S>& mblock) {
    m_io = m_adios.DeclareIO("WriteKokkos");
    m_io.SetEngine("HDF5");
    adios2::Dims shape, start, count;
    for (short d = 0; d < (short)D; ++d) {
      shape.push_back(mblock.Ni(d) + 2 * N_GHOSTS);
      count.push_back(mblock.Ni(d) + 2 * N_GHOSTS);
      start.push_back(0);
    }
    auto isLayoutRight
      = std::is_same<typename ndfield_t<D, 6>::array_layout, Kokkos::LayoutRight>::value;

    if (isLayoutRight) {
      m_io.DefineAttribute<int>("LayoutRight", 1);
    } else {
      std::reverse(shape.begin(), shape.end());
      std::reverse(count.begin(), count.end());
      m_io.DefineAttribute<int>("LayoutRight", 0);
    }

    m_io.DefineVariable<int>("Step");
    m_io.DefineVariable<real_t>("Time");

    m_io.DefineAttribute<std::string>("Metric", mblock.metric.label);
    m_io.DefineAttribute<std::string>("Engine", stringizeSimulationEngine(S));
    if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("X1Min", mblock.metric.x1_min);
      m_io.DefineAttribute<real_t>("X1Max", mblock.metric.x1_max);

      auto x1 = new real_t[mblock.Ni1() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni1(); ++i) {
        auto x_ = mblock.metric.x1_min
                  + (mblock.metric.x1_max - mblock.metric.x1_min) * i / mblock.Ni1();
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[0] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x1[i] = xph[0];
      }
      m_io.DefineAttribute<real_t>("X1", x1, mblock.Ni1() + 1);
    }
    if constexpr (D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("X2Min", mblock.metric.x2_min);
      m_io.DefineAttribute<real_t>("X2Max", mblock.metric.x2_max);

      auto x2 = new real_t[mblock.Ni2() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni2(); ++i) {
        auto x_ = mblock.metric.x2_min
                  + (mblock.metric.x2_max - mblock.metric.x2_min) * i / mblock.Ni2();
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[1] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x2[i] = xph[1];
      }
      m_io.DefineAttribute<real_t>("X2", x2, mblock.Ni2() + 1);
    }
    if constexpr (D == Dim3) {
      m_io.DefineAttribute<real_t>("X3Min", mblock.metric.x3_min);
      m_io.DefineAttribute<real_t>("X3Max", mblock.metric.x3_max);

      auto x3 = new real_t[mblock.Ni3() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni3(); ++i) {
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[2] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x3[i] = xph[2];
      }
      m_io.DefineAttribute<real_t>("X3", x3, mblock.Ni3() + 1);
    }
    m_io.DefineAttribute<int>("NGhosts", N_GHOSTS);
    m_io.DefineAttribute<int>("Dimension", (int)D);

    if constexpr (S == GRPICEngine) {
      m_io.DefineAttribute<real_t>("a", mblock.metric.spin());
    }

    m_io.DefineAttribute<real_t>("Timestep", mblock.timestep());

    for (auto& var : params.outputFields()) {
      m_fields.push_back(InterpretInputField(var));
    }
    for (auto& fld : m_fields) {
      for (std::size_t i { 0 }; i < fld.comp.size(); ++i) {
        m_io.DefineVariable<real_t>(fld.name(i), shape, start, count, adios2::ConstantDims);
      }
    }
  }

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::~Writer() {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams& params,
                                 Meshblock<D, S>&        mblock,
                                 const real_t&           time,
                                 const std::size_t&      tstep) {
    NTTLog();
    m_writer = m_io.Open(params.title() + ".flds.h5", m_mode);
    m_mode   = adios2::Mode::Append;

    m_writer.BeginStep();

    int step = (int)tstep;
    m_writer.Put<int>(m_io.InquireVariable<int>("Step"), &step);
    m_writer.Put<real_t>(m_io.InquireVariable<real_t>("Time"), &time);

    // traverse all the fields and put them. ...
    // ... also make sure that the fields are ready for output, ...
    // ... i.e. they have been written into proper arrays
    for (auto& fld : m_fields) {
      fld.put<D, S>(m_io, m_writer, params, mblock);
    }

    m_writer.EndStep();
    m_writer.Close();
  }

#else
  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::Initialize(const SimulationParams&, const Meshblock<D, S>&) {}

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::~Writer() {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams&,
                                 Meshblock<D, S>&,
                                 const real_t&,
                                 const std::size_t&) {}

#endif

}    // namespace ntt