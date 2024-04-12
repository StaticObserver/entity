#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "output/fields.h"
#include "output/writer.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
auto readVar(adios2::IO& io, const std::string& name) -> T {
  T              var;
  adios2::Engine reader = io.Open("test.h5", adios2::Mode::Read);
  reader.BeginStep();
  adios2::Variable<T> Var = io.InquireVariable<T>(name);
  reader.Get(Var, var);
  reader.EndStep();
  reader.Close();
  return var;
}

void cleanup() {
  namespace fs = std::filesystem;
  fs::path tempfile_path { "test.h5" };
  fs::remove(tempfile_path);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;
    auto writer = out::Writer("hdf5");
    writer.defineFieldLayout({ 10, 10, 10 }, { 0, 0, 0 }, { 10, 10, 10 }, false);
    writer.defineFieldOutputs(SimEngine::SRPIC, { "E", "B", "Rho_1_3", "N_2" });

    ndfield_t<Dim::_3D, 3> field { "fld",
                                   10 + 2 * N_GHOSTS,
                                   10 + 2 * N_GHOSTS,
                                   10 + 2 * N_GHOSTS };
    Kokkos::parallel_for(
      "fill",
      CreateRangePolicy<Dim::_3D>({ N_GHOSTS, N_GHOSTS, N_GHOSTS },
                                  { 10 + N_GHOSTS, 10 + N_GHOSTS, 10 + N_GHOSTS }),
      Lambda(index_t i1, index_t i2, index_t i3) {
        field(i1, i2, i3, 0) = i1 + i2 + i3;
        field(i1, i2, i3, 1) = i1 * i2 / i3;
        field(i1, i2, i3, 2) = i1 / i2 * i3;
      });
    // writer.fieldWriters().
    std::vector<std::string> names;
    std::vector<std::size_t> addresses;
    for (auto i = 0; i < 3; ++i) {
      names.push_back(writer.fieldWriters()[0].name(i));
      addresses.push_back(i);
    }
    writer.beginWriting("test", 0, 0.0);
    writer.writeField<Dim::_3D, 3>(names, field, addresses);
    writer.endWriting();

    writer.beginWriting("test", 1, 0.1);
    writer.writeField<Dim::_3D, 3>(names, field, addresses);
    writer.endWriting();

    {
      // read
      adios2::ADIOS adios;
      adios2::IO    io = adios.DeclareIO("read-test");
      io.SetEngine("hdf5");

      raise::ErrorIf(readVar<std::size_t>(io, "Step") != 0, "Step is not 0", HERE);
      raise::ErrorIf(readVar<real_t>(io, "Time") != (real_t)0.0,
                     "Time is not 0.0",
                     HERE);

      adios2::Engine reader = io.Open("test.h5", adios2::Mode::Read);
      std::size_t    step   = 0;
      for (; reader.BeginStep() == adios2::StepStatus::OK; ++step) {
        std::size_t                   step_read;
        adios2::Variable<std::size_t> stepVar = io.InquireVariable<std::size_t>(
          "Step");
        reader.Get(stepVar, step_read);

        real_t                   time_read;
        adios2::Variable<real_t> timeVar = io.InquireVariable<real_t>("Time");
        reader.Get(timeVar, time_read);
        raise::ErrorIf(step_read != step, "Step is not correct", HERE);
        raise::ErrorIf(time_read != (real_t)step / 10, "Time is not correct", HERE);

        for (const auto& name : names) {
          auto data = io.InquireVariable<real_t>(name);
          raise::ErrorIf(data.Shape().size() != 3,
                         fmt::format("%s is not 3D", name),
                         HERE);

          auto        dims = data.Shape();
          std::size_t nx1  = dims[0];
          std::size_t nx2  = dims[1];
          std::size_t nx3  = dims[2];
          raise::ErrorIf((nx1 != 10) || (nx2 != 10) || (nx3 != 10),
                         fmt::format("%s is not 10x10x10", name.c_str()),
                         HERE);
        }
        reader.EndStep();
      }
      reader.Close();
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    cleanup();
    Kokkos::finalize();
    return 1;
  }
  cleanup();
  Kokkos::finalize();
  return 0;
}