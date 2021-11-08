#include "global.h"
#include "cargs.h"
#include "input.h"
#include "simulation.h"

#include <toml/toml.hpp>

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <vector>
#include <stdexcept>

using plog_t = plog::ColorConsoleAppender<plog::NTTFormatter>;

void initLogger(plog_t* console_appender);

// Logging is done via `plog` library...
// ... Use the following commands:
//  `PLOGI << ...` for general info
//  `PLOGF << ...` for fatal error messages (development)
//  `PLOGD << ...` for debug messages (development)
//  `PLOGE << ...` for simple error messages
//  `PLOGW << ...` for warnings

auto main(int argc, char* argv[]) -> int {
  plog_t console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::DEF_input_filename);
    auto outputpath = cl_args.getArgument("-output", ntt::DEF_output_path);
    auto inputdata = toml::parse(static_cast<std::string>(inputfilename));
    short res = static_cast<short>(
        ntt::readFromInput<std::vector<std::size_t>>(inputdata, "domain", "resolution").size());

    if (res == 1) {
      ntt::Simulation<ntt::ONE_D> sim(inputdata);
      sim.run(inputfilename, outputpath);
    } else if (res == 2) {
      ntt::Simulation<ntt::TWO_D> sim(inputdata);
      sim.run(inputfilename, outputpath);
    } else if (res == 3) {
      ntt::Simulation<ntt::THREE_D> sim(inputdata);
      sim.run(inputfilename, outputpath);
    } else {
      throw std::logic_error("# Error: wrong dimension specified.");
    }
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
