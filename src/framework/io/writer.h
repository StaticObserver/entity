#ifndef IO_WRITER_H
#define IO_WRITER_H

#include "wrapper.h"

#include "sim_params.h"

#include "communications/metadomain.h"
#include "io/output.h"
#include "meshblock/meshblock.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#  ifdef MPI_ENABLED
#    include <mpi.h>
#  endif
#endif

#include <toml.hpp>

#include <map>
#include <string>

namespace ntt {

  template <Dimension D, SimulationEngine S>
  class Writer {
#ifdef OUTPUT_ENABLED

#  ifndef MPI_ENABLED
    adios2::ADIOS m_adios;
#  else
    adios2::ADIOS m_adios { MPI_COMM_WORLD };
#  endif
    adios2::IO                   m_io;
    adios2::Engine               m_writer;

    std::vector<OutputField>     m_fields;
    std::vector<OutputParticles> m_particles;

    real_t                       m_last_output_time { -1.0 };
#endif

  public:
    Writer()  = default;
    ~Writer() = default;

    void Initialize(const SimulationParams&, const Metadomain<D>&, const Meshblock<D, S>&);
    void WriteAll(const SimulationParams&,
                  const Metadomain<D>&,
                  Meshblock<D, S>&,
                  const real_t&,
                  const std::size_t&);

    void Finalize() {
#ifdef OUTPUT_ENABLED
      m_writer.Close();
#endif
    }

    void WriteTomlAttribute(const std::string&, const toml::value&);
    void WriteMeshGrid(const Metadomain<D>&);

    void WriteFields(const SimulationParams&,
                     Meshblock<D, S>&,
                     const real_t&,
                     const std::size_t&);

    void WriteParticles(const SimulationParams&,
                        const Metadomain<D>&,
                        Meshblock<D, S>&,
                        const real_t&,
                        const std::size_t&);
  };

}    // namespace ntt

#endif    // IO_WRITER_H
