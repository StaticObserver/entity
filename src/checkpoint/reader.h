/**
 * @file checkpoint/reader.h
 * @brief Function for reading field & particle data from checkpoint files using HDF5
 * @implements
 *   - checkpoint::ReadFields -> void
 *   - checkpoint::ReadParticleData -> void
 *   - checkpoint::ReadParticleCount -> std::pair<std::size_t, std::size_t>
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_READER_H
#define CHECKPOINT_READER_H

#include "arch/kokkos_aliases.h"
#include "global.h" // for Dimension, etc.

#include <H5Cpp.h>

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  /**
   * @brief Read a portion of a multi-dimensional field dataset from an HDF5 file.
   *
   * @tparam D Dimension of the field.
   * @tparam N Number of components in the field.
   * @param file Reference to an opened H5::H5File object.
   * @param dataset_name The name of the dataset to read from.
   * @param start Starting indices (for each dimension) of the hyperslab to read.
   * @param count Number of elements (for each dimension) to read.
   * @param field The ndfield_t object where data will be stored.
   */
  template <Dimension D, int N>
  void ReadFields(H5::H5File& file,
                  const std::string& dataset_name,
                  const std::vector<hsize_t>& start,
                  const std::vector<hsize_t>& count,
                  ndfield_t<D, N>& field);

  /**
   * @brief Read the total particle count and determine local offset from an HDF5 file.
   *
   * @param file Reference to an opened H5::H5File object.
   * @param dataset_name The name of the dataset holding particle data (e.g., a 1D dataset of particles).
   * @param some_param1 Some parameter (previously unsigned short), usage depends on original logic.
   * @param some_param2 Some parameter (std::size_t), usage depends on original logic.
   * @param some_param3 Some parameter (std::size_t), usage depends on original logic.
   * @return A pair {global_particle_count, local_offset}.
   */
  std::pair<std::size_t, std::size_t> ReadParticleCount(H5::H5File& file,
                                                        const std::string& dataset_name,
                                                        unsigned short some_param1,
                                                        std::size_t some_param2,
                                                        std::size_t some_param3);

  /**
   * @brief Read a portion of particle data from an HDF5 file.
   *
   * @tparam T Data type of the particle attribute to read.
   * @param file Reference to an opened H5::H5File object.
   * @param dataset_name The name of the dataset from which particles are read.
   * @param species_id A parameter (previously used in ADIOS2 code), usage depends on original logic.
   * @param array The Kokkos array_t where data will be stored.
   * @param offset The starting offset in the dataset to read from.
   * @param local_count The number of elements to read.
   */
  template <typename T>
  void ReadParticleData(H5::H5File& file,
                        const std::string& dataset_name,
                        unsigned short species_id,
                        array_t<T*>& array,
                        std::size_t offset,
                        std::size_t local_count);

} // namespace checkpoint

#endif // CHECKPOINT_READER_H
