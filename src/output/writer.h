#ifndef OUTPUT_WRITERH5_H
#define OUTPUT_WRITERH5_H

#include <string>
#include <vector>
#include <map>

#include "enums.h"
#include "utils/param_container.h"
#include "utils/tools.h"

// HDF5 C++接口
#include <H5Cpp.h>

namespace out {

  class Writer {
    H5::H5File* m_file { nullptr };

    std::string m_fname;
    std::string m_engine;  // 可能不再需要，但保留以兼容已有逻辑

    bool m_writing_mode { false };

    // global shape of the fields array to output
    std::vector<std::size_t> m_flds_g_shape;
    // local corner of the fields array to output
    std::vector<std::size_t> m_flds_l_corner;
     // local shape of the fields array to output
    std::vector<std::size_t> m_flds_l_shape;

    // downsampling factors for each dimension
    std::vector<unsigned int> m_dwn;
    // starting cell in each dimension (not including ghosts)
    std::vector<std::size_t>  m_flds_l_first;

    bool m_flds_ghosts { false };

    std::map<std::string, tools::Tracker> m_trackers;

    std::vector<OutputField>   m_flds_writers;
    std::vector<OutputSpecies> m_prtl_writers;
    std::vector<OutputSpectra> m_spectra_writers;

    // 这里需要存储 DataSet 的信息（如 step, time, 等）
    // 可在 init 时创建基础数据集或在 beginWriting 时动态创建

  public:
    Writer() {}
    ~Writer() {
      if (m_file) {
        delete m_file;
        m_file = nullptr;
      }
    }

    Writer(Writer&&) = default;

    void init(const std::string& title); 
    // 不需要 adios2::ADIOS* 和 engine 参数了

    void setMode(); // 不需要 adios2::Mode 参数，可根据需要自定义写模式

    void addTracker(const std::string&, std::size_t, long double);
    auto shouldWrite(const std::string&, std::size_t, long double) -> bool;

    void writeAttrs(const prm::Parameters&);

    void defineMeshLayout(const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          const std::vector<unsigned int>&,
                          bool,
                          Coord);

    void defineFieldOutputs(const SimEngine&, const std::vector<std::string>&);
    void defineParticleOutputs(Dimension, const std::vector<unsigned short>&);
    void defineSpectraOutputs(const std::vector<unsigned short>&);

    void writeMesh(unsigned short, const array_t<real_t*>&, const array_t<real_t*>&);

    template <Dimension D, int N>
    void writeField(const std::vector<std::string>&,
                    const ndfield_t<D, N>&,
                    const std::vector<std::size_t>&);

    void writeParticleQuantity(const array_t<real_t*>&,
                               std::size_t,
                               std::size_t,
                               const std::string&);
    void writeSpectrum(const array_t<real_t*>&, const std::string&);
    void writeSpectrumBins(const array_t<real_t*>&, const std::string&);

    void beginWriting(std::size_t, long double);
    void endWriting();

    auto fname() const -> const std::string& {
      return m_fname;
    }
    auto fieldWriters() const -> const std::string& {
      return m_flds_writers;
    }
    auto speciesWriters() const -> const std::vector<OutputSpecies>& {
      return m_prtl_writers;
    }
    auto spectraWriters() const -> const std::vector<OutputSpectra>& {
      return m_spectra_writers;
    }
  };

} // namespace out

#endif // OUTPUT_WRITERH5_H
