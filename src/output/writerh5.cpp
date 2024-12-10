#include "writerh5.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/param_container_h5.h"
#include "utils/tools.h"

#include <Kokkos_Core.hpp>
#include <H5Cpp.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
  #include "arch/mpi_aliases.h"
#endif

#include <string>
#include <vector>
#include <cstdio>  // for std::snprintf


namespace out {

void WriterH5::init(const std::string& title) {
    m_fname = title + ".h5";
    m_file = new H5::H5File(m_fname, H5F_ACC_TRUNC);
}

void WriterH5::addTracker(const std::string& type,
                          std::size_t interval,
                          long double interval_time) {
    m_trackers.insert({ type, tools::Tracker(type, interval, interval_time) });
}

bool WriterH5::shouldWrite(const std::string& type,
                           std::size_t step,
                           long double time) {
    if (m_trackers.find(type) != m_trackers.end()) {
        return m_trackers.at(type).shouldWrite(step, time);
    } else {
        raise::Error(fmt::format("Tracker type {} not found", type.c_str()), HERE);
        return false;
    }
}

void WriterH5::setMode() {
    // 在 HDF5 中模式由文件和数据集创建方式决定，此处无需实现
}

void WriterH5::writeAttrs(const prm::Parameters& params) {
    H5::Group root = m_file->openGroup("/");
    params.writeHDF5(root);  // 假设您实现了 params.writeHDF5 方法
}

void WriterH5::defineMeshLayout(const std::vector<std::size_t>&  glob_shape,
                                const std::vector<std::size_t>&  loc_corner,
                                const std::vector<std::size_t>&  loc_shape,
                                const std::vector<unsigned int>& dwn,
                                bool                             incl_ghosts,
                                Coord                            coords) {
    m_flds_ghosts = incl_ghosts;
    m_dwn         = dwn;

    m_flds_g_shape  = glob_shape;
    m_flds_l_corner = loc_corner;
    m_flds_l_shape  = loc_shape;

    m_flds_g_shape_dwn.clear();
    m_flds_l_corner_dwn.clear();
    m_flds_l_first.clear();
    m_flds_l_shape_dwn.clear();

    for (std::size_t i = 0; i < glob_shape.size(); ++i) {
        raise::ErrorIf(dwn[i] != 1 && incl_ghosts,
                       "Downsampling with ghosts not supported",
                       HERE);

        const double g = static_cast<double>(glob_shape[i]);
        const double dd = static_cast<double>(m_dwn[i]);
        const double l = static_cast<double>(loc_corner[i]);
        const double n = static_cast<double>(loc_shape[i]);
        const double f = math::ceil(l / dd) * dd - l;  // 确保 math::ceil 在 tools.h 有定义

        m_flds_g_shape_dwn.push_back(static_cast<std::size_t>(math::ceil(g / dd)));
        m_flds_l_corner_dwn.push_back(static_cast<std::size_t>(math::ceil(l / dd)));
        m_flds_l_first.push_back(static_cast<std::size_t>(f));
        m_flds_l_shape_dwn.push_back(static_cast<std::size_t>(math::ceil((n - f) / dd)));
    }

    H5::Group meshGroup;
    try {
        meshGroup = m_file->openGroup("/Mesh");
    } catch (...) {
        meshGroup = m_file->createGroup("/Mesh");
    }

    {
        H5::DataSpace scalar_space(H5S_SCALAR);

        int nghosts = incl_ghosts ? N_GHOSTS : 0;
        {
            auto attr = meshGroup.createAttribute("NGhosts", H5::PredType::NATIVE_INT, scalar_space);
            attr.write(H5::PredType::NATIVE_INT, &nghosts);
        }

        int dimension = static_cast<int>(m_flds_g_shape.size());
        {
            auto attr = meshGroup.createAttribute("Dimension", H5::PredType::NATIVE_INT, scalar_space);
            attr.write(H5::PredType::NATIVE_INT, &dimension);
        }

        {
            std::string coord_str = coords.to_string();
            H5::StrType str_type(0, H5T_VARIABLE);
            auto attr = meshGroup.createAttribute("Coordinates", str_type, scalar_space);
            attr.write(str_type, coord_str);
        }
    }

    int layoutRight = 0;
    if constexpr (std::is_same<typename ndfield_t<Dim::_3D, 6>::array_layout,
                               Kokkos::LayoutRight>::value) {
        layoutRight = 1;
    } else {
        std::reverse(m_flds_g_shape_dwn.begin(), m_flds_g_shape_dwn.end());
        std::reverse(m_flds_l_corner_dwn.begin(), m_flds_l_corner_dwn.end());
        std::reverse(m_flds_l_shape_dwn.begin(), m_flds_l_shape_dwn.end());
    }

    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto attr = meshGroup.createAttribute("LayoutRight", H5::PredType::NATIVE_INT, scalar_space);
        attr.write(H5::PredType::NATIVE_INT, &layoutRight);
    }
}

void WriterH5::defineFieldOutputs(const SimEngine& S,
                                  const std::vector<std::string>& flds_out) {
    m_flds_writers.clear();

    raise::ErrorIf((m_flds_g_shape_dwn.empty()) ||
                   (m_flds_l_corner_dwn.empty()) ||
                   (m_flds_l_shape_dwn.empty()),
                   "Mesh layout must be defined before field output", HERE);

    for (const auto& fld : flds_out) {
        m_flds_writers.emplace_back(S, fld);
    }

    H5::Group fieldsGroup;
    try {
        fieldsGroup = m_file->openGroup("/Fields");
    } catch (...) {
        fieldsGroup = m_file->createGroup("/Fields");
    }

    std::string all_fields;
    for (size_t i = 0; i < m_flds_writers.size(); ++i) {
        if (i > 0) all_fields += ",";
        all_fields += m_flds_writers[i].name();
    }

    {
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::StrType str_type(0, H5T_VARIABLE);
        auto attr = fieldsGroup.createAttribute("FieldList", str_type, attr_space);
        attr.write(str_type, all_fields);
    }
}

void WriterH5::defineParticleOutputs(Dimension                          dim,
                                     const std::vector<unsigned short>& specs) {
    m_prtl_writers.clear();
    for (auto s : specs) {
        m_prtl_writers.emplace_back(s);
    }

    H5::Group prtlGroup;
    try {
        prtlGroup = m_file->openGroup("/Particles");
    } catch (...) {
        prtlGroup = m_file->createGroup("/Particles");
    }

    {
        H5::DataSpace scalar_space(H5S_SCALAR);
        int dim_int = static_cast<int>(dim);
        auto attr = prtlGroup.createAttribute("Dimension", H5::PredType::NATIVE_INT, scalar_space);
        attr.write(H5::PredType::NATIVE_INT, &dim_int);
    }

    {
        hsize_t dims[1] = { specs.size() };
        H5::DataSpace space(1, dims);
        auto attr = prtlGroup.createAttribute("SpeciesList", H5::PredType::NATIVE_USHORT, space);
        attr.write(H5::PredType::NATIVE_USHORT, specs.data());
    }
}

void WriterH5::defineSpectraOutputs(const std::vector<unsigned short>& specs) {
    m_spectra_writers.clear();
    for (auto s : specs) {
        m_spectra_writers.emplace_back(s);
    }

    H5::Group specGroup;
    try {
        specGroup = m_file->openGroup("/Spectra");
    } catch (...) {
        specGroup = m_file->createGroup("/Spectra");
    }

    {
        hsize_t dims[1] = { specs.size() };
        H5::DataSpace space(1, dims);
        auto attr = specGroup.createAttribute("SpeciesList", H5::PredType::NATIVE_USHORT, space);
        attr.write(H5::PredType::NATIVE_USHORT, specs.data());
    }
}

void WriterH5::writeMesh(unsigned short dim,
                         const array_t<real_t*>& xc,
                         const array_t<real_t*>& xe) {
    H5::Group meshGroup;
    try {
        meshGroup = m_file->openGroup("/Mesh");
    } catch (...) {
        meshGroup = m_file->createGroup("/Mesh");
    }

    std::string varc_name = "X" + std::to_string(dim + 1);
    std::string vare_name = "X" + std::to_string(dim + 1) + "e";

    auto xc_h = Kokkos::create_mirror_view(xc);
    auto xe_h = Kokkos::create_mirror_view(xe);
    Kokkos::deep_copy(xc_h, xc);
    Kokkos::deep_copy(xe_h, xe);

    hsize_t size_c = static_cast<hsize_t>(xc.extent(0));
    hsize_t size_e = static_cast<hsize_t>(xe.extent(0));
    H5::DataSpace dsp_c(1, &size_c);
    H5::DataSpace dsp_e(1, &size_e);

    H5::DataSet ds_c;
    H5::DataSet ds_e;
    bool c_exists = false, e_exists = false;
    try {
        ds_c = meshGroup.openDataSet(varc_name);
        c_exists = true;
    } catch (...) {}

    if (!c_exists) {
        ds_c = meshGroup.createDataSet(varc_name, H5::PredType::NATIVE_DOUBLE, dsp_c);
    }

    try {
        ds_e = meshGroup.openDataSet(vare_name);
        e_exists = true;
    } catch (...) {}

    if (!e_exists) {
        ds_e = meshGroup.createDataSet(vare_name, H5::PredType::NATIVE_DOUBLE, dsp_e);
    }

    ds_c.write(xc_h.data(), H5::PredType::NATIVE_DOUBLE);
    ds_e.write(xe_h.data(), H5::PredType::NATIVE_DOUBLE);
}


template <Dimension D, int N>
void WriterH5::writeField(const std::vector<std::string>& names,
                          const ndfield_t<D, N>&          fld,
                          const std::vector<std::size_t>& addresses) {
    raise::ErrorIf(addresses.size() > static_cast<std::size_t>(N),
                   "addresses vector size must be less or equal to N",
                   HERE);
    raise::ErrorIf(names.size() != addresses.size(),
                   "# of names != # of addresses ",
                   HERE);

    H5::Group fieldsGroup;
    try {
        fieldsGroup = m_file->openGroup("/Fields");
    } catch (...) {
        fieldsGroup = m_file->createGroup("/Fields");
    }

    auto get_dwn = [&](size_t i) { return m_dwn[i]; };
    auto get_first = [&](size_t i) { return m_flds_l_first[i]; };

    std::vector<hsize_t> dims;
    if constexpr (D == Dim::_1D) {
        double nx1_full = static_cast<double>(fld.extent(0)-2*N_GHOSTS);
        double dwn1 = static_cast<double>(get_dwn(0));
        double f1 = static_cast<double>(get_first(0));
        hsize_t nx1_dwn = static_cast<hsize_t>(std::ceil((nx1_full - f1) / dwn1));
        dims = {nx1_dwn};
    } else if constexpr (D == Dim::_2D) {
        double nx1_full = static_cast<double>(fld.extent(0)-2*N_GHOSTS);
        double nx2_full = static_cast<double>(fld.extent(1)-2*N_GHOSTS);
        double dwn1 = static_cast<double>(get_dwn(0));
        double dwn2 = static_cast<double>(get_dwn(1));
        double f1 = static_cast<double>(get_first(0));
        double f2 = static_cast<double>(get_first(1));
        hsize_t nx1_dwn = static_cast<hsize_t>(std::ceil((nx1_full - f1) / dwn1));
        hsize_t nx2_dwn = static_cast<hsize_t>(std::ceil((nx2_full - f2) / dwn2));
        dims = {nx1_dwn, nx2_dwn};
    } else if constexpr (D == Dim::_3D) {
        double nx1_full = static_cast<double>(fld.extent(0)-2*N_GHOSTS);
        double nx2_full = static_cast<double>(fld.extent(1)-2*N_GHOSTS);
        double nx3_full = static_cast<double>(fld.extent(2)-2*N_GHOSTS);
        double dwn1 = static_cast<double>(get_dwn(0));
        double dwn2 = static_cast<double>(get_dwn(1));
        double dwn3 = static_cast<double>(get_dwn(2));
        double f1 = static_cast<double>(get_first(0));
        double f2 = static_cast<double>(get_first(1));
        double f3 = static_cast<double>(get_first(2));
        hsize_t nx1_dwn = static_cast<hsize_t>(std::ceil((nx1_full - f1) / dwn1));
        hsize_t nx2_dwn = static_cast<hsize_t>(std::ceil((nx2_full - f2) / dwn2));
        hsize_t nx3_dwn = static_cast<hsize_t>(std::ceil((nx3_full - f3) / dwn3));
        dims = {nx1_dwn, nx2_dwn, nx3_dwn};
    }

    for (std::size_t i = 0; i < addresses.size(); ++i) {
        const auto& varname = names[i];
        std::size_t comp = addresses[i];

        size_t total_size = 1;
        for (auto d : dims) total_size *= d;

        std::vector<real_t> buffer(total_size);

        if constexpr (D == Dim::_1D) {
            hsize_t nx = dims[0];
            double dwn1 = static_cast<double>(get_dwn(0));
            double f1 = static_cast<double>(get_first(0));
            for (hsize_t ix = 0; ix < nx; ix++) {
                size_t src_i = static_cast<size_t>(f1 + ix * dwn1 + N_GHOSTS);
                buffer[ix] = fld(src_i, comp);
            }
        } else if constexpr (D == Dim::_2D) {
            hsize_t nx1 = dims[0];
            hsize_t nx2 = dims[1];
            double dwn1 = static_cast<double>(get_dwn(0));
            double dwn2 = static_cast<double>(get_dwn(1));
            double f1 = static_cast<double>(get_first(0));
            double f2 = static_cast<double>(get_first(1));
            for (hsize_t i1 = 0; i1 < nx1; i1++) {
                for (hsize_t i2 = 0; i2 < nx2; i2++) {
                    size_t src_i1 = static_cast<size_t>(f1 + i1 * dwn1 + N_GHOSTS);
                    size_t src_i2 = static_cast<size_t>(f2 + i2 * dwn2 + N_GHOSTS);
                    buffer[i1*nx2 + i2] = fld(src_i1, src_i2, comp);
                }
            }
        } else if constexpr (D == Dim::_3D) {
            hsize_t nx1 = dims[0];
            hsize_t nx2 = dims[1];
            hsize_t nx3 = dims[2];
            double dwn1 = static_cast<double>(get_dwn(0));
            double dwn2 = static_cast<double>(get_dwn(1));
            double dwn3 = static_cast<double>(get_dwn(2));
            double f1 = static_cast<double>(get_first(0));
            double f2 = static_cast<double>(get_first(1));
            double f3 = static_cast<double>(get_first(2));
            for (hsize_t i1 = 0; i1 < nx1; i1++) {
                for (hsize_t i2 = 0; i2 < nx2; i2++) {
                    for (hsize_t i3 = 0; i3 < nx3; i3++) {
                        size_t src_i1 = static_cast<size_t>(f1 + i1 * dwn1 + N_GHOSTS);
                        size_t src_i2 = static_cast<size_t>(f2 + i2 * dwn2 + N_GHOSTS);
                        size_t src_i3 = static_cast<size_t>(f3 + i3 * dwn3 + N_GHOSTS);
                        buffer[(i1*nx2 + i2)*nx3 + i3] = fld(src_i1, src_i2, src_i3, comp);
                    }
                }
            }
        }

        H5::DataSpace dataspace(dims.size(), dims.data());
        H5::DataSet dataset;
        bool exists = false;
        try {
            dataset = fieldsGroup.openDataSet(varname);
            exists = true;
        } catch (...) {}

        if (!exists) {
            dataset = fieldsGroup.createDataSet(varname, H5::PredType::NATIVE_DOUBLE, dataspace);
        }

        dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
    }
}

template void WriterH5::writeField<Dim::_1D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 3>&,
                                                const std::vector<std::size_t>&);
template void WriterH5::writeField<Dim::_1D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_1D, 6>&,
                                                const std::vector<std::size_t>&);
template void WriterH5::writeField<Dim::_2D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 3>&,
                                                const std::vector<std::size_t>&);
template void WriterH5::writeField<Dim::_2D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_2D, 6>&,
                                                const std::vector<std::size_t>&);
template void WriterH5::writeField<Dim::_3D, 3>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 3>&,
                                                const std::vector<std::size_t>&);
template void WriterH5::writeField<Dim::_3D, 6>(const std::vector<std::string>&,
                                                const ndfield_t<Dim::_3D, 6>&,
                                                const std::vector<std::size_t>&);

void WriterH5::writeParticleQuantity(const array_t<real_t*>& array,
                                     std::size_t             glob_total,
                                     std::size_t             loc_offset,
                                     const std::string&      varname) {
    H5::Group prtlGroup;
    try {
        prtlGroup = m_file->openGroup("/Particles");
    } catch (...) {
        prtlGroup = m_file->createGroup("/Particles");
    }

    auto array_h = Kokkos::create_mirror_view(array);
    Kokkos::deep_copy(array_h, array);

    hsize_t global_size = static_cast<hsize_t>(glob_total);
    hsize_t local_size = static_cast<hsize_t>(array.extent(0));

    H5::DataSet dataset;
    bool exists = false;
    try {
        dataset = prtlGroup.openDataSet(varname);
        exists = true;
    } catch (...) {}

    if (!exists) {
        H5::DataSpace fspace(1, &global_size);
        dataset = prtlGroup.createDataSet(varname, H5::PredType::NATIVE_DOUBLE, fspace);
    }

    H5::DataSpace fspace = dataset.getSpace();

    hsize_t start[1] = { loc_offset };
    hsize_t count[1] = { local_size };
    fspace.selectHyperslab(H5S_SELECT_SET, count, start);

    H5::DataSpace mspace(1, &local_size);
    dataset.write(array_h.data(), H5::PredType::NATIVE_DOUBLE, mspace, fspace);
}

void WriterH5::writeSpectrum(const array_t<real_t*>& counts,
                             const std::string&      varname) {
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto counts_h = Kokkos::create_mirror_view(counts);
    Kokkos::deep_copy(counts_h, counts);

    std::vector<real_t> counts_all(counts.extent(0), 0.0);
    MPI_Reduce(counts_h.data(),
               counts_all.data(),
               static_cast<int>(counts_h.extent(0)),
               mpi::get_type<real_t>(),
               MPI_SUM,
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);

    if (rank == MPI_ROOT_RANK) {
        H5::Group specGroup;
        try {
            specGroup = m_file->openGroup("/Spectra");
        } catch (...) {
            specGroup = m_file->createGroup("/Spectra");
        }

        hsize_t size = static_cast<hsize_t>(counts_all.size());
        H5::DataSpace dsp(1, &size);

        H5::DataSet dataset;
        bool exists = false;
        try {
            dataset = specGroup.openDataSet(varname);
            exists = true;
        } catch (...) {}

        if (!exists) {
            dataset = specGroup.createDataSet(varname, H5::PredType::NATIVE_DOUBLE, dsp);
        }

        dataset.write(counts_all.data(), H5::PredType::NATIVE_DOUBLE);
    }

#else
    auto counts_h = Kokkos::create_mirror_view(counts);
    Kokkos::deep_copy(counts_h, counts);

    H5::Group specGroup;
    try {
        specGroup = m_file->openGroup("/Spectra");
    } catch (...) {
        specGroup = m_file->createGroup("/Spectra");
    }

    hsize_t size = static_cast<hsize_t>(counts_h.extent(0));
    H5::DataSpace dsp(1, &size);

    H5::DataSet dataset;
    bool exists = false;
    try {
        dataset = specGroup.openDataSet(varname);
        exists = true;
    } catch (...) {}

    if (!exists) {
        dataset = specGroup.createDataSet(varname, H5::PredType::NATIVE_DOUBLE, dsp);
    }

    dataset.write(counts_h.data(), H5::PredType::NATIVE_DOUBLE);

#endif
}

void WriterH5::writeSpectrumBins(const array_t<real_t*>& e_bins,
                                 const std::string&      varname) {
#if defined(MPI_ENABLED)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != MPI_ROOT_RANK) {
        return;
    }
#endif

    auto e_bins_h = Kokkos::create_mirror_view(e_bins);
    Kokkos::deep_copy(e_bins_h, e_bins);

    H5::Group specGroup;
    try {
        specGroup = m_file->openGroup("/Spectra");
    } catch (...) {
        specGroup = m_file->createGroup("/Spectra");
    }

    hsize_t size = static_cast<hsize_t>(e_bins.extent(0));
    H5::DataSpace dsp(1, &size);

    H5::DataSet dataset;
    bool exists = false;
    try {
        dataset = specGroup.openDataSet(varname);
        exists = true;
    } catch (...) {}

    if (!exists) {
        dataset = specGroup.createDataSet(varname, H5::PredType::NATIVE_DOUBLE, dsp);
    }

    dataset.write(e_bins_h.data(), H5::PredType::NATIVE_DOUBLE);
}

void WriterH5::beginWriting(std::size_t tstep, long double time) {
    if (m_writing_mode) {
        raise::Fatal("Already writing", HERE);
    }
    m_writing_mode = true;

    char stepname[64];
    std::snprintf(stepname, sizeof(stepname), "/Step%08zu", tstep);
    H5::Group stepGroup;
    try {
        stepGroup = m_file->openGroup(stepname);
    } catch (...) {
        stepGroup = m_file->createGroup(stepname);
    }

    {
        H5::DataSpace scalar_space(H5S_SCALAR);

        {
            auto attr = stepGroup.createAttribute("Step", H5::PredType::NATIVE_ULLONG, scalar_space);
            unsigned long long step_ull = static_cast<unsigned long long>(tstep);
            attr.write(H5::PredType::NATIVE_ULLONG, &step_ull);
        }

        {
            auto attr = stepGroup.createAttribute("Time", H5::PredType::NATIVE_LDOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_LDOUBLE, &time);
        }
    }
}

void WriterH5::endWriting() {
    if (!m_writing_mode) {
        raise::Fatal("Not writing", HERE);
    }

    m_writing_mode = false;

    m_file->flush(H5F_SCOPE_GLOBAL);
}

// 确保命名空间结束
} // namespace out

