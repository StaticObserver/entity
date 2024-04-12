#include "framework/logistics/grid.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace ntt {

  template <>
  auto Grid<Dim::_1D>::rangeAllCells() -> range_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeAllCells() -> range_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeAllCells() -> range_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::allLayer,
                                    CellLayer::allLayer,
                                    CellLayer::allLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_1D>::rangeActiveCells() -> range_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeActiveCells() -> range_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeActiveCells() -> range_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::activeLayer,
                                    CellLayer::activeLayer,
                                    CellLayer::activeLayer };
    return rangeCells(region);
  }

  template <Dimension D>
  auto Grid<D>::rangeCells(const box_region_t<D>& region) -> range_t<D> {
    tuple_t<std::size_t, D> imin, imax;
    for (unsigned short i = 0; i < (unsigned short)D; i++) {
      switch (region[i]) {
        case CellLayer::allLayer:
          imin[i] = 0;
          imax[i] = n_all(i);
          break;
        case CellLayer::activeLayer:
          imin[i] = i_min(i);
          imax[i] = i_max(i);
          break;
        case CellLayer::minGhostLayer:
          imin[i] = 0;
          imax[i] = i_min(i);
          break;
        case CellLayer::minActiveLayer:
          imin[i] = i_min(i);
          imax[i] = i_min(i) + N_GHOSTS;
          break;
        case CellLayer::maxActiveLayer:
          imin[i] = i_max(i) - N_GHOSTS;
          imax[i] = i_max(i);
          break;
        case CellLayer::maxGhostLayer:
          imin[i] = i_max(i);
          imax[i] = n_all(i);
          break;
        default:
          raise::Error("Invalid cell layer", HERE);
          throw;
      }
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  // !TODO: too ugly, implement a better solution (combine with device)
  template <Dimension D>
  auto Grid<D>::rangeCellsOnHost(const box_region_t<D>& region) -> range_h_t<D> {
    tuple_t<std::size_t, D> imin, imax;
    for (unsigned short i = 0; i < (unsigned short)D; i++) {
      switch (region[i]) {
        case CellLayer::allLayer:
          imin[i] = 0;
          imax[i] = n_all(i);
          break;
        case CellLayer::activeLayer:
          imin[i] = i_min(i);
          imax[i] = i_max(i);
          break;
        case CellLayer::minGhostLayer:
          imin[i] = 0;
          imax[i] = i_min(i);
          break;
        case CellLayer::minActiveLayer:
          imin[i] = i_min(i);
          imax[i] = i_min(i) + N_GHOSTS;
          break;
        case CellLayer::maxActiveLayer:
          imin[i] = i_max(i) - N_GHOSTS;
          imax[i] = i_max(i);
          break;
        case CellLayer::maxGhostLayer:
          imin[i] = i_max(i);
          imax[i] = n_all(i);
          break;
        default:
          raise::Error("Invalid cell layer", HERE);
      }
    }
    return CreateRangePolicyOnHost<D>(imin, imax);
  }

  template <>
  auto Grid<Dim::_1D>::rangeAllCellsOnHost() -> range_h_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeAllCellsOnHost() -> range_h_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::allLayer, CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeAllCellsOnHost() -> range_h_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::allLayer,
                                    CellLayer::allLayer,
                                    CellLayer::allLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_1D>::rangeActiveCellsOnHost() -> range_h_t<Dim::_1D> {
    box_region_t<Dim::_1D> region { CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_2D>::rangeActiveCellsOnHost() -> range_h_t<Dim::_2D> {
    box_region_t<Dim::_2D> region { CellLayer::activeLayer, CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <>
  auto Grid<Dim::_3D>::rangeActiveCellsOnHost() -> range_h_t<Dim::_3D> {
    box_region_t<Dim::_3D> region { CellLayer::activeLayer,
                                    CellLayer::activeLayer,
                                    CellLayer::activeLayer };
    return rangeCellsOnHost(region);
  }

  template <Dimension D>
  auto Grid<D>::rangeCells(const tuple_t<list_t<int, 2>, D>& ranges)
    -> range_t<D> {
    tuple_t<std::size_t, D> imin, imax;
    for (unsigned short i = 0; i < (unsigned short)D; i++) {
      raise::ErrorIf((ranges[i][0] < -(int)N_GHOSTS) ||
                       (ranges[i][1] > (int)N_GHOSTS),
                     "Invalid cell layer picked",
                     HERE);
      imin[i] = i_min(i) + ranges[i][0];
      imax[i] = i_max(i) + ranges[i][1];
      raise::ErrorIf(imin[i] >= imax[i], "Invalid cell layer picked", HERE);
    }
    return CreateRangePolicy<D>(imin, imax);
  }

  template struct Grid<Dim::_1D>;
  template struct Grid<Dim::_2D>;
  template struct Grid<Dim::_3D>;

} // namespace ntt