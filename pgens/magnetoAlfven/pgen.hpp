#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"

#include "framework/domain/metadomain.h"

#include <string>

namespace user {
  using namespace ntt;

  enum class FieldGeometry {
    dipole,
    monopole
  };

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bsurf, real_t rstar, const std::string& field_geometry)
      : Bsurf { bsurf }
      , Rstar { rstar }
      , field_geom { field_geometry == "monopole" ? FieldGeometry::monopole
                                                  : FieldGeometry::dipole } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return Bsurf / SQR(x_Ph[0] / Rstar);
      } else {
        return Bsurf * math::cos(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      if (field_geom == FieldGeometry::monopole) {
        return ZERO;
      } else {
        return Bsurf * HALF * math::sin(x_Ph[1]) / CUBE(x_Ph[0] / Rstar);
      }
    }

  private:
    const real_t        Bsurf, Rstar;
    const FieldGeometry field_geom;
  };

  template <Dimension D>
  struct DriveFields : public InitFields<D> {
    DriveFields(real_t bsurf, real_t rstar, real_t omega, const std::string& field_geometry)
      : InitFields<D> { bsurf, rstar, field_geometry }
      , Omega { omega } {}

    using InitFields<D>::bx1;
    using InitFields<D>::bx2;

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return Omega * bx2(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return -Omega * bx1(x_Ph) * x_Ph[0] * math::sin(x_Ph[1]);
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t Omega;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Spherical, Metric::QSpherical> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D> {}
    };

    static constexpr auto D { M::Dim };

    const SimulationParams& params;

    const real_t      Bsurf, Rstar, Omega, Temperature, N0;
    const std::string field_geom;
    InitFields<D>     init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& m)
      : params { p }
      , Bsurf { p.template get<real_t>("setup.Bsurf", ONE) }
      , Rstar { m.mesh().extent(in::x1).first }
      , Omega { p.template get<real_t>("setup.Omega") }
      , Temperature { p.template get<real_t>("setup.temperature") }
      , N0 { p.template get<real_t>("setup.N0") }
      , field_geom { p.template get<std::string>("setup.field_geometry", "dipole") }
      , init_flds { Bsurf, Rstar, field_geom } {}

    inline void InitPrtls(Domain<S, M>& local_domain) {
      const auto energy_dist = arch::energy_dist::Maxwellian<D, M::CoordType>(
        local_domain.random_pool(), Temperature);
      const auto use_weights = params.template get<bool>("particles.use_weights");
      arch::InjectUniform<S, M>(
        params,
        local_domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        N0,
        use_weights);
    }

    auto AtmFields(real_t time) const -> DriveFields<D> {
      return DriveFields<D> { Bsurf, Rstar, Omega, field_geom };
    }

    auto MatchFields(real_t) const -> InitFields<D> {
      return InitFields<D> { Bsurf, Rstar, field_geom };
    }
  };

} // namespace user

#endif
