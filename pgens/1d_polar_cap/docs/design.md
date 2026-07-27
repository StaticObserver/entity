# 1D Polar-Cap Design

## 1. Goal

Implement a local one-dimensional pulsar polar-cap discharge on Entity v1.4.4:
charge starvation produces `E_parallel`, accelerated pairs emit curvature
photons, photons accumulate magnetic-conversion opacity, and converted pairs
feed back through the standard SRPIC current deposition.

The first version uses a prescribed constant field-line curvature radius and
does not model transverse Landau levels, varying magnetic geometry, GR, or a
general-purpose QED module.

## 2. Basic Configuration

Status: implemented, statically checked, not compiled.

- Entity baseline: v1.4.4.
- Engine: SRPIC.
- Metric and dimension: 1D Minkowski/Cartesian.
- Current production build contract: Esirkepov current deposition with
  third-order particle shape (`SHAPE_ORDER=3`).
- Field-output particle moments use third-order spline smoothing so diagnostic
  density/charge moments use the same shape order as deposition.
- QED-off species order: electron `1`, positron `2`; no photon container is
  allocated.
- QED-on species order: electron `1`, positron `2`, photon `3`.
- QED-on runtime data: `data/curvature_ccdf.tsv`; QED-off construction neither
  reads this table nor allocates its device arrays.
- Production normalization remains open. The checked-in values are migration
  baselines reconstructed from the old prototype.

## 3. Initial Electromagnetic Fields

Status: implemented, not runtime-verified.

`InitialFields` sets constant `B_x = setup.polar_cap.B0`. The excess charge is
a unit plateau through the atmosphere, so `E_x = 0` up to
`x_surface + grid.boundaries.atmosphere.ds`. Beyond that edge, `E_x` is the
analytic integral of the S-shaped excess-positron decline and the prescribed
background. After the excess profile reaches its explicit cutoff at
`x_surface + 1.33 ds`, only the background remains and `E_x` is linear. The
atmosphere and match boundaries impose `E_x = 0` and the same background `B_x`.

For `x >= x_surface`, this field obeys
`dE_x/dx = rho_excess - rho_GJ` when
`extra_positron_density = initial_e_coefficient`, where the fixed prescribed
background is `rho_GJ = initial_e_coefficient`. The default parameters satisfy
this continuous relation. The background is part of the physical
interpretation and is not represented by a dynamic Entity species.

## 4. Initial Particles

Status: implemented, not runtime-verified.

The Entity nonuniform injector creates a neutral electron-positron atmosphere.
Its configured `density` is the total neutral density, divided equally between
the two species. For an intended neutral edge density `n_edge`, its scale
height must satisfy
`height = ds / log(density / n_edge)`; the production choice
`density = 100`, `ds = 0.6`, and `n_edge = 1` gives
`height = 0.1302883446`.

A PGen-local single-species injector adds the positron excess required by the
migrated initial state. Its normalized density is exactly one from the surface
through `x_surface + ds`, then follows an S-shaped logistic decline with scale
`0.03 ds`, and is zero from `x_surface + 1.33 ds` onward. It uses stochastic
rounding, explicit capacity checks, and updates `npart`, `counter`, and sorting
state.

Both paths retain Entity's standard 1D3V Maxwellian. The prototype's global
injector modification that forced `u2 = u3 = 0` is not restored.

For the excess-positron dilute tail, densities below
`initial_injection.minimum_density` are omitted. If the target PPC is below
`initial_injection.minimum_ppc`, the injector keeps that fixed macro-particle
count and uses weight `target_ppc / minimum_ppc`; stochastic rounding is used
only above this floor. Position and thermal-momentum sampling remain random.
The stored weight participates in SRPIC current deposition and is inherited by
emitted photons and converted pairs. The Cartesian atmosphere path now reads
the global `particles.use_weights` setting, which is enabled for this case, so
atmosphere replenishment and standard `N_*` moments use the same weighted
density convention.

## 5. Boundaries

Status: implemented through standard Entity boundaries, not runtime-verified.

- `x1 min`: field and particle atmosphere.
- `x1 max`: matched fields and absorbing particles.
- No global boundary or solver behavior is modified.

## 6. Custom Behavior

Status: implemented, reference-tested, not compiled.

`setup.polar_cap.external_current` is the desired constant tetrad current
normalized to `n0 q0 c`. Entity adds `ext_current.jx1()` directly to the
deposited contravariant current, so the PGen stores and returns
`J^1 = J^(hat 1) / scales.dx0`. The `ppc0` multiplier in the Ampere kernel is
cancelled by the `1/ppc0` dependence of `scales.q0` and does not replace this
coordinate-basis conversion.

When QED is enabled, electron and positron species select
`emission = "custom"`. Their `EmissionPolicy` provides:

- continuous curvature recoil proportional to `-gamma^3 u / rho_c^2`;
- unbiased stochastic rounding of the expected macro-photon count;
- curvature-spectrum energy sampling truncated at the parent kinetic energy;
- complete photon payload initialization.

Entity's built-in synchrotron reaction is intentionally unused. It depends on
local electromagnetic acceleration, vanishes for ideal parallel motion, and
does not represent a prescribed field-line curvature radius.

When QED is disabled, the TOML contains only the electron and positron species,
both with `emission = "none"`. Entity then dispatches its no-emission policy,
the PGen does not access species `3`, and no photon particle container or
curvature-spectrum device arrays are allocated.

The independent TOML switches are `curvature_drag`,
`curvature_emission`, and `magnetic_pair_creation`; `enable = false` disables
the complete QED path and is the baseline setting until normalization is fixed.

The photon `CustomParticleUpdate` advances:

```text
pld_r[:, 0] = epsilon_gamma
pld_r[:, 1] = tau_pair
pld_r[:, 2] = theta_B
```

At emission, `theta_B` is initialized from the full photon direction relative
to the `x1` magnetic field. The update then uses an even-substep composite
Simpson rule, full photon trajectory length for opacity, `x1` displacement for
the curvature-angle increment, and `abs(sin(theta_B))`.
`CustomPostStep` converts photons satisfying both
`epsilon_gamma * abs(sin(theta_B)) >= 2` and the configured optical-depth
threshold. Each conversion creates one electron and one positron with equal
weight and half the photon energy.

Pairs are created after the standard field/current step and participate from
the next timestep.

## 7. Custom Output

Status: not-used.

The case requests standard Entity fields and species densities. No
`CustomFieldOutput` or `CustomStat` is defined in the first version.

## 8. PGen-TOML Contract

- Electron and positron species must remain indices `1` and `2`, massive and
  oppositely charged.
- With `qed.enable = false`, exactly two species are allowed; both charged
  species use `emission = "none"` and output must not request `N_3`.
- With `qed.enable = true`, exactly three species are required; both charged
  species use `emission = "custom"`, while photon species `3` is massless,
  neutral, uses the photon pusher, and provides at least three real payloads.
- `rho_c`, `gamma_emit`, `photon_energy_min`, `b_over_bq`,
  `max_photons_per_particle_step`, `opacity_substeps`, and
  `conversion_optical_depth` must be positive.
- `opacity_substeps` must be even.
- `extra_positron_density = initial_e_coefficient` is required for the default
  continuous initial Gauss closure.
- `grid.boundaries.atmosphere.height` controls only the neutral atmosphere. It
  must not be fitted to the sum of neutral particles and excess positrons.
- `external_current` is a tetrad current normalized to `n0 q0 c`; the PGen
  converts it to the internal contravariant `x1` component using `scales.dx0`.
- `initial_injection.minimum_density` must be non-negative and
  `initial_injection.minimum_ppc` must be a positive integer.
- Production builds use `deposit=esirkepov` and `shape_order=3`; matching TOML
  output smoothing is `method="spline"` and `order=3`.
- `spectrum_table` is resolved first from the launch directory and then
  relative to the PGen directory.
- No `radiative_drag = "synchrotron"` entry belongs on the charged species.

## 9. Current Status

Completed:

- v1.4.4 PGen, TOML, curvature spectrum table and generator;
- custom curvature emission and continuous recoil;
- photon angle/opacity update;
- deterministic magnetic pair conversion and particle bookkeeping;
- non-compiling Python reference checks.

Pending:

- Entity compilation and runtime smoke tests;
- single-particle drag and emission tests inside Kokkos;
- discrete Gauss/Ampere checks for the initial state;
- MPI payload transport and multi-domain conversion checks;
- production normalization and ensemble radiation-energy calibration.

## 10. Important Changes

- Replaced old global pusher/engine modifications with PGen-local v1.4.4
  policies.
- Made the species allocation conditional: QED-off runs now allocate only the
  electron and positron containers.
- Kept curvature radiation custom despite built-in synchrotron support because
  the physical accelerations are different.
- Replaced the old two-table spectrum reader with one validated CCDF table.
- Fixed missing photon configuration, payload initialization, pair-kernel
  launch, threshold survival, signed opacity, Simpson parity, capacity checks,
  counters, and propagation direction.
