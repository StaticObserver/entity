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

The filled start (CPT2024-style: Chernoglazov, Philippov & Timokhin 2024;
Witte et al. 2026, arXiv:2609.08840 Sec. IV) is the only initial condition.
`InitialFields` sets constant `B_x = setup.polar_cap.B0` and `E_x = 0`
everywhere: the domain is plasma-filled and Gauss-consistent. The deposited
net charge equals the prescribed GJ background pointwise,

```text
rho_net(x) = rho_GJ0 * (1 + rho_gj_gradient * (x - x_surface) / L),
L = x_max - x_surface,
```

so `dE_x/dx = rho_deposited - rho_GJ = 0` and the discharge is driven out of
equilibrium by the prescribed `external_current` through Ampere's law, not by
an initial vacuum field. `rho_gj_gradient = 0.8` reproduces the reference
`(1 + 0.8x/L)` Lense-Thirring-inspired gradient; `0` gives the uniform
background. `rho_GJ0` is the `extra_positron_density` scale (default `1`);
the background is part of the physical interpretation and is not represented
by a dynamic Entity species. The prescribed external current is spatially
constant across the whole domain, including the MATCH layer: a graded static
background charge plus a graded prescribed current would violate continuity.

The atmosphere boundary imposes `E_x = 0` and the background `B_x`. The right
MATCH boundary relaxes `B_x` to `B0` and `E_x` to zero; the `E_x -> 0` target
keeps a sheath field from accumulating on the absorbing edge.

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
initial state. Its normalized density follows the prescribed GJ gradient,
`1 + rho_gj_gradient * (x - x_surface) / L`, from the surface through the
whole domain, so the deposited net charge equals the GJ background pointwise
(Sec. 3). It uses stochastic rounding, explicit capacity checks, and updates
`npart`, `counter`, and sorting state.

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

A uniform neutral electron-positron bulk fill of
`initial_injection.fill_density` (in n0 units) is injected from `x_surface`
through the active domain via the same `InjectNonUniform` path, mimicking the
plasma left behind by a previous discharge (the reference value is 10 n_GJ).
All filled particles use the same non-relativistic Maxwellian
(`setup.polar_cap.temperature`), matching the reference practice of
initializing charges at non-relativistic velocities.

## 5. Boundaries

Status: implemented and runtime-verified (five-run A100/V100 boundary
campaign, Phase 2).

- `x1 min`: field and particle atmosphere. The engine additionally confines
  the atmosphere with a constant gravity inside the layer
  (`x <= x_surface + ds`): `g = temperature / height` is derived from the
  atmosphere parameters (not a separate TOML key) and applied as a
  Strang-split half-kick bracketing the Boris push
  (`engines/srpic/particle_pusher.h`, `kernels/pushers/sr.hpp`). This is the
  same hydrostatic confinement the reference Tristan setup applies explicitly
  via `userDriveParticles`.
- `x1 max`: two cooperating mechanisms, each owning one aspect of the
  boundary:
  - fields `MATCH`, relaxing `B_x` to `B0` and `E_x` to zero through the
    layer. The `E_x -> 0` target suppresses the DC sheath that hard
    absorption would otherwise leave at the edge;
  - particles `ABSORB` with flux-conserving bookkeeping in the SR engine: the
    pusher clamps boundary-crossing particles onto the boundary face, the
    standard deposit counts their final displacement, and
    `srpic::BoundaryAbsorb` (src/kernels/boundary_absorb.hpp) tags them dead
    after `CurrentsDeposit`, before particle communication. `x2`/`x3` absorb
    branches keep the old kill-in-pusher behavior;
- Engine-level flux-conserving absorption was verified on A100/V100: edge
  `E_x` sheath 1.79e-6 -> 1.14e-6 and inward electron flux 23% -> 6.9%
  relative to the unrepaired no-`E_x`-match boundary.

## 6. Custom Behavior

Status: implemented, reference-tested, not compiled.

`setup.polar_cap.external_current` is the prescribed constant tetrad current
normalized to `n0 q0 c`. Entity adds `ext_current.jx1()` directly to the
deposited contravariant current, so the PGen stores
`J^1 = J^(hat 1) / scales.dx0` throughout the domain. It is not tapered in the
right MATCH layer: a spatial taper would introduce `dJ_ext/dx != 0` without an
evolving prescribed charge that satisfies continuity. The `ppc0` multiplier
in the Ampere kernel is
cancelled by the `1/ppc0` dependence of `scales.q0` and does not replace the
coordinate-basis conversion.

When QED is enabled, electron and positron species select
`emission = "custom"`. Their `EmissionPolicy` provides:

- continuous curvature recoil normalized by `qed.gamma_rad` and
  `qed.reference_electric_field`;
- unbiased stochastic rounding of the expected macro-photon count;
- curvature-spectrum energy sampling truncated at the parent kinetic energy;
- optional removal of the spectrum interval that cannot reach the magnetic
  pair threshold before an absorbing `x1` boundary;
- complete photon payload initialization.

When `filter_nonconverting_photons = true`, the emission policy evaluates

```text
s_max = max_path abs(sin(theta_B))
epsilon_keep = max(photon_energy_min, 2 / s_max)
```

along the photon's remaining straight path to the appropriate global `x1`
boundary. The curvature-number CCDF is integrated and sampled only above
`epsilon_keep`. This preserves every photon that can acquire nonzero magnetic
pair opacity while omitting photons whose opacity is identically zero
throughout the domain. Continuous curvature recoil is unchanged, so omitted
escaping radiation is not represented by individual macro-photons.

For QED curvature drag, the per-step coefficient is

```text
dt * omegaB0 * abs(qed.reference_electric_field) / qed.gamma_rad^4
```

so the intended radiation-reaction energy is independent of `ppc0` and
macro-particle charge.

Entity's built-in synchrotron reaction is intentionally unused. It depends on
local electromagnetic acceleration, vanishes for ideal parallel motion, and
does not represent a prescribed field-line curvature radius.

When QED is disabled, the TOML contains only the electron and positron
species, both with `emission = "none"`. Entity then dispatches its
no-emission policy, the PGen does not access species `3`, and no photon
particle container or curvature-spectrum device arrays are allocated.

The independent TOML switches are `curvature_drag`,
`curvature_emission`, and `magnetic_pair_creation`; `enable = false` disables
the complete QED path and is the baseline setting until normalization is fixed.
When `curvature_drag = true`, QED also requires `gamma_rad` and
`reference_electric_field`.

The photon `CustomParticleUpdate` advances:

```text
pld_r[:, 0] = epsilon_gamma
pld_r[:, 1] = tau_pair
pld_r[:, 2] = theta_B
```

At emission, `theta_B` is initialized from the full photon direction relative
to the `x1` magnetic field. The update then uses an even-substep composite
Simpson rule, full photon trajectory length for opacity, signed `x1`
displacement for the curvature-angle increment, and
`abs(sin(theta_B))`.
`CustomPostStep` converts photons satisfying both
`epsilon_gamma * abs(sin(theta_B)) >= 2` and the configured optical-depth
threshold. Each conversion creates one electron and one positron with equal
weight and half the photon energy.

Pairs are created after the standard field/current step and participate from
the next timestep. The photon container is marked unsorted after conversion
and compacted after each complete `photon_recycle_interval`, so converted and
boundary-absorbed photons are reclaimed after pair conversion rather than just
before it. Optimized QED-on inputs set the photon species' standard
`clear_interval` to zero to avoid a redundant pre-conversion compaction.

## 7. Custom Output

Status: not-used.

The case requests standard Entity fields and species densities. No
`CustomFieldOutput` or `CustomStat` is defined in the first version.

## 8. PGen-TOML Contract

- Electron and positron species must remain indices `1` and `2`, massive and
  oppositely charged.
- With `qed.enable = false`, exactly two species are allowed, both charged
  species must use `emission = "none"`, and output must not request `N_3`.
- With `qed.enable = true`, exactly three species are required; both charged
  species use `emission = "custom"`, while photon species `3` is massless,
  neutral, uses the photon pusher, and provides at least three real payloads.
- `rho_c`, `gamma_emit`, `photon_energy_min`, `b_over_bq`,
  `max_photons_per_particle_step`, `opacity_substeps`, and
  `conversion_optical_depth` must be positive.
- When magnetic pair creation is enabled, `photon_recycle_interval` must be a
  positive integer. QED-on inputs that use PGen-managed post-conversion
  recycling set photon-species `clear_interval = 0`. Modes without magnetic
  pair creation do not require this interval.
- `filter_nonconverting_photons` is valid only with magnetic pair creation and
  absorbing or atmosphere particle boundaries on both `x1` sides.
- `opacity_substeps` must be even.
- `qed.gamma_rad > 1`, `qed.reference_electric_field > 0`, and
  `0 < qed.max_drag_fraction < 1` are required when QED curvature drag is
  enabled.
- `extra_positron_density` (default `1`) must be positive: it is the GJ
  net-charge scale, so the deposited net charge equals the GJ background
  pointwise and the Gauss-consistent initial `E_x` vanishes.
- `grid.boundaries.atmosphere.height` controls only the neutral atmosphere. It
  must not be fitted to the sum of neutral particles and excess positrons.
- `external_current` is the interior tetrad current normalized to `n0 q0 c`;
  the PGen converts it to a spatially constant internal contravariant `x1`
  component using `scales.dx0`.
- The filled start is the only initial condition; there is no
  `initial_condition` key.
- `initial_injection.fill_density` (default `0`) is the uniform neutral bulk
  fill density in n0 units.
- `rho_gj_gradient` (default `0`, must be non-negative) is the GJ charge
  gradient amplitude applied to the excess-charge profile.
- `initial_injection.minimum_density` must be non-negative and
  `initial_injection.minimum_ppc` must be a positive integer.
- Production builds use `deposit=esirkepov` and `shape_order=3`; matching TOML
  output smoothing is `method="spline"` and `order=3`.
- `spectrum_table` is resolved first from the launch directory and then
  relative to the PGen directory.
- No `radiative_drag = "synchrotron"` entry belongs on the charged species.
- `debug_edge_fields` (default `false`, interval
  `debug_edge_fields_interval`) prints the upstream `E_x` reference 100 cells
  inside the right edge, the last eight active `E_x`/`J_x` faces, and the
  inward/total weight fraction of each charged species in the 0.2-wide region
  inside the right edge (`A_sp1_inw`/`A_sp2_inw`) to stdout for boundary
  diagnostics. It changes no physics.

## 9. Current Status

Completed:

- v1.4.4 PGen, TOML, curvature spectrum table and generator;
- custom curvature emission and continuous recoil;
- photon angle/opacity update;
- deterministic magnetic pair conversion and particle bookkeeping;
- path-aware removal of photons that cannot reach the magnetic pair threshold;
- post-conversion photon recycling and non-duplicating per-species framework
  cleanup;
- QED-off construction does not require inactive QED-only parameters such as
  `b_over_bq`;
- Python reference checks, including counter-propagating photon curvature;
- A100 QED-off electron-beam force balance at
  `j_ext = 1.5 j_GJ` and `gamma_rad = 1.6e5`;
- bounded A100 QED-on emission and two-species force-balance measurement
  through `t = 0.2`;
- engine-level flux-conserving absorption (kill-after-deposit), verified on
  A100/V100: edge `E_x` sheath 1.79e-6 -> 1.14e-6 and inward electron flux
  23% -> 6.9% relative to the unrepaired no-`E_x`-match boundary;
- filled-start initial condition (CPT2024 / Witte+2026 style), now the only
  initial condition: E_x = 0 plasma-filled Gauss-consistent start with uniform
  neutral bulk fill (`initial_injection.fill_density`) and optional GJ charge
  gradient (`rho_gj_gradient`); three ladder inputs `filled_start_e1_ic.toml`,
  `filled_start_e2_atmosphere.toml`, `filled_start_e3_cpt2024.toml` with
  decision cards; Python reference checks. Not yet compiled or run.

Pending:

- A100 production verification run 4b2cb7763820783a;
- Entity compilation and runtime smoke tests for the photon-filter branch;
- single-particle emission tests inside Kokkos;
- discrete Gauss/Ampere checks for the initial state;
- MPI payload transport and multi-domain conversion checks;
- production normalization and ensemble radiation-energy calibration;
- clean process teardown after output closure.

## 10. Important Changes

- 2026-10-15: reduced the case to the filled start as the only initial
  condition (branch `dev/1d-filled-start`, commit 832aa403). Removed the
  vacuum-gap initial condition together with the `initial_condition` and
  `initial_e_coefficient` keys (`InitialFields` now returns uniform `B_x`
  and `E_x = 0`; `extra_positron_density` defaults to `1`); the MATCH-layer
  external-current taper (`external_current` is spatially uniform across the
  whole domain); the PGen-level boundary-flux compensation
  (`boundary_flux_compensation`, obsolete since the engine's flux-conserving
  absorption); the PGen-level `E_x` damping (`boundary_ex_damping`, replaced
  by the `E_x -> 0` right MATCH target); and the QED-off explicit
  radiation-reaction mode (`radiation_reaction.*`, superseded by QED
  curvature drag — QED-off runs must set both charged species to
  `emission = "none"`). The `debug_edge_fields` output dropped the
  missing-flux field and now reports the inward/total weight fractions
  `A_sp1_inw`/`A_sp2_inw` in a 0.2-wide region inside the right edge.

- 2026-09-18: added the filled-start initial condition (branch
  `dev/1d-filled-start`), following the CPT2024 / Witte+2026 (arXiv:2609.08840)
  1D Tristan setup: plasma-filled Gauss-consistent start with `E_x = 0`,
  uniform neutral bulk fill (`initial_injection.fill_density`, reference
  10 n_GJ), and an optional GJ charge gradient (`rho_gj_gradient`, reference
  0.8) applied to the excess-charge profile. `InitialFields` returns `E_x = 0`
  in this mode. The external current stays spatially constant (continuity).
  Motivation: the vacuum-gap start drives a large initial spark and the
  discharged region never damps its broadband electrostatic oscillations,
  while the reference setup settles into damped quasi-periodic cycles. The
  atmosphere confinement gravity turned out to be engine-built-in
  (`g = temperature / height`, derived in `parameters/grid.cpp` and applied in
  the pusher), so the quiet-atmosphere experiment only changes density,
  temperature, and height.

- Replaced old global pusher/engine modifications with PGen-local v1.4.4
  policies.
- Made the species allocation conditional: QED-off runs now allocate only the
  electron and positron containers.
- Kept curvature radiation custom despite built-in synchrotron support because
  the physical accelerations are different.
- Applied the same explicit `gamma_rad` normalization to QED-on curvature
  recoil, replacing the migrated macro-particle-charge coefficient.
- Made inactive QED-only constructor parameters optional so a QED-off input
  does not need placeholder QED configuration.
- Replaced the old two-table spectrum reader with one validated CCDF table.
- Fixed missing photon configuration, payload initialization, pair-kernel
  launch, threshold survival, signed opacity, Simpson parity, capacity checks,
  counters, and propagation direction.
- Added an explicit counter-propagating photon regression test so the signed
  curvature-angle convention cannot silently revert to `abs(ux1)`.
- Clamp pair-threshold roundoff with an explicit device-visible `real_t(0.0)`
  bound before evaluating the child momentum square root.
- Added a path-aware curvature-spectrum lower bound that removes only photons
  whose magnetic pair opacity remains identically zero before escape.
- Moved periodic photon reclamation behind pair conversion and fixed the
  framework cleanup loop so one due species no longer clears every species.
- Replaced kill-before-deposit absorption with flux-conserving absorption in
  the SR engine: the pusher clamps `x1` boundary crossers onto the boundary
  face (unique `dx1` markers) and the new `srpic::BoundaryAbsorb` pass tags
  them dead after `CurrentsDeposit`, before particle communication.
