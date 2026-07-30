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
The atmosphere boundary imposes `E_x = 0` and the background `B_x`. The right
MATCH boundary relaxes only `B_x`; it deliberately leaves the longitudinal
`E_x` untouched so the sponge does not violate the one-dimensional Gauss
constraint.

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
- `x1 max`: `B_x` is matched, longitudinal `E_x` is not matched, and particles
  are absorbed.
- No global boundary or solver behavior is modified.

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
macro-particle charge. QED and the separate QED-off radiation-reaction mode
remain mutually exclusive.

Entity's built-in synchrotron reaction is intentionally unused. It depends on
local electromagnetic acceleration, vanishes for ideal parallel motion, and
does not represent a prescribed field-line curvature radius.

When QED and explicit radiation reaction are both disabled, the TOML contains
only the electron and positron species, both with `emission = "none"`. Entity
then dispatches its no-emission policy, the PGen does not access species `3`,
and no photon particle container or curvature-spectrum device arrays are
allocated.

A QED-off radiation-reaction test still contains exactly two charged species
and no photon container, but selects `emission = "custom"` to invoke a
drag-only policy. Its independent parameters are
`radiation_reaction.enable`, `radiation_reaction.gamma_rad`,
`radiation_reaction.reference_electric_field`, and
`radiation_reaction.max_drag_fraction`. The per-step coefficient is

```text
dt * omegaB0 * abs(reference_electric_field) / gamma_rad^4
```

so curvature drag balances electric acceleration at `gamma_rad` when the local
parallel field equals the configured reference field. It is independent of
`ppc0`, creates no photons, and cannot be enabled simultaneously with QED.

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
- With `qed.enable = false`, exactly two species are allowed and output must
  not request `N_3`. Charged species use `emission = "custom"` only when
  `radiation_reaction.enable = true`; otherwise they use `emission = "none"`.
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
- `radiation_reaction.gamma_rad > 1`,
  `radiation_reaction.reference_electric_field > 0`, and
  `0 < radiation_reaction.max_drag_fraction < 1` are required when explicit
  radiation reaction is enabled.
- `qed.enable` and `radiation_reaction.enable` are mutually exclusive.
- `extra_positron_density = initial_e_coefficient` is required for the default
  continuous initial Gauss closure.
- `grid.boundaries.atmosphere.height` controls only the neutral atmosphere. It
  must not be fitted to the sum of neutral particles and excess positrons.
- `external_current` is the interior tetrad current normalized to `n0 q0 c`;
  the PGen converts it to a spatially constant internal contravariant `x1`
  component using `scales.dx0`.
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
- path-aware removal of photons that cannot reach the magnetic pair threshold;
- post-conversion photon recycling and non-duplicating per-species framework
  cleanup;
- QED-off construction does not require inactive QED-only parameters such as
  `b_over_bq`;
- Python reference checks, including counter-propagating photon curvature;
- A100 QED-off electron-beam force balance at
  `j_ext = 1.5 j_GJ` and `gamma_rad = 1.6e5`;
- bounded A100 QED-on emission and two-species force-balance measurement
  through `t = 0.2`.

Pending:

- Entity compilation and runtime smoke tests for the photon-filter branch;
- single-particle emission tests inside Kokkos;
- discrete Gauss/Ampere checks for the initial state;
- MPI payload transport and multi-domain conversion checks;
- production normalization and ensemble radiation-energy calibration;
- clean process teardown after output closure.

## 10. Important Changes

- Replaced old global pusher/engine modifications with PGen-local v1.4.4
  policies.
- Made the species allocation conditional: QED-off runs now allocate only the
  electron and positron containers.
- Kept curvature radiation custom despite built-in synchrotron support because
  the physical accelerations are different.
- Added a photon-free, QED-off curvature-drag mode normalized by an explicit
  `gamma_rad`, removing macro-particle-charge dependence from the balance
  energy.
- Applied the same explicit `gamma_rad` normalization to QED-on curvature
  recoil, replacing the migrated macro-particle-charge coefficient.
- Made inactive QED-only constructor parameters optional so a QED-off
  radiation-reaction input does not need placeholder QED configuration.
- Replaced the old two-table spectrum reader with one validated CCDF table.
- Fixed missing photon configuration, payload initialization, pair-kernel
  launch, threshold survival, signed opacity, Simpson parity, capacity checks,
  counters, and propagation direction.
- Removed the prescribed-current taper and excluded longitudinal `E_x` from
  the right MATCH target so the boundary no longer creates a Gauss-law source
  across the sponge layer.
- Added an explicit counter-propagating photon regression test so the signed
  curvature-angle convention cannot silently revert to `abs(ux1)`.
- Clamp pair-threshold roundoff with an explicit device-visible `real_t(0.0)`
  bound before evaluating the child momentum square root.
- Added a path-aware curvature-spectrum lower bound that removes only photons
  whose magnetic pair opacity remains identically zero before escape.
- Moved periodic photon reclamation behind pair conversion and fixed the
  framework cleanup loop so one due species no longer clears every species.
