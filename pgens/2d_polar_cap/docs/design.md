# 2d_polar_cap Design

## 1. Goal

Local two-dimensional pulsar polar-cap discharge on Entity v1.4.4, targeting
reproduction of the 2D setup of Chernoglazov, Philippov & Timokhin (2024,
ApJL, arXiv:2409.15409): a Cartesian slab with x1 along the background
magnetic field (gap direction) and x2 transverse across the polar cap
(periodic). The physics goal is the transverse coherence of pair discharges:
with a transversely varying magnetospheric current j_mag(x2), discharges on
adjacent field lines must evolve independently, with coherence scale of order
the gap height.

v1 scope (this implementation): the transversely uniform baseline — same
initial/boundary physics as `pgens/1d_polar_cap`, lifted to 2D, plus an
x2-dependent external-current profile. A transversely uniform 2D run must
reduce column-by-column to the validated 1D results; this is the primary
acceptance test.

Deferred (planned, not in v1):

- patch desynchronization protocol (dense plasma patches with time-scheduled
  keep-alive pair injection, CPT2024 Sec. II.5);
- paper-style full-domain plasma fill carrying j = j_mag with deltaE = 0;
- rho_GJ gradient along x1 (`1 + 0.8 z/Lz`);
- dipole-like curvature radius varying with x2 (`SCLFsmallCurv`);
- transverse pair momentum at conversion (paper Eq. A15, u_parallel ~ 1/sin
  psi);
- 3D.

## 2. Basic Configuration

- Engine SRPIC, metric Minkowski (Cartesian), 2D (`Dim::_2D` only); traits in
  `pgen.hpp` statically restrict dimension, engine, and metric.
- Production build contract (same as 1D): Esirkepov current deposition,
  third-order particle shapes (`-Ddeposit=esirkepov -Dshape_order=3`),
  `[output.fields.smoothing]` spline order 3.
- Numerical-dispersion control is configuration, not code: Entity's
  `[algorithms.fieldsolver]` implements the Blinne et al. (2018) modified
  stencil (coefficients `delta_x`, `delta_y`, `beta_xy`, `beta_yx`; all-zero
  is standard Yee). v1 TOML keeps Yee defaults; stencil coefficients and
  current-filter count are tuned in the empty-domain wave test before
  production (open item, Sec. 9).
- Species contract identical to 1D: 1 = electron, 2 = positron, 3 = photon
  (QED-enabled only), photon real payloads `[energy, tau_pair, theta_B]`.
- Normalization coefficients are migrated from the 1D baseline
  (`larmor0 = 4e-4`, `skindepth0 = 0.02`); production normalization against
  CPT2024 Table 1 (gamma_pc, gamma_rad, gamma_emit, rho_c, B/Bq) is a
  separate calibration step.

## 3. Initial Electromagnetic Fields

State: implemented (ported from 1D; Gauss normalization fixed 2026-08-24)

- Uniform `B_x1 = B0`.
- `E_x1(x1)` is the same analytic profile as 1D (logistic excess-positron
  profile minus fixed GJ background), transversely uniform. With
  `extra_positron_density = initial_e_coefficient` it satisfies the per-column
  Gauss closure `dE_x1/dx1 = rho_excess - rho_GJ`.
- Gauss-law normalization: with E in units of B0 and x in code lengths,
  `dE/dx = (larmor0/skindepth0^2) * rho` for `rho` in units of n0. The
  `InitialFields` constructor therefore receives
  `initial_e_coefficient * larmor0/skindepth0^2` as the field-side
  coefficient; the particle-side densities keep `initial_e_coefficient`
  (n0 units). The factor is 1 for the legacy normalization
  (`larmor0 = skindepth0^2`, satisfied by all pre-2026-08-24 TOMLs), so the
  fix leaves the smoke and trigger-test setups bit-identical; without it the
  initial E was wrong by `skindepth0^2/larmor0` for any other normalization.
- Transverse field components are initially zero.

## 4. Initial Particles

State: implemented (ported from 1D)

- Neutral atmosphere via `InjectNonUniform` (dimension-generic archetype),
  split equally between electrons and positrons, exponential scale height
  from `grid.boundaries.atmosphere.height`; transversely uniform.
- Excess positrons via the local `SingleSpeciesInjector`, generalized to the
  2D active-cell range (`operator()(i1, i2)`); same dilute-tail weight
  encoding (`minimum_density`, `minimum_ppc`) as 1D. Transversely uniform in
  v1.
- Both paths use the standard 2D3V Maxwellian; `particles.use_weights =
  true` as in 1D.

## 5. Boundaries

State: implemented (ported from 1D; x2 is new)

- x1 min: fields ATMOSPHERE, particles ATMOSPHERE (continuous neutral
  atmosphere replenishment).
- x1 max: fields MATCH, particles ABSORB.
- x2: PERIODIC fields and particles.
- ATMOSPHERE is allowed by Entity in at most one direction; here it is x1.
- `CustomPostStep` applies the per-column `Ex1` damping inside the right
  MATCH layer (2D generalization of the 1D `BoundaryExDamper`: the damping
  loop runs over transverse cells and damps the last
  `boundary_ex_damping.cells` active x1 faces in each column). The deprecated
  1D boundary-flux compensation is dropped entirely (the engine deposits
  clamped boundary particles before killing them).
- RS-style ("anti-pulsar", no surface supply) runs are configuration-only:
  particles ABSORB at x1 min, no atmosphere, sign-flipped `B0` /
  `external_current`; not exercised in v1.

## 6. Custom Behavior

State: implemented (ported from 1D; j_mag x2-profile is new)

- External magnetospheric current, added through Entity's Ampere hook as in
  1D (tetrad component normalized to `n0 q0 c`, converted with
  `J^1 = J^(hat1)/dx`). v1 supports two transverse profiles selected by
  `setup.polar_cap.current_profile`:
  - `"uniform"` (default): constant, identical to the 1D behavior;
  - `"philippov2024"`: CPT2024 Eq. 4,
    `j(x) = j_scale * (2 - C1 x^4 + C2 x^6 + C3 x^40)` with
    `x = (x2 - x2_center) / x2_halfwidth` clamped to `[-1, 1]`, and the
    constraint solution C1 = 100/3, C2 = 560/17, C3 = -82/51 (zero net
    current, j(1) = j'(1) = 0, j(0) = 2, super-GJ for |x| <~ 0.54).
    `j_scale` is `setup.polar_cap.external_current`.
- QED path, ported with the same three independent switches under
  `qed.enable`:
  - `curvature_drag`: continuous recoil `-gamma^3 u / rho_c^2` with
    `max_drag_fraction` cap (unchanged, already fully 3D in momentum);
  - `curvature_emission`: macro-photon sampling from the tabulated curvature
    CCDF (same `data/curvature_ccdf.tsv` and generator script); photons
    inherit the parent's 3D direction, so oblique emission works unchanged
    in 2D. `filter_nonconverting_photons` keeps its x1-based exit-angle
    estimate, which stays correct in 2D because the periodic x2 boundary
    never lets photons escape transversely;
  - `magnetic_pair_creation`: photon opacity integrated along the full
    trajectory with `theta_B` evolving only with the x1 displacement
    (`dtheta = dx1 / rho_c`) — this matches CPT2024's prescription, which
    accumulates the angle along the field-line direction and neglects
    transverse photon displacement. Conversion at
    `epsilon * |sin theta_B| >= 2` and `tau >= conversion_optical_depth`.
  - 2D pair-placement closure: pairs are placed at the photon's (i1, i2)
    cell and move along the local B axis (x1) in the photon's x1-propagation
    direction, each with half the photon energy — the direct lift of the 1D
    closure. Transverse momentum at conversion (paper Eq. A15) is deferred.
- `boundary_ex_damping`: see Sec. 5.

## 7. Custom Output

State: not-used

Standard field and species-moment output only (`N_1, N_2, N_3, E, B, J`).
Transverse coherence diagnostics (autocorrelation of E_x1 / density) are
post-processing on the 2D output, not PGen code.

## 8. PGen-TOML Contract

- Species indices, masses, charges, pushers, and emission policies as in 1D;
  QED-off runs must have exactly two species and must not request `N_3`.
- `grid.resolution`, `grid.extent`, and boundary lists are two-element
  (x1, x2); x2 boundaries must both be PERIODIC.
- All `setup.polar_cap.*` keys keep their 1D names and types; new keys:
  - `setup.polar_cap.current_profile` = `"uniform" | "philippov2024"`;
  - `setup.polar_cap.current_profile_center` (code units, default: domain
    x2 midpoint) and `setup.polar_cap.current_profile_halfwidth` (default:
    half the domain x2 extent) define the profile mapping.
- `extra_positron_density = initial_e_coefficient` for the Gauss closure.
- `external_current` is the interior tetrad-normalized current in units of
  `n0 q0 c`; for `"philippov2024"` it multiplies the dimensionless profile.
- GJ-unit scaling (derived 2026-08-24, anchored by the validated trigger
  test): at fixed `larmor0`, the Goldreich-Julian density in code units
  scales as `n_GJ/n0 = (skindepth0/5e-4)^2` (= 1 for the legacy
  normalization `larmor0 = 2.5e-7, skindepth0 = 5e-4`, which also fixes the
  factor-2 convention). All GJ-normalized inputs must be rescaled
  accordingly: `initial_e_coefficient`, `extra_positron_density`,
  `external_current` (`j_GJ = n_GJ q0 c`), and `grid.boundaries.atmosphere
  .density` (100 n_GJ in the validated setups). Conversely, the QED
  `emission_coefficient`/`pair_coefficient` scale linearly with
  `skindepth0` to keep the per-code-length photon emission rate and
  conversion opacity invariant (the runtime divides both by `skindepth0`),
  while `gamma_emit`, `gamma_rad`, `rho_c`, `b_over_bq`,
  `reference_electric_field` and the drag normalization
  (`dt * omegaB0 * E_ref / gamma_rad^4` with `omegaB0` fixed by `larmor0`)
  are unchanged. WARNING: `larmor0/skindepth0^2` (the
  `initial_e_coefficient` fallback formula) is the *inverse* of the true
  n_GJ scaling and equals it only at the legacy normalization; do not use
  it as a scaling law.
- `boundary_ex_damping` requires fields MATCH at x1 max and Cartesian
  coordinates, as in 1D, but is valid for the 2D build.
- QED validation rules (positive `rho_c`, `gamma_emit > 1`, even
  `opacity_substeps`, etc.) are unchanged.
- `x_surface` derives from `algorithms.current_filters + 2` vs 5 buffer
  cells, as in 1D.

## 9. Current Status

Completed:

- Design (this document).
- pgen.hpp + helpers + TOML implementation (see Sec. 10).
- `tests/reference_models.py`: philippov2024 profile and CCDF-table checks
  pass.
- CUDA build on m87 (`polar-cap-2d-v144-001`, ADA89, Esirkepov, shape order
  3, single precision, output ON) from immutable snapshot
  `source-2d-polar-cap-20260820-079c19ec`: compile and link pass.
- GPU smoke run on m87 (362 steps, QED-on switches but sub-threshold
  energies): all steps complete; 2D HDF5 output valid; `fE1` transversely
  uniform to ~5e-5 relative (particle noise) with the expected 1D-like
  longitudinal profile. The process aborts with "double free or corruption"
  during teardown after the final output — the same symptom exists in the
  previous 1D m87 runs (exit 134), so it is a pre-existing site/engine
  teardown issue, not introduced by this PGen.

Pending / open:

- QED event verification (photon emission/conversion) needs a run that
  reaches `gamma_emit`; the smoke grid does not.
- `boundary_ex_damping` 2D kernel compiled but not yet exercised.
- Column-wise 2D-vs-1D equivalence test (needs run infrastructure).
- Blinne stencil coefficients + current-filter count for 2D, to be fixed by
  an empty-domain electromagnetic-wave test before production.
- Normalization calibration against CPT2024 Table 1.

Deferred to later versions: see Sec. 1 (patches, full fill, rho_GJ gradient,
x2-dependent curvature, transverse pair momentum, 3D).

## 10. Important Changes

- 2026-08-20: initial design. New PGen forked conceptually from
  `pgens/1d_polar_cap` (branch `dev/2d-polar-cap`); the 1D case remains
  untouched. Drops the deprecated boundary-flux compensation; generalizes
  the MATCH-layer Ex damping to 2D; adds the x2-dependent
  `"philippov2024"` current profile.
- 2026-08-20: smoke TOML `output.format` changed `"disabled"` -> `"hdf5"`;
  `"disabled"` is rejected by output=ON builds (ADIOS2 engine error), hdf5
  works for both build modes. Smoke TOML also adds the QED keys
  `gamma_rad = 6.7e5` and `reference_electric_field = 2.0`, which the QED-on
  constructor requires but the 1D smoke TOML lacked.
- 2026-08-24: `InitialFields` Gauss normalization fix. The initial `E_x1`
  slope was `initial_e_coefficient` regardless of `larmor0/skindepth0`,
  which is only correct when `larmor0 = skindepth0^2` (all legacy TOMLs);
  the field-side coefficient is now `initial_e_coefficient *
  larmor0/skindepth0^2`. Legacy TOMLs are bit-identical; any other
  normalization previously got an initial E wrong by `skindepth0^2/larmor0`.
  Found when a rescaled production TOML (`skindepth0 = 10x` legacy) showed
  an initial E 700x weaker than the trigger test and no cascade ignition.
- 2026-08-24: added `qed_cascade_test.toml` (20000x16 j=1.5 trigger test,
  validated on lambda H100: photons 0 -> 2.4e7 quasi-steady, pair spectra
  cut off at `gamma_rad = 1.6e5`) and `qed_cascade_jmag_production.toml`
  (full-cap philippov2024 run; see Sec. 8 for the GJ scaling law used to
  set its densities, current, and QED coefficients).
