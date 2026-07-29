# 1D polar-cap

This case implements a local one-dimensional pulsar polar-cap discharge on
Entity v1.4.4. The first implementation is deliberately scoped to SRPIC,
Minkowski coordinates, and a constant magnetic-field curvature radius.

## Species and payloads

Species indices are fixed by the case contract:

1. electron;
2. positron;
3. photon.

The photon species has three real payloads:

```text
pld_r[:, 0] = photon energy
pld_r[:, 1] = accumulated magnetic pair-production optical depth
pld_r[:, 2] = angle between the photon and the local magnetic field
```

Initial and replenished atmosphere particles use Entity's standard 1D3V
Maxwellian. A new photon's initial `theta_B` is computed from its full emission
direction relative to the `x1` magnetic field; it is not assumed to be zero.
Opacity is integrated over the photon's full `dt` trajectory, while field-line
curvature changes `theta_B` according to the photon's `x1` displacement.

The excess-positron initial injector has explicit dilute-tail controls. Cells
below `initial_injection.minimum_density` receive no particles. When the target
particles per cell are below `initial_injection.minimum_ppc`, the injector
creates exactly `minimum_ppc` macro-particles and assigns each the weight
`target_ppc / minimum_ppc`. This avoids random zero-particle cells while
preserving the represented density. Particle positions and Maxwellian momenta
remain sampled; only the low-density particle count is deterministic.

SRPIC current deposition and all QED descendants use the stored particle
weight. This branch also makes Cartesian `ATMOSPHERE` injection follow the
global `particles.use_weights` setting, which is enabled by this case. The
atmosphere density feedback and standard `N_*` output moments therefore use
the same weighted-density convention as the dilute-tail injector.

## Step ordering

The case uses Entity's standard SRPIC step. Electron and positron pushers apply
the custom curvature-emission policy, then the photon pusher advances all
photons and applies the opacity update. `CustomPostStep` converts eligible
photons into pairs. New pairs start contributing to the pusher and current
deposit on the following timestep.

No global Entity pusher, injector, enum, or engine behavior is modified.

## Initial-state closure

For `x >= x_surface`, the migrated field and excess-positron profiles obey

```text
dE_x/dx = rho_excess - rho_GJ
```

when `extra_positron_density = initial_e_coefficient`, with the prescribed
Goldreich-Julian background `rho_GJ = initial_e_coefficient`. The checked-in
defaults satisfy this relation. `rho_GJ` is a fixed background in the physical
interpretation, not an Entity particle species; changing either coefficient
independently intentionally breaks this initial continuous Gauss closure.

The excess-positron profile is a unit plateau through
`x_surface + grid.boundaries.atmosphere.ds`. That atmosphere edge is the start,
not the midpoint, of its S-shaped decline. The profile is explicitly zero from
`x_surface + 1.33 ds` onward. The neutral atmosphere scale height is independent
of this excess component: for total neutral surface density `n0` and requested
neutral edge density `n_edge`, use
`height = ds / log(n0 / n_edge)`.

## Radiation model

The charged species use `emission = "custom"`. Entity v1.4.4 also provides
built-in synchrotron drag and emission, but those depend on the local
electromagnetic acceleration. They vanish for ideal motion parallel to the
field and do not contain the prescribed field-line curvature radius `rho_c`.
This case therefore implements curvature radiation separately and does not
enable `radiative_drag = "synchrotron"`.

The QED path has independent switches:

- `curvature_drag` applies continuous curvature-radiation recoil;
- `curvature_emission` creates statistically sampled macro-photons;
- `magnetic_pair_creation` evolves photon opacity and converts eligible
  photons.

The baseline TOML sets the top-level `enable` switch to `false` until the
normalization is fixed. This supports QED-off,
drag-only, emission-only, and full-cascade reference configurations without
changing the species contract.

Continuous recoil is not tied to an individual sampled macro-photon. It
represents the mean radiative loss, while macro-photons sample the associated
radiation field. The sampled spectrum is truncated so that an individual
photon cannot exceed its parent particle's kinetic energy. The ensemble energy
budget must still be checked after the normalization parameters are fixed; the
current migration defaults are not an event-by-event energy-conserving Monte
Carlo prescription.

For pair-cascade runs,
`filter_nonconverting_photons = true` additionally removes the part of the
curvature-number spectrum that cannot satisfy
`epsilon_gamma * abs(sin(theta_B)) >= 2` anywhere before the photon reaches an
absorbing `x1` boundary. The cutoff is computed from the exact maximum of
`abs(sin(theta_B))` along the remaining path, so it does not remove any photon
that can acquire nonzero magnetic pair opacity. It does remove per-particle
records of low-energy escaping radiation; continuous curvature recoil remains
unchanged.

## Curvature spectrum

`data/curvature_ccdf.tsv` stores the normalized complementary CDF of the
curvature-radiation photon-number spectrum,

```text
C(x) = [3 / (5 pi)] integral_x^infinity (t - x) K_(5/3)(t) dt.
```

The normalization integral is `5 pi / 3`. Regenerate the table with:

```bash
python3 pgens/1d_polar_cap/generate_curvature_table.py
```

The runtime table reader checks positivity and monotonicity. It first resolves
the configured path from the launch directory, then from the PGen directory.
Inverse sampling uses log-log interpolation inside the table, the analytic
small-argument approximation, and a log-linear continuation of the
exponentially falling upper tail. There is only one tabulated source of truth.

Reference-level checks that do not compile Entity can be run with:

```bash
python3 pgens/1d_polar_cap/tests/reference_models.py
```

The production build uses Esirkepov deposition with third-order particle
shapes. `[output.fields.smoothing]` therefore uses the spline method with
`order = 3`, keeping output particle moments consistent with the compiled
deposition shape.

## QED coefficients

The current TOML preserves the structure of the old `dev/1dsr` prototype while
making every coefficient explicit:

- `emission_coefficient` controls the curvature photon number rate before the
  timestep, skin-depth, and `5 pi / 3` factors are applied;
- `pair_coefficient` controls the magnetic conversion opacity before the
  `0.23 pi sqrt(3) (B/B_Q) / skindepth0` factor;
- `b_over_bq` is a positive magnetic-field magnitude and is independent of the
  sign of `setup.polar_cap.B0`;
- `rho_c` is a positive curvature radius in code length units;
- `external_current` is the interior tetrad current normalized to `n0 q0 c`.
  Entity's Ampere kernel adds external current directly to the deposited
  contravariant current, so this 1D Minkowski PGen converts
  `J^(hat 1) -> J^1` by dividing by `scales.dx0`. Across the right MATCH layer
  it multiplies this value by `tanh(4 (x_max - x) / match_ds)`, matching the
  electric-field boundary profile and reaching zero at the outer edge. The
  kernel's `ppc0` factor cancels against the `ppc0` dependence of `scales.q0`;
  it is independent of this coordinate-basis conversion.

The numerical defaults are migration baselines, not a validated pulsar model.
The physical conversion from a chosen pulsar field, curvature radius, and
fiducial Entity scales must be fixed before production runs.

The checked-in coefficient `2.6926062e7` is reconstructed from the old
`QED_process.cpp` reference values `gamma_emit=3e4`, `gamma_rad=6.7e5`, and
`gamma_pc=7.2e7` using

```text
(3/2)^2 sqrt(3/2) / pi sqrt(gamma_pc)
  * [(gamma_emit/gamma_rad)^2 gamma_emit]^2.
```

## Pair conversion model

The first implementation retains the prototype's deterministic conversion
condition: a photon converts when both
`epsilon_gamma * abs(sin(theta_B)) >= 2` and
`tau >= conversion_optical_depth`. The generated electron and positron each
receive half the photon energy and move in the photon's parallel direction.
This gives an explicit energy-conserving 1D closure while leaving transverse
Landau-level physics outside the first implementation.

Converted and boundary-absorbed photons are compacted after pair conversion
every `photon_recycle_interval` completed steps. QED-on inputs that enable this
path set the photon species' standard `clear_interval = 0`, avoiding redundant
pre-conversion cleanup. The interval is required only when magnetic pair
creation is enabled.
