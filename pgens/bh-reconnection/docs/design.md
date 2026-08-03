# BH Reconnection PGen Design

## 1. Goal

Implement a first runnable 2D axisymmetric GRPIC case for equatorial magnetic reconnection in a Kerr magnetosphere. The initial field is an unsmoothed split monopole; no sheet-width parameter or prescribed equilibrium current is introduced.

## 2. Basic Configuration

Status: implemented

- Entity checkout: v1.4.4 at `b1018183bc6a334293ad6dc8a0834241834bdc8b`
- Engine/dimension/metric: GRPIC, 2D, `qkerr_schild`
- Basis: GR contravariant coordinate components on the staggered mesh
- Requirements beyond the standard Entity build: none

## 3. Initial Electromagnetic Fields

Status: statically verified

The only nonzero potential component is

\[
A_\phi(\theta)=A_*\left(1-|\cos\theta|\right),
\qquad |\Phi_{\rm hemi}|=2\pi A_*.
\]

`A_3()` converts the QKerr code coordinate to physical \(\theta\) before evaluating this expression. The staggered finite-difference curl gives opposite radial polarity across the equator and \(B^{x^2}=0\) away from the cusp. The polar value uses the metric limit through `sqrt_det_h_tilde`; it is not hard-coded. Initially \(B^{x^3}=0\) and \(D^i=0\).

TOML parameters: `setup.flux0 = A_*` and `setup.m_eps`, the code-coordinate curl step. `m_eps` is not a current-sheet thickness.

## 4. Initial Particles

Status: implemented

Zero-drift Maxwellian electron-positron pairs (species 1 and 2) initialize every active cell across the full grid domain with the radial density profile

\[
\frac{n_{\rm floor}(r)}{n_0}
= n_{\rm floor,ref}\left(\frac{r_{\rm ref}}{r}\right)^{3/2},
\qquad
n_{\rm floor,ref}
= \texttt{density\_mult}\,\frac{B_0 d_0^2}{n_0}.
\]

The measured density is the total pair number density `N_1 + N_2`, not the cancelling net charge density. Each deficient cell receives exactly `injection_pairs_per_cell` electron-positron macro-particle pairs. For \(N_{\rm pair}\) fixed pairs, the density represented at unit weight is

\[
n_{\rm inj,unit}=\frac{2N_{\rm pair}}{\mathrm{ppc0}},
\qquad
w_{\rm deficit}=\frac{n_{\rm floor}-n}{n_{\rm inj,unit}},
\]

before Entity applies the metric volume factor. Therefore the injected weighted density is \(n_{\rm inj,unit}w_{\rm deficit}=n_{\rm floor}-n\), independent of `ppc0`. `temperature` sets the Maxwellian temperature.

At initialization the injector receives no explicit box, so Entity selects all active cells (no ghost cells), spanning the configured radial extent `r = [1, 15]` and the full polar range. Because the initial density is zero, each cell receives one electron-positron macro-particle pair whose weight represents the local \(r^{-3/2}\) target. The configured `xi_min`/`xi_max` box is used only for subsequent density-floor replenishment.

## 5. Boundaries

Status: implemented

GRPIC supplies the horizon and polar-axis treatment. The outer field boundary is `MATCH`, so it reuses the split-monopole `init_flds` over a layer with `grid.boundaries.match.ds = 1.0`. The outer particle boundary is `ABSORB` with `grid.boundaries.absorb.ds = 1.0`.

## 6. Custom Behavior

Status: implemented

`CustomPostStep` recomputes `N_1 + N_2` and injects only cells below 90% of the target floor inside `xi_min`/`xi_max` (`r = [1.2, 10.0]` and the polar range `\theta \in [0.01, \pi - 0.01]`). `ReplenishFixedPairs` returns `{1, w_deficit}`: the first component fixes the macro-particle pair count, while the second puts the entire density deficit into particle weight. No external current, external force, field drive, custom particle update, or imposed relaxation time is used.

## 7. Custom Output

Status: not-used

No custom output hook is defined. TOML requests built-in `D`, `B`, `J`, `N_1`, `N_2`, and `A`; particle and spectrum output are disabled.

## 8. PGen-TOML Contract

| Contract | TOML key |
|---|---|
| GRPIC / 2D / QKerr-Schild traits | `simulation.engine`, grid shape, `grid.metric.metric` |
| Split-monopole amplitude | `setup.flux0` |
| Curl difference step, not sheet width | `setup.m_eps` |
| Initial pair region | all active cells from `grid.extent` |
| Replenishment region | `setup.xi_min`, `setup.xi_max` |
| Pair density floor and temperature | `setup.density_mult`, `setup.r_ref`, `setup.temperature` |
| Fixed macro-particle pairs per deficient cell | `setup.injection_pairs_per_cell` |
| Electron then positron species indices | first and second `particles.species` tables |
| Matched field / absorbed particles and layer widths | `grid.boundaries`, `grid.boundaries.match.ds`, `grid.boundaries.absorb.ds` |

`flux0`, `m_eps`, `density_mult`, and `r_ref` must be positive; `injection_pairs_per_cell` must be a positive integer. The coordinate vectors have two entries, species order is fixed, and weighted injection requires `particles.use_weights = true`. With `grid.extent = [1, 15]` and `grid.boundaries.absorb.ds = 1`, `xi_max[0] = 10` stays well inside the largest allowed radial injection edge of 14, and `xi_min[1] = 0.01`/`xi_max[1] = \pi - 0.01` keep replenishment away from the polar axis.

## 9. Current Status

- Implemented: canonical PGen, TOML, split-monopole field, full-domain initial \(r^{-3/2}\) pair profile, fixed-pair weighted density-floor replenishment, and boundaries.
- Statically verified: full-active-domain selection, fixed pair count, adaptive weight normalization, metric-volume cancellation, and the PGen-TOML-design contract; earlier fixed-weight evidence is recorded in `verification/pgen-fixed-pair-weight-2026-07-16.md`.
- Pending owner: remote compile/smoke verification through the Router if requested.
- Not verified: the revised injection behavior at runtime, numerical stability, relaxed-state properties, or reconnection physics.

## 10. Important Changes

- Replaced the `accretion` Wald-like field with the unsmoothed split-monopole potential.
- Added `setup.flux0`; added no sheet-width parameter.
- Restricted the PGen trait to the QKerr-Schild metric used by this case.
- Replaced the magnetization/net-charge trigger with a radial `N_1 + N_2` density floor and expanded the radial injection box to `[1.2, 10.0]`.
- Decoupled injected macro-particle count from density by fixing `injection_pairs_per_cell` and encoding the entire deficit in particle weight.
- Expanded only the initial injection to every active cell; retained `[1.2, 10.0]` as the subsequent replenishment region.
- Set both the outer MATCH field layer and ABSORB particle layer widths to `ds = 1.0`.
- Rescaled the case for the 2048x1024 run: `runtime = 100.0`, `resolution = [2048, 1024]`, `extent = [1, 15]`, `larmor0 = 1e-4`, `skindepth0 = 1e-3`, `current_filters = 16`, `maxnpart = 4e8` per species, `density_mult = 100.0`, `temperature = 0.1`, and replenishment box `xi_min = [1.2, 1e-2]` / `xi_max = [10.0, 3.14159264]`.
