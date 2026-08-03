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

The simulation starts with no particles. `InitPrtls` is intentionally a no-op; all electron-positron pairs are created dynamically by the local field trigger described below. The injected momenta are drawn from a zero-drift Maxwellian with `setup.temperature`.

## 5. Boundaries

Status: implemented

GRPIC supplies the horizon and polar-axis treatment. The outer field boundary is `MATCH`, so it reuses the split-monopole `init_flds` over a layer with `grid.boundaries.match.ds = 1.0`. The outer particle boundary is `ABSORB` with `grid.boundaries.absorb.ds = 1.0`.

## 6. Custom Behavior

Status: implemented

After every step, cells inside `xi_min`/`xi_max` (`r = [1.2, 10.0]`, `\theta \in [0.01, \pi - 0.01]`) are tested using local EM-array values at one grid index. No staggered-field interpolation or tetrad conversion is performed. The code lowers the magnetic field directly with the spatial metric,

\[
B_i=\gamma_{ij}B^j,
\qquad
D\cdot B=D^iB_i,
\qquad
B^2=B^iB_i,
\]

and activates injection when

\[
\frac{|D\cdot B|}{B^2}>\epsilon_{DB},
\qquad
B^2>f_{\sigma}\,\widetilde\rho.
\]

Here `FldsID::Rho` is computed from both massive species and is already normalized by Entity's `1/n0` moment coefficient. Consequently `\widetilde\rho` must not be divided by `scales.n0` again. The second condition is the normalized form of the Parfrey et al. magnetization guard `sigma > sigma0/20` when `f_sigma = 0.05`.

Every triggered cell receives exactly one electron macro-particle and one positron macro-particle. Following the Parfrey et al. local prescription, each member of the pair represents the normalized density

\[
\Delta\widetilde n
=R\,\widetilde n_{\rm GJ}\,
  \frac{|D\cdot B|}{\sqrt{B^2}},
\qquad
\widetilde n_{\rm GJ}=B_0 d_0^2
=\texttt{scales.B0}\,\texttt{scales.skindepth0}^2.
\]

`InjectNonUniform` is called with `number_density = 2 / ppc0`, which makes its internal count exactly one pair per accepted cell. The spatial distribution returns

\[
\{1,w\},\qquad w=\mathrm{ppc0}\,\Delta\widetilde n.
\]

For either unit-mass species, Entity's density moment multiplies the stored particle weight by `1/n0/sqrt(det(h))`; the non-Cartesian injector stores an additional `sqrt(det(h))/V0`, and `n0 = ppc0/V0`. Thus each particle deposits exactly \(w/\mathrm{ppc0}=\Delta\widetilde n\), while the pair deposits \(2\Delta\widetilde n\). This is why `ppc0` controls sampling capacity but does not change the physical injected density. Non-finite values and cells with vanishing or non-positive `B^2` are rejected. No external current, external force, field drive, custom particle update, or imposed relaxation time is used.

## 7. Custom Output

Status: not-used

No custom output hook is defined. TOML requests built-in `D`, `B`, `J`, `N_1`, `N_2`, and `A`; particle and spectrum output are disabled.

## 8. PGen-TOML Contract

| Contract | TOML key |
|---|---|
| GRPIC / 2D / QKerr-Schild traits | `simulation.engine`, grid shape, `grid.metric.metric` |
| Split-monopole amplitude | `setup.flux0` |
| Curl difference step, not sheet width | `setup.m_eps` |
| Initial particles | none; `InitPrtls` is a no-op |
| Dynamic injection region | `setup.xi_min`, `setup.xi_max` |
| Pair-creation rate | `setup.pair_creation_rate` |
| Parallel-field trigger | `setup.ddotb_threshold` |
| Magnetization guard fraction | `setup.sigma_min_fraction` |
| Injected Maxwellian temperature | `setup.temperature` |
| Fixed macro-particle count | one electron-positron pair per triggered cell per step |
| Electron then positron species indices | first and second `particles.species` tables |
| Matched field / absorbed particles and layer widths | `grid.boundaries`, `grid.boundaries.match.ds`, `grid.boundaries.absorb.ds` |

`flux0`, `m_eps`, `pair_creation_rate`, and `ddotb_threshold` must be positive; `sigma_min_fraction` must be non-negative. The coordinate vectors have two entries, species order is fixed, and weighted injection requires `particles.use_weights = true`. With `grid.extent = [1, 15]` and `grid.boundaries.absorb.ds = 1`, `xi_max[0] = 10` stays well inside the largest allowed radial injection edge of 14, and `xi_min[1] = 0.01`/`xi_max[1] = \pi - 0.01` keep injection away from the polar axis.

## 9. Current Status

- Implemented: canonical PGen, TOML, split-monopole field, vacuum particle initialization, local \(D\cdot B/B^2\)-triggered fixed-pair weighted injection, and boundaries.
- Statically verified: direct coordinate-basis contractions, one-pair count, weight normalization, metric-volume cancellation, density normalization, and the PGen-TOML-design contract.
- Locally verified: Entity v1.4.4 CPU/Serial, single precision, MPI/OFF, output/OFF, debug build compiled successfully. A 16x8 two-step smoke run from vacuum completed normally; the electron and positron counts remained paired and grew from 56 each after step 0 to 121 each after step 1, demonstrating that the local trigger and fixed-pair injection path execute.
- Not verified: production V100 compilation, production-scale numerical stability, relaxed-state properties, or reconnection physics.

## 10. Important Changes

- Replaced the `accretion` Wald-like field with the unsmoothed split-monopole potential.
- Added `setup.flux0`; added no sheet-width parameter.
- Restricted the PGen trait to the QKerr-Schild metric used by this case.
- Replaced the radial density floor with a local `abs(D dot B) / B^2` trigger plus a total-mass-density magnetization guard.
- Switched to vacuum initialization and retained `[1.2, 10.0]` as the dynamic injection region.
- Fixed the macro-particle count at one electron-positron pair per triggered cell per step and encoded the local Parfrey density in particle weight.
- Used direct coordinate-basis metric contractions and local EM-array grid values without tetrad conversion or staggered interpolation.
- Set both the outer MATCH field layer and ABSORB particle layer widths to `ds = 1.0`.
- Kept the 2048x1024 production geometry and scales unchanged while setting `pair_creation_rate = 0.5`, `ddotb_threshold = 1e-2`, `sigma_min_fraction = 0.05`, `temperature = 0.1`, and injection box `xi_min = [1.2, 1e-2]` / `xi_max = [10.0, 3.14159264]`.
