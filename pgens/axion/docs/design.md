# Axion PGen Design

## 1. Goal

Model a prescribed travelling axion background coupled to a 2D pair plasma in
SRPIC. The current build target is a double-precision CUDA/MPI executable for
V100 GPUs. The scalar field is prescribed rather than dynamically evolved.

## 2. Basic Configuration

- Entity: v1.4.3 checkout, PGen `axion`
- Engine/metric/dimension: SRPIC, Minkowski, 2D Cartesian
- Boundaries: periodic for fields and particles in both dimensions
- Build intent: double precision, CUDA `VOLTA70`, MPI enabled, output enabled
- Active build-baseline TOML: `../2d_bx2_cowan_N2048.toml`

## 3. Initial Electromagnetic Fields

Status: implemented.

The prescribed phase is `a = cos(k x1 - omega t)`. Initial fields are uniform
`Bx1=B0`, `Bx2=B0_perp`, `Bx3=0`, with
`Ex1=-epsilon B0 cos(k x1)` and other electric components zero. `InitFields`
returns code-normalized fields directly, without an additional `larmor0`
factor.

## 4. Initial Particles

Status: implemented.

When `setup.density > 0`, initialize neutral electron-positron species 1 and 2
with Maxwellian temperature `setup.temperature`. An optional x1 density
profile (`setup.profile` = `uniform`/`ramp`/`barrier`, with `n_min`, `n_max`,
`ramp_W`, `ramp_xa`, `ramp_xb`) varies the injected ppc with position while
keeping particle weights constant, so N_D = ppc·(lambda_D/dx)^2 stays uniform
along the gradient. `profile = "uniform"` keeps the pre-profile code path
byte-identical. No particles are loaded for non-positive density. Dynamic
pair injection is not used.

## 5. Boundaries

Status: configured.

The matching TOML uses `PERIODIC` field and particle boundaries in both
dimensions. No custom boundary hook is present.

## 6. Custom Behavior

Status: implemented.

`ext_current` injects the full
`J_a = epsilon * (partial_t(a) * B + grad(a) x E)`, using the evolved
electromagnetic field. The returned current includes the required
`skindepth0^2/larmor0` pre-compensation. In 2D the stored basis is
`[ex1, ex2, bx1, bx2] = physical/dx`, `[ex3, bx3] = physical`; the
`grad(a) x E` term crosses indices and carries an explicit `ctx.dx`
correction (`jx2 += ... ex3 / dx`, `jx3 -= ... ex2 * dx`; 2D-specific,
rederive for 3D). The term can be disabled with
`setup.use_grad_a_cross_e = false` for ON/OFF discrimination runs.
Verification: `test_gradaE_on/off.toml` (vacuum, uniform `E0_x3` seed)
against the analytic forced wave `max<E2^2> = 2*(eps*k*E0/omega)^2`.

## 7. Custom Output

Status: not-used.

The TOML requests standard Entity field quantities; the PGen defines no custom
field or statistics output.

## 8. PGen-TOML Contract

- Required setup keys: `epsilon`, `omega_ratio`, `k`.
- Optional/defaulted setup keys: `B0`, `B0_perp`, `temperature`, `density`,
  `use_grad_a_cross_e` (default true), `E0_x3` (default 0),
  `profile` (default "uniform"), `n_min`/`n_max` (default 1),
  `ramp_W`/`ramp_xa`/`ramp_xb`.
- `omega = omega_ratio/skindepth0` and must satisfy `omega*dt < 1`.
- `particles.use_weights` must be true and species 1/2 must be electron and
  positron entries compatible with the initialization order.
- `2d_bx2_cowan_N2048.toml` currently selects a 2048x2048 pair-plasma baseline
  with `density=1`, `B0_perp=0`, and the Cowan 2D field-solver stencil.
- Test TOMLs: `test_gradaE_on.toml` / `test_gradaE_off.toml` (vacuum ∇a×E
  forced wave ON/OFF), `test_profile_barrier.toml` (density profile smoke).

## 9. Current Status

- PGen and baseline TOML are consistent for the current double build.
- Build verification on pi2 is pending.
- Higher-resolution run parameters and Slurm resource counts remain pending
  user confirmation; no job may be submitted from this build step.

## 10. Important Changes

- Dynamic pair injection was removed; initial density now controls plasma
  loading.
- Initial `Ex1` no longer divides by `larmor0`.
- External current reads the evolved magnetic field and retains the Ampere
  normalization compensation.
- `grad(a) x E` term added to `ext_current` (2026-09-01, re-implementing the
  H100-campaign PGen from `lambda/axion-plasma-coupling-report-v2.md`),
  with 2D dx-basis correction and a `use_grad_a_cross_e` switch. This
  changes the physics of existing TOMLs (default ON).
- Density profiles along x1 (`setup.profile`) added the same day; `uniform`
  keeps the old code path.
- Optional uniform `E0_x3` seed added to `InitFields` for the vacuum
  forced-wave test.
