# Session Summary — Axion-PIC Normalization Fix & Plasma Tests

## Key Achievement: Fixed larmor0 Normalization Bug

### Root Cause
Entity's Ampere kernel has `coeff = -dt * larmor0 / (ppc0 * skindepth0²)` — the `larmor0` factor (from `B0 = 1/larmor0`) is baked into the engine and cannot be changed from pgen.

### Fix Applied to pgen.hpp

**InitFields::ex1** — REMOVED `/larmor0`:
```cpp
// Before (WRONG):
return -epsilon * b_para * math::cos(k * x_Ph[0]) / larmor0;
// After (CORRECT):
return -epsilon * b_para * math::cos(k * x_Ph[0]);
```

**AxionExternalCurrent** — KEPT `coef = skindepth0²/larmor0` (this is CORRECT and necessary):
```cpp
coef = SQR(skindepth0) / larmor0  // compensates Ampere's larmor0
```

### Why coef is needed
- Ampere dE/dt = `-larmor0/skindepth0² * j_ext`
- Traveling wave requires dE/dt = `-ε·ω·B·sin(kx-ωt)`
- Therefore j_ext must have `skindepth0²/larmor0` factor

### Verified with two vacuum test cases:
| Case | skindepth0 | larmor0 | coef | ω_a | Ex ✓ | Jx ✓ | DivE ✓ |
|------|-----------|---------|------|-----|------|------|--------|
| skin=1.0 | 1.0 | 0.01 | 100 | 0.01 | ✓ | ✓ | ✓ |
| skin=0.5 | 0.5 | 0.01 | 25 | 0.02 | ✓ | ✓ | ✓ |

---

## Analysis Script Normalization (for video scripts)

| Quantity | Formula | Needs NORM? | Reason |
|----------|---------|-------------|--------|
| E_a (axion E-field) | `-ε·B·cos(kx-ωt)` | No | E is set directly, matches sim Ex |
| J_a (axion current) | `ε·ω·B·sin(kx-ωt) * NORM` | **Yes** | J goes through Ampere normalization |
| ρ_a (axion charge) | `ε·k·B·sin(kx-ωt) * NORM` | **Yes** | Matches sim Charge normalization |
| Charge (sim output) | As-is from sim | No | Already in same units as ρ_a*NORM |
| DivE (sim output) | As-is from sim | No | = Ex(i)-Ex(i-1), direct derivative |

NORM = skindepth0² / larmor0

---

## Energy Normalization (for video scripts)

Physical units (Lorentz-Heaviside, c=1):
- `E_phys = ∫(Ex²+Ey²+Ez²)/2 dx * B0²` where B0 = 1/larmor0
- T00 deposition per particle already includes `weight = density/ppc0` and `inv_n0`
- `particle_phys = ∫Ttt dx` — resolution-independent, no PPC/dx factor needed
  - Verified: n=16392 and n=3072 give identical ∫Ttt dx at t=0 (191.3 in both cases)
  - Old formula `∫Ttt dx * PPC/dx` was WRONG — it counted total particle count, not total mass
- Subtract initial values to show energy GAIN

---

## Current Simulation Parameters

### plasma_larmor1e-2.toml (latest runs)
```
n16392 run:
  name = "plasma_larmor1e-2_or0.5_n16392"
  resolution = [16392], runtime = 1000.0
n3072 run:
  name = "plasma_larmor1e-2_or0.5_t2000"
  resolution = [3072], runtime = 2000.0

Common:
  larmor0 = 0.01, skindepth0 = 0.5
  omega_ratio = 0.5 → ω_a = 1.0, ω_p = 2.0
  epsilon = 0.01, k = 0.1, B0 = 1.0, theta = 0
  temperature = 0.01, density = 1.0, ppc0 = 128
  current_filters = 16
  interval_time = 0.5
  output smoothing: order=3, method=spline
```

### Build configuration on m87
- SHAPE_ORDER=3 (compile-time, Esirkepov deposition)
- CUDA, GPU: RTX 4070 Ti (ADA89)
- Kokkos 5.1.0, ADIOS2 2.11.0, GCC 12, CUDA 13.0
- Binary: `~/entity/problems/axion/build/axion/src/entity_axion.xc`

---

## Video Scripts on m87

All scripts accept CASE path as first argument:

1. **make_plasma_larmor1e-2_video.py** — 4-panel (Ex, Jx, Charge, Energy)
   - Hardcoded: RUNTIME=1000, STRIDE=4
   - Charge uses raw data (no smoothing)
   - Energy panel: E_phys = ∫E²/2 dx × B₀², par_phys = ∫Ttt dx (no PPC/dx)
   - Fixed np.trapz→np.trapezoid for NumPy 2.x

2. **make_vacuum_larmor1e-2_video.py** — 4-panel (Ex, Jx, DivE, E² energy)
   - Configurable SKIN/LARMOR

3. **make_energy_video.py** — Single panel (particle vs E-field energy gain)
   - Uses σ=(SKIN/LARMOR)² scaling for E-field
   - Subtracts initial values
   - Takes RUNTIME and STRIDE as args 2,3
   - NOTE: par_phys formula here may also need fix (PPC/dx issue)

---

## Analysis Results Location
```
analysis_results/
├── plasma_larmor1e-2_or0.5_t2000.mp4   — 4-panel, t=0~2000
├── energy_t2000.mp4                     — energy evolution, t=0~2000
├── vacuum_larmor1e-2.mp4               — vacuum skin=1.0
├── vacuum_larmor1e-2_skin0.1.mp4       — vacuum skin=0.1
├── vacuum_larmor1e-2_skin0.5.mp4       — vacuum skin=0.5
├── plasma_larmor1e-2_or0.5.mp4         — plasma t=100
├── plasma_larmor1e-2_or0.5_t200.mp4    — plasma t=200
└── plasma_larmor1e-2_or0.5_t500.mp4    — plasma t=500
```

---

## Known Issues
1. Entity exits with "double free or corruption" on cleanup — data is complete, just a destructor bug
2. Binary is `entity_axion.xc` in `build/axion/src/`, NOT `entity.xc`
3. TOML files must be accessed from git repo path: `~/entity/1.4.3/pgens/axion/`
4. Output goes to CWD: must `cd ~/entity/problems/axion` before running

## Workflow (from local → m87)
```bash
# 1. Edit files locally in code/entity-1.4.3/pgens/axion/
# 2. Commit and push
git -C code/entity-1.4.3 add pgens/axion/... && git -C code/entity-1.4.3 commit -m "..." && git -C code/entity-1.4.3 push origin pgen/axion
# 3. Pull on m87, compile if needed, run
ssh m87 "cd ~/entity/1.4.3 && git pull origin pgen/axion && source ~/entity/problems/axion/_build/env.sh && cd ~/entity/problems/axion/build && cmake --build . -j \$(nproc)"
ssh m87 "source ~/entity/problems/axion/_build/env.sh && cd ~/entity/problems/axion && nohup ~/entity/problems/axion/build/axion/src/entity_axion.xc -input ~/entity/1.4.3/pgens/axion/<toml> > <log> 2>&1 &"
```
