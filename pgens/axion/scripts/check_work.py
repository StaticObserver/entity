import nt2, numpy as np

data = nt2.Data('/home/staticobserver/entity/problems/axion/plasma_larmor1e-2_or0.5_n16392')
fields = data.fields
x = fields.x.values
Ex_all = fields['Ex'].sel(t=slice(None, 1000.0)).values
t = fields.t.sel(t=slice(None, 1000.0)).values.ravel()
dt = t[1] - t[0]

EPS, OMEGA_RATIO, K, B0 = 0.01, 0.5, 0.1, 1.0
SKIN, LARMOR = 0.5, 0.01
OMEGA = OMEGA_RATIO / SKIN
NORM = SKIN**2 / LARMOR
B0_PHYS = 1.0 / LARMOR

# Ja_all = theory with NORM factor (as plotted in Jx panel)
Ja_all = EPS * OMEGA * B0 * np.sin(K * x[None, :] - OMEGA * t[:, None]) * NORM
# Ja_theory_raw = without NORM
Ja_raw = EPS * OMEGA * B0 * np.sin(K * x[None, :] - OMEGA * t[:, None])

# spatial integral of Ex * Ja
P_with_norm = np.trapezoid(Ex_all * Ja_all, x, axis=1)  # Ex * Ja_all
P_from_raw = np.trapezoid(Ex_all * Ja_raw, x, axis=1)  # Ex * Ja_raw

# Physical power: Ex_phys = Ex * 1/larmor, Ja_phys = Ja_raw * skindepth0²/larmor
# P_phys = (skindepth0²/larmor²) * integral(Ex * Ja_raw)
P_phys = (SKIN**2 / LARMOR**2) * P_from_raw

# Alternative: P_phys = (1/larmor) * integral(Ex * Ja_all) since Ja_all=Ja_raw*NORM
P_phys_alt = B0_PHYS * P_with_norm

print(f'P_phys (from raw): mean={np.mean(P_phys):.4f}, std={np.std(P_phys):.4f}')
print(f'P_phys (alt calc): mean={np.mean(P_phys_alt):.4f}, std={np.std(P_phys_alt):.4f}')
print(f'Match: {np.allclose(P_phys, P_phys_alt)}')

# Cumulative work
W_phys = np.cumsum(P_phys) * dt

print(f'\nW_phys[t=0]: {W_phys[0]:.4f}')
print(f'W_phys[t=500]: {W_phys[len(t)//2]:.4f}')
print(f'W_phys[t=1000]: {W_phys[-1]:.4f}')

# Compare to other energies
Ttt = fields['Ttt'].sel(t=slice(None, 1000.0)).values
Ey = fields['Ey'].sel(t=slice(None, 1000.0)).values
Ez = fields['Ez'].sel(t=slice(None, 1000.0)).values
E_phys = np.trapezoid(Ex_all**2 + Ey**2 + Ez**2, x, axis=1) * 0.5 * B0_PHYS**2
par_phys = np.trapezoid(Ttt, x, axis=1)
dPar = par_phys - par_phys[0]

print(f'\nComparison at t=1000:')
print(f'  E_phys: {E_phys[-1]:.1f}')
print(f'  dPar: {dPar[-1]:.1f}')
print(f'  dPar+dE: {dPar[-1] + (E_phys[-1]-E_phys[0]):.1f}')
print(f'  W_phys (axion work): {W_phys[-1]:.1f}')
print(f'  Ratio W/(dPar+dE): {W_phys[-1]/(dPar[-1]+(E_phys[-1]-E_phys[0])):.4f}')
