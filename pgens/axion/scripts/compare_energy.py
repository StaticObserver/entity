import nt2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_energy(case_path, PPC=128.0, SKIN=0.5, LARMOR=0.01, RUNTIME=None):
    data = nt2.Data(case_path)
    fields = data.fields
    x = fields.x.values; dx = x[1] - x[0]
    if RUNTIME is not None:
        Ex = fields['Ex'].sel(t=slice(None, RUNTIME)).values
        Ey = fields['Ey'].sel(t=slice(None, RUNTIME)).values
        Ez = fields['Ez'].sel(t=slice(None, RUNTIME)).values
        Ttt = fields['Ttt'].sel(t=slice(None, RUNTIME)).values
        t = fields.t.sel(t=slice(None, RUNTIME)).values.ravel()
    else:
        Ex = fields['Ex'].values
        Ey = fields['Ey'].values
        Ez = fields['Ez'].values
        Ttt = fields['Ttt'].values
        t = fields.t.values.ravel()

    B0_PHYS = 1.0 / LARMOR
    SIGMA = (SKIN/LARMOR)**2
    E_phys = np.trapezoid(Ex**2 + Ey**2 + Ez**2, x, axis=1) * 0.5 * B0_PHYS**2 * SIGMA
    par_phys = np.trapezoid(Ttt, x, axis=1) * PPC / dx
    dPar = par_phys - par_phys[0]
    return t, E_phys, dPar, len(x)

BASE = '/home/staticobserver/entity/problems/axion'
t1, E1, dPar1, nx1 = load_energy(f'{BASE}/plasma_larmor1e-2_or0.5_n16392', RUNTIME=1000.0)
t2, E2, dPar2, nx2 = load_energy(f'{BASE}/plasma_larmor1e-2_or0.5_t2000', RUNTIME=1000.0)

# Interpolate to common time grid
t_common = np.linspace(max(t1[0], t2[0]), min(t1[-1], t2[-1]), 500)
E1_i = interp1d(t1, E1, axis=0)(t_common)
E2_i = interp1d(t2, E2, axis=0)(t_common)
dPar1_i = interp1d(t1, dPar1, axis=0)(t_common)
dPar2_i = interp1d(t2, dPar2, axis=0)(t_common)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: E-field energy
ax = axes[0, 0]
ax.plot(t1, E1, 'C0', lw=1.2, label=f'n={nx1}')
ax.plot(t2, E2, 'C1', lw=1.2, label=f'n={nx2}')
ax.set(xlabel=r'time [$\omega_p^{-1}$]', ylabel='E-field Energy (LH)',
       title='E-field Total Energy')
ax.legend(title='Resolution'); ax.grid(True, alpha=0.3)

# Top-right: Particle energy gain
ax = axes[0, 1]
ax.plot(t1, dPar1, 'C0', lw=1.2, label=f'n={nx1}')
ax.plot(t2, dPar2, 'C1', lw=1.2, label=f'n={nx2}')
ax.set(xlabel=r'time [$\omega_p^{-1}$]', ylabel='Particle Energy Gain (LH)',
       title='Particle Energy Gain')
ax.legend(title='Resolution'); ax.grid(True, alpha=0.3)

# Bottom-left: Ratio dPar / E
ax = axes[1, 0]
ax.plot(t_common, dPar1_i/E1_i, 'C0', lw=1.2, label=f'n={nx1}')
ax.plot(t_common, dPar2_i/E2_i, 'C1', lw=1.2, label=f'n={nx2}')
ax.set(xlabel=r'time [$\omega_p^{-1}$]', ylabel='dPar / E',
       title='Particle Gain / E-field Energy Ratio')
ax.legend(title='Resolution'); ax.grid(True, alpha=0.3)

# Bottom-right: key numbers
ax = axes[1, 1]
ax.axis('off')

txt = (
    "Resolution Comparison (t=0..1000)\n"
    "=" * 45 + "\n"
    f"n={nx1} (dx={188.496/nx1:.4f}):\n"
    f"  E[t=0]:      {E1[0]:.1f}\n"
    f"  E[final]:    {E1[-1]:.1f}\n"
    f"  dPar[final]: {dPar1[-1]:.0f}\n"
    f"  ratio:       {dPar1[-1]/E1[-1]:.2f}\n\n"
    f"n={nx2} (dx={188.496/nx2:.4f}):\n"
    f"  E[t=0]:      {E2[0]:.1f}\n"
    f"  E[final]:    {E2[-1]:.1f}\n"
    f"  dPar[final]: {dPar2[-1]:.0f}\n"
    f"  ratio:       {dPar2[-1]/E2[-1]:.2f}\n\n"
    f"Ratio of ratios: {(dPar1[-1]/E1[-1])/(dPar2[-1]/E2[-1]):.2f}x"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
OUT = f'{BASE}/energy_resolution_compare.png'
plt.savefig(OUT, dpi=150)
print(f'Saved: {OUT}')
