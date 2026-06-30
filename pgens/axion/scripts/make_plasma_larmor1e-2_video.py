import numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nt2.plotters.export import makeFramesAndMovie
import nt2

CASE = sys.argv[1] if len(sys.argv) > 1 else '/home/staticobserver/entity/problems/axion/plasma_larmor1e-2'
EPS, OMEGA_RATIO, K, B0 = 0.0001, 10.0, 0.1, 1.0
SKIN, LARMOR = 0.5, 0.01
OMEGA = OMEGA_RATIO / SKIN
NORM = SKIN**2 / LARMOR
RUNTIME = 10
STRIDE = 4

print(f'Loading plasma from {CASE} (t <= {RUNTIME:.0f})...')
print(f'omega_a={OMEGA}, NORM={NORM}')
data = nt2.Data(CASE)
fields = data.fields

Ex_all = fields['Ex'].sel(t=slice(None, RUNTIME)).values
Jx_all = fields['Jx'].sel(t=slice(None, RUNTIME)).values
Charge_all = fields['Charge'].sel(t=slice(None, RUNTIME)).values

# Energy quantities
Ey_all = fields['Ey'].sel(t=slice(None, RUNTIME)).values
Ez_all = fields['Ez'].sel(t=slice(None, RUNTIME)).values
Ttt_all = fields['Ttt'].sel(t=slice(None, RUNTIME)).values

x = fields.x.values
dx = x[1] - x[0]
t_all_raw = fields.t.sel(t=slice(None, RUNTIME)).values
t_all = np.asarray(t_all_raw).ravel()
t_to_idx = {float(t): i for i, t in enumerate(t_all)}
print(f'  {len(t_all)} timesteps, {len(x)} cells')

B0_PHYS = 1.0 / LARMOR
SIGMA = (SKIN/LARMOR)**2

# Axion-theory predictions
rho_a_all = EPS * K * B0 * np.sin(K * x[None, :] - OMEGA * t_all[:, None]) * NORM
Ja_all = EPS * OMEGA * B0 * np.sin(K * x[None, :] - OMEGA * t_all[:, None]) * NORM
E_a_all = -EPS * B0 * np.cos(K * x[None, :] - OMEGA * t_all[:, None])
E_init = -EPS * B0 * np.cos(K * x)

# Energy computation (physical units)
# T00 includes weight=density/ppc0, so integral is resolution-independent
E_phys = np.trapezoid(Ex_all**2 + Ey_all**2 + Ez_all**2, x, axis=1) * 0.5 * B0_PHYS**2
par_phys = np.trapezoid(Ttt_all, x, axis=1)
dPar = par_phys - par_phys[0]   # particle energy GAIN
total = dPar + E_phys            # absolute E-field + particle gain


def ylim(arr, margin=0.1):
    lo, hi = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
    span = max(hi - lo, 1e-10)
    return (lo - span * margin, hi + span * margin)

YLIM_EX = ylim(Ex_all)
YLIM_JX = ylim(Jx_all)
YLIM_RHO = ylim(Charge_all)
YLIM_ENE = (0, total.max() * 1.1)

OUT = f'{CASE}/movie'

def plot_frame(t):
    i_t = t_to_idx[float(t)]
    t_val = float(t)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Plasma larmor0={LARMOR} skin={SKIN}  t={t_val:.1f}', fontsize=12)

    ax = axes[0, 0]
    ax.plot(x, E_init, '--', color='gray', alpha=0.3, lw=1, label=r'$E_x(t{=}0)$')
    ax.plot(x, Ex_all[i_t], 'C0', lw=1.2, label='Ex plasma')
    ax.plot(x, E_a_all[i_t], '-', color='red', alpha=0.8, lw=1.5, label=r'$E_a(t)$')
    ax.set(xlabel='x [d_e]', ylabel='Ex', title=f'Ex  (t={t_val:.1f})')
    ax.legend(fontsize=7, loc='upper right'); ax.set_ylim(*YLIM_EX); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, Jx_all[i_t], 'C1', lw=1.2, label='Jx plasma')
    ax.plot(x, Ja_all[i_t], '-', color='red', alpha=0.8, lw=1.5, label=r'$J_a$ (axion)')
    ax.set(xlabel='x [d_e]', ylabel='Jx', title=f'Jx  (t={t_val:.1f})')
    ax.legend(fontsize=7, loc='upper right'); ax.set_ylim(*YLIM_JX); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, Charge_all[i_t], 'C2', lw=1.2, label=r'$\rho_q$')
    ax.plot(x, rho_a_all[i_t], '-', color='red', alpha=0.8, lw=1.5, label=r'$\rho_a$ (axion)')
    ax.set(xlabel='x [d_e]', ylabel=r'$\rho$', title=f'Charge  (t={t_val:.1f})')
    ax.legend(fontsize=7, loc='upper right'); ax.set_ylim(*YLIM_RHO); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_all[:i_t+1], E_phys[:i_t+1], 'C0', lw=1.5, label='E-field Energy')
    ax.plot(t_all[:i_t+1], dPar[:i_t+1], 'C1', lw=1.5, label='Particle Gain')
    ax.plot(t_all[:i_t+1], total[:i_t+1], 'k--', lw=1, alpha=0.6, label='Total')
    ax.axvline(x=t_val, color='gray', ls='--', alpha=0.5)
    ax.set(xlabel=r'time [$\omega_p^{-1}$]', ylabel='Energy (LH units)',
           title='Energy Evolution')
    ax.legend(fontsize=7, loc='upper left'); ax.set_ylim(*YLIM_ENE); ax.grid(True, alpha=0.3)

    plt.tight_layout()

times_subset = t_all[::STRIDE]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
print(f'Generating {len(times_subset)} frames (stride={STRIDE})...')
makeFramesAndMovie(name=OUT, plot=plot_frame, times=times_subset, framerate=30, remove_frames=False)
print(f'Done: {OUT}.mp4')
