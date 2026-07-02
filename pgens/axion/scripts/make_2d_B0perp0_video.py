"""6-panel E/B evolution video — B0_perp=0.0, 2D data averaged over x2."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nt2

# ── Config ──────────────────────────────────────────────────────────
DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum"
OUT    = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum"
EPS    = 0.01
K      = 0.1
OMEGA_RATIO = 0.5
SKIN   = 0.5
LARMOR = 0.01
B0     = 1.0
B0_PERP = 0.0

OMEGA_A = OMEGA_RATIO / SKIN  # = 1.0

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1 = len(x1)
print(f"nt={nt}, nx1={nx1}")

# Average over x2 for quasi-1D analysis
def avg_x2(name, it):
    """Load one timestep and average over x2."""
    v = ds[name].isel(t=it).values  # (nx2, nx1)
    return v.mean(axis=0)  # average over x2 -> (nx1,)

# Calibrate y-axis limits
print("Calibrating limits...")
sample_idxs = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
vlims = {}
for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
    lo, hi = np.inf, -np.inf
    for i in sample_idxs:
        v = avg_x2(name, i)
        lo, hi = min(lo, v.min()), max(hi, v.max())
    m = max(abs(lo), abs(hi)) * 1.15
    vlims[name] = (-m if m > 1e-10 else -1e-10, m if m > 1e-10 else 1e-10)
    print(f"  {name}: ±{m:.4e}")

# Stride
stride = max(1, nt // 300)
n_frames = nt // stride
print(f"Stride={stride}, Frames={n_frames}")

# ── Build figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f'2D Axion-PIC  Bperp={B0_PERP}  density=1.0  PPC=16  '
    rf'$\varepsilon$={EPS}  $\omega_a/\omega_p$={OMEGA_RATIO}  k={K}',
    fontsize=13)

# E-field row
lines = {}
for ax, name, ylabel in zip(
    [axes[0, 0], axes[0, 1], axes[0, 2],
     axes[1, 0], axes[1, 1], axes[1, 2]],
    ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz'],
    ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']):
    y0 = avg_x2(name, 0)
    line, = ax.plot(x1, y0, 'b-', lw=0.8)
    lines[name] = line
    ax.set(xlim=(x1[0], x1[-1]), ylim=vlims[name],
           xlabel='x1', ylabel=ylabel, title=f'{name}  t={t_all[0]:.2f}')
    ax.grid(alpha=0.3)

# Analytical Ex at t=0 (dashed reference)
ex_analytic = -EPS * B0 * np.cos(K * x1)
axes[0, 0].plot(x1, ex_analytic, 'r--', lw=0.8, alpha=0.5, label='Ex(t=0) analytic')
axes[0, 0].legend(fontsize=7)

# Bx constant reference
axes[1, 0].axhline(B0, color='r', ls='--', lw=0.8, alpha=0.5, label='B0=1.0')
axes[1, 0].legend(fontsize=7)

plt.tight_layout()

# ── Animation ───────────────────────────────────────────────────────
def update(fi):
    i = fi * stride
    ti = t_all[i]
    for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
        v = avg_x2(name, i)
        lines[name].set_ydata(v)
    axes[0, 0].set_title(f'Ex  t={ti:.2f}')
    axes[0, 1].set_title(f'Ey  t={ti:.2f}')
    axes[0, 2].set_title(f'Ez  t={ti:.2f}')
    axes[1, 0].set_title(f'Bx  t={ti:.2f}')
    axes[1, 1].set_title(f'By  t={ti:.2f}')
    axes[1, 2].set_title(f'Bz  t={ti:.2f}')
    return list(lines.values())

ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=True)
out_path = f"{OUT}/2d_evolution_B0perp0.mp4"
ani.save(out_path, writer='ffmpeg', fps=20, dpi=120,
         savefig_kwargs={'facecolor': 'white'})
print(f"Saved: {out_path}")
