"""2D E/B evolution video — 2 rows x 3 cols, E top, B bottom."""
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
B0     = 1.0
B0_PERP = 0.0

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values
x2 = ds.y.values
t_all = ds.t.values.ravel()
nt = len(t_all)
print(f"nt={nt}, nx1={len(x1)}, nx2={len(x2)}")

# Calibrate color limits
print("Calibrating color limits...")
sample_idxs = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
vlims = {}
for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
    lo, hi = np.inf, -np.inf
    for i in sample_idxs:
        v = ds[name].isel(t=i).values
        lo, hi = min(lo, v.min()), max(hi, v.max())
    m = max(abs(lo), abs(hi)) * 1.05
    if m < 1e-15:
        m = 1e-15
    vlims[name] = (-m, m)
    print(f"  {name}: ±{m:.4e}")

stride = max(1, nt // 200)
n_frames = nt // stride
print(f"Stride={stride}, Frames={n_frames}")

# ── Build figure: 2 rows x 3 cols ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f'2D Axion-PIC  Bperp={B0_PERP}  density=0  vacuum  '
    rf'$\varepsilon$={EPS}  $\omega_a/\omega_p$={OMEGA_RATIO}  k={K}',
    fontsize=13)

E_names = ['Ex', 'Ey', 'Ez']
B_names = ['Bx', 'By', 'Bz']
meshes = {}

for ax, name in zip(axes[0], E_names):
    v0 = ds[name].isel(t=0).values
    im = ax.pcolormesh(x1, x2, v0, shading='auto', cmap='RdBu_r',
                       vmin=vlims[name][0], vmax=vlims[name][1])
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xlabel='x1', ylabel='x2', title=f'{name}  t={t_all[0]:.2f}')
    meshes[name] = im

for ax, name in zip(axes[1], B_names):
    v0 = ds[name].isel(t=0).values
    im = ax.pcolormesh(x1, x2, v0, shading='auto', cmap='RdBu_r',
                       vmin=vlims[name][0], vmax=vlims[name][1])
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set(xlabel='x1', ylabel='x2', title=f'{name}  t={t_all[0]:.2f}')
    meshes[name] = im

plt.tight_layout()

# ── Animation ───────────────────────────────────────────────────────
def update(fi):
    i = fi * stride
    ti = t_all[i]
    for name in E_names + B_names:
        v = ds[name].isel(t=i).values
        meshes[name].set_array(v.ravel())
        meshes[name].axes.set_title(f'{name}  t={ti:.2f}')
    return list(meshes.values())

ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=True)
out_path = f"{OUT}/2d_evolution_B0perp0.mp4"
ani.save(out_path, writer='ffmpeg', fps=20, dpi=100,
         savefig_kwargs={'facecolor': 'white'})
print(f"Saved: {out_path}")
