"""E/B 2D colormesh video — B0_perp=0.0, GridSpec(3,2) layout."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    vlims[name] = (-m, m)
    print(f"  {name}: ±{m:.4e}")

stride = max(1, nt // 200)
n_frames = nt // stride
print(f"Stride={stride}, Frames={n_frames}")

# ── Build figure: GridSpec(3, 2) ────────────────────────────────────
fig = plt.figure(figsize=(12, 16))
fig.suptitle(
    f'2D Axion-PIC  Bperp={B0_PERP}  density=1.0  PPC=16  '
    rf'$\varepsilon$={EPS}  $\omega_a/\omega_p$={OMEGA_RATIO}  k={K}',
    fontsize=13)

gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

E_names = ['Ex', 'Ey', 'Ez']
B_names = ['Bx', 'By', 'Bz']
meshes = {}
cbfs   = {}

for row, (en, bn) in enumerate(zip(E_names, B_names)):
    # E panel (left column)
    ax_e = fig.add_subplot(gs[row, 0])
    v0_e = ds[en].isel(t=0).values
    im_e = ax_e.pcolormesh(x1, x2, v0_e, shading='auto', cmap='RdBu_r',
                           vmin=vlims[en][0], vmax=vlims[en][1])
    cbfs[en] = plt.colorbar(im_e, ax=ax_e, fraction=0.046)
    ax_e.set(xlabel='x1', ylabel='x2', title=f'{en}  t={t_all[0]:.2f}')
    meshes[en] = im_e

    # B panel (right column)
    ax_b = fig.add_subplot(gs[row, 1])
    v0_b = ds[bn].isel(t=0).values
    im_b = ax_b.pcolormesh(x1, x2, v0_b, shading='auto', cmap='RdBu_r',
                           vmin=vlims[bn][0], vmax=vlims[bn][1])
    cbfs[bn] = plt.colorbar(im_b, ax=ax_b, fraction=0.046)
    ax_b.set(xlabel='x1', ylabel='x2', title=f'{bn}  t={t_all[0]:.2f}')
    meshes[bn] = im_b

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
