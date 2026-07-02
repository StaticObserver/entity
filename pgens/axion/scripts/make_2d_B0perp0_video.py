"""6-panel 2D E/B colormesh video — B0_perp=0.0, full 2D fields."""
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
B0     = 1.0
B0_PERP = 0.0

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values
x2 = ds.y.values
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1, nx2 = len(x1), len(x2)
print(f"nt={nt}, nx1={nx1}, nx2={nx2}")

# Calibrate color limits
print("Calibrating color limits...")
sample_idxs = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
vlims = {}
for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
    lo, hi = np.inf, -np.inf
    for i in sample_idxs:
        v = ds[name].isel(t=i).values  # (nx2, nx1)
        lo, hi = min(lo, v.min()), max(hi, v.max())
    m = max(abs(lo), abs(hi)) * 1.05
    vlims[name] = (-m, m)
    print(f"  {name}: ±{m:.4e}")

# Stride for ~200 frames
stride = max(1, nt // 200)
n_frames = nt // stride
print(f"Stride={stride}, Frames={n_frames}")

# ── Build figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    f'2D Axion-PIC  Bperp={B0_PERP}  density=1.0  PPC=16  '
    rf'$\varepsilon$={EPS}  $\omega_a/\omega_p$={OMEGA_RATIO}  k={K}',
    fontsize=13)

fields = ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
meshes = {}
for ax, name in zip(axes.flat, fields):
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
    for name in fields:
        v = ds[name].isel(t=i).values
        meshes[name].set_array(v.ravel())
        # Update title for the parent axes of each mesh
        meshes[name].axes.set_title(f'{name}  t={ti:.2f}')
    return list(meshes.values())

ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=True)
out_path = f"{OUT}/2d_evolution_B0perp0.mp4"
ani.save(out_path, writer='ffmpeg', fps=20, dpi=100,
         savefig_kwargs={'facecolor': 'white'})
print(f"Saved: {out_path}")
