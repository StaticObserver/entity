"""2D evolution video: Ex, Bz, DivE, Jy — B0_perp=0.0, density=1.0, PPC=16."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nt2

# Config
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
nx1 = len(x1)
nx2 = len(x2)
L1 = x1[-1] - x1[0]
L2 = x2[-1] - x2[0]
print(f"Shape: nt={nt}, nx1={nx1}, nx2={nx2}")

# Pre-scan y-axis limits from 5 samples
print("Calibrating color limits...")
sample_idxs = [0, nt//4, nt//2, 3*nt//4, nt-1]
def _vrange(name, idxs):
    lo, hi = np.inf, -np.inf
    for i in idxs:
        v = ds[name].isel(t=i).values
        lo, hi = min(lo, v.min()), max(hi, v.max())
    return lo, hi

vlims = {}
for name in ['Ex', 'Bz', 'DivE', 'Jy']:
    lo, hi = _vrange(name, sample_idxs)
    m = max(abs(lo), abs(hi)) * 1.05
    vlims[name] = (-m, m)
    print(f"  {name}: ±{m:.4f}")

# Stride for ~200 frames
stride = max(1, nt // 200)
n_frames = nt // stride
print(f"Stride={stride}, Frames={n_frames}")

# Build figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'2D Axion-PIC  density=1.0  PPC=16  eps={EPS}  wa/wp={OMEGA_RATIO}  Bperp={B0_PERP}',
             fontsize=13)

# Initialize empty pcolormesh
imgs = {}
titles = ['Ex (axion-driven)', 'Bz (transverse)', 'DivE (Gauss law)', 'Jy (current)']
for ax, name, title in zip(axes.flat, ['Ex', 'Bz', 'DivE', 'Jy'], titles):
    v0 = ds[name].isel(t=0).values
    im = ax.pcolormesh(x1, x2, v0, shading='auto', cmap='RdBu_r',
                       vmin=vlims[name][0], vmax=vlims[name][1])
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='x1', ylabel='x2', title=f'{title}  t=0.00')
    imgs[name] = im

plt.tight_layout()

def update(fi):
    i = fi * stride
    ti = t_all[i]
    for name in ['Ex', 'Bz', 'DivE', 'Jy']:
        v = ds[name].isel(t=i).values
        imgs[name].set_array(v.ravel())
    axes.flat[0].set_title(f'Ex (axion-driven)  t={ti:.2f}')
    axes.flat[1].set_title(f'Bz (transverse)  t={ti:.2f}')
    axes.flat[2].set_title(f'DivE (Gauss law)  t={ti:.2f}')
    axes.flat[3].set_title(f'Jy (current)  t={ti:.2f}')
    return list(imgs.values())

ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=True)
out_path = f"{OUT}/2d_evolution_B0perp0.mp4"
ani.save(out_path, writer='ffmpeg', fps=20, dpi=100,
         savefig_kwargs={'facecolor': 'white'})
print(f"Saved: {out_path}")
