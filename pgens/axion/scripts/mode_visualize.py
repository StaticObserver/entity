"""k-omega dispersion visualization with peak overlay — angular freq."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift, fftfreq, rfft, rfftfreq
from scipy.ndimage import maximum_filter
import nt2

DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum_d1.0_Bperp0.0"
OUT    = f"{DATA}/mode_diagnosis"
import os; os.makedirs(OUT, exist_ok=True)

OMEGA_A = 1.0
K_A = 0.1

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values; x2 = ds.y.values
t_all = ds.t.values.ravel()
nt = len(t_all); nx1, nx2 = len(x1), len(x2)
dx1, dx2 = x1[1]-x1[0], x2[1]-x2[0]
dt_out = np.median(np.diff(t_all))
T_total = t_all[-1]-t_all[0]
domega = 2*np.pi/T_total
print(f"nt={nt}, domega(ang)={domega:.4f}, T={T_total:.1f}")

# 3D FFT
ss = 2
print("Computing 3D FFT...")
field_3d = {}
for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
    print(f"  {name}...")
    fsub = ds[name].values[:, ::ss, ::ss]
    for i in range(fsub.shape[1]):
        for j in range(fsub.shape[2]):
            ts = fsub[:, i, j]
            ts = ts - np.mean(ts)
            ts = ts - np.polyval(np.polyfit(t_all, ts, 1), t_all)
            fsub[:, i, j] = ts * np.hanning(nt)
    f3d = rfft(fsub, axis=0)
    f3d = fftshift(fft2(f3d, axes=(1,2)), axes=(1,2))
    field_3d[name] = np.abs(f3d)

# ANGULAR frequencies
f_arr = rfftfreq(nt, d=dt_out)
omega_arr = f_arr * 2*np.pi
k1_arr = fftshift(fftfreq(nx1//ss, d=dx1*ss)) * 2*np.pi
k2_arr = fftshift(fftfreq(nx2//ss, d=dx2*ss)) * 2*np.pi
ik2_z = len(k2_arr)//2
ik1_z = len(k1_arr)//2

P_E = field_3d['Ex']**2 + field_3d['Ey']**2 + field_3d['Ez']**2
P_B = field_3d['Bx']**2 + field_3d['By']**2 + field_3d['Bz']**2
P_tot = P_E + P_B

# Peak finder
def find_peaks_2d(spec, k_arr, w_arr, thresh=0.08, min_dist=3):
    max_val = spec.max()
    threshold = max_val * thresh
    fp = np.ones((min_dist*2+1, min_dist*2+1))
    mask = (maximum_filter(spec, footprint=fp) == spec) & (spec > threshold)
    pi, pk = np.where(mask)
    vals = spec[pi, pk]
    order = np.argsort(vals)[::-1]
    return [(w_arr[pi[i]], k_arr[pk[i]]) for i in order[:15]]

# ── Figure 1: k1-omega spectrogram with peak overlay ────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(f'k₁-ω Dispersion  density=1.0  Bperp=0.0  (angular ω)', fontsize=14)

# Left: P_tot
ax = axes[0]
w_pos = slice(1, None)
spec = P_tot[w_pos, :, ik2_z]  # (n_omega-1, nk1)
im = ax.pcolormesh(k1_arr, omega_arr[w_pos], spec,
                    shading='gouraud', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046, label='P_total')

# Overlay peaks
for name, spec_arr, color, marker in [
    ('P_tot', P_tot[:,:,ik2_z], 'cyan', 'o'),
    ('Bz', field_3d['Bz'][:,:,ik2_z], 'lime', 's'),
    ('Ex', field_3d['Ex'][:,:,ik2_z], 'magenta', '^'),
]:
    peaks = find_peaks_2d(spec_arr, k1_arr, omega_arr, thresh=0.05)
    for w, k in peaks[:10]:
        ax.plot(k, w, marker, color=color, markersize=4, alpha=0.7)

ax.axhline(OMEGA_A, color='white', ls='--', alpha=0.6, lw=1)
ax.axhline(2*OMEGA_A, color='white', ls=':', alpha=0.4, lw=1)
ax.axvline(K_A, color='white', ls='--', alpha=0.4, lw=0.8)
ax.set(xlabel='k₁', ylabel='ω (angular)', title='P_total(k₁,ω) + peak overlay')
# Mark axion source
ax.plot(K_A, OMEGA_A, 'r*', markersize=15, markeredgecolor='white', zorder=5)

# Right: zoom on key region
ax = axes[1]
w_mask = (omega_arr > 0.3) & (omega_arr < 2.2)
k_mask = (k1_arr > -2) & (k1_arr < 2)
spec_zoom = P_tot[w_mask, :, ik2_z][:, k_mask]
im = ax.pcolormesh(k1_arr[k_mask], omega_arr[w_mask], spec_zoom,
                    shading='gouraud', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046, label='P_total')

# Overlay Bz peaks specifically
bz_peaks = find_peaks_2d(field_3d['Bz'][:,:,ik2_z], k1_arr, omega_arr, thresh=0.03)
for w, k in bz_peaks[:12]:
    ax.plot(k, w, 's', color='lime', markersize=6, alpha=0.8, mec='white', mew=0.5)

# Light line at v_ph=1.0
k_line = np.linspace(-2, 2, 100)
ax.plot(k_line, np.abs(k_line), 'w--', alpha=0.2, lw=1, label='v_ph=1 (light line)')
ax.plot(k_line, 2*np.abs(k_line), 'w:', alpha=0.15, lw=1, label='v_ph=2')

ax.axhline(OMEGA_A, color='cyan', ls='--', alpha=0.5, label='ω_a=1.0')
ax.axhline(2*OMEGA_A, color='cyan', ls=':', alpha=0.3, label='2ω_a')
ax.plot(K_A, OMEGA_A, 'r*', markersize=12, zorder=5, label='axion source')
ax.set(xlabel='k₁', ylabel='ω (angular)', title='Zoom: Bz peaks + light line',
       xlim=(-2, 2), ylim=(0.3, 2.2))
ax.legend(fontsize=6, loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUT}/06_dispersion_peaks.png", dpi=150)
print(f"Saved: {OUT}/06_dispersion_peaks.png")

# ── Figure 2: Peak dispersion diagram ───────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Collect all Bz peaks
bz_spec = field_3d['Bz'][:,:,ik2_z]
bz_peaks_all = find_peaks_2d(bz_spec, k1_arr, omega_arr, thresh=0.03)

# Plot Bz peaks as colored by frequency
ws = [p[0] for p in bz_peaks_all]
ks = [abs(p[1]) for p in bz_peaks_all]
sc = ax.scatter(ks, ws, c=ws, cmap='plasma', s=50, edgecolors='white', linewidth=0.5)
plt.colorbar(sc, ax=ax, label='ω (angular)')

# Light line
k_line = np.linspace(0, 2.5, 200)
ax.plot(k_line, k_line, 'k--', alpha=0.3, lw=1, label='v_ph=c=1')

# Axion markers
ax.axhline(OMEGA_A, color='red', ls='--', alpha=0.5, label='ω_a=1.0')
ax.axvline(K_A, color='red', ls=':', alpha=0.3, label='k_a=0.1')
ax.axhline(2*OMEGA_A, color='orange', ls=':', alpha=0.3, label='2ω_a')

ax.set(xlabel='|k₁|', ylabel='ω (angular)', title='Bz Dispersion Relation  k₂=0',
       xlim=(0, 2.2), ylim=(0, 2.2))
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT}/06_dispersion_diagram.png", dpi=150)
print(f"Saved: {OUT}/06_dispersion_diagram.png")

# ── Figure 3: Field-by-field breakdown ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('k₁-ω Spectrum by Field  (angular ω)', fontsize=14)

field_pairs = [
    ('Ex', field_3d['Ex'][:,:,ik2_z]),
    ('Ey', field_3d['Ey'][:,:,ik2_z]),
    ('Ez', field_3d['Ez'][:,:,ik2_z]),
    ('Bx', field_3d['Bx'][:,:,ik2_z]),
    ('By', field_3d['By'][:,:,ik2_z]),
    ('Bz', field_3d['Bz'][:,:,ik2_z]),
]

for ax, (name, spec) in zip(axes.flat, field_pairs):
    im = ax.pcolormesh(k1_arr, omega_arr[w_pos], spec[w_pos, :],
                        shading='gouraud', cmap='inferno', norm=LogNorm())
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axhline(OMEGA_A, color='cyan', ls='--', alpha=0.5, lw=1)
    ax.axhline(2*OMEGA_A, color='cyan', ls=':', alpha=0.3, lw=1)
    ax.set(xlabel='k₁', ylabel='ω', title=name,
           xlim=(-2.5, 2.5), ylim=(0.1, 2.2))

plt.tight_layout()
plt.savefig(f"{OUT}/06_field_breakdown.png", dpi=150)
print(f"Saved: {OUT}/06_field_breakdown.png")

print("\nDone!")
