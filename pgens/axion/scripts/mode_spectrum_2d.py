"""2D mode diagnosis: data quality → spatial spectrum → k-omega spectrum.
Follows 二维场模式诊断计划.md sections 1-5."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift, fftfreq, rfft, rfftfreq, fft, ifft
from scipy.signal import find_peaks
import nt2

# ── Config ──────────────────────────────────────────────────────────
DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum_d1.0_Bperp0.0"
OUT    = f"{DATA}/mode_diagnosis"
import os; os.makedirs(OUT, exist_ok=True)

EPS    = 0.01
K_A    = 0.1       # axion wavenumber along x1
OMEGA_RATIO = 0.5
SKIN   = 0.5
OMEGA_A = OMEGA_RATIO / SKIN  # = 1.0
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
dx1 = x1[1] - x1[0]
dx2 = x2[1] - x2[0]
L1 = x1[-1] - x1[0]
L2 = x2[-1] - x2[0]
print(f"nt={nt}, nx1={nx1}, nx2={nx2}, dx1={dx1:.4f}, dx2={dx2:.4f}")

# =========================================================================
# SECTION 1: Data Quality Checks
# =========================================================================
print("\n" + "="*60)
print("SECTION 1: Data Quality Checks")
print("="*60)

# 1.1 Field availability
print("\n1.1 Field availability:")
print("  Ex, Ey, Ez, Bx, By, Bz, DivE, Jx, Jy, Jz, Ttt")

# 1.2 Time sampling
dt_out = np.median(np.diff(t_all))
T_total = t_all[-1] - t_all[0]
domega = 2 * np.pi / T_total
omega_nyq = np.pi / dt_out
print(f"\n1.2 Time sampling:")
print(f"  dt_out = {dt_out:.4f}")
print(f"  T = {T_total:.2f}")
print(f"  domega = {domega:.4f} rad")
print(f"  omega_Nyquist = {omega_nyq:.4f} rad")
print(f"  omega_a = {OMEGA_A:.4f} (< Nyquist: {OMEGA_A < omega_nyq})")
print(f"  omega_a / domega = {OMEGA_A/domega:.1f} bins (>>1: OK)")

# 1.3 Constraint baselines — compute RMS time series
print("\n1.3 Constraint baselines (RMS over space)...")
def rms(v):
    return np.sqrt(np.mean(v**2))

rms_divE = np.array([rms(ds['DivE'].isel(t=i).values) for i in range(0, nt, 4)])
rms_Ex   = np.array([rms(ds['Ex'].isel(t=i).values) for i in range(0, nt, 4)])
rms_Ey   = np.array([rms(ds['Ey'].isel(t=i).values) for i in range(0, nt, 4)])
rms_Ez   = np.array([rms(ds['Ez'].isel(t=i).values) for i in range(0, nt, 4)])
rms_Bx   = np.array([rms(ds['Bx'].isel(t=i).values - B0) for i in range(0, nt, 4)])
rms_By   = np.array([rms(ds['By'].isel(t=i).values) for i in range(0, nt, 4)])
rms_Bz   = np.array([rms(ds['Bz'].isel(t=i).values) for i in range(0, nt, 4)])
t_rms    = t_all[::4]

# Energy
from numpy.fft import rfftn
Esq_tot = np.array([0.5 * np.mean(ds['Ex'].isel(t=i).values**2 +
                                   ds['Ey'].isel(t=i).values**2 +
                                   ds['Ez'].isel(t=i).values**2) * L1 * L2
                    for i in range(0, nt, 4)])
Bsq_tot = np.array([0.5 * np.mean(ds['Bx'].isel(t=i).values**2 +
                                   ds['By'].isel(t=i).values**2 +
                                   ds['Bz'].isel(t=i).values**2) * L1 * L2
                    for i in range(0, nt, 4)])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Data Quality: RMS & Energy Baselines  density=1.0  Bperp=0.0', fontsize=13)

ax = axes[0, 0]
ax.plot(t_rms, rms_Ex, label='Ex')
ax.plot(t_rms, rms_Ey, label='Ey')
ax.plot(t_rms, rms_Ez, label='Ez')
ax.legend(); ax.set(xlabel='t', ylabel='RMS', title='E-field RMS'); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_rms, rms_Bx, label='Bx-B0')
ax.plot(t_rms, rms_By, label='By')
ax.plot(t_rms, rms_Bz, label='Bz')
ax.legend(); ax.set(xlabel='t', ylabel='RMS', title='B-field RMS'); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(t_rms, rms_divE)
ax.set(xlabel='t', ylabel='RMS(DivE)', title='Gauss Law Violation'); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t_rms, Esq_tot, label='E')
ax.plot(t_rms, Bsq_tot, label='B')
ax.plot(t_rms, Esq_tot + Bsq_tot, 'k--', alpha=0.5, label='Total EM')
ax.legend(); ax.set(xlabel='t', ylabel='Energy', title='EM Energy'); ax.grid(alpha=0.3)

ax = axes[1, 1]
if 'Jx' in ds:
    rms_Jx = np.array([rms(ds['Jx'].isel(t=i).values) for i in range(0, nt, 4)])
    rms_Jy = np.array([rms(ds['Jy'].isel(t=i).values) for i in range(0, nt, 4)])
    rms_Jz = np.array([rms(ds['Jz'].isel(t=i).values) for i in range(0, nt, 4)])
    ax.plot(t_rms, rms_Jx, label='Jx')
    ax.plot(t_rms, rms_Jy, label='Jy')
    ax.plot(t_rms, rms_Jz, label='Jz')
    ax.legend()
ax.set(xlabel='t', ylabel='RMS', title='Current RMS'); ax.grid(alpha=0.3)

ax = axes[1, 2]
if 'Ttt' in ds:
    rms_Ttt = np.array([rms(ds['Ttt'].isel(t=i).values) for i in range(0, nt, 4)])
    ax.plot(t_rms, rms_Ttt)
ax.set(xlabel='t', ylabel='RMS(Ttt)', title='Particle Energy Density'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/01_data_quality.png", dpi=150)
print(f"Saved: {OUT}/01_data_quality.png")

print(f"\n  RMS(DivE) = {rms_divE[-1]:.4e} (final)")
print(f"  RMS(By)   = {rms_By[-1]:.4e} (final)")
print(f"  RMS(Bz)   = {rms_Bz[-1]:.4e} (final)")
print(f"  E growth: {Esq_tot[-1]/Esq_tot[0]:.2f}x")
print(f"  B growth: {Bsq_tot[-1]/Bsq_tot[0]:.2f}x")

# =========================================================================
# SECTION 3: Single-point spectrum (5 probe points)
# =========================================================================
print("\n" + "="*60)
print("SECTION 3: Single-point Time Spectrum")
print("="*60)

probe_points = {
    'center':     (nx1//2, nx2//2),
    'x1_left':    (nx1//4, nx2//2),
    'x1_right':   (3*nx1//4, nx2//2),
    'x2_bottom':  (nx1//2, nx2//4),
    'x2_top':     (nx1//2, 3*nx2//4),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Single-point Ey Spectrum  5 Probe Points', fontsize=13)

for (label, (ix, iy)), ax in zip(probe_points.items(), axes.flat[:5]):
    ey = ds['Ey'].values[:, iy, ix]
    ey_ac = ey - np.mean(ey)
    ey_ac = ey_ac - np.polyval(np.polyfit(t_all, ey_ac, 1), t_all)  # detrend
    window = np.hanning(nt)
    ey_fft = np.abs(rfft(ey_ac * window))
    freqs = rfftfreq(nt, d=dt_out)
    ax.semilogy(freqs[freqs > 0], ey_fft[freqs > 0], lw=0.8)
    ax.axvline(OMEGA_A, color='red', ls='--', alpha=0.5, label=f'wa={OMEGA_A:.3f}')
    ax.axvline(2*OMEGA_A, color='orange', ls='--', alpha=0.4, label=f'2wa={2*OMEGA_A:.3f}')
    ax.axvline(domega, color='gray', ls=':', alpha=0.4, label=f'dw={domega:.3f}')
    ax.set(xlabel='omega', ylabel='|FFT(Ey)|', title=f'Ey @ {label} ({ix},{iy})',
           xlim=(0, freqs[-1]))
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 6th panel: Ex at center for comparison
ax = axes.flat[5]
ex = ds['Ex'].values[:, nx2//2, nx1//2]
ex_ac = ex - np.mean(ex)
ex_ac = ex_ac - np.polyval(np.polyfit(t_all, ex_ac, 1), t_all)
ex_fft = np.abs(rfft(ex_ac * np.hanning(nt)))
frets = rfftfreq(nt, d=dt_out)
ax.semilogy(frets[frets > 0], ex_fft[frets > 0], lw=0.8, color='purple')
ax.axvline(OMEGA_A, color='red', ls='--', alpha=0.5, label=f'wa={OMEGA_A:.3f}')
ax.set(xlabel='omega', ylabel='|FFT(Ex)|', title='Ex @ center',
       xlim=(0, freqs[-1]))
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/03_single_point_spectrum.png", dpi=150)
print(f"Saved: {OUT}/03_single_point_spectrum.png")

# Find peaks for Ey at center
ey_center = ds['Ey'].values[:, nx2//2, nx1//2]
ey_center_ac = ey_center - np.mean(ey_center)
ey_center_ac = ey_center_ac - np.polyval(np.polyfit(t_all, ey_center_ac, 1), t_all)
ey_fft_center = np.abs(rfft(ey_center_ac * np.hanning(nt)))
peaks_idx, _ = find_peaks(np.log10(ey_fft_center[1:] + 1e-20), height=-10, distance=2)
peaks_freq = freqs[1:][peaks_idx]
peaks_amp  = ey_fft_center[1:][peaks_idx]
order = np.argsort(peaks_amp)[::-1][:8]
print("\nEy center frequency peaks:")
for f, a in zip(peaks_freq[order], peaks_amp[order]):
    print(f"  w={f:.4f}, amp={a:.2e}")

# =========================================================================
# SECTION 4: Spatial Spectrum
# =========================================================================
print("\n" + "="*60)
print("SECTION 4: Spatial Spectrum")
print("="*60)

k1 = fftshift(fftfreq(nx1, d=dx1)) * 2 * np.pi
k2 = fftshift(fftfreq(nx2, d=dx2)) * 2 * np.pi

# Pick key times
t_key_idxs = [0, nt//4, nt//2, 3*nt//4, nt-1]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Spatial Spectrum |FFT(Bz)|  k-space  t = {t_all[t_key_idxs[-1]]:.1f}', fontsize=13)

# Bz spatial spectrum at last time (2D)
bz_last = ds['Bz'].isel(t=nt-1).values
bz_fft_2d = fftshift(np.abs(fft2(bz_last)))

ax = axes[0, 0]
im = ax.pcolormesh(k1, k2, bz_fft_2d, shading='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046)
ax.axvline(K_A, color='cyan', ls='--', alpha=0.7, label=f'ka={K_A}')
ax.axvline(-K_A, color='cyan', ls='--', alpha=0.7)
ax.set(xlabel='k1', ylabel='k2', title=f'|FFT(Bz)| 2D, t={t_all[-1]:.1f}')
ax.legend(fontsize=7)

# k1 profile (k2=0)
ax = axes[0, 1]
ik2_zero = nx2 // 2  # fftshifted index for k2=0
bz_k1_profile = bz_fft_2d[ik2_zero, :]
ax.semilogy(k1[k1 >= 0], bz_k1_profile[k1 >= 0], lw=0.8)
ax.axvline(K_A, color='red', ls='--', alpha=0.5, label=f'ka={K_A}')
ax.set(xlabel='k1', ylabel='|FFT(Bz)|', title='k1 profile (k2=0)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# k2 profile (k1=0)
ax = axes[0, 2]
ik1_zero = nx1 // 2
bz_k2_profile = bz_fft_2d[:, ik1_zero]
ax.semilogy(k2[k2 >= 0], bz_k2_profile[k2 >= 0], lw=0.8, color='green')
ax.set(xlabel='k2', ylabel='|FFT(Bz)|', title='k2 profile (k1=0)')
ax.grid(alpha=0.3)

# Ey spatial spectrum at last time
ey_last = ds['Ey'].isel(t=nt-1).values
ey_fft_2d = fftshift(np.abs(fft2(ey_last)))

ax = axes[1, 0]
im = ax.pcolormesh(k1, k2, ey_fft_2d, shading='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046)
ax.axvline(K_A, color='cyan', ls='--', alpha=0.7)
ax.axvline(-K_A, color='cyan', ls='--', alpha=0.7)
ax.set(xlabel='k1', ylabel='k2', title=f'|FFT(Ey)| 2D, t={t_all[-1]:.1f}')

# Ex spatial spectrum
ex_last = ds['Ex'].isel(t=nt-1).values
ex_fft_2d = fftshift(np.abs(fft2(ex_last)))

ax = axes[1, 1]
im = ax.pcolormesh(k1, k2, ex_fft_2d, shading='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046)
ax.axvline(K_A, color='cyan', ls='--', alpha=0.7)
ax.axvline(-K_A, color='cyan', ls='--', alpha=0.7)
ax.set(xlabel='k1', ylabel='k2', title=f'|FFT(Ex)| 2D, t={t_all[-1]:.1f}')

# Angular average P(|k|)
ax = axes[1, 2]
k_radial = np.sqrt(k1[:, None]**2 + k2[None, :]**2)  # (nx1, nx2)
k_bins = np.linspace(0, k_radial.max(), 50)
bz_radial = np.array([bz_fft_2d[(k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])].mean()
                      for i in range(len(k_bins)-1)])
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
ax.semilogy(k_centers, bz_radial, lw=1, label='Bz')
ey_radial = np.array([ey_fft_2d[(k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])].mean()
                      for i in range(len(k_bins)-1)])
ax.semilogy(k_centers, ey_radial, lw=1, label='Ey')
ax.axvline(K_A, color='red', ls='--', alpha=0.5, label=f'ka={K_A}')
ax.set(xlabel='|k|', ylabel='<|FFT|>', title='Azimuthal Average')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/04_spatial_spectrum.png", dpi=150)
print(f"Saved: {OUT}/04_spatial_spectrum.png")

# =========================================================================
# SECTION 5: k-omega Spectrum (3D FFT core diagnosis)
# =========================================================================
print("\n" + "="*60)
print("SECTION 5: k-omega Spectrum")
print("="*60)

# Compute full 3D FFT for key fields
# Strategy: rFFT in time (real→half), FFT in x1, FFT in x2
# Since we have 768 time steps, full 3D FFT is manageable

print("Computing 3D FFT (t→omega, x1→k1, x2→k2)...")
n_omega = nt // 2 + 1
# Sub-sample spatially to make 3D FFT cheaper: take every 2nd point
ss = 2
fields_3d = {}
for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
    print(f"  Processing {name}...")
    # Load subset: every ss in x1, every ss in x2
    field_sub = ds[name].values[:, ::ss, ::ss]  # (nt, nx1//ss, nx2//ss)
    # Detrend and window in time
    for i in range(field_sub.shape[1]):
        for j in range(field_sub.shape[2]):
            ts = field_sub[:, i, j]
            ts = ts - np.mean(ts)
            ts = ts - np.polyval(np.polyfit(t_all, ts, 1), t_all)
            field_sub[:, i, j] = ts * np.hanning(nt)
    # 3D FFT: rFFT in t, then FFT2 in space
    f3d = rfft(field_sub, axis=0, n=nt)  # (n_omega, nx1//ss, nx2//ss)
    f3d = fftshift(fft2(f3d, axes=(1, 2)), axes=(1, 2))  # shift spatial axes
    fields_3d[name] = np.abs(f3d)

omega_arr = rfftfreq(nt, d=dt_out)
k1_sub = fftshift(fftfreq(nx1//ss, d=dx1*ss)) * 2 * np.pi
k2_sub = fftshift(fftfreq(nx2//ss, d=dx2*ss)) * 2 * np.pi
print(f"  3D spectrum shape: (n_omega={len(omega_arr)}, nk1={len(k1_sub)}, nk2={len(k2_sub)})")

# P_E and P_B combined power
P_E = fields_3d['Ex']**2 + fields_3d['Ey']**2 + fields_3d['Ez']**2
P_B = fields_3d['Bx']**2 + fields_3d['By']**2 + fields_3d['Bz']**2

# Plot: P(k1, omega) at k2=0
ik2_z = len(k2_sub) // 2  # k2=0 index
ik1_z = len(k1_sub) // 2  # k1=0 index

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('k-omega Spectrum  density=1.0  Bperp=0.0', fontsize=14)

kw_plots = [
    ('P_E(k1,w) k2=0', P_E[:, :, ik2_z], omega_arr, k1_sub, 'P_E'),
    ('P_E(k2,w) k1=0', P_E[:, ik1_z, :], omega_arr, k2_sub, 'P_E'),
    ('P_B(k1,w) k2=0', P_B[:, :, ik2_z], omega_arr, k1_sub, 'P_B'),
    ('P_B(k2,w) k1=0', P_B[:, ik1_z, :], omega_arr, k2_sub, 'P_B'),
    ('|Ey|(k1,w) k2=0', fields_3d['Ey'][:, :, ik2_z], omega_arr, k1_sub, '|Ey|'),
    ('|Bz|(k1,w) k2=0', fields_3d['Bz'][:, :, ik2_z], omega_arr, k1_sub, '|Bz|'),
]

for (title, data_2d, w_arr, k_arr, label), ax in zip(kw_plots, axes.flat):
    # data_2d shape: (n_omega, nk)
    # Only show positive omega
    w_pos = slice(1, None)  # skip omega=0
    im = ax.pcolormesh(k_arr, w_arr[w_pos], data_2d[w_pos, :].T,
                        shading='auto', cmap='inferno', norm=LogNorm())
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axhline(OMEGA_A, color='cyan', ls='--', alpha=0.6, label=f'wa={OMEGA_A:.3f}')
    ax.axvline(K_A, color='lime', ls='--', alpha=0.6, label=f'ka={K_A}')
    ax.set(xlabel='k', ylabel='omega', title=title)
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f"{OUT}/05_k_omega_spectrum.png", dpi=150)
print(f"Saved: {OUT}/05_k_omega_spectrum.png")

# k1-omega zoom on Ey low-frequency region
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
data_ey = fields_3d['Ey'][:, :, ik2_z]  # (n_omega, nk1)
w_zoom = slice(1, 60)  # zoom on low omega
im = ax.pcolormesh(k1_sub, omega_arr[w_zoom], data_ey[w_zoom, :].T,
                    shading='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046)
ax.axhline(OMEGA_A, color='cyan', ls='--', alpha=0.6, label=f'wa={OMEGA_A:.3f}')
ax.axvline(K_A, color='lime', ls='--', alpha=0.6, label=f'ka={K_A}')
ax.set(xlabel='k1', ylabel='omega', title='|Ey|(k1,omega) k2=0 — Low-freq Zoom')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/05_k_omega_Ey_zoom.png", dpi=150)
print(f"Saved: {OUT}/05_k_omega_Ey_zoom.png")

# k1-omega zoom on Bz
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
data_bz = fields_3d['Bz'][:, :, ik2_z]
im = ax.pcolormesh(k1_sub, omega_arr[w_zoom], data_bz[w_zoom, :].T,
                    shading='auto', cmap='inferno', norm=LogNorm())
plt.colorbar(im, ax=ax, fraction=0.046)
ax.axhline(OMEGA_A, color='cyan', ls='--', alpha=0.6, label=f'wa={OMEGA_A:.3f}')
ax.axvline(K_A, color='lime', ls='--', alpha=0.6, label=f'ka={K_A}')
ax.set(xlabel='k1', ylabel='omega', title='|Bz|(k1,omega) k2=0 — Low-freq Zoom')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/05_k_omega_Bz_zoom.png", dpi=150)
print(f"Saved: {OUT}/05_k_omega_Bz_zoom.png")

print("\nDone! Output in:", OUT)
