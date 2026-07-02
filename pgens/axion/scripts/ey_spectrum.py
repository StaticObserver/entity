"""Ey frequency spectrum analysis — density=1.0, B0_perp=0.0."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import nt2

# ── Config ──────────────────────────────────────────────────────────
DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum_d1.0_Bperp0.0"
OUT    = DATA
OMEGA_RATIO = 0.5
SKIN   = 0.5
OMEGA_A = OMEGA_RATIO / SKIN  # = 1.0

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1, nx2 = ds['Ex'].shape[1], ds['Ex'].shape[2]
print(f"nt={nt}, nx1={nx1}, nx2={nx2}")

# Probe Ey at center of domain
ix, iy = nx1 // 2, nx2 // 2
ey_probe = ds['Ey'].values[:, iy, ix]  # x2-avg or center point

dt = t_all[1] - t_all[0]
freqs = rfftfreq(nt, d=dt)
ey_fft = np.abs(rfft(ey_probe))

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Ey Spectrum  density=1.0  wa={OMEGA_A:.3f}  center=({ix},{iy})', fontsize=13)

ax = axes[0]
ax.plot(t_all, ey_probe, lw=0.5)
ax.set(xlabel='t', ylabel='Ey', title=f'Ey timeseries at center')
ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(freqs[freqs > 0], ey_fft[freqs > 0], lw=0.8)
ax.axvline(OMEGA_A, color='red', ls='--', alpha=0.6, label=f'axion wa={OMEGA_A:.3f}')
ax.set(xlabel='omega', ylabel='|FFT(Ey)|', title='Frequency Spectrum',
       xlim=(0, freqs[-1]))
ax.legend()
ax.grid(alpha=0.3)

# Find peaks
peaks_idx, props = find_peaks(np.log10(ey_fft[1:] + 1e-20), height=-10, distance=2)
peaks_freq = freqs[1:][peaks_idx]
peaks_amp = ey_fft[1:][peaks_idx]
order = np.argsort(peaks_amp)[::-1][:10]
print("\nTop frequency peaks (omega, amplitude):")
for f, a in zip(peaks_freq[order], peaks_amp[order]):
    print(f"  w={f:.4f}, amp={a:.2e}")

plt.tight_layout()
out_path = f"{OUT}/ey_spectrum.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
