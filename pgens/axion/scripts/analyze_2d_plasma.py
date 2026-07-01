"""2D plasma analysis: energy budget, Bz spectrum, frequency peaks, mode growth."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.fft import fft2, fftshift, fftfreq, rfft, rfftfreq
from scipy.signal import find_peaks
import pandas as pd
import nt2

# ── config ──────────────────────────────────────────────────────────
DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum"
OUT    = "/home/staticobserver/entity/problems/axion/2d_bx2_spectrum_d1_ppc16"
EPS    = 0.01
K      = 0.1
OMEGA_RATIO = 0.5
SKIN   = 0.5
LARMOR = 0.01
B0     = 1.0
B0_PERP = 0.05
SIGMA  = (SKIN / LARMOR) ** 2

OMEGA_A = OMEGA_RATIO / SKIN  # 1.0

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values
x2 = ds.y.values if hasattr(ds, "y") else x1
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1 = len(x1)
nx2 = len(x2)
print(f"Shape: nt={nt}, nx1={nx1}, nx2={nx2}")

stats = pd.read_csv(f"{DATA}/2d_bx2_spectrum_stats.csv")
stats.columns = stats.columns.str.strip()
t_stats = stats["time"].values

# ── 1. Energy diagnostics ───────────────────────────────────────────
print("\n=== Energy Diagnostics ===")
L1 = x1[-1] - x1[0]
L2 = x2[-1] - x2[0]
area = L1 * L2

B1sq = stats["B1^2"].values
B2sq = stats["B2^2"].values
B3sq = stats["B3^2"].values
E1sq = stats["E1^2"].values
E2sq = stats["E2^2"].values
E3sq = stats["E3^2"].values
T00  = stats["T00"].values

B_phys_scale = (1.0 / LARMOR) ** 2
E_tot = (E1sq + E2sq + E3sq) * area * 0.5 * B_phys_scale
B_tot = (B1sq + B2sq + B3sq) * area * 0.5 * B_phys_scale

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
suptitle = f"2D Axion-PIC  density=1.0 PPC=16 eps={EPS} wa/wp={OMEGA_RATIO} Bperp={B0_PERP}"
fig.suptitle(suptitle, fontsize=13)

ax = axes[0, 0]
ax.plot(t_stats, E1sq, label="E1^2")
ax.plot(t_stats, E2sq, label="E2^2")
ax.plot(t_stats, E3sq, label="E3^2")
ax.legend()
ax.set(xlabel="t", ylabel="<E^2>", title="E-field squared")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_stats, B1sq, label="B1^2")
ax.plot(t_stats, B2sq, label="B2^2")
ax.plot(t_stats, B3sq, label="B3^2")
ax.legend()
ax.set(xlabel="t", ylabel="<B^2>", title="B-field squared")
ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(t_stats, E_tot, label="E")
ax.plot(t_stats, B_tot, label="B")
ax.plot(t_stats, E_tot + B_tot, "k--", alpha=0.5, label="Total EM")
ax.legend()
ax.set(xlabel="t", ylabel="Energy (phys)", title="Total EM Energy")
ax.grid(alpha=0.3)

ax = axes[1, 0]
dE = E_tot - E_tot[0]
dB = B_tot - B_tot[0]
ax.plot(t_stats, dE, label="dE")
ax.plot(t_stats, dB, label="dB")
ax.plot(t_stats, dE + dB, "k--", alpha=0.5, label="d(E+B)")
ax.legend()
ax.set(xlabel="t", ylabel="delta Energy", title="EM Energy Change")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(t_stats, T00)
ax.set(xlabel="t", ylabel="T00", title="Particle Energy")
ax.grid(alpha=0.3)

ax = axes[1, 2]
ExB1 = stats["ExB1"].values
ExB2 = stats["ExB2"].values
ExB3 = stats["ExB3"].values
ax.plot(t_stats, ExB1, label="ExB1")
ax.plot(t_stats, ExB2, label="ExB2")
ax.plot(t_stats, ExB3, label="ExB3")
ax.legend()
ax.set(xlabel="t", ylabel="Poynting", title="ExB Flux")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/energy_diagnostics.png", dpi=150)
print(f"Saved: {OUT}/energy_diagnostics.png")

print(f"\nE-field energy growth: {E_tot[-1]/E_tot[0]:.1f}x")
print(f"B-field energy growth: {B_tot[-1]/B_tot[0]:.5f}x")
print(f"T00 growth: {T00[-1]/T00[0]:.1f}x")
print(f"B3^2 final: {B3sq[-1]:.6f}")

# ── 2. Bz spatial profile + power spectrum ──────────────────────────
print("\n=== Bz Spectrum ===")
Bz = ds["Bz"].values
bz_last = Bz[-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
if bz_last.ndim == 2:
    im = ax.pcolormesh(x1, x2, bz_last, shading="auto", cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Bz")
else:
    ax.plot(x1, bz_last)
ax.set(xlabel="x1", ylabel="x2" if bz_last.ndim == 2 else "Bz",
       title=f"Bz at t={t_all[-1]:.1f}")
ax.grid(alpha=0.3)

ax = axes[1]
if bz_last.ndim == 2:
    bz_fft = fftshift(np.abs(fft2(bz_last)))
    kx1 = fftshift(fftfreq(nx1, d=(x1[1]-x1[0]))) * 2 * np.pi
    kx2 = fftshift(fftfreq(nx2, d=(x2[1]-x2[0]))) * 2 * np.pi
    im = ax.pcolormesh(kx1, kx2, bz_fft, shading="auto", cmap="inferno", norm=LogNorm())
    plt.colorbar(im, ax=ax, label="|FFT(Bz)|")
    ax.set(xlabel="k1", ylabel="k2")
else:
    bz_fft = np.abs(rfft(bz_last))
    kx = rfftfreq(nx1, d=(x1[1]-x1[0])) * 2 * np.pi
    ax.semilogy(kx, bz_fft)
    ax.set(xlabel="k")
ax.set(title=f"Power Spectrum of Bz, t={t_all[-1]:.1f}")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/bz_spectrum.png", dpi=150)
print(f"Saved: {OUT}/bz_spectrum.png")

# ── 3. Frequency analysis at center ─────────────────────────────────
print("\n=== Frequency Analysis ===")
if bz_last.ndim == 2:
    ix, iy = nx1 // 2, nx2 // 2
    ey_probe = ds["Ey"].values[:, iy, ix]
    label = f"Ey at center ({ix},{iy})"
else:
    ix = nx1 // 2
    ey_probe = ds["Ey"].values[:, ix]
    label = f"Ey at x={x1[ix]:.1f}"

dt = t_all[1] - t_all[0]
freqs = rfftfreq(nt, d=dt)
ey_fft = np.abs(rfft(ey_probe))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(t_all, ey_probe)
ax.axvline(OMEGA_A, color="red", ls="--", alpha=0.5, label=f"axion wa={OMEGA_A:.3f}")
ax.set(xlabel="t", ylabel="Ey", title=label)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.semilogy(freqs[freqs > 0], ey_fft[freqs > 0])
ax.axvline(OMEGA_A, color="red", ls="--", alpha=0.5, label=f"axion wa={OMEGA_A:.3f}")
ax.set(xlabel="omega", ylabel="|FFT(Ey)|", title="Frequency Spectrum of Ey",
       xlim=(0, freqs[-1]))
ax.legend()
ax.grid(alpha=0.3)

# Find peaks
try:
    peaks_idx, props = find_peaks(np.log10(ey_fft[1:] + 1e-20), height=-10, distance=2)
    peaks_freq = freqs[1:][peaks_idx]
    peaks_amp = ey_fft[1:][peaks_idx]
    order = np.argsort(peaks_amp)[::-1][:10]
    top_peaks = list(zip(peaks_freq[order], peaks_amp[order]))
    print("Top frequency peaks (omega, amplitude):")
    for f, a in top_peaks[:8]:
        print(f"  w={f:.4f}, amp={a:.2e}")
except Exception:
    print("  (peak finding skipped)")

plt.tight_layout()
plt.savefig(f"{OUT}/ey_frequency.png", dpi=150)
print(f"Saved: {OUT}/ey_frequency.png")

# ── 4. Bz mode growth over time ─────────────────────────────────────
print("\n=== Bz Mode Growth ===")
bz_all = ds["Bz"].values
n_modes_track = 5
mode_amps = np.zeros((nt, n_modes_track))
mode_kx = np.zeros(n_modes_track)

for it in range(nt):
    bz = bz_all[it]
    if bz.ndim == 2:
        fft = np.abs(rfft(bz.mean(axis=0)))
        k_all = rfftfreq(nx1, d=(x1[1] - x1[0])) * 2 * np.pi
    else:
        fft = np.abs(rfft(bz))
        k_all = rfftfreq(nx1, d=(x1[1] - x1[0])) * 2 * np.pi
    if it == 0:
        top_idx = np.argsort(fft[1:])[::-1][:n_modes_track] + 1
        mode_kx = k_all[top_idx]
    mode_amps[it] = fft[top_idx]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i in range(n_modes_track):
    ax.semilogy(t_all, mode_amps[:, i], label=f"k={mode_kx[i]:.3f}")
ax.set(xlabel="t", ylabel="|FFT(Bz)|", title="Bz Fourier Mode Amplitudes Over Time")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/bz_mode_growth.png", dpi=150)
print(f"Saved: {OUT}/bz_mode_growth.png")

print("\nDone!")
