"""Extract mode peak table from 3D k-omega spectrum — numerical output."""
import numpy as np
from numpy.fft import fft2, fftshift, fftfreq, rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
import nt2

DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum_d1.0_Bperp0.0"
OMEGA_A = 1.0
K_A = 0.1

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values
x2 = ds.y.values
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1, nx2 = len(x1), len(x2)
dx1, dx2 = x1[1]-x1[0], x2[1]-x2[0]
dt_out = np.median(np.diff(t_all))
domega = 2*np.pi/(t_all[-1]-t_all[0])
print(f"nt={nt}, nx1={nx1}, nx2={nx2}, domega={domega:.4f}, dt={dt_out:.4f}")

# 3D FFT with spatial subsample
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

omega_arr = rfftfreq(nt, d=dt_out)
k1_arr = fftshift(fftfreq(nx1//ss, d=dx1*ss)) * 2*np.pi
k2_arr = fftshift(fftfreq(nx2//ss, d=dx2*ss)) * 2*np.pi
ik2_z = len(k2_arr)//2
ik1_z = len(k1_arr)//2

P_E = field_3d['Ex']**2 + field_3d['Ey']**2 + field_3d['Ez']**2
P_B = field_3d['Bx']**2 + field_3d['By']**2 + field_3d['Bz']**2
P_tot = P_E + P_B

# ── Peak Finding in k1-omega slice (k2=0) ───────────────────────────
def find_2d_peaks(spec, k_arr, w_arr, threshold_factor=0.1, min_dist=3):
    """Find local maxima in 2D spectrum, return sorted list."""
    max_val = spec.max()
    threshold = max_val * threshold_factor
    # Local max filter
    footprint = np.ones((min_dist*2+1, min_dist*2+1))
    local_max = maximum_filter(spec, footprint=footprint) == spec
    peaks_mask = local_max & (spec > threshold)
    pi, pk = np.where(peaks_mask)
    vals = spec[pi, pk]
    order = np.argsort(vals)[::-1]
    results = []
    for idx in order[:20]:
        results.append({
            'omega': w_arr[pi[idx]],
            'k1': k_arr[pk[idx]],
            'power': vals[idx],
            'rel_power': vals[idx]/max_val
        })
    return results

print("\n" + "="*70)
print("PEAK TABLE: P_E(k1, omega) at k2=0")
print("="*70)
pe_peaks = find_2d_peaks(P_E[:, :, ik2_z], k1_arr, omega_arr)
for i, p in enumerate(pe_peaks[:12]):
    print(f"  M{i+1:2d}: w={p['omega']:.4f}, k1={p['k1']:.4f}, "
          f"power={p['power']:.2e}, rel={p['rel_power']:.3f}")

print("\n" + "="*70)
print("PEAK TABLE: P_B(k1, omega) at k2=0")
print("="*70)
pb_peaks = find_2d_peaks(P_B[:, :, ik2_z], k1_arr, omega_arr)
for i, p in enumerate(pb_peaks[:12]):
    print(f"  M{i+1:2d}: w={p['omega']:.4f}, k1={p['k1']:.4f}, "
          f"power={p['power']:.2e}, rel={p['rel_power']:.3f}")

# ── Field-specific peaks: Ey ────────────────────────────────────────
print("\n" + "="*70)
print("PEAK TABLE: |Ey|(k1, omega) at k2=0")
print("="*70)
ey_kw = field_3d['Ey'][:, :, ik2_z]
ey_peaks = find_2d_peaks(ey_kw, k1_arr, omega_arr, threshold_factor=0.05)
for i, p in enumerate(ey_peaks[:12]):
    k_abs = abs(p['k1'])
    phase_v = p['omega']/k_abs if k_abs > 0.001 else np.inf
    print(f"  M{i+1:2d}: w={p['omega']:.4f}, k1={p['k1']:.4f}, "
          f"v_ph={phase_v:.2f}, power={p['power']:.2e}, rel={p['rel_power']:.3f}")

# ── Field-specific peaks: Bz ────────────────────────────────────────
print("\n" + "="*70)
print("PEAK TABLE: |Bz|(k1, omega) at k2=0")
print("="*70)
bz_kw = field_3d['Bz'][:, :, ik2_z]
bz_peaks = find_2d_peaks(bz_kw, k1_arr, omega_arr, threshold_factor=0.05)
for i, p in enumerate(bz_peaks[:12]):
    k_abs = abs(p['k1'])
    phase_v = p['omega']/k_abs if k_abs > 0.001 else np.inf
    print(f"  M{i+1:2d}: w={p['omega']:.4f}, k1={p['k1']:.4f}, "
          f"v_ph={phase_v:.2f}, power={p['power']:.2e}, rel={p['rel_power']:.3f}")

# ── Ex at axion frequency ───────────────────────────────────────────
print("\n" + "="*70)
print("CHECK: Ex power at axion (w=1.0, k1=0.1)")
print("="*70)
ex_kw = field_3d['Ex'][:, :, ik2_z]
# Find index closest to wa=1.0 and ka=0.1
iw_a = np.argmin(np.abs(omega_arr - OMEGA_A))
ik_a = np.argmin(np.abs(k1_arr - K_A))
print(f"  Ex(w={omega_arr[iw_a]:.4f}, k1={k1_arr[ik_a]:.4f}) = {ex_kw[iw_a, ik_a]:.4e}")
print(f"  Max Ex overall: {ex_kw.max():.4e} at w={omega_arr[np.unravel_index(ex_kw.argmax(), ex_kw.shape)[0]]:.4f}")
print(f"  P_E at (wa,ka): {P_E[iw_a, :, ik2_z].max():.4e}")

# ── Dominant field at each peak frequency ────────────────────────────
print("\n" + "="*70)
print("DOMINANT FIELD per frequency band")
print("="*70)
# Check at center of k-space for low-freq bands
freq_bands = [(0.05, 0.12), (0.12, 0.20), (0.20, 0.35), (0.35, 0.50), (0.90, 1.10)]
for f_lo, f_hi in freq_bands:
    i_lo = np.argmin(np.abs(omega_arr - f_lo))
    i_hi = np.argmin(np.abs(omega_arr - f_hi))
    band_slice = slice(min(i_lo, i_hi), max(i_lo, i_hi)+1)
    band_powers = {}
    for name in ['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']:
        band_powers[name] = field_3d[name][band_slice, :, :].sum()
    dominant = max(band_powers, key=band_powers.get)
    total = sum(band_powers.values())
    print(f"  Band [{f_lo:.2f},{f_hi:.2f}]: dominant={dominant} "
          f"({band_powers[dominant]/total*100:.0f}%)")
    for name in ['Ex','Ey','Ez','Bx','By','Bz']:
        print(f"    {name}: {band_powers[name]/total*100:5.1f}%")

# ── Polarization at key k-omega points ───────────────────────────────
print("\n" + "="*70)
print("POLARIZATION at Ey peak (w≈0.155)")
print("="*70)
iw = np.argmin(np.abs(omega_arr - 0.155))
# Sum over k-space near k=0 region
k_center = slice(ik1_z-10, ik1_z+10)
for name in ['Ex', 'Ey', 'Ez']:
    pwr = (field_3d[name][iw, k_center, ik2_z]**2).sum()
    print(f"  P_{name} = {pwr:.4e}")
Epar_pwr = (field_3d['Ex'][iw, k_center, ik2_z]**2).sum()
Eperp_pwr = (field_3d['Ey'][iw, k_center, ik2_z]**2).sum()
Ez_pwr = (field_3d['Ez'][iw, k_center, ik2_z]**2).sum()
Etot = Epar_pwr + Eperp_pwr + Ez_pwr
print(f"  E_parallel (Ex): {Epar_pwr/Etot*100:.1f}%")
print(f"  E_perp_xy (Ey):  {Eperp_pwr/Etot*100:.1f}%")
print(f"  E_z (Ez):        {Ez_pwr/Etot*100:.1f}%")
B_pwr_w = (field_3d['Bz'][iw, k_center, ik2_z]**2).sum()
print(f"  P_B / (P_E+P_B) = {B_pwr_w/(Etot+B_pwr_w)*100:.1f}%")

print("\nDone!")
