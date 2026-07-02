"""Extract mode peak table from 3D k-omega spectrum — CORRECT angular freq.
FFT rfftfreq returns f (cycles/time). Multiply by 2π to get angular ω."""
import numpy as np
from numpy.fft import fft2, fftshift, fftfreq, rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter
import nt2

DATA   = "/home/staticobserver/entity/problems/axion/build/2d_bx2_spectrum_d1.0_Bperp0.0"
OMEGA_A = 1.0   # angular frequency
K_A = 0.1       # wavenumber

print("Loading data...")
data = nt2.Data(DATA)
ds = data.fields
x1 = ds.x.values; x2 = ds.y.values
t_all = ds.t.values.ravel()
nt = len(t_all)
nx1, nx2 = len(x1), len(x2)
dx1, dx2 = x1[1]-x1[0], x2[1]-x2[0]
dt_out = np.median(np.diff(t_all))
T_total = t_all[-1] - t_all[0]
# ANGULAR frequency resolution
domega_ang = 2*np.pi / T_total
print(f"nt={nt}, T={T_total:.1f}, domega_ang={domega_ang:.4f} rad")

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

# KEY FIX: rfftfreq returns f (cycles/time). Convert to angular ω.
f_arr = rfftfreq(nt, d=dt_out)
omega_arr = f_arr * 2 * np.pi  # ANGULAR frequency
k1_arr = fftshift(fftfreq(nx1//ss, d=dx1*ss)) * 2*np.pi
k2_arr = fftshift(fftfreq(nx2//ss, d=dx2*ss)) * 2*np.pi
ik2_z = len(k2_arr)//2
ik1_z = len(k1_arr)//2

print(f"\n  omega_Nyquist_ang = {omega_arr[-1]:.2f} rad")
print(f"  omega_a = {OMEGA_A:.2f}, domega = {domega_ang:.4f}")
print(f"  wa/domega = {OMEGA_A/domega_ang:.1f} bins")
print(f"  FFT bin nearest to wa: omega[{np.argmin(np.abs(omega_arr-OMEGA_A))}]={omega_arr[np.argmin(np.abs(omega_arr-OMEGA_A))]:.4f}")

P_E = field_3d['Ex']**2 + field_3d['Ey']**2 + field_3d['Ez']**2
P_B = field_3d['Bx']**2 + field_3d['By']**2 + field_3d['Bz']**2
P_tot = P_E + P_B

# ── Peak Finding ────────────────────────────────────────────────────
def find_2d_peaks(spec, k_arr, w_arr, threshold_factor=0.1, min_dist=3):
    max_val = spec.max()
    threshold = max_val * threshold_factor
    footprint = np.ones((min_dist*2+1, min_dist*2+1))
    local_max = maximum_filter(spec, footprint=footprint) == spec
    peaks_mask = local_max & (spec > threshold)
    pi, pk = np.where(peaks_mask)
    vals = spec[pi, pk]
    order = np.argsort(vals)[::-1]
    results = []
    for idx in order[:20]:
        results.append({
            'omega_ang': w_arr[pi[idx]],
            'k1': k_arr[pk[idx]],
            'power': vals[idx],
            'rel_power': vals[idx]/max_val
        })
    return results

for label, spec in [('P_E', P_E[:,:,ik2_z]), ('P_B', P_B[:,:,ik2_z]),
                     ('|Ey|', field_3d['Ey'][:,:,ik2_z]),
                     ('|Bz|', field_3d['Bz'][:,:,ik2_z]),
                     ('|Ex|', field_3d['Ex'][:,:,ik2_z])]:
    print(f"\n{'='*70}")
    print(f"PEAK TABLE: {label}(k1, ω) at k2=0  [ω = ANGULAR frequency]")
    print(f"{'='*70}")
    peaks = find_2d_peaks(spec, k1_arr, omega_arr)
    for i, p in enumerate(peaks[:10]):
        k_abs = abs(p['k1'])
        v_ph = p['omega_ang']/k_abs if k_abs > 0.001 else np.inf
        near_wa = " *** AXION ***" if abs(p['omega_ang']-OMEGA_A)/OMEGA_A < 0.15 else ""
        near_2wa = " 2wa?" if abs(p['omega_ang']-2*OMEGA_A)/(2*OMEGA_A) < 0.15 else ""
        print(f"  {i+1:2d}: ω={p['omega_ang']:.4f}, k1={p['k1']:+.4f}, "
              f"v_ph={v_ph:.3f}, power={p['power']:.2e}, rel={p['rel_power']:.3f}{near_wa}{near_2wa}")

# ── Ex at axion (ω_a=1.0, k_a=0.1) ──────────────────────────────────
print(f"\n{'='*70}")
print("Ex power at AXION (ω_a=1.0, k_a=0.1)")
print(f"{'='*70}")
ex_kw = field_3d['Ex'][:,:,ik2_z]
iw_a = np.argmin(np.abs(omega_arr - OMEGA_A))
ik_a = np.argmin(np.abs(k1_arr - K_A))
print(f"  Ex(ω={omega_arr[iw_a]:.4f}, k1={k1_arr[ik_a]:.4f}) = {ex_kw[iw_a, ik_a]:.4e}")
print(f"  Max Ex: {ex_kw.max():.2e} at ω={omega_arr[np.unravel_index(ex_kw.argmax(), ex_kw.shape)[0]]:.4f}")
print(f"  Ex(wa,ka) / Ex_max = {ex_kw[iw_a, ik_a]/ex_kw.max():.2e}")

# ── Dominant field per angular frequency band ───────────────────────
print(f"\n{'='*70}")
print("DOMINANT FIELD per angular frequency band")
print(f"{'='*70}")
bands = [(0.2,0.5), (0.5,0.8), (0.8,1.2), (1.2,1.8), (1.8,2.5)]
for w_lo, w_hi in bands:
    i_lo = np.argmin(np.abs(omega_arr - w_lo))
    i_hi = np.argmin(np.abs(omega_arr - w_hi))
    bs = slice(min(i_lo,i_hi), max(i_lo,i_hi)+1)
    powers = {n: field_3d[n][bs,:,:].sum() for n in ['Ex','Ey','Ez','Bx','By','Bz']}
    total = sum(powers.values())
    dominant = max(powers, key=powers.get)
    ax_flag = " ← AXION" if abs((w_lo+w_hi)/2 - OMEGA_A)/OMEGA_A < 0.2 else ""
    print(f"  ω∈[{w_lo:.1f},{w_hi:.1f}]{ax_flag}: dominant={dominant} ({powers[dominant]/total*100:.0f}%)")
    for n in ['Ex','Ey','Ez','Bx','By','Bz']:
        print(f"    {n}: {powers[n]/total*100:5.1f}%")

# ── Mode Classification ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("MODE CLASSIFICATION")
print(f"{'='*70}")

# Find the strongest peak overall
all_peaks = find_2d_peaks(P_tot[:,:,ik2_z], k1_arr, omega_arr)
print(f"\n  Dominant mode: ω={all_peaks[0]['omega_ang']:.4f}, k1={all_peaks[0]['k1']:.4f}")
# Check if this is near ω_a
w_peak = all_peaks[0]['omega_ang']
near_wa = abs(w_peak - OMEGA_A)/OMEGA_A < 0.15
print(f"  Δω/ω_a = {abs(w_peak-OMEGA_A)/OMEGA_A:.3f} → {'AXION-FORCED' if near_wa else 'other'}")

# Check Bz harmonic chain
bz_kw = field_3d['Bz'][:,:,ik2_z]
bz_peaks = find_2d_peaks(bz_kw, k1_arr, omega_arr, threshold_factor=0.05)
print(f"\n  Bz peaks (ω_ang, k1, v_ph):")
for p in bz_peaks[:8]:
    print(f"    ω={p['omega_ang']:.4f}, k1={p['k1']:+.4f}, v_ph={p['omega_ang']/abs(p['k1']):.3f}")

print("\nDone!")
