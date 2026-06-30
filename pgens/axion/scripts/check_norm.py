import nt2, numpy as np

def load(case, runtime=None):
    data = nt2.Data(case); f = data.fields
    x = f.x.values
    if runtime:
        T = f['Ttt'].sel(t=slice(None, runtime)).values
        E = f['Ex'].sel(t=slice(None, runtime)).values
        Ey = f['Ey'].sel(t=slice(None, runtime)).values
        Ez = f['Ez'].sel(t=slice(None, runtime)).values
    else:
        T = f['Ttt'].values
        E = f['Ex'].values; Ey = f['Ey'].values; Ez = f['Ez'].values
    B0_PHYS = 100.0; SIGMA = 2500.0
    E_phys = np.trapezoid(E**2 + Ey**2 + Ez**2, x, axis=1) * 0.5 * B0_PHYS**2 * SIGMA
    T_sum = np.trapezoid(T, x, axis=1)
    return T_sum, E_phys, len(x)

BASE = '/home/staticobserver/entity/problems/axion'
T1, E1, nx1 = load(f'{BASE}/plasma_larmor1e-2_or0.5_n16392', 1000.0)
T2, E2, nx2 = load(f'{BASE}/plasma_larmor1e-2_or0.5_t2000', 1000.0)

print('=== Without PPC/dx factor (resolution-independent) ===')
print(f'{"":25s} {"n="+str(nx1):>15s} {"n="+str(nx2):>15s} {"ratio":>10s}')
print('-' * 65)
print(f'{"Int Ttt dx [t=0]":25s} {T1[0]:15.1f} {T2[0]:15.1f} {T1[0]/T2[0]:10.2f}')
print(f'{"Int Ttt dx [final]":25s} {T1[-1]:15.1f} {T2[-1]:15.1f} {T1[-1]/T2[-1]:10.2f}')

dT1 = T1[-1] - T1[0]
dT2 = T2[-1] - T2[0]
print(f'{"dT (gain)":25s} {dT1:15.1f} {dT2:15.1f} {dT1/dT2:10.2f}')
print(f'{"E_phys [t=0]":25s} {E1[0]:15.1f} {E2[0]:15.1f} {E1[0]/E2[0]:10.2f}')
print(f'{"E_phys [final]":25s} {E1[-1]:15.1f} {E2[-1]:15.1f} {E1[-1]/E2[-1]:10.2f}')
print()

print('=== Per-particle check ===')
ppt1 = dT1 / (nx1 * 128)
ppt2 = dT2 / (nx2 * 128)
print(f'dT per particle (n={nx1}): {ppt1:.6f}')
print(f'dT per particle (n={nx2}): {ppt2:.6f}')
print(f'ratio: {ppt1/ppt2:.4f}')
print()

print('=== Key ratios with correct normalization ===')
print(f'dT/E_phys[final] (n={nx1}): {dT1/E1[-1]:.4f}')
print(f'dT/E_phys[final] (n={nx2}): {dT2/E2[-1]:.4f}')
