import nt2, numpy as np

def load_energy(case_path, PPC=128.0, SKIN=0.5, LARMOR=0.01, RUNTIME=None):
    data = nt2.Data(case_path)
    fields = data.fields
    x = fields.x.values; dx = x[1] - x[0]
    if RUNTIME is not None:
        Ex = fields['Ex'].sel(t=slice(None, RUNTIME)).values
        Ey = fields['Ey'].sel(t=slice(None, RUNTIME)).values
        Ez = fields['Ez'].sel(t=slice(None, RUNTIME)).values
        Ttt = fields['Ttt'].sel(t=slice(None, RUNTIME)).values
    else:
        Ex = fields['Ex'].values; Ey = fields['Ey'].values
        Ez = fields['Ez'].values; Ttt = fields['Ttt'].values

    B0_PHYS = 1.0 / LARMOR; SIGMA = (SKIN/LARMOR)**2
    E_phys = np.trapezoid(Ex**2 + Ey**2 + Ez**2, x, axis=1) * 0.5 * B0_PHYS**2 * SIGMA
    par_phys = np.trapezoid(Ttt, x, axis=1) * PPC / dx
    dPar = par_phys - par_phys[0]
    return E_phys, dPar, len(x)

BASE = '/home/staticobserver/entity/problems/axion'
E1, dPar1, nx1 = load_energy(f'{BASE}/plasma_larmor1e-2_or0.5_n16392', RUNTIME=1000.0)
E2, dPar2, nx2 = load_energy(f'{BASE}/plasma_larmor1e-2_or0.5_t2000', RUNTIME=1000.0)

dE1 = E1[-1] - E1[0]
dE2 = E2[-1] - E2[0]
ratio1 = dPar1[-1] / E1[-1]
ratio2 = dPar2[-1] / E2[-1]

print(f'=== Resolution Comparison (t=0..1000) ===')
print(f'')
print(f'{"Metric":25} {"n="+str(nx1):>16} {"n="+str(nx2):>16} {"ratio":>10}')
print(f'{"-"*67}')
print(f'{"E-field[t=0]":25} {E1[0]:16.1f} {E2[0]:16.1f} {E1[0]/E2[0]:10.2f}')
print(f'{"E-field[final]":25} {E1[-1]:16.1f} {E2[-1]:16.1f} {E1[-1]/E2[-1]:10.2f}')
print(f'{"dE (final-init)":25} {dE1:16.1f} {dE2:16.1f} {dE1/dE2:10.2f}')
print(f'{"dPar[final]":25} {dPar1[-1]:16.0f} {dPar2[-1]:16.0f} {dPar1[-1]/dPar2[-1]:10.2f}')
print(f'{"dPar / E[final]":25} {ratio1:16.4f} {ratio2:16.4f} {ratio1/ratio2:10.2f}')
print(f'')
print(f'dPar/dE (n={nx1}): {dPar1[-1]/dE1:.2f}')
print(f'dPar/dE (n={nx2}): {dPar2[-1]/dE2:.2f}')
