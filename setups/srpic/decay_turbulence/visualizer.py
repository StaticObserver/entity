import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

def compute_spectrum(field, dx, num_bins):
    Ny, Nx = field.shape
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    dkx = 2 * np.pi / (Nx * dx)
    dky = 2 * np.pi / (Ny * dx)
    kx, ky = np.meshgrid(
        np.linspace(-Nx/2 * dkx, Nx/2 * dkx - dkx, Nx),
        np.linspace(-Ny/2 * dky, Ny/2 * dky - dky, Ny)
    )
    k_mag = np.sqrt(kx**2 + ky**2).flatten()
    power_spectrum_flatten = power_spectrum.flatten()
    k_bins = np.logspace(np.log10(2 * np.pi / (Nx * dx)), np.log10(2 * np.pi / dx), num=num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    k_indices = np.digitize(k_mag, k_bins)
    power_spectrum_binned = np.array([
        np.sum(power_spectrum_flatten[k_indices == i]) if np.any(k_indices == i) else 0 
        for i in range(1, len(k_bins))
    ])
    power_spectrum_binned /= (Nx * Ny)
    return k_bin_centers, power_spectrum_binned

def characterL(field, dx):
    k_bin_centers, power_spectrum_binned = compute_spectrum(field, dx, 200)
    return np.dot(1.0 / k_bin_centers, power_spectrum_binned) / np.sum(power_spectrum_binned) 


@xr.register_dataset_accessor('visualizer')
class VisualizerAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.times = self._obj.coords['t'].values
        self.dx = self._obj.coords['x'].values[1] - self._obj.coords['x'].values[0]
        self.d0 = None  
        self.rho0 = None
        self.field_map = {
            'B2': lambda data: (data.Bx**2 + data.By**2 + data.Bz**2) ,
            'E2': lambda data: (data.Ex**2 + data.Ey**2 + data.Ez**2) ,
            'EM_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                             data.Bx**2 + data.By**2 + data.Bz**2) * (self.d0 / self.rho0)**2,
            'Prtl_Energy': lambda data: data.T00,
            'Total_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                                data.Bx**2 + data.By**2 + data.Bz**2) * (self.d0 / self.rho0)**2 + data.T00,
            'N' : lambda data: data.N_1 + data.N_2,
            'Bxy_Energy' : lambda data: (data.Bx**2 + data.By**2) * (self.d0 / self.rho0)**2, 
        }
        self.norm_map = {
            'norm': lambda vmin, vmax: mpl.colors.Normalize(vmin, vmax),
            'log': lambda vmin, vmax: mpl.colors.LogNorm(vmin, vmax),
        }

    def _check_params_set(self):
        if self.d0 is None or self.rho0 is None:
            raise ValueError("d0 and rho0 must be set before calling this method.")
        
    def get_data(self, name):
        if name not in self.field_map:
            raise ValueError("Invalid type.")
        else:
            return self.field_map[name](self._obj)

    def set_params(self, d0, rho0):
        self.d0 = d0
        self.rho0 = rho0
    
    def decay_rate(self, name, **kwargs):
        if name not in self.field_map:
            raise ValueError("Invalid type.")
        data = self.field_map[name](self._obj)
        ts = self.times[1:]
        Qs = np.array([data.sel({'t': t}, method="nearest").mean(('x', 'y')).compute().item() for t in tqdm(self.times[1:], desc="Computing means")])
        rate = np.array([np.log(Qs[i-1] / Qs[i+1]) / np.log(ts[i+1] / ts[i-1]) for i in range(1, len(ts) - 1)])
        plt.plot(ts[1:-1], rate, **kwargs)
        plt.savefig("decay_{}.png".format(name), dpi=300, bbox_inches="tight")
        np.savetxt("decay_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))
        
    def dLdt(self, name, **kwargs):
        if name in self.field_map:
            data = self.field_map[name](self._obj)
        elif hasattr(self._obj, name):
            data = getattr(self._obj, name)
        else:
            raise ValueError("Invalid type.")
        ts = self.times[1:]
        Qs = np.array([characterL(data.sel({'t': t}, method="nearest").values, self.dx) for t in tqdm(self.times[1:], desc="Computing means")])
        rate = np.array([np.log(Qs[i+1] / Qs[i-1]) / np.log(ts[i+1] / ts[i-1]) for i in range(1, len(ts) - 1)])
        plt.plot(ts[1:-1], rate, **kwargs)
        plt.savefig("dLdt_{}.png".format(name), dpi=300, bbox_inches="tight")
        np.savetxt("dLdt_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))
    
    def plot_mean(self, name, times, scale=None, **kwargs):
        self._check_params_set()
        if hasattr(self._obj, name):
            plot_data = getattr(self._obj, name)
        elif name in self.field_map:
            plot_data = self.field_map[name]
        else:
            raise ValueError("Invalid type.")
        means = [plot_data.sel({'t': t}, method="nearest").mean(('x', 'y')).compute().item() for t in tqdm(times, desc="Computing mean")]
        plt.plot(times, means, **kwargs)
        if scale == 'log':
            plt.xscale('log')
            plt.yscale('log')
        plt.savefig("mean_{}.png".format(name), dpi=300, bbox_inches="tight")
        plt.close()
        np.savetxt("mean_{}.dat".format(name), np.column_stack((times, means)))
    
    def plot_spectrum(self, t, name, num_bins=200, y_min=None, y_max=None, **kwargs):
        self._check_params_set()
        if hasattr(self._obj, name):
            field = getattr(self._obj, name)
        elif name in self.field_map:
            field = self.field_map[name]
        else:
            raise ValueError("Invalid type.")
        k_bin_centers, power_spectrum_binned = compute_spectrum(field, self.dx, num_bins)
        fig, ax = plt.subplots()
        ax.plot(k_bin_centers, power_spectrum_binned, '-', **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|k|$')
        ax.set_ylabel('Energy Spectrum Density')
        ax.set_title('t = {:.2f}'.format(t))
        ax.grid(True)
        if y_max is None:
            y_max = np.max(power_spectrum_binned)
        if y_min is None:
            y_min = y_max / 1e6
        ax.set_ylim(bottom=y_min, top=y_max)
        plt.savefig("spectrum_{}_t{:.2f}.png".format(name, t), dpi=300, bbox_inches="tight")
        plt.close()

