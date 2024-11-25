import nt2.read as nt2r
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import nt2.export as exp
import matplotlib
import xarray as xr

def plot_mean(data, name, d0, rho0):
    if name == 'B2':
        bx = getattr(data, 'Bx')
        by = getattr(data, 'By')
        bz = getattr(data, 'Bz')
        plot_data = bx**2 + by**2 + bz**2
    elif name == 'E2':
        ex = getattr(data, 'Ex')
        ey = getattr(data, 'Ey')
        ez = getattr(data, 'Ez')
        plot_data = ex**2 + ey**2 + ez**2
    elif name == 'EM_Energy':
        ex = getattr(data, 'Ex')
        ey = getattr(data, 'Ey')
        ez = getattr(data, 'Ez')
        bx = getattr(data, 'Bx')
        by = getattr(data, 'By')
        bz = getattr(data, 'Bz')
        plot_data = 2.0 * np.pi * (ex**2 + ey**2 + ez**2 + bx**2 + by**2 + bz**2) * (d0 / rho0)**2
    elif name == 'Prtl_Energy':
        plot_data = getattr(data, 'T00')
    elif name == 'Total_Energy':
        ex = getattr(data, 'Ex')
        ey = getattr(data, 'Ey')
        ez = getattr(data, 'Ez')
        bx = getattr(data, 'Bx')
        by = getattr(data, 'By')
        bz = getattr(data, 'Bz')
        plot_data = 2.0 * np.pi * (ex**2 + ey**2 + ez**2 + bx**2 + by**2 + bz**2) * (d0 / rho0)**2 + getattr(data, 'T00')
    elif name == 'N':
        plot_data = getattr(data, 'N_1') + getattr(data, 'N_2')
    else:
        raise ValueError("Invalid field name")
    #plot_data = np.log10(plot_data)
    plot_data.mean(('x', 'y')).plot()
    plt.savefig("mean_" + name + ".png", dpi=300, bbox_inches="tight")
    plt.close() 
           
def plot_attr(steps, data, name, norm_t = 'norm', vmin = 0.0, vmax = 1.0, num_cpus=None):
    if norm_t == 'norm':
       normalize = mpl.colors.Normalize(vmin, vmax)
    elif norm_t == 'log':
       if (vmin * vmax) <= 0:
          raise ValueError("vmin and vmax must be positive for log normlization")
       normalize = mpl.colors.LogNorm(vmin, vmax)
    else:
       raise ValueError("Invalid normalization type.")
    if name == 'B2':
       bx = getattr(data, 'Bx')
       by = getattr(data, 'By')
       bz = getattr(data, 'Bz')
       plot_data = bx**2 + by**2 + bz**2
    elif name == 'Rho':
       plot_data = getattr(data, 'N_1') + getattr(data, 'N_2') 
    else:
       plot_data = getattr(data, name)
    exp.makeFrames(lambda t, data: data.sel({'t':t}, method="nearest").plot(norm = normalize, cmap="jet"), 
                   steps, 
                   name + "/frames", 
                   plot_data,
                   num_cpus)


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

@xr.register_dataarray_accessor("spectrum")
class SpectrumAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, num_bins=200, ax=None, y_min=None, y_max=None, **kwargs):
        field = self._obj.values
        x_coords = self._obj.coords['x'].values
        y_coords = self._obj.coords['y'].values

        # 确定步长和网格数量（假设 dx = dy）
        dx = x_coords[1] - x_coords[0]
        Ny, Nx = field.shape

        # Calculate the power spectrum
        power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
       
        
        # Calculate k-space grids
        dkx = 2 * np.pi / (Nx * dx)
        dky = 2 * np.pi / (Ny * dx)
        kx, ky = np.meshgrid(np.linspace(-Nx/2 * dkx, Nx/2 * dkx - dkx, Nx),
                             np.linspace(-Ny/2 * dky, Ny/2 * dky - dky, Ny))

        # Calculate magnitude of k-vector
        k_mag = np.sqrt(kx**2 + ky**2).flatten()
        power_spectrum_flatten = power_spectrum.flatten()

        # Set k_bins range from 2π/(Nx*dx) to 2π/dx
        k_min = 2 * np.pi / (Nx * dx)
        k_max = 2 * np.pi / dx
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num=num_bins)
        k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        k_indices = np.digitize(k_mag, k_bins)
        power_spectrum_binned = np.array([np.mean(power_spectrum_flatten[k_indices == i]) if np.any(k_indices == i) else 0 for i in range(1, len(k_bins))])
        power_spectrum_binned = power_spectrum_binned / (Nx * Ny)
        
        # Use provided axis or create a new one if not provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plotting on the given or newly created axis
        ax.plot(k_bin_centers, power_spectrum_binned, '-', **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$|k|$')
        ax.set_ylabel('Power Spectrum')
        ax.set_title('Power Spectrum vs. $|k|$')
        ax.grid(True)
        
        # Set y-axis limits with default values if not specified
        if y_max is None:
            y_max = np.max(power_spectrum_binned)
        if y_min is None:
            y_min = y_max / (Nx * Ny)
        ax.set_ylim(bottom=y_min, top=y_max)

        # Show the plot if a new figure was created
        if ax is None:
            plt.show()

