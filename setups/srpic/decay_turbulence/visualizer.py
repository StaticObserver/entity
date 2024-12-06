import numpy as np
import xarray as xr
import nt2.read as nt2r
import matplotlib.pyplot as plt

def parallel(func, steps, dataset, num_cpus=None):
    from tqdm import tqdm
    import multiprocessing as mp
    import numpy as np  # 添加numpy导入
    
    if num_cpus is None:
        num_cpus = mp.cpu_count()

    global calculate
    def calculate(t):
        try:
            value = func(t, dataset)
        except Exception as e:
            print(f"Error in processing {t}: {e}")
            return t, None
        return t, value
    
    # 初始化多进程池
    pool = mp.Pool(num_cpus)
    try:
        # 添加进度条
        results = [pool.apply_async(calculate, args=(t,)) for t in tqdm(steps)]
        pool.close()  # 关闭输入通道
        pool.join()   # 等待所有进程完成
    except Exception as e:
        pool.terminate()  # 遇到异常时终止所有进程
        print(f"Error during multiprocessing: {e}")
        raise
    
    # 获取结果并排序
    sorted_results = sorted([r.get() for r in results], key=lambda x: x[0])
    
    # 提取结果值为numpy数组
    return np.array([value for t, value in sorted_results])
            
    
def compute_spectrum(field, dx, num_bins=200):
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
    k_bins = np.linspace(2 * np.pi / (Nx * dx), 2 * np.pi / dx, num=num_bins)
    # k_bins = np.logspace(np.log10(2 * np.pi / (Nx * dx)), np.log10(2 * np.pi / dx), num=num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    k_indices = np.digitize(k_mag, k_bins)
    power_spectrum_binned = np.array([
        np.sum(power_spectrum_flatten[k_indices == i]) if np.any(k_indices == i) else 0 
        for i in range(1, len(k_bins))
    ])
    power_spectrum_binned /= (Nx * Ny)
    return k_bin_centers, power_spectrum_binned

def compute_L(field, dx, num_bins=200):
    k_bin_centers, power_spectrum_binned = compute_spectrum(field, dx, num_bins)
    return np.dot(1.0 / k_bin_centers, power_spectrum_binned) / np.sum(power_spectrum_binned)
    
class Visualizer(nt2r.Data):
    def __init__(self, filename, d0, rho0, num_cpus):
        super().__init__(filename)
        self.sigma0 = (d0 / rho0)**2
        self.num_cpus = num_cpus
        self.times = self.coords['t'].values
        self.dx = self.coords['x'].values[1] - self.coords['x'].values[0]
        self.field_map = {
            'B2': lambda data: (data.Bx**2 + data.By**2 + data.Bz**2) ,
            'E2': lambda data: (data.Ex**2 + data.Ey**2 + data.Ez**2) ,
            'EM_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                             data.Bx**2 + data.By**2 + data.Bz**2) * self.sigma0,
            'Prtl_Energy': lambda data: data.T00,
            'Total_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                                data.Bx**2 + data.By**2 + data.Bz**2) * self.sigma0+ data.T00,
            'N' : lambda data: data.N_1 + data.N_2,
            'Bxy_Energy' : lambda data: 0.5 * (data.Bx**2 + data.By**2) * self.sigma0, 
        }
    
    def set_cpu(self, num_cpus):
        self.num_cpus = num_cpus
    
    def set_para(self, d0, rho0):
        self.sigma0 = (d0 / rho0)**2
    
    def get_field(self, name):
        if name in self.field_map:
            return self.field_map[name](self)
        else:   
            raise ValueError(f"{name} not found.")
        
    def get_means(self, name, times=None):
        if times is None:
            times = self.times
        if name in self.field_map:
            field = self.field_map[name](self)
        elif hasattr(self, name):
            field = getattr(self, name)
        else:
            raise ValueError("Invalid type.")
        return  parallel(lambda t, data: data.sel({'t':t}, method='nearest').mean(('x', 'y')).compute().item(),
                         times, 
                         field,
                         self.num_cpus)
    
    def get_spectrum(self, name, t, num_bins=200):
        if name in self.field_map:
            field = self.field_map[name](self)
        elif hasattr(self, name):
            field = getattr(self, name)
        else:
            raise ValueError("Invalid type.")
        return compute_spectrum(field.sel({'t': t}, method='nearest'), self.dx, num_bins)
    
    def get_L(self, name, times=None, num_bins=200):
        if times is None:
            times = self.times
        if name in self.field_map:
            field = self.field_map[name](self)
        elif hasattr(self, name):
            field = getattr(self, name)
        else:
            raise ValueError("Invalid type.")
        return parallel(lambda t, data: compute_L(data.sel({'t': t}, method='nearest'), self.dx, num_bins),
                        times,
                        field,
                        self.num_cpus)
        
    def decay_rate(self, name, times=None, **kwargs):
        if times is None:
            times = self.times
        ts = times[1:]
        Qs = self.get_means(name, ts)
        rate = np.array([np.log(Qs[i-1] / Qs[i+1]) / np.log(ts[i+1] / ts[i-1]) for i in range(1, len(ts) - 1)])
        plt.plot(ts[1:-1], rate, **kwargs)
        plt.savefig("decay_{}.png".format(name), dpi=300, bbox_inches="tight")
        np.savetxt("decay_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))
        
    def increase_rate_L(self, name, times=None, num_bins=200, **kwargs):
        if times is None:
            times = self.times
        ts = times[1:]
        Qs = self.get_L(name, ts, num_bins)
        rate = np.array([np.log(Qs[i+1] / Qs[i-1]) / np.log(ts[i+1] / ts[i-1]) for i in range(1, len(ts) - 1)])
        plt.plot(ts[1:-1], rate, **kwargs)
        plt.savefig("L_{}.png".format(name), dpi=300, bbox_inches="tight")
        np.savetxt("L_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))
        
    def plot_mean(self, name, times=None, xscale=None, yscale=None, **kwargs):
        if times is None:
            times = self.times
        means = self.get_means(name, times)
        plt.plot(times, means, **kwargs)
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)
        plt.savefig("mean_{}.png".format(name), dpi=300, bbox_inches="tight")
        plt.close()
        np.savetxt("mean_{}.dat".format(name), np.column_stack((times, means)))
    
    def plot_spectrum(self, name, t, num_bins=200, y_min=None, y_max=None, **kwargs):
        k_bins, powers = self.get_spectrum(name, t, num_bins)
        fig, ax = plt.subplots()
        ax.plot(k_bins, powers, '-', **kwargs)
        ax.set_xlabel(r'$|k|$')
        ax.set_ylabel('Energy Spectrum Density')
        ax.set_title('t = {:.2f}'.format(t))
        ax.grid(True)
        if y_max is None:
            y_max = np.max(powers)
        if y_min is None:
            y_min = y_max / 1e6
        ax.set_ylim(bottom=y_min, top=y_max)






