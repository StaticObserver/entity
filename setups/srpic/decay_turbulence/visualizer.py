import numpy as np
import xarray as xr
import nt2.read as nt2r
import matplotlib.pyplot as plt
from tqdm import tqdm

def parallel(func, steps, dataset, num_cpus=None):
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
            
    
def compute_spectrum(field, dx, k_min=None, k_max=None, num_bins=200):
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
    if k_max is None:
        k_max = np.max(k_mag)
    if k_min is None:
        k_min = np.min(k_mag[k_mag > 0])
    k_bins = np.linspace(k_min, k_max, num=num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    dk = k_bins[1] - k_bins[0]
    k_indices = np.digitize(k_mag, k_bins, right=False)
    power_spectrum_binned = np.array([
        np.sum(power_spectrum_flatten[k_indices == i]) if np.any(k_indices == i) else 0 
        for i in range(1, len(k_bins))
    ])
    power_spectrum_binned /= (Nx * Ny )
    print(power_spectrum_binned)
    return k_bin_centers, power_spectrum_binned

def compute_L(field, dx, k_min=None, k_max=None, num_bins=200):
    k_bin_centers, power_spectrum_binned = compute_spectrum(field, dx, k_min, k_max, num_bins)
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
    
    def get_spectrum(self, name, t, k_min=None, k_max=None, num_bins=200):
        if name in self.field_map:
            field = self.field_map[name](self)
        elif hasattr(self, name):
            field = getattr(self, name)
        else:
            raise ValueError("Invalid type.")
        field = field.sel({'t': t}, method='nearest').values
        print(field)
        k_bins, powers = compute_spectrum(field, self.dx, k_min, k_max, num_bins)
        return k_bins, powers
    
    def get_L(self, name, times=None, k_min=None, k_max=None, num_bins=200):
        if times is None:
            times = self.times
        if name in self.field_map:
            field = self.field_map[name](self)
        elif hasattr(self, name):
            field = getattr(self, name)
        else:
            raise ValueError("Invalid type.")
        return parallel(lambda t, data: compute_L(data.sel({'t': t}, method='nearest'), self.dx, k_min, k_max, num_bins),
                        times,
                        field,
                        self.num_cpus)
        
    def decay_rate(self, name, times=None, **kwargs):
        if times is None:
            times = self.times
        ts = times[1:]
        Qs = self.get_means(name, ts)
        rate = np.array([np.log(Qs[i-1] / Qs[i+1]) / np.log(ts[i+1] / ts[i-1]) for i in tqdm(range(1, len(ts) - 1))])
        plt.plot(ts[1:-1], rate, **kwargs)
        plt.savefig("decay_{}.png".format(name), dpi=300, bbox_inches="tight")
        np.savetxt("decay_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))
        
    def increase_rate_L(self, name, times=None, k_min=None, k_max=None, num_bins=200, **kwargs):
        if times is None:
            times = self.times
        ts = times[1:]
        Qs = self.get_L(name, ts, k_min, k_max, num_bins)
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
        k_bins, powers = self.get_spectrum(name, t, num_bins=num_bins)
        print(powers)
        plt.plot(k_bins, powers, **kwargs)
        if y_max is None:
            y_max = np.max(powers)
        if y_min is None:
            y_min = y_max / 1e6
        plt.ylim([y_min, y_max])
        plt.xscale('log')
        plt.yscale('log')
        
    def field_line(self, name, t, density=2):
        x = np.array(self.coords['x'].values)
        y = np.array(self.coords['y'].values)
        X, Y = np.meshgrid(x,y)
        if name=='magnetic':
            vx = self.Bx.sel({'t':t}, method='nearest')
            vy = self.By.sel({'t':t}, method='nearest')
        else:
            raise ValueError("Invalid type.")
        plt.figure(figsize=(8, 6))
        plt.streamplot(X, Y, vx, vy, density=density)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')

        
        






