import numpy as np
import xarray as xr
import nt2.read as nt2r

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

class dataMap:
    def __init__(self, d0, rho0):
        self.sigma0 = (d0 / rho0)**2
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
    
    def get_field(self, name, dataset):
        if name in self.field_map:
            return self.field_map[name](dataset)
        elif hasattr(dataset, name):
            return getattr(dataset, name)
        else:   
            raise ValueError(f"{name} not found in dataset or field_map.")
            

@xr.register_dataset_accessor('proccesser')
class ProccesserAccessor:
    def __init__(self, dataset):
        self._obj = dataset
    
    def compute_spectrum(self, dx, num_bins=200):
        Ny, Nx = self._obj.shape
        power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(self._obj)))**2
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
    
    def compute_L(self, dx, num_bins=200):
        k_bin_centers, power_spectrum_binned = self.compute_spectrum(dx, num_bins)
        return np.dot(1.0 / k_bin_centers, power_spectrum_binned) / np.sum(power_spectrum_binned)
    
    
def plot_mean(dataset, times, num_cpus=None, **kwargs):
    import matplotlib.pyplot as plt
    
    values = parallel(lambda t, data: data.sel({'t':t}).mean(('x', 'y')).compute().item(), times, dataset, num_cpus)
    
    plt.plot(times, values, **kwargs)
    