import nt2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm

sigma0 = 16

field_map = {
            'B2': lambda data: (data.Bx**2 + data.By**2 + data.Bz**2) ,
            'E2': lambda data: (data.Ex**2 + data.Ey**2 + data.Ez**2) ,
            'EM_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                             data.Bx**2 + data.By**2 + data.Bz**2) * sigma0,
            'Prtl_Energy': lambda data: data.T00,
            'Total_Energy': lambda data: 0.5 * (data.Ex**2 + data.Ey**2 + data.Ez**2 +
                                                data.Bx**2 + data.By**2 + data.Bz**2) * sigma0+ data.T00,
            'N' : lambda data: data.N_1 + data.N_2,
            'Bxy_Energy' : lambda data: 0.5 * (data.Bx**2 + data.By**2) * sigma0, 
        }

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


def get_means(data, times, name, num_cpus=4):
    if name in field_map:
        field = field_map[name](data)
    elif hasattr(data, name):
        field = getattr(data, name)
    else:
        raise ValueError("Invalid name.")
    return parallel(lambda t, data: data.sel({'t':t}, method='nearest').mean(('x', 'y')).compute().item(),
                         times, 
                         field,
                         num_cpus)

def plot_means(data, times, name, num_cpus=4):
    means = get_means(data, times, name, num_cpus)
    plt.plot(times, means)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("mean_{}.png".format(name), dpi=100, bbox_inches="tight")
    plt.close()
    np.savetxt("mean_{}.dat".format(name), np.column_stack((times, means)))
   

def decay_rate(data, name, ts, num_cpus=4):
    Qs = get_means(data, ts, name, num_cpus)
    rate = np.array([np.log(Qs[i-1] / Qs[i+1]) / np.log(ts[i+1] / ts[i-1]) for i in tqdm(range(1, len(ts) - 1))])
    plt.plot(ts[1:-1], rate)
    plt.savefig("decay_{}.png".format(name), dpi=100, bbox_inches="tight")
    np.savetxt("decay_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))

def compute_spectrum(field, dx, k_min=None, k_max=None, num_bins=200, use_log_bins=True):
    """
    计算二维场的径向功率谱。
    
    参数:
        field: 2D numpy数组，输入场
        dx: 空间分辨率
        k_min: 最小波数 (可选)
        k_max: 最大波数 (可选)
        num_bins: 波数bin的数量
        use_log_bins: 是否使用对数间隔的波数bins
        
    返回:
        k_bin_centers: 波数bin中心点
        power_spectrum_binned: 相应的功率谱密度
    """
    Ny, Nx = field.shape
    
    # 计算FFT并取幅度平方
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(field)))**2
    
    # 计算波数网格
    dkx = 2 * np.pi / (Nx * dx)
    dky = 2 * np.pi / (Ny * dx)
    kx, ky = np.meshgrid(
        np.linspace(-Nx/2 * dkx, Nx/2 * dkx - dkx, Nx),
        np.linspace(-Ny/2 * dky, Ny/2 * dky - dky, Ny)
    )
    
    # 计算波数幅度
    k_mag = np.sqrt(kx**2 + ky**2).flatten()
    power_spectrum_flatten = power_spectrum.flatten()
    
    # 设置波数范围
    if k_max is None:
        k_max = np.max(k_mag)
    if k_min is None:
        k_min = 0.0  
    
    
    k_bins = np.linspace(k_min, k_max, num=num_bins)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    power_spectrum_binned = np.zeros(len(k_bin_centers))

    for i in range(len(k_bins) - 1):
            bin_mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            if np.sum(bin_mask) > 0:
                 power_spectrum_binned[i] = np.sum(power_spectrum_flatten[bin_mask])
    
    # 归一化
    power_spectrum_binned /= (Nx * Ny)  # 归一化FFT
    
    
    return k_bin_centers, power_spectrum_binned

def spectrum(t, data):
    frame = data.sel({'t':t}, method="nearest")
    dx = data.coords['x'].values[1] - data.coords['x'].values[0]
    k_bins, powers = compute_spectrum(frame.values, dx, None, None, 200, True)
    plt.plot(k_bins, powers)
    #plt.xscale("log")
    plt.yscale("log")
    

def compute_L(field, dx, k_min=None, k_max=None, num_bins=200):
    k_bin_centers, power_spectrum_binned = compute_spectrum(field, dx, k_min, k_max, num_bins)
    return np.dot(1.0 / k_bin_centers, power_spectrum_binned) / np.sum(power_spectrum_binned)

def get_L(data, name, times, k_min=None, k_max=None, num_bins=200, num_cpus=4):
    if name in field_map:
        field = field_map[name](data)
    elif hasattr(data, name):
        field = getattr(data, name)
    else:
        raise ValueError("Invalid type.")
    dx = data.coords['x'].values[1] - data.coords['x'].values[0]
    return parallel(lambda t, data: compute_L(data.sel({'t': t}, method='nearest'), dx, k_min, k_max, num_bins),
                    times,
                    field,
                    num_cpus)
    
def increase_rate_L(data, name, ts, k_min=None, k_max=None, num_bins=200, num_cpus=4):
    Qs = get_L(data, name, ts, k_min, k_max, num_bins, num_cpus)
    rate = np.array([np.log(Qs[i+1] / Qs[i-1]) / np.log(ts[i+1] / ts[i-1]) for i in range(1, len(ts) - 1)])
    plt.plot(ts[1:-1], rate)
    plt.savefig("L_{}.png".format(name), dpi=300, bbox_inches="tight")
    np.savetxt("L_{}.dat".format(name), np.column_stack((ts[1:-1], rate)))

def plot_spectra(t, data):
    frame = data.sel({'t':t}, method="nearest")
    plt.figure(figsize=(6, 3))
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(frame.e, frame.n_1 + frame.n_2, c="r")
    plt.ylim(10, 1e8)
    plt.xlabel(r"$\gamma - 1$")
    plt.xlim(frame.e.min(), 1e3)

def plot_func(t, fld):
    vmin = 0.0
    vmax = 3.0
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    colormap = 'inferno'
    #colormap ='jet'
    fld.sel({'t':t}, method='nearest').plot(ax=ax, norm=mpl.colors.Normalize(vmin, vmax), cmap=colormap)

def main():
    data = nt2.Data(path="turbulence")
    num_cpus = 32
    times = np.linspace(0, 1000, 200)
    plot_means(data.fields, times, 'EM_Energy', num_cpus=num_cpus)
    plot_means(data.fields, times, 'Total_Energy', num_cpus=num_cpus)
    plot_means(data.fields, times, 'Prtl_Energy', num_cpus=num_cpus)
    #sp = data.spectra
    
    #nt2.export.makeFrames(plot_spectra, times, 'spectra', sp, num_cpus=num_cpus)
    #nt2.export.makeFrames(plot_func, times, 'N', data.fields.N, num_cpus=num_cpus)

if __name__ == '__main__':
    main()
