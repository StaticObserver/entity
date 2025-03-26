import nt2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data = nt2.Data(path="turbulence")

#data.fields.inspect.plot(name="inspect", only_fields=["N", "Jz"])

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
    num_cpus = 32
    times = np.linspace(0, 1000, 200)
    #sp = data.spectra
    
    #nt2.export.makeFrames(plot_spectra, times, 'spectra', sp, num_cpus=num_cpus)
    #nt2.export.makeFrames(plot_func, times, 'N', data.fields.N, num_cpus=num_cpus)
    print(data)
if __name__ == '__main__':
    main()
