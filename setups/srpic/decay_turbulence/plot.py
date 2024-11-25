import sys
sys.path.append("/work/home/acdnhs4d3w/Entity-dev/decay_turbulence/")
import visualize as vis
import nt2.read as nt2r
import nt2.export as exp
import numpy as np

data = nt2r.Data("turb-decay.h5")

max_t = data.t.max().item()
steps = np.linspace(0, max_t, len(data.t))


def main():
    field = 'Bx'
    means = ['Prtl_Energy', 'Total_Energy', 'B2'] 
    d = 1.0
    rho = 0.2
    num_cpus = 4
    #for mean in means:
     # vis.plot_mean(data, mean, d, rho) 
    #vis.plot_mean(data, 'Total_Energy', d, rho)
    #vis.plot_attr(steps, data, field, 'norm', 0.0, 3.0, num_cpus=num_cpus)
    #exp.makeMovie(input=field+"/frames/", overwrite=True, output=field + "/" + field + ".mp4", number=5)
    
    plot_data = getattr(data, field)
    exp.makeFrames(lambda t, data: data.sel({'t':t}, method="nearest").spectrum.plot(),
                   steps,
                   field + "_spec/frames",
                   plot_data,
                   num_cpus=num_cpus)
    #print(getattr(data,'Bx').sel({'t':1.0}, method="nearest"))

if __name__ == '__main__':
    main()
