# source ~/venvs/mda/bin/activate

import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from MDAnalysis.tests.datafiles import RANDOM_WALK_TOPO, RANDOM_WALK

u = mda.Universe("md_water-T_12k300.0.xyz", format="XYZ")
MSD_calc = msd.EinsteinMSD(u, select='all', msd_type='xyz', fft=True)
MSD_calc.run()

msd =  MSD_calc.results.timeseries

import matplotlib.pyplot as plt
import numpy as np
nframes = MSD_calc.n_frames
lagtimes = np.arange(nframes)*timestep 
fig = plt.figure()
ax = plt.axes()
ax.plot(lagtimes, msd, ls="-", label=r'D = $D')
plt.show()


from scipy.stats import linregress
start_time = 50
start_index = int(start_time/timestep)
end_time = 250
linear_model = linregress(lagtimes[start_index:-1],
                                              msd[start_index:-1])
slope = linear_model.slope
error = linear_model.stderr
D = slope * 1/(6)
