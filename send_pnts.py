import numpy as np
import shutil
import os


tt = np.arange(1652,1694,1)

ff = ['wy2017_2021_eco_pnts.{:08d}.vtk'.format(t) for t in tt]

# Create a new directory
dr_new = 'sendme'
if os.path.exists(dr_new) and os.path.isdir(dr_new):
    shutil.rmtree(dr_new)
os.makedirs(dr_new)

for f in ff:
    shutil.copy(f, dr_new)

