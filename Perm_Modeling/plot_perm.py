import numpy as np
import pandas as pd
import pickle as pk
import os

from parflowio.pyParflowio import PFData
#import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.patches as patches
plt.rcParams['font.size'] = 14



#
# Read in Permeability File
#

f = 'subsurface_lc102a0.mat'

with open(f, 'r') as ff:
    ll = ff.readlines()
    
    

ncells = None

for i in range(len(ll)):
    try: 
        ll[i].split()[1]
    except IndexError:
        pass
    else:
        if ll[i].split()[1] == 'dzScale.nzListNumber':
            ncells = int(ll[i].split()[2])
            

dz    = []             
K_mhr = []


for i in range(len(ll)):
    try: 
        ll[i].split()[1]
    except IndexError:
        pass
    else:
        for j in range(ncells):
            if '.'.join(ll[i].split()[1].split('.')[:3]) == 'Cell.{}.dzScale'.format(j):
                #print (ll[i].split()[-1])
                dz.append(float(ll[i].split()[-1]))
            if '.'.join(ll[i].split()[1].split('.')[:4]) == 'Geom.i{}.Perm.Value'.format(j):
                #print (ll[i].split()[-1])
                K_mhr.append(float(ll[i].split()[-1]))


# Mangle Units

cell_multiplier = 10.0 

cells_ = dz + [0]
cells  = np.array(cells_) * cell_multiplier

Z = np.flip(cells).cumsum() # depth below land surface for each layer


K_ = [K_mhr[0]] +  K_mhr
K  = np.array(K_) / 3600 # m/sec

lK = np.log10(K)

# plots


import matplotlib
from matplotlib.ticker import LogFormatter


cmap = matplotlib.cm.coolwarm
normalize = matplotlib.colors.LogNorm(vmin=K.min(), vmax=K.max())
normalize = matplotlib.colors.LogNorm(vmin=1.e-9, vmax=1.e-4)

fig, ax = plt.subplots()
for i in range(len(Z)-1):
    ax.fill_between(x=[0,1], y1=Z[i], y2=Z[i+1], color=cmap(normalize(K[i])))
ax.invert_yaxis()
cbax = fig.add_axes([0.85, 0.12, 0.05, 0.78])
cb = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=normalize, orientation='vertical')#, format=formatter)
cb.set_label('K (m/s)', rotation=270, labelpad=15)
plt.show()









