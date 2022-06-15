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
plt.rcParams['font.size'] = 16


import matplotlib
from matplotlib.ticker import LogFormatter


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
        for j in range(ncells+1):
            if '.'.join(ll[i].split()[1].split('.')[:3]) == 'Cell.{}.dzScale'.format(j):
                #print (ll[i].split()[-1])
                dz.append(float(ll[i].split()[-1]))
            if '.'.join(ll[i].split()[1].split('.')[:4]) == 'Geom.i{}.Perm.Value'.format(j):
                #print (ll[i].split()[-1])
                K_mhr.append(float(ll[i].split()[-1]))


# Mangle Units
cell_multiplier = 10.0 
cells  = np.array(dz) * cell_multiplier
cells_centered = np.cumsum(cells).max() - (np.cumsum(cells) - np.array(dz)*10/2)

Z  = np.flip(cells).cumsum() # depth below land surface for each layer
Z_ = np.flip(cells_centered)

# Pad the first value so land surface is at 0 meters
Z = np.concatenate((np.array([0]), Z))
Z_ = np.concatenate((np.array([0]), Z_))

#K_ = [K_mhr[0]] +  K_mhr
K_  = K_mhr + [K_mhr[-1]] # Not sure, but this aligns everything up correct
K  = np.array(K_) / 3600 # m/sec

lK = np.log10(K)


#
# plots
#
cmap = matplotlib.cm.coolwarm
normalize = matplotlib.colors.LogNorm(vmin=K.min(), vmax=K.max())
normalize = matplotlib.colors.LogNorm(vmin=1.e-8, vmax=1.e-4)


fig, ax = plt.subplots(figsize=(3,4))
fig.subplots_adjust(left=0.4, right=0.6, top=0.97, bottom=0.05)
for i in range(len(Z)-1):
    ax.fill_between(x=[0,1], y1=Z[i], y2=Z[i+1], color=cmap(normalize(K[i])))
ax.invert_yaxis()
ax.set_ylim(100,0)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

ax.set_ylabel('Depth (m)', labelpad=0.05)
ax.tick_params(axis='x',which='both',labelbottom=False, bottom=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cbax = fig.add_axes([0.61, 0.15, 0.05, 0.7])
cb = matplotlib.colorbar.ColorbarBase(cbax, cmap=cmap, norm=normalize, orientation='vertical')
cb.set_label('K (m/s)', rotation=270, labelpad=25)
plt.savefig('figures/perm_plot.png',dpi=300)
plt.show()









