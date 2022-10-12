# Alters the orignal drv_vegm.dat file to change the plant-type distribution

import numpy as np
import pandas as pd
import os
from parflowio.pyParflowio import PFData
import pyvista


# Read in CLM plant type info
clm = pd.read_csv('clm_inputs/drv_vegm.dat', skiprows=2, header=None, index_col=0, delim_whitespace=True)
clm.columns = np.concatenate((['y','lat','lon','sand','clay','color'],np.arange(1,19)))
clm.index.name = 'x'

# Array with the plant type index
v_type = clm[np.arange(1,19).astype(str)]
vind = []
vnum = []
for i in range(1, len(v_type)+1):
    v_ind = v_type.columns.astype(float) * v_type.loc[i,:]
    # test how many indices there are
    vnum.append((v_type.loc[i,:] > 0.0).sum())
    
    vind.append(v_ind.max())
    #v_type.loc[i, 'vind'] = v_ind.max()
#v_type['vind'] = vind


# Make a raster
v_arr = np.zeros((32,559))
v_arr[-20:,:] = vind



# Pyvista to create new vtk file
# First read in a permeability field vtk file created by vis.vtk.tcl scipt
dom = pyvista.read('tfg.out.Perm.vtk')
dom.clear_arrays()



# Add new arrays
dom['vegind'] = v_arr.ravel()
dom.save('veg_ind.vtk')



"""
# write a new file
clm2 = clm.copy()
clm2.loc[:, np.arange(1,19).astype(str)] *= 0 
clm2.loc[:517, '10'] = 1
clm2.loc[518:, '6'] = 1
clm2.insert(0, 'x', clm2.index)

# write a new file
# first read in original header
with open('clm_inputs/drv_vegm.dat', 'r') as f:
    l = f.readlines()
head = l[:2]    
head = ''.join(head)[:-1]

# new one
fmt = ['%1d','%1d','%.5f','%.5f','%.2f','%.2f'] + ['%1d']*19
np.savetxt('drv_vegm.dat.v2', clm2.to_numpy(), fmt=fmt, header=head, comments='')

#with open('drv_vegm.dat.v2', 'w') as ff:
#    ff.writelines(head)
""" 
    
    
    
