# Reads in a Parflow velocity .pfb file for a single timestep
# Writes a vtk file with the velocity vectors

import numpy as np
import os
from parflowio.pyParflowio import PFData
import pyvista


vx_f = './wy_2017_2021/wy_2017_2021.out.velx.01683.pfb'
vy_f = './wy_2017_2021/wy_2017_2021.out.vely.01683.pfb'
vz_f = './wy_2017_2021/wy_2017_2021.out.velz.01683.pfb'

def read_pfb(fname):
    pfdata = PFData(fname)
    pfdata.loadHeader()
    pfdata.loadData()
    return pfdata.copyDataArray()

# Velocity field shapes do not acutally match domain shape
# need to clip, not positive on clipping first or last index
vx = read_pfb(vx_f)[:,:,1:] #[:,:,:-1]
vy = read_pfb(vy_f)[:,1:,:]
vz = read_pfb(vz_f)[1:,:,:]

if vz.shape == vy.shape == vz.shape:
    pass
else:
    print ('Check Shapes')


# Pyvista to create new vtk file
# First read in a permeability field vtk file created by vis.vtk.tcl scipt
dom = pyvista.read('tfg.out.Perm.vtk')
dom.clear_arrays()

# Add new arrays
dom['vx'] = vx.ravel()
dom['vy'] = vy.ravel()
dom['vz'] = vz.ravel()

dom.save('vel_comps.vtk')






