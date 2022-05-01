# Script to process the Parflow velocity fields



# To-do:
# Plot the Vz velocity magnitudes without first calculating the entire Vx,Vz vector 




import numpy as np
import pandas as pd
import os
import pickle 
import glob


from parflowio.pyParflowio import PFData
import pyvista as pv

import parflow.tools as pftools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.patches as patches
plt.rcParams['font.size'] = 14
import matplotlib.colors as colors





#------------------
#
# Utilities
#
#------------------


def read_pfb(fname):
    '''Read in a .pfb file'''
    pfdata = PFData(fname)
    pfdata.loadHeader()
    pfdata.loadData()
    return pfdata.copyDataArray()



class hydro_utils():
    def __init__(self, dz_scale):
        # define file names
        #self.timestep = None
        #self.press = None
        #self.sat   = None
        #self.specific_storage = None
        #self.porosity = None
        
        self.dz_scale = dz_scale
        
    def read_fields(self, timestep, directory, header):
        self.timestep = timestep
        
        fn_press = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'press',timestep))
        self.press = read_pfb(fn_press)
        
        fn_sat = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'satur',timestep))
        self.sat = read_pfb(fn_sat)
        
        fn_spc_stor = os.path.join(directory, '{}.out.{}.pfb'.format(header,'specific_storage'))
        self.specific_storage = read_pfb(fn_spc_stor)
        
        fn_porosity = os.path.join(directory, '{}.out.{}.pfb'.format(header,'porosity'))
        self.porosity = read_pfb(fn_porosity)
        
        # Need to think about shapes here
        # This indexing is based on matching z velocity with K constrasts...
        fn_velx = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'velx',timestep))
        self.velx = read_pfb(fn_velx)[:,:,1:]

        fn_vely = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'vely',timestep))
        self.vely = read_pfb(fn_vely)[:,1:,:]
        
        fn_velz = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'velz',timestep))
        self.velz = read_pfb(fn_velz)[1:,:,:]
	
        fn_et = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'evaptrans',timestep))
        self.et = read_pfb(fn_et)       

 

    def pull_wtd(self):
        wtd = pftools.hydrology.calculate_water_table_depth(self.press, self.sat, self.dz_scale)
        return wtd.ravel()

    def pull_storage(self):
        return pftools.hydrology.calculate_subsurface_storage(self.porosity, self.press, self.sat, self.specific_storage, 1.5125, 1.0, self.dz_scale)
        
    def pull_et(self):
        return pftools.hydrology.calculate_evapotranspiration(self.et, 1.5125, 1.0, self.dz_scale)    


    def vel_bedrock_layer(self, bedrock_mbls):
        #Z  = np.flip(self.dz_scale).cumsum() # depth below land surface for each layer
        Z_ = self.dz_scale.sum() - self.dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up
        bedrock_ind  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer

        # Velocity field shapes do not acutally match domain shape
        # need to clip, not positive on clipping first or last index
        #vx = read_pfb(vx_f)[:,:,1:] #[:,:,:-1]
        #vy = read_pfb(vy_f)[:,1:,:]
        #vz = read_pfb(vz_f)[1:,:,:]

        # Velocity at first bedrock layer
        Vx_bed = self.velx[bedrock_ind,0,:]
        Vz_bed = self.velz[bedrock_ind,0,:]
        
        
        # Calculate velocity component that is below the land surface slope
        #Vz_bed_ = Vz_bed - np.tan(-1*slope*np.pi/180) * Vx_bed
        return  [Vx_bed, Vz_bed]

    def vel_soil_layer(self, bedrock_mbls):
        #Z  = np.flip(self.dz_scale).cumsum() # depth below land surface for each layer
        Z_ = self.dz_scale.sum() - self.dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up
        bedrock_ind  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer

        # Velocity at first bedrock layer
        Vx_bed = self.velx[bedrock_ind+1,0,:]
        Vz_bed = self.velz[bedrock_ind+1,0,:]

        return  [Vx_bed, Vz_bed]


        


# Parflow variable dz
dz = np.array([1.00, 1.00, 1.00, 1.00, 1.00,       # 52.0 - 102.0
               0.80, 0.80,                         # 36.0 - 52.0
               0.60, 0.60,                         # 24.0 - 36.0
               0.40, 0.40,                         # 16.0 - 24.0
               0.20, 0.20, 0.20,                   # 10.0 - 16.0  -- 2m layers down to 16 m
               0.10, 0.10,                         # 8.0  - 10.0
               0.10, 0.05, 0.05,                   # 6.0  - 8.0   -- 0.5m res possible down to 7.0 m
               0.05, 0.05, 0.05, 0.05,             # 4.0  - 6.0
               0.05, 0.05, 0.05, 0.05,             # 2.0  - 4.0
               0.05, 0.05, 0.05, 0.025, 0.025])    # 0.0  - 2.0  
dz_scale = 10 * dz



"""
# Define timesteps and depth of bedrock in the model
ts   = 1683
bedrock_mbls=9.0


# Run the functions
hut = hydro_utils(dz_scale=dz_scale)
hut.read_fields(1683, 'wy_2017_2021', 'wy_2017_2021')

wtd = hut.pull_wtd()
specific_storage = hut.pull_storage() 
velx_bed,  velz_bed  = hut.vel_bedrock_layer(bedrock_mbls)
velx_soil, velz_soil = hut.vel_soil_layer(bedrock_mbls)
"""


#
# Loop Through Transient Files
#

bedrock_mbls=9.0

directory = 'wy_2017_2021'
header    = 'wy_2017_2021'

pf_out_dict = {'bedrock_mbls':bedrock_mbls,
               'wtd':{},
               'specific_storage':{},
               'et':{},
               'velbed':{},
               'velsoil':{}}

# Use only files that exist



ff = glob.glob(os.path.join(directory,'*press*'))
ts_list_ = [int(i.split('.')[-2]) for i in ff]
ts_list_.sort()


hut = hydro_utils(dz_scale=dz_scale)
for i in ts_list_:
    print ('working on {}/{}'.format(i, len(ts_list_)))
    try:
        hut.read_fields(i, directory, header)
        
        pf_out_dict['wtd'][i] = hut.pull_wtd()
        pf_out_dict['specific_storage'][i] = hut.pull_storage()
        pf_out_dict['et'][i] = hut.et()
        pf_out_dict['velbed'][i] = hut.vel_bedrock_layer(bedrock_mbls)
        pf_out_dict['velsoil'][i] = hut.vel_soil_layer(bedrock_mbls)
    except TypeError:
        pass
    
with open('parflow_out/pf_out_dict.pk', 'wb') as ff_:
    pickle.dump(pf_out_dict, ff_)
    


#
# Single Timestep
#
# Run the functions
#hut = hydro_utils(dz_scale=dz_scale)
#hut.read_fields(1683, 'wy_2017_2021', 'wy_2017_2021')

#wtd = hut.pull_wtd()
#specific_storage = hut.pull_storage() 
#velx_bed,  velz_bed  = hut.vel_bedrock_layer(bedrock_mbls)
#velx_soil, velz_soil = hut.vel_soil_layer(bedrock_mbls)



"""

#
# Post-processing and plotting
#

#
# Read in Land Surface Slope
#
slope = read_pfb('slope_x_v4.pfb')
slope = slope[0,0,:] * 180/np.pi



#
# Parflow Grid info
#
dom = pv.read('tfg.out.Perm.vtk')
cell_bounds = np.array([np.array(dom.cell_bounds(i))[[0,1,4,5]] for i in range(dom.GetNumberOfCells())])
cell_center = np.column_stack((cell_bounds[:,[0,1]].mean(axis=1), cell_bounds[:,[2,3]].mean(axis=1)))

xx_ = np.flip(cell_center[:,0].reshape(32,559), axis=0)
zz_ = np.flip(cell_center[:,1].reshape(32,559), axis=0)
xx  = xx_[0,:]
land_surf = zz_[0,:]

wtd_elev = land_surf - wtd



#
# Subsurface Storage
#
# First create masks for the bedrock and soil layers
#
# CHANGE ME #
#
#bedrock_bls = 9.0  # fractured bedrock starts 9 mbls

Z  = np.flip(dz_scale).cumsum() # depth below land surface for each layer
Z_ = dz_scale.sum() - dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up

bedrock_ind  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer

por = hut.porosity
bedrock_mask = np.zeros_like(por)
bedrock_mask[0:bedrock_ind+1,:] = 1 # remember need +1 when indexing with a list

soil_mask = np.zeros_like(por)
soil_mask[bedrock_ind+1:,:] = 1

map_check = np.column_stack((np.arange(len(Z_)), Z_, por[:,0,0], bedrock_mask[:,0,0], soil_mask[:,0,0])) # check to make sure indexing is correct



bedrock_storage = specific_storage.copy()
bedrock_storage[bedrock_mask==0] = 0

soil_storage = specific_storage.copy()
soil_storage[bedrock_mask==1] = 0




#
# Plotting
#

fig, axes = plt.subplots(2,1)
#
# Plot Z-velocity 
#
ax = axes[0]
ax.plot(xx, velz_bed*8760, label='top bedrock')
ax.plot(xx, velz_soil*8760, label='bottom soil')
#ax.plot(xx, velz_bed_*8760)
#
ax.set_ylabel('Recharge (m/year)')
ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
#
# Plot Topography Profile
#
ax = axes[1]
ax.plot(xx, land_surf, color='black', label='land surface')
ax.plot(xx, wtd_elev, color='lightblue', linestyle='--', alpha=1.0, zorder=8.0, label='water table')
ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', label='bedrock')
# 
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
ax.set_ylim(2750-20, 2950)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
# Cleanup
[axes[i].margins(x=0.0) for i in [0,1]]
[axes[i].minorticks_on() for i in [0,1]]
[axes[i].xaxis.set_major_locator(ticker.MultipleLocator(100)) for i in [0,1]] 
[axes[i].tick_params(which='both', axis='both', top=True, right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
fig.tight_layout()
plt.show()

"""
















