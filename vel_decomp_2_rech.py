# Script to read in and process the Parflow .pfb files for all timesteps
# Saves output to 'parflow_out/pf_out_dict.pk' 
# Outputs in pf_out_dict.pk: WTD, specific storage, gw velocities, ET, and pressures
# Notes: Need to manually specify ParFlow time and domain details below



import numpy as np
import pandas as pd
import os
import pickle 
import glob
import pdb

from parflowio.pyParflowio import PFData
import pyvista as pv
import parflow.tools as pftools





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
       
        fn_K = os.path.join(directory, '{}.out.{}.pfb'.format(header,'perm_x'))
        self.K = read_pfb(fn_K) 
       
        fn_et = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'evaptrans',timestep))
        self.et = read_pfb(fn_et)
 
        # Need to think about shapes here
        # This indexing is based on matching z velocity with K constrasts...
        fn_velx = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'velx',timestep))
        self.velx = read_pfb(fn_velx)[:,:,1:]

        fn_vely = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'vely',timestep))
        self.vely = read_pfb(fn_vely)[:,1:,:]
        
        fn_velz = os.path.join(directory, '{}.out.{}.{:05d}.pfb'.format(header,'velz',timestep))
        self.velz = read_pfb(fn_velz)[1:,:,:]
        

    def pull_wtd(self):
        wtd = pftools.hydrology.calculate_water_table_depth(self.press, self.sat, self.dz_scale)
        return wtd.ravel()

    def pull_storage(self):
        return pftools.hydrology.calculate_subsurface_storage(self.porosity, self.press, self.sat, self.specific_storage, 1.5125, 1.0, self.dz_scale)
       
    def pull_et(self):
         return pftools.hydrology.calculate_evapotranspiration(self.et, 1.5125, 1.0, self.dz_scale)    
    
    def pull_bedrock_ind(self):
        '''Find index where porosity changes, take this as bedrock. Soil is +1'''
        #pdb.set_trace()
        self.bedrock_ind = np.where(self.porosity[:,0,0]==self.porosity[:,0,0].min())[0].max()
    
    def vel_bedrock_layer(self, bedrock_mbls):
        self.pull_bedrock_ind()
        
        # Testing....
        Z  = np.flip(self.dz_scale).cumsum() # depth below land surface for each layer
        Z_ = self.dz_scale.sum() - self.dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up
        bedrock_ind_  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer
        
        if bedrock_ind_ != self.bedrock_ind:
            print ('bedrock depth not matching porosity')
        
        # Velocity field shapes do not acutally match domain shape
        # need to clip, not positive on clipping first or last index
        #vx = read_pfb(vx_f)[:,:,1:] #[:,:,:-1]
        #vy = read_pfb(vy_f)[:,1:,:]
        #vz = read_pfb(vz_f)[1:,:,:]

        # Velocity at first bedrock layer
        Vx_bed = self.velx[self.bedrock_ind,0,:]
        Vz_bed = self.velz[self.bedrock_ind,0,:]
        
        # Calculate velocity component that is below the land surface slope
        #Vz_bed_ = Vz_bed - np.tan(-1*slope*np.pi/180) * Vx_bed
        return  [Vx_bed, Vz_bed]

    def vel_soil_layer(self, bedrock_mbls):
        #Z  = np.flip(self.dz_scale).cumsum() # depth below land surface for each layer
        #Z_ = self.dz_scale.sum() - self.dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up
        #bedrock_ind  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer

        #pdb.set_trace()
        # Velocity at first bedrock layer
        Vx_bed = self.velx[self.bedrock_ind+1,0,:]
        Vz_bed = self.velz[self.bedrock_ind+1,0,:]

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




## Define timesteps and depth of bedrock in the model
#ts   = 1683
#bedrock_mbls = 9.0
#
#directory = 'wy_2017_2021'
#header    = 'wy_2017_2021'
#
## Run the functions
#hut = hydro_utils(dz_scale=dz_scale)
#hut.read_fields(ts, directory, header)
#
#wtd = hut.pull_wtd()
#specific_storage = hut.pull_storage() 
#velx_bed,  velz_bed  = hut.vel_bedrock_layer(bedrock_mbls)
#velx_soil, velz_soil = hut.vel_soil_layer(bedrock_mbls)



#
# Loop Through Transient Files
#
bedrock_mbls = 9.0

pf_out_dict = {'bedrock_mbls':bedrock_mbls,
               'wtd':{},
               'specific_storage':{},
               'velbed':{},
               'velsoil':{},
               'et':{},
               'sat':{},
               'press':{}}

# Use only files that exist
directory = 'wy_2000_2016'
header    = 'wy_2000_2016'

ff = glob.glob(os.path.join(directory,'*velx*'))[1:]
ts_list1 = [int(i.split('.')[-2]) for i in ff]
ts_list1.sort()

hut = hydro_utils(dz_scale=dz_scale)
print ('Working on WY 2000-2016 velocity files')
for i in ts_list1:
    #print ('working on {}/{}'.format(i, len(ts_list1)))
    try:
        hut.read_fields(i, directory, header)
        pf_out_dict['wtd'][i] = hut.pull_wtd()
        pf_out_dict['specific_storage'][i] = hut.pull_storage()
        pf_out_dict['velbed'][i] = hut.vel_bedrock_layer(bedrock_mbls)
        pf_out_dict['velsoil'][i] = hut.vel_soil_layer(bedrock_mbls)
        pf_out_dict['et'][i] = hut.pull_et()
        pf_out_dict['sat'][i] = hut.sat
        pf_out_dict['press'][i] = hut.press
    except TypeError:
        pass
    
#
# Now WY 2017-2021
#

directory = 'wy_2017_2021'
header    = 'wy_2017_2021'

ff = glob.glob(os.path.join(directory,'*velx*'))[1:]
ts_list2 = [int(i.split('.')[-2]) for i in ff]
ts_list2.sort()
_ts_list2 = np.array(ts_list2) + max(ts_list1)

hut = hydro_utils(dz_scale=dz_scale)
print ('Working on WY 2017-2021 velocity files')
for i,ii in zip(ts_list2, _ts_list2):
    #print ('working on {}/{}'.format(i, len(ts_list2)))
    try:
        hut.read_fields(i, directory, header)
        pf_out_dict['wtd'][ii] = hut.pull_wtd()
        pf_out_dict['specific_storage'][ii] = hut.pull_storage()
        pf_out_dict['velbed'][ii] = hut.vel_bedrock_layer(bedrock_mbls)
        pf_out_dict['velsoil'][ii] = hut.vel_soil_layer(bedrock_mbls)
        pf_out_dict['et'][ii] = hut.pull_et()
        pf_out_dict['sat'][ii] = hut.sat
        pf_out_dict['press'][ii] = hut.press
    except TypeError:
        pass

with open('parflow_out/pf_out_dict_0021.pk', 'wb') as ff_:
    pickle.dump(pf_out_dict, ff_)
    










'''
ck_mbls = 9.0

directory = 'wy_2017_2021'
header    = 'wy_2017_2021'

pf_out_dict = {'bedrock_mbls':bedrock_mbls,
               'wtd':{},
               'specific_storage':{},
               'velbed':{},
               'velsoil':{},
               'et':{},
               'sat':{},
               'press':{}}

# Use only files that exist
ff = glob.glob(os.path.join(directory,'*velx*'))
ts_list_ = [int(i.split('.')[-2]) for i in ff]
ts_list_.sort()

hut = hydro_utils(dz_scale=dz_scale)
print ('Working on WY 2017-2021 velocity files')
for i in ts_list_:
    #print ('working on {}/{}'.format(i, len(ts_list_)))
    try:
        hut.read_fields(i, directory, header)
        
        pf_out_dict['wtd'][i] = hut.pull_wtd()
        pf_out_dict['specific_storage'][i] = hut.pull_storage()
        pf_out_dict['velbed'][i] = hut.vel_bedrock_layer(bedrock_mbls)
        pf_out_dict['velsoil'][i] = hut.vel_soil_layer(bedrock_mbls)
        pf_out_dict['et'][i] = hut.pull_et()
        pf_out_dict['sat'][i] = hut.sat
        pf_out_dict['press'][i] = hut.press
    except TypeError:
        pass
    
with open('parflow_out/pf_out_dict.pk', 'wb') as ff_:
    pickle.dump(pf_out_dict, ff_)
#
# Single Timestep
#
# Run the functions
hut = hydro_utils(dz_scale=dz_scale)
hut.read_fields(1683, 'wy_2017_2021', 'wy_2017_2021')
wtd = hut.pull_wtd()
specific_storage = hut.pull_storage() 
velx_bed,  velz_bed  = hut.vel_bedrock_layer(bedrock_mbls)
velx_soil, velz_soil = hut.vel_soil_layer(bedrock_mbls)
'''


















