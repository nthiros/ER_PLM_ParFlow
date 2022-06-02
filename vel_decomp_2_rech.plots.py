# Script to process the Parflow velocity.pfb fields
# Generates plots of recharge fluxes, and storage changes
# Need to run 'vel_decomp_2_rech.py' first







import numpy as np
import pandas as pd
import os
import pickle 
import glob

import pdb

from parflowio.pyParflowio import PFData
import pyvista as pv

import parflow.tools as pftools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.patches as patches
plt.rcParams['font.size'] = 14
import matplotlib.colors as colors





#---------------------------------
#
# Utilities
#
#---------------------------------

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
    


        

#
# Parflow variable dz
#
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



#
# Single Timestep to produce some fields
#
ts   = 1683
# CHANGE ME #
# Note, it using the porosity field now
bedrock_mbls=9.0

hut = hydro_utils(dz_scale=dz_scale)
hut.read_fields(ts, 'wy_2017_2021', 'wy_2017_2021')

wtd = hut.pull_wtd()
specific_storage = hut.pull_storage() 
velx_bed,  velz_bed  = hut.vel_bedrock_layer(bedrock_mbls)
velx_soil, velz_soil = hut.vel_soil_layer(bedrock_mbls)



#
# Post-processing and plotting
#
# Read in pickle
pf = pd.read_pickle('parflow_out/pf_out_dict.pk')
#
timekeys = list(pf['wtd'].keys())
#
bedrock_mbls = pf['bedrock_mbls']
# Read in Land Surface Slope
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
# Dates for ET and CLM arrays
#
def pf_2_dates(startdate, enddate, f):
    '''Assumes ParFlow outputs every 24 hours'''
    s = pd.to_datetime(startdate)
    e = pd.to_datetime(enddate)
    d_list = pd.date_range(start=s, end=e, freq=f)
    # Drop Leap years again
    d_list_ = d_list[~((d_list.month == 2) & (d_list.day == 29))]
    return d_list_

nsteps = 1794

yrs = [2017,2018,2019,2020,2021] # Calender years within timeseries

dates = pf_2_dates('2016-10-01', '2021-08-29', '24H')
find_date_ind = lambda d: (dates==d).argmax()

# Water year indexes for et_out and clm_out
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

wy17 = wy_inds_[0]
wy18 = wy_inds_[1]
wy19 = wy_inds_[2]
wy20 = wy_inds_[3]
wy21 = wy_inds_[4]

wy_mapper = {2017:wy17,2018:wy18,2019:wy19,2020:wy20,2021:wy21}

month_map  = {1:'Oct',2:'Nov',3:'Dec',4:'Jan',5:'Feb',6:'Mar',7:'Apr',8:'May',9:'Jun',10:'Jul',11:'Aug',12:'Sep'}
months     = list(month_map.values())
months_num = np.array(list(month_map.keys()))




#
# Subsurface Storage
#
# First create masks for the bedrock and soil layers
Z  = np.flip(dz_scale).cumsum() # depth below land surface for each layer
Z_ = dz_scale.sum() - dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up

#bedrock_ind  = abs(Z_ - bedrock_mbls).argmin() # index of first bedrock layer
bedrock_ind = hut.bedrock_ind

por = hut.porosity
bedrock_mask = np.zeros_like(por)
bedrock_mask[0:bedrock_ind+1,:] = 1 # remember need +1 when indexing with a list

soil_mask = np.zeros_like(por)
soil_mask[bedrock_ind+1:,:] = 1

map_check = np.column_stack((np.arange(len(Z_)), Z_, por[:,0,0], bedrock_mask[:,0,0], soil_mask[:,0,0])) # check to make sure indexing is correct

# Single Timestep
bedrock_storage = specific_storage.copy()
bedrock_storage[bedrock_mask==0] = 0

soil_storage = specific_storage.copy()
soil_storage[bedrock_mask==1] = 0


#
# Transient Analysis
#
# Total of the spatial bedrock and storage
bedrock_ss = []
soil_ss    = []

for i in timekeys:
    ss  = pf['specific_storage'][i].copy()
    ss[bedrock_mask==0] = 0
    bedrock_ss.append(ss.ravel().sum())
    
    ss  = pf['specific_storage'][i].copy()
    ss[soil_mask==0] = 0
    soil_ss.append(ss.ravel().sum())
bedrock_ss = np.array(bedrock_ss)[1:] # clipping because do not exactly match with dates
soil_ss = np.array(soil_ss)[1:] # clipping because do not exactly match with dates

# Total Bedrock Recharge Fluxes
bedrock_zvel = []
bedrock_xvel = []
for i in timekeys:
    vvx_, vvz_ = pf['velbed'][i]
    bedrock_xvel.append(vvx_)
    bedrock_zvel.append(vvz_)
bedrock_xvel = np.array(bedrock_xvel)[1:] # clipping because do not exactly match with dates
bedrock_zvel = np.array(bedrock_zvel)[1:]

# Total Soil Recharge Fluxes
soil_zvel = []
soil_xvel = []
for i in timekeys:
    vvx_, vvz_ = pf['velsoil'][i]
    soil_xvel.append(vvx_)
    soil_zvel.append(vvz_)
soil_xvel = np.array(soil_xvel)[1:] # clipping because do not exactly match with dates
soil_zvel = np.array(soil_zvel)[1:]

# Water table depths
wtd_temp = []
for i in timekeys:
    wtd_temp.append(pf['wtd'][i])
wtd_temp = np.array(wtd_temp)[1:]





#dem = pd.read_csv('Perm_Modeling/elevation.sa', skiprows=1, header=None, names=['Z'])
#dem['X'] = dem.index * 1.5125





#-----------------------------------------------------
#
# Plotting
#
#-----------------------------------------------------

# Veloicity field at first bedrock layer
# Single Timestep simple plot
fig, axes = plt.subplots(2,1, figsize=(5,4))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.98, hspace=0.3)
#
# Plot Z-velocity 
#
ax = axes[0]
#ax.plot(xx, -1*velz_bed*24*1000/3, linewidth=2.0, label='top bedrock')
ax.plot(xx, -1*velz_soil*24*1000/3, linewidth=2.0, label='bottom soil')
ax.axhline(y=0, color='black', linestyle='--')
#
ax.set_ylabel('Recharge\n(mm/day)')
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
#
# Plot Topography Profile
#
ax = axes[1]
ax.plot(xx, land_surf, color='black', linewidth=2.0, label='land surface')
ax.plot(xx, wtd_elev, color='skyblue', linestyle='--', alpha=1.0, zorder=8.0, linewidth=2.0, label='water table')
ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', linewidth=2.0, label='bedrock')
# 
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
ax.set_ylim(land_surf.min()-50, land_surf.max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
# Cleanup
[axes[i].margins(x=0.0) for i in [0,1]]
[axes[i].minorticks_on() for i in [0,1]]
[axes[i].xaxis.set_major_locator(ticker.MultipleLocator(200)) for i in [0,1]] 
[axes[i].xaxis.set_minor_locator(ticker.MultipleLocator(50)) for i in [0,1]] 
[axes[i].tick_params(which='both', axis='both', top=True, right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.show()






#
# Transient ensembles
#
wy_ = 2019
wy = wy_mapper[wy_]

dd_ = dates[wy]
first_month = [(dd_.month==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]

# pick either first bedrock layer or last soil layer?
vel_master = bedrock_zvel[wy]
#vel_master = soil_zvel[wy]

wtd_master = wtd_temp[wy]


#
# Transient Plot
#
fig, axes = plt.subplots(2,1, figsize=(5,4))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.98, hspace=0.3)
#vz_ = bedrock_zvel[wy]
#vz_ = soil_zvel[wy]
vz_ = vel_master.copy()

fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.98, hspace=0.35)
#
# Plot Z-velocity 
#
ax = axes[0]
# ensemble of all timesteps
for i in range(len(vz_)):
    ax.plot(xx, vz_[i]*24*1000/3, color='grey', alpha=0.5)
ax.plot(xx, vz_.mean(axis=0)*24*1000/3, color='black', alpha=0.75)
ax.axhline(y=0.0, color='black', linestyle='--')
ax.set_ylabel('Bedrock\nRecharge\n(mm/day)')
ax.minorticks_on()
#ax.set_xlabel('Distance (m)')
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
#
# Plot Topography Profile
#
ax = axes[1]
ax.plot(xx, land_surf, color='black', linewidth=2.0, label='land surface')
ax.plot(xx, land_surf-wtd_temp[wy].mean(axis=0), color='skyblue', linestyle='--', alpha=1.0, zorder=8.0, linewidth=2.0, label='water table')
ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', linewidth=2.0, zorder=10.0, label='bedrock')
for i in range(len(wtd_master)):
    ax.plot(xx, land_surf-wtd_master[i], color='skyblue', linestyle='-', alpha=0.25, zorder=5.0, linewidth=1.0)
# 
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
ax.set_ylim(land_surf.min()-50, land_surf.max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
#
# Cleanup
#
[axes[i].margins(x=0.0) for i in [0,1]]
[axes[i].tick_params(which='both', axis='both', top=True, right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
[axes[i].xaxis.set_major_locator(ticker.MultipleLocator(200)) for i in [0,1]]
[axes[i].xaxis.set_minor_locator(ticker.MultipleLocator(50)) for i in [0,1]]
#fig.tight_layout()
plt.savefig('./figures/velz_temporal.png', dpi=300)
plt.show()






#
# Timeseries 
#
fig, axes = plt.subplots(2,1, figsize=(5,4))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.95, hspace=0.55)

vz_ = vel_master.copy().mean(axis=1)*24*1000/3
#
# Plot Z-velocity timeseries at discrete points
#
ax = axes[0]
# timeseries
ax.plot(range(len(vz_)), vz_, color='black', alpha=1.0, linewidth=2.0)
ax.set_ylabel('Bedrock\nRecharge\n(mm/day)')
ax.set_xticks(first_month)
ax.set_xticklabels(labels=months, rotation=45)
ax.set_xlim(0,366)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.axhline(0.0, color='black', linestyle='--')
#
# Plot Topography Profile
#
ax = axes[1]
ax.plot(xx, land_surf, color='black', linewidth=2.0, label='land surface')
ax.plot(xx, land_surf-wtd_temp[wy].mean(axis=0), color='skyblue', linestyle='--', alpha=1.0, zorder=8.0, label='water table')
ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', label='bedrock')
# 
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
ax.set_ylim(land_surf.min()-50, land_surf.max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200)) 
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50)) 
#
# Cleanup
#
[axes[i].margins(x=0.0) for i in [0,1]]
[axes[i].tick_params(which='both', axis='both', top=True, right=True) for i in [0,1]]
#[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
#plt.savefig('./figures/velz_temporal_pnts.png', dpi=300)
plt.show()





#
# Timeseries at Discrete Points
#
x_pnts = [265, 404, 494, 530] # these are the cell numbers, not x-distance
#x_pnts = (np.array([400, 500, 600, 700, 800])/1.5125).astype(int)

#cc = plt.cm.turbo(np.linspace(0,1,len(x_pnts)))
cc = plt.cm.viridis(np.linspace(0,1,len(x_pnts)))

fig, axes = plt.subplots(2,1, figsize=(5,4))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.98, hspace=0.3)
vz_ = vel_master.copy()
#vz_ = bedrock_zvel[wy]
#vz_ = soil_zvel[wy]
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.95, hspace=0.55)
#
# Plot Z-velocity timeseries at discrete points
#
ax = axes[0]
# ensemble of all timesteps
for i in range(len(x_pnts)):
    ax.plot(range(len(dd_)), vz_[:,x_pnts[i]]*24*1000/3, color=cc[i], alpha=1.0, linewidth=2.0)
ax.set_ylabel('Bedrock\nRecharge\n(mm/day)')
ax.set_xticks(first_month)
ax.set_xticklabels(labels=months, rotation=45)
ax.set_xlim(0,366)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.axhline(0.0, color='black', linestyle='--')
#
# Plot Topography Profile
#
ax = axes[1]
ax.plot(xx, land_surf, color='black', linewidth=2.0, label='land surface')
ax.plot(xx, land_surf-wtd_temp[wy].mean(axis=0), color='skyblue', linestyle='--', alpha=1.0, zorder=8.0, label='water table')
ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', label='bedrock')
# 
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
ax.set_ylim(land_surf.min()-50, land_surf.max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(200)) 
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50)) 
for i in range(len(x_pnts)):
    ax.axvline(x=x_pnts[i]*1.5125, color=cc[i], linestyle='--', linewidth=2.5)
#
# Cleanup
#
[axes[i].margins(x=0.0) for i in [0,1]]
[axes[i].tick_params(which='both', axis='both', top=True, right=True) for i in [0,1]]
#[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.savefig('./figures/velz_temporal_pnts.png', dpi=300)
plt.show()






# 
# Integrated velocity distribution over both space and time
#
fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.22, right=0.95, hspace=0.3)
vz_ = vel_master.copy().ravel() *24*1000/3 # 365 days x 559 cells

ax.hist(x=vz_, bins=30, density=True)
ax.set_ylabel('Density')
ax.set_xlabel('Bedrock Recharge (mm/day)')
ax.minorticks_on()
ax.grid()
plt.savefig('./figures/velz_distribution.png', dpi=300)
plt.show()











#
# Plot changes in storage
#
bss_ = bedrock_ss[wy]
sss_ = soil_ss[wy]
#
fig, axes = plt.subplots(1,1, figsize=(6.2,3))
fig.subplots_adjust(top=0.96, bottom=0.2, left=0.2, right=0.8)
ax = axes
l1 = ax.plot(np.arange(len(dd_)), bss_, color='black', label='Bedrock')
ax.set_ylabel('Bedrock Storage (m$^{3}$)')        
ax.set_xticks(first_month)
ax.set_xticklabels(labels=months, rotation=45)
ax.set_xlim(0,366)
ax2 = ax.twinx()
l2 = ax2.plot(np.arange(len(dd_)), sss_, color='black', linestyle='--', label='Soil')
ax2.set_ylabel('Soil Storage (m$^{3}$)') 
#
# Cleanup
#
fig.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5, loc='upper left', bbox_to_anchor=(0.2, 0.95))
#fig.tight_layout()
plt.show()


#
# Plot changes in storage as percent changes
#
bss_ = bedrock_ss[wy]
sss_ = soil_ss[wy]
# percent change
bssp = 100*(bss_ - bss_.min())/bss_.min()
sssp = 100*(sss_ - sss_.min())/sss_.min()
#
fig, axes = plt.subplots(1,1, figsize=(6.2,3))
fig.subplots_adjust(top=0.96, bottom=0.2, left=0.2, right=0.8)
ax = axes
l1 = ax.plot(np.arange(len(dd_)), bssp, color='black', label='Bedrock')

ax.set_ylabel('Bedrock Storage\n(% Change)')        
ax.set_xticks(first_month)
ax.set_xticklabels(labels=months, rotation=45)
ax.set_xlim(0,366)

ax2 = ax.twinx()
l2 = ax2.plot(np.arange(len(dd_)), sssp, color='black', linestyle='--', label='Soil')
ax2.set_ylabel('Soil Storage\n(% Change)') 
#
# Cleanup
#
fig.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5, loc='upper left', bbox_to_anchor=(0.2, 0.95))
#fig.tight_layout()
plt.savefig('./figures/specific_storage.png', dpi=300)
plt.show()





#
# Plot of water table depths
#
fig, axes = plt.subplots(1,1, figsize=(6,3))
wtd_ = wtd_temp[wy]
fig.subplots_adjust(top=0.96, bottom=0.2, left=0.2, right=0.95)
#
# Plot wtd
#
ax = axes
for i in range(len(vz_)):
    ax.plot(xx, wtd_[i], color='grey', alpha=0.5)
ax.plot(xx, wtd_.mean(axis=0), color='black', alpha=0.75)
ax.axhline(y=bedrock_mbls, linestyle='--', color='peru')
ax.set_ylabel('Water Table Depth (m)')        
ax.set_xlabel('Distance (m)')
#
# Cleanup
#
ax.invert_yaxis()
ax.minorticks_on()
ax.grid()
ax.margins(x=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
#fig.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5, loc='upper left', bbox_to_anchor=(0.2, 0.95))
#fig.tight_layout()
plt.show()













