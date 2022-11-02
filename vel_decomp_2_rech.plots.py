# Cleanup - 09/29/2022
# Update  - 10/14/2022 - plottoing 2000-2021
# Plotting script coupled to vel_decomp_2_rech.py
# Processes the Parflow*.pfb fields
# Generates plots of spatial and temporal recharge fluxes and storage changes
# Run 'vel_decomp_2_rech.py' first within the server directory with all runs
#     This will produce pf_out_dict.pk, which is used here



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
import matplotlib.dates as mdates

import matplotlib as mpl




#---------------------------------
#
# Utilities
#
#---------------------------------
# This is all run in vel_decomp_2_rech.py
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
    



def pf_2_dates(startdate, enddate, f):
    '''Assumes ParFlow outputs every 24 hours'''
    s = pd.to_datetime(startdate)
    e = pd.to_datetime(enddate)
    d_list = pd.date_range(start=s, end=e, freq=f)
    # Drop Leap years again
    d_list_ = d_list[~((d_list.month == 2) & (d_list.day == 29))]
    return d_list_


# set ticks to beginning of water year (october 01)
def set_wy(df):
    dates     = df.copy().index
    yrs       = dates.year
    yrs_      = np.unique(yrs)[1:]
    wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs_]
    wy_inds   = np.array([wy_inds_[i]*yrs_[i] for i in range(len(yrs_))]).sum(axis=0)
    first_yrs = [(wy_inds==i).argmax() for i in yrs_]
    return list(wy_inds), list(first_yrs)



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
# Start post-processing output from vel_decomp_2_rech.py
#
# Read in pickle - generated by vel_decomp_2_rech.py
#pf   = pd.read_pickle('parflow_out/pf_out_dict.pk')
#pf0016 = pd.read_pickle('parflow_out/pf_out_dict_0016.pk')
#pf1721 = pd.read_pickle('parflow_out/pf_out_dict_1721.pk')
pf   = pd.read_pickle('parflow_out/pf_out_dict_0021.pk')

dates = pf_2_dates('1999-10-01', '2021-08-29', '24H')



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

topo = land_surf.copy()
xrng = xx.copy()



#
#
#
"""yrs = [2017,2018,2019,2020,2021] # Calender years within timeseries
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
months_simp = ['Oct','','Dec','','Feb','','Apr','','Jun','','Aug','']

# For plotting -- index that corresponds to first of month, starting with october 1st
first_month = [(dates[wy17].month==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]
"""


## Expanding to more dates
yrs = np.unique(dates.year)[1:] # Calender years within timeseries
find_date_ind = lambda d: (dates==d).argmax()

# Water year indexes for et_out and clm_out
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

wy_mapper = lambda _yr: np.where(wy_inds==_yr, True, False)

month_map  = {1:'Oct',2:'Nov',3:'Dec',4:'Jan',5:'Feb',6:'Mar',7:'Apr',8:'May',9:'Jun',10:'Jul',11:'Aug',12:'Sep'}
months     = list(month_map.values())
months_num = np.array(list(month_map.keys()))
months_simp = ['Oct','','Dec','','Feb','','Apr','','Jun','','Aug','']

# For plotting -- index that corresponds to first of month, starting with october 1st
first_month = [(dates[wy_mapper(2017)].month==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]
dates_1yr   = dates[wy_mapper(2017)]










#-------------------------------------------------------------
#
# Subsurface Storage and Bedrock Fluxes
#
#-------------------------------------------------------------

# First create masks for the bedrock and soil layers
Z  = np.flip(dz_scale).cumsum() # depth below land surface for each layer
Z_ = dz_scale.sum() - dz_scale.cumsum() + dz_scale/2 # cell-centered z value, starting at base of the domain then going up

# initialize functions
hut = hydro_utils(dz_scale=dz_scale)
hut.read_fields(1683, 'wy_2017_2021', 'wy_2017_2021')
hut.pull_bedrock_ind()
# pull info
bedrock_ind = hut.bedrock_ind
por = hut.porosity
bedrock_mask = np.zeros_like(por)
bedrock_mask[0:bedrock_ind+1,:] = 1 # remember need +1 when indexing with a list

soil_mask = np.zeros_like(por)
soil_mask[bedrock_ind+1:,:] = 1
map_check = np.column_stack((np.arange(len(Z_)), Z_, por[:,0,0], bedrock_mask[:,0,0], soil_mask[:,0,0])) # check to make sure indexing is correct

#
# Total of the spatial bedrock and soil storage
#
ss_tot     = [] # total subsurface storage
ss_bedrock = [] # storage in bedrock
ss_soil    = [] # storage in soil and saprolite
for i in timekeys:
    ss_  = pf['specific_storage'][i].copy()
    ss_tot.append(ss_.ravel().sum())
    ss_[bedrock_mask==0] = 0
    ss_bedrock.append(ss_.ravel().sum())
    
    ss_  = pf['specific_storage'][i].copy()
    ss_[soil_mask==0] = 0
    ss_soil.append(ss_.ravel().sum())
ss_tot     = np.array(ss_tot)
ss_bedrock = np.array(ss_bedrock)
ss_soil    = np.array(ss_soil)

#
# Total Bedrock Recharge Fluxes -- ParFlow units should be m/hr
#
bedrock_zvel = []
bedrock_xvel = []
for i in timekeys:
    vvx_, vvz_ = pf['velbed'][i]
    bedrock_xvel.append(vvx_)
    bedrock_zvel.append(vvz_)
bedrock_xvel = np.array(bedrock_xvel)*1000*24 # clipping to match dates (mm/day). Shape [timesteps,domain x]
bedrock_zvel = np.array(bedrock_zvel)*1000*24 # 

#
# Total Soil Recharge Fluxes -- ParFlow units should be m/hr
#
soil_zvel = []
soil_xvel = []
for i in timekeys:
    vvx_, vvz_ = pf['velsoil'][i]
    soil_xvel.append(vvx_)
    soil_zvel.append(vvz_)
soil_xvel = np.array(soil_xvel)*1000*24 # clipping to match dates (mm/day
soil_zvel = np.array(soil_zvel)*1000*24

#
# Water table depths
#
wtd_temp = []
for i in timekeys:
    wtd_temp.append(pf['wtd'][i])
wtd_temp = np.array(wtd_temp) # meters below land surface



# Write all these to dictionary for later use
vel_decomp_dict = {'ss_tot':ss_tot,
                   'ss_bed':ss_bedrock,
                   'ss_soil':ss_soil,
                   'bed_xvel':bedrock_xvel,
                   'bed_zvel':bedrock_zvel,
                   'soil_xvel':soil_xvel,
                   'soil_zvel':soil_zvel,
                   'wtd':wtd_temp,
                   'dates':dates,
                   'wy':wy_inds}

with open('vel_decomp_dict.pk', 'wb') as handle:
    pickle.dump(vel_decomp_dict, handle)




#-----------------------------------------------------
#
# Plotting
#
#-----------------------------------------------------


#
# Plots of Vertical Bedrock Fluxes
#
wy_ = [2017, 2018, 2019, 2020]

fig, axes = plt.subplots(5,1, figsize=(5.5, 6.5))
fig.subplots_adjust(top=0.98, bottom=0.1, left=0.3, right=0.98, hspace=0.15)
# Plot Z-velocity (Flux) for each water year
for y in range(len(wy_)):
    # ensemble of all timesteps
    ax = axes[y]
    vz_ = bedrock_zvel[wy_mapper(wy_[y])]
    for i in range(len(vz_)):
        ax.plot(xx, vz_[i], color='C{}'.format(y), alpha=0.5) #color='grey'
    ax.plot(xx, vz_.mean(axis=0), color='grey', alpha=0.75, label=wy_[y])
    ax.text(0.08, 0.88, wy_[y], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    # Plot Topography Profile with wtd
    #ax = axes[4]
    #wtd_ = wtd_temp[wy_mapper[wy_[y]]]
    #for i in range(len(wtd_)):
    #    ax.plot(xx, land_surf-wtd_[i], color='skyblue', linestyle='-', alpha=0.25, zorder=5.0, linewidth=1.0)
    #ax.plot(xx, land_surf-wtd_.mean(axis=0), color='skyblue', linestyle='--', alpha=1.0, zorder=8.0, linewidth=2.0, label='water table')
#
ax = axes[4]
ax.plot(xx, land_surf, color='black', linewidth=2.5, label='land surface') 
#ax.plot(xx, land_surf-Z_[bedrock_ind], color='peru', linewidth=2.0, zorder=10.0, label='bedrock')
ax.set_ylabel('Elevation (m)')
ax.set_xlabel('Distance (m)')
#ax.set_ylim(land_surf.min()-10, land_surf.max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.minorticks_on()
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
#
[axes[i].set_ylim(bedrock_zvel.min(), bedrock_zvel.max()) for i in range(len(wy_))]
[axes[i].axhline(y=0.0, color='black', linestyle='--') for i in range(len(wy_))]
[axes[i].minorticks_on() for i in range(len(wy_))]
[axes[i].tick_params(axis='x', labelbottom=False) for i in range(len(wy_))]
#[axes[i].set_ylabel('Bedrock Flux\n(mm/day)') for i in range(len(wy_))]
axes[1].set_ylabel('Bedrock Flux (mm/day)')
axes[1].yaxis.set_label_coords(-0.18, -0.25)
# Cleanup
[axes[i].margins(x=0.0) for i in range(len(axes))]
[axes[i].tick_params(which='both', axis='both', top=False, right=True) for i in range(len(axes))]
[axes[i].grid() for i in range(len(axes))]
plt.savefig('./figures/velz_spatial.png', dpi=300)
plt.show()


#
# Timeseries Plots -- Spatially averaged vertical fluxes for the hillslope
#
fig, ax = plt.subplots(1,1, figsize=(4.5, 3.0))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.3, right=0.98)

for y in range(len(wy_)):
    vz_ = bedrock_zvel[wy_mapper(wy_[y])].mean(axis=1)
    ax.plot(range(len(vz_)), vz_, color='C{}'.format(y), alpha=1.0, lw=1.25, label=wy_[y]) #color='grey'
    ax.fill_between(range(len(vz_)), y2=0, y1=vz_, color='C{}'.format(y), alpha=0.15)
    #ax.text(0.08, 0.88, wy_[y], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months_simp, rotation=45)
    ax.set_xlim(0,366)
ax.set_ylabel('Bedrock Flux\n(mm/day)')
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.axhline(0.0, color='black', linestyle='--')
ax.margins(x=0)
ax.grid()
ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
plt.savefig('./figures/velz_temporal.png', dpi=300)
plt.show()





#
# Timeseries Plots -- Spatially averaged vertical fluxes for the hillslope
# Same, but all water years
#
wy_ = np.arange(2009, 2021)

cmap = plt.cm.jet  # define the colormap
cmap = plt.cm.tab20  # define the colormap
# extract all colors from the .jet map
#cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist = [cmap(i) for i in range(len(wy_))]
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(wy_.min(), wy_.max()+1, len(wy_)+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


# Plot 1
fig, axes = plt.subplots(2,1, figsize=(5.0, 5.5))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.3, right=0.85, hspace=0.3)
# Curves
for y in range(len(wy_)):
    vz_ = bedrock_zvel[wy_mapper(wy_[y])].mean(axis=1)
    ax = axes[0]
    #ax.plot(range(len(vz_)), vz_, color='C0', alpha=0.0, label=wy_[y])
    ax.fill_between(range(len(vz_)), y2=0, y1=vz_, color='C0', alpha=0.2)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months_simp, rotation=45)
    ax.set_xlim(0,366)
for label in ax.get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
ax.tick_params(axis='x', pad=0.1)
ax.set_ylabel('Bedrock Flux\n(mm/day)')
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.axhline(0.0, color='black', linestyle='--')
ax.margins(x=0)
ax.grid()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.remove()
# Distribution in peak timing vs magnitude
ax = axes[1]
vz_max_t = np.array([np.argmin(bedrock_zvel[wy_mapper(y)].mean(axis=1)) for y in wy_]) # date of peak
vz_max   = np.array([np.min(bedrock_zvel[wy_mapper(y)].mean(axis=1)) for y in wy_]) # value of peak
vz_sum   = np.array([(bedrock_zvel[wy_mapper(y)].mean(axis=1)).sum() for y in wy_]) # sum of temporal
#d_ = ax.scatter(dates_1yr[vz_max_t], vz_max, c=wy_, cmap=cmap, norm=norm, label=wy_)
d_ = ax.scatter(dates_1yr[vz_max_t], vz_sum, c=wy_, cmap=cmap, norm=norm, label=wy_)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
#ax.xaxis.set_minor_locator(mdates.DayLocator(interval=15))
ax.set_xlim(pd.to_datetime('2017-04-01'), pd.to_datetime('2017-07-01'))
ax.tick_params(axis='x',rotation=45, pad=0.1)
for label in ax.get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
ax.set_ylabel('Total Bedrock Flux\n(mm/day)')
ax.set_xlabel('Peak Bedrock Flux Timing')
ax.grid()
## colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(d_, cax=cax, ticks=wy_[::2])
cb.set_ticks(wy_+0.5)
cb.set_ticklabels(wy_)
for label in cb.ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)
#ax.legend(handlelength=1.0, labelspacing=0.25, handletextpad=0.5)
plt.savefig('./figures/velz_temporal_0021.png', dpi=300)
plt.show()


# Plot 2
fig, axes = plt.subplots(2,1, figsize=(5.5, 4.5))
fig.subplots_adjust(top=0.96, bottom=0.15, left=0.25, right=0.98, hspace=0.1)
# Distribution in peak timing vs magnitude
vz_max_t = np.array([np.argmin(bedrock_zvel[wy_mapper(y)].mean(axis=1)) for y in wy_]) # date of peak
vz_max   = np.array([np.min(bedrock_zvel[wy_mapper(y)].mean(axis=1)) for y in wy_]) # value of peak
vz_sum   = np.array([(bedrock_zvel[wy_mapper(y)].mean(axis=1)).sum() for y in wy_]) # sum of temporal
[axes[1].scatter(wy_[y], vz_sum[y], color=cmaplist[y], s=65) for y in range(len(wy_))]
axes[1].set_ylabel('Total Bedrock Flux\n(mm/day)')
[axes[0].scatter(wy_[y], vz_max[y], color=cmaplist[y], s=65) for y in range(len(wy_))]
axes[0].set_ylabel('Peak Bedrock Flux\n(mm/day)')
for i in [0,1]:
    axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(1))
    for label in axes[i].get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid()
axes[0].tick_params(axis='x', labelbottom=False)
plt.savefig('./figures/velz_temporal_0021.sums.png', dpi=300)
plt.show()




# 
# Integrated velocity distribution over both space and time
#
fig, axes = plt.subplots(1, len(wy_), figsize=(7, 3))
fig.subplots_adjust(top=0.88, bottom=0.22, left=0.05, right=0.98, hspace=0.3, wspace=0.08)
for y in range(len(wy_)):
    ax = axes[y]
    vz_ = bedrock_zvel[wy_mapper(wy_[y])].ravel()
    ax.hist(x=vz_, bins=30, density=True, color='C{}'.format(y))
    #ax.set_ylabel('Density')
    #ax.set_xlabel('Bedrock Flux (mm/day)')
    ax.minorticks_on()
    ax.grid()
    ax.tick_params(which='both', axis='y', left=False, labelleft=False)
    ax.set_xlim(bedrock_zvel.min(), bedrock_zvel.max())
    ax.set_title(wy_[y])
    ax.text(0.02, 0.88, '$\mu$={:.3f}\n$\sigma$={:.3f}'.format(vz_.mean(), vz_.std()),
            horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
axes[0].set_ylabel('Density')
fig.text(0.5, 0.04, 'Bedrock Flux (mm/day)', ha='center')
#plt.savefig('./figures/velz_distribution.png', dpi=300)
plt.show()







#
# Plot changes in storage
#

# wy2008-2021
_dates      = np.where(dates>'2008-09-30', True, False)
_ss_tot     = ss_tot[_dates]
ss_df       = pd.DataFrame(index=dates[_dates], data=_ss_tot)
ss_df['wy'] = set_wy(ss_df)[0]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 4), gridspec_kw={'height_ratios': [2,3]})
fig.subplots_adjust(top=0.96, bottom=0.2, left=0.2, right=0.96, hspace=0.1)
# Storage Values
ax[1].plot(dates[_dates], _ss_tot, color='black')
ax[1].set_ylabel('Subsurface Storage\n(m$^{3}$)') 
ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
#ax[i].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax[1].margins(x=0.01)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
for y in np.arange(2009,2021).reshape(6,2):
    ax[1].axvspan(pd.to_datetime('{}-10-01'.format(y[0])), pd.to_datetime('{}-09-30'.format(y[1])), alpha=0.04, color='red')
for label in ax[1].get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
#ax[1].set_xlim()
# Plot percent Change
ss_ = ss_df.groupby(by='wy') 
_inds = ss_.min().index
ax[0].scatter(_inds-0.5, 100*(ss_.max() - ss_.min())/ss_.min(), color='black')
ax[0].set_ylabel('Annual %\nChange')
ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].set_xlim(2007.9, 2021.05)
#ax[0].margins(x=0.05)
ax[0].tick_params(axis='x', labelbottom=False)
for y in np.arange(2009,2021).reshape(6,2):
    ax[0].axvspan(y[0], y[1], alpha=0.04, color='red')
# Cleanup
for i in [0,1]:
    ax[i].tick_params(axis='x', which='major', length=4.0, labelrotation=45, pad=0.1)
    ax[i].grid(axis='x')
plt.savefig('./figures/subsurface_storage_0821.png', dpi=300)
plt.show()





#
# Soil and Bedrock Storage WY 2008-2021
#
_dates = np.where(dates>'2008-09-30', True, False)
_ss_soil    = ss_soil[_dates]
_ss_bed     = ss_bedrock[_dates]
ss_df       = pd.DataFrame(index=dates[_dates], data=np.column_stack([_ss_soil,_ss_bed]), columns=['soil','bed'])
ss_df['wy'] = set_wy(ss_df)[0]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 5), gridspec_kw={'height_ratios': [2,3,3]})
fig.subplots_adjust(top=0.96, bottom=0.12, left=0.2, right=0.8, hspace=0.1)
# --- Soil Storage  ---
ax[1].plot(dates[_dates], _ss_soil, color='C0')
ax[1].set_ylabel('Soil\nStorage\n(m$^{3}$)') 
ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
#ax[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax[1].margins(x=0.01)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[1].tick_params(axis='x', labelbottom=False)
[ax[1].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['soil'], linestyle='--', alpha=0.4, color='C0') for y in ss_df['wy'].unique()]
[ax[1].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['bed'], linestyle='--', alpha=0.4, color='C1') for y in ss_df['wy'].unique()]
for y in np.arange(2009,2021).reshape(6,2):
    ax[1].axvspan(pd.to_datetime('{}-10-01'.format(y[0])), pd.to_datetime('{}-09-30'.format(y[1])), alpha=0.04, color='red')
# --- Bedrock Storage ---  
ax[2].plot(dates[_dates], _ss_bed, color='C1')
ax[2].set_ylabel('Bedrock\nStorage\n(m$^{3}$)') 
ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
#ax[2].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax[2].margins(x=0.01)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
[ax[2].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['soil'], linestyle='--', alpha=0.4, color='C0') for y in ss_df['wy'].unique()]
[ax[2].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['bed'], linestyle='--', alpha=0.4, color='C1') for y in ss_df['wy'].unique()]
for y in np.arange(2009,2021).reshape(6,2):
    ax[2].axvspan(pd.to_datetime('{}-10-01'.format(y[0])), pd.to_datetime('{}-09-30'.format(y[1])), alpha=0.04, color='red')
# --- Percent Change --
ss_ = ss_df.groupby(by='wy') 
_inds = ss_.min().index
pchange = (100*(ss_.max()['soil'] - ss_.min()['soil'])/ss_.min()['soil']).to_numpy()
ax[0].scatter(_inds-0.5, pchange, marker='o', facecolor='none', color='C0', label='soil')
ax[0].set_ylabel('Soil\nAnnual %\nChange')
ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].set_xlim(2007.95, 2021.05)
ax[0].tick_params(axis='x', labelbottom=False)
ax[0].set_ylim(0.95*np.floor(ax[0].get_ylim()[0]), 1.05*np.ceil(ax[0].get_ylim()[1]))
#ax[0].text(_inds[0]-0.4, pchange[0], 'soil', horizontalalignment='left', verticalalignment='center', color='C0', fontsize=12)
for y in np.arange(2017,2021).reshape(2,2):
    ax[0].axvspan(y[0], y[1], alpha=0.04, color='red')
ax2 = ax[0].twinx()
pchange = (100*(ss_.max()['bed'] - ss_.min()['bed'])/ss_.min()['bed']).to_numpy()
ax2.scatter(_inds-0.5, pchange, marker='D',facecolor='none', color='C1', label='bedrock')
ax2.set_ylabel('Bedrock\nAnnual %\nChange')
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax2.set_ylim(0.95*np.floor(ax2.get_ylim()[0]), 1.05*np.ceil(ax2.get_ylim()[1]))
#ax2.text(_inds[0]-0.4, pchange[0], 'bedrock', horizontalalignment='left', verticalalignment='center', color='C1', fontsize=12)
# --- Cleanup ---
for i in [0,1,2]:
    ax[i].tick_params(axis='x', which='major', length=4.0, labelrotation=0, pad=0.1)
    ax[i].grid(axis='x')
for label in ax[2].get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")
ax[2].tick_params(axis='x', rotation=45)
fig.legend(loc='upper left', bbox_to_anchor=(0.8, 0.75), fontsize=12.5, handlelength=1.0, labelspacing=0.25, handletextpad=0.25)
plt.savefig('./figures/subsurface_storage_0021.png', dpi=300)
plt.show()









#
# Soil and Bedrock Storage WY 2017-2021
#
wy_ = [2016, 2017, 2018, 2019, 2020]
_dates = np.where(dates>'2016-09-30', True, False)
_ss_soil    = ss_soil[_dates]
_ss_bed     = ss_bedrock[_dates]
ss_df       = pd.DataFrame(index=dates[_dates], data=np.column_stack([_ss_soil,_ss_bed]), columns=['soil','bed'])
ss_df['wy'] = set_wy(ss_df)[0]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), gridspec_kw={'height_ratios': [2,3,3]})
fig.subplots_adjust(top=0.96, bottom=0.12, left=0.2, right=0.8, hspace=0.1)
# --- Soil Storage  ---
ax[1].plot(dates[_dates], _ss_soil, color='C0')
ax[1].set_ylabel('Soil\nStorage\n(m$^{3}$)') 
ax[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
ax[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax[1].margins(x=0.01)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[1].tick_params(axis='x', labelbottom=False)
[ax[1].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['soil'], linestyle='--', alpha=0.4, color='C0') for y in ss_df['wy'].unique()]
[ax[1].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['bed'], linestyle='--', alpha=0.4, color='C1') for y in ss_df['wy'].unique()]
for y in np.arange(2017,2021).reshape(2,2):
    ax[1].axvspan(pd.to_datetime('{}-10-01'.format(y[0])), pd.to_datetime('{}-09-30'.format(y[1])), alpha=0.04, color='red')
#for label in ax[1].get_xticklabels(which='major'):
#    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
# --- Bedrock Storage ---  
ax[2].plot(dates[_dates], _ss_bed, color='C1')
ax[2].set_ylabel('Bedrock\nStorage\n(m$^{3}$)') 
ax[2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
ax[2].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax[2].margins(x=0.01)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
[ax[2].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['soil'], linestyle='--', alpha=0.4, color='C0') for y in ss_df['wy'].unique()]
[ax[2].axvline(ss_df[np.where((ss_df['wy']==y) & (ss_df.index>'{}-04-01'.format(y)), True, False)].idxmax()['bed'], linestyle='--', alpha=0.4, color='C1') for y in ss_df['wy'].unique()]
for y in np.arange(2017,2021).reshape(2,2):
    ax[2].axvspan(pd.to_datetime('{}-10-01'.format(y[0])), pd.to_datetime('{}-09-30'.format(y[1])), alpha=0.04, color='red')
# --- Percent Change --
ss_ = ss_df.groupby(by='wy') 
_inds = ss_.min().index
pchange = (100*(ss_.max()['soil'] - ss_.min()['soil'])/ss_.min()['soil']).to_numpy()
ax[0].scatter(_inds-0.5, pchange, marker='o', facecolor='none', color='C0', label='soil')
ax[0].set_ylabel('Soil\nAnnual %\nChange')
ax[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
ax[0].set_xlim(2015.95, 2020.95)
ax[0].tick_params(axis='x', labelbottom=False)
ax[0].set_ylim(0.95*np.floor(ax[0].get_ylim()[0]), 1.05*np.ceil(ax[0].get_ylim()[1]))
#ax[0].text(_inds[0]-0.4, pchange[0], 'soil', horizontalalignment='left', verticalalignment='center', color='C0', fontsize=12)
for y in np.arange(2017,2021).reshape(2,2):
    ax[0].axvspan(y[0], y[1], alpha=0.04, color='red')
ax2 = ax[0].twinx()
pchange = (100*(ss_.max()['bed'] - ss_.min()['bed'])/ss_.min()['bed']).to_numpy()
ax2.scatter(_inds-0.5, pchange, marker='D',facecolor='none', color='C1', label='bedrock')
ax2.set_ylabel('Bedrock\nAnnual %\nChange')
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax2.set_ylim(0.95*np.floor(ax2.get_ylim()[0]), 1.05*np.ceil(ax2.get_ylim()[1]))
#ax2.text(_inds[0]-0.4, pchange[0], 'bedrock', horizontalalignment='left', verticalalignment='center', color='C1', fontsize=12)
# --- Cleanup ---
for i in [0,1,2]:
    ax[i].tick_params(axis='x', which='major', length=4.0, labelrotation=0, pad=0.5)
    ax[i].grid(axis='x')
fig.legend(loc='upper left', bbox_to_anchor=(0.8, 0.75), fontsize=12.5, handlelength=1.0, labelspacing=0.25, handletextpad=0.25)
plt.savefig('./figures/subsurface_storage_1721.png', dpi=300)
plt.show()













#------------------------------------------------------------
#
# Water Tables
#
#------------------------------------------------------------

# Read in Raster Fields
#prs = pd.read_pickle('press_out_dict.pk')
#sat = pd.read_pickle('sat_out_dict.pk')

# Shape for pressure and saturatio field is 32 rows by 559 columns
#   sat[0,:] is the bottom layer of the domain, sat[31,:] is land surface
#   sat[31,0] is top left corner, sat[31,558] is top right corner (stream)


# Parflow Domain 
# these are cell centered values, Z_layer_num corresponds to rows in sat and prs
z_info = pd.read_csv('../utils/plm_grid_info_v3b.csv') 

#
# Using the Parflow pftools scipt
wtd = pd.DataFrame(wtd_temp)
wtd.index = dates



#
# Plot Water Table Depths
#
colors = plt.cm.twilight(np.linspace(0,1,len(months)+1))

fig, axes = plt.subplots(nrows=len(wy_), ncols=1, figsize=(6, 6))
fig.subplots_adjust(top=0.96, bottom=0.1, left=0.2, right=0.78, hspace=0.1)
for y in range(len(wy_)):
    ax = axes[y]
    wy = wy_[y]
    wt = wtd.iloc[wy_mapper(wy),:]
    # Plot all timesteps
    ax.fill_between(x=xx, y1=wt.min(axis=0), y2=wt.max(axis=0), color='grey', alpha=0.3)
    # Plot the first of the months
    for j in range(len(first_month)):
        jj = first_month[j]
        ax.plot(xx, wt.iloc[jj,:], color=colors[j], alpha=1.0, label='{}'.format(months[j]))
    #ax.text(0.1, 0.85, wy_[y], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.minorticks_on()
    ax.set_ylim(wtd.to_numpy().min()-1, wtd.to_numpy().max()+2)
    ax.invert_yaxis()
    ax.margins(x=0.01)
    ax.set_ylabel(wy_[y])
    if y != len(axes)-1:
        ax.tick_params(axis='x', labelbottom=False)
    ax.grid()
axes[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), fontsize=12.5, handlelength=1, labelspacing=0.25)
fig.text(0.04, 0.5, 'Water Table Depth (mbls)', va='center', rotation='vertical')
axes[3].set_xlabel('Distance (m)')
plt.savefig('./figures/wt_spatial.png',dpi=300)
plt.show()





