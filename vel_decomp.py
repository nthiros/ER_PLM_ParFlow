import numpy as np
import pandas as pd
import os
from parflowio.pyParflowio import PFData
import pyvista as pv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.patches as patches
plt.rcParams['font.size'] = 14



#-------------------------------------------
# Velocity Fields
#-------------------------------------------
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
dom = pv.read('tfg.out.Perm.vtk')
dom.clear_arrays()

# Add new arrays
dom['vx'] = vx.ravel()
dom['vy'] = vy.ravel()
dom['vz'] = vz.ravel()
dom.save('vel_comps.vtk')


# Calculate velocity magnitude and directions
Vx = vx[:,0,:] + 1.e-20
Vz = vz[:,0,:] + 1.e-20

mag = np.sqrt(Vx**2+Vz**2)
ang = np.arctan2(Vz,Vx) * 180/np.pi
# These angles are measured as degrees form the line at (1,0) --> a horizontal line in plus x direction is the azimuth
# Positive means the vector is pointing above horizontal 
# Negative means the vector is pointing below horizontal
# abs(ang) <= 90 means the vector is pointing to right direction
# abs(ang) >= 90 means the vector is pointing to left  direction
# ang=90 is straight up
# ang=-90 is straight down
vtk = dom.copy()
vtk.clear_arrays()
ang_ = ang.copy()
ang_[-1,0] = -180
ang_[-1,-1] = 180
vtk['Velocity Angle'] = ang_[:,np.newaxis,:].ravel()

#-----------
# Plots
#fig, ax = plt.subplots()
#ax.hist(ang.ravel(), bins=50)
#plt.show()

"""
#-----------
# VTK plot
sargs = dict(
    title_font_size=20,
    label_font_size=16,
    shadow=False,
    color='black',
    n_labels=9,
    fmt="%.0f")
#vtk.plot(cpos='XZ', background='white', scalar_bar_args=sargs, screenshot='./test.png')
plotter = pv.Plotter()
plotter.window_size = 800, 800
plotter.background_color = 'w'
plotter.add_mesh(vtk, scalar_bar_args=sargs)
plotter.show(auto_close=False, cpos='XZ')#, screenshot='./test.png')
#plotter.save_graphic('test.pdf', title='tester')
#plotter.save_graphic('/Users/nicholasthiros/Documents/SCGSR/Modeling/PLM_transect.v5/RunB.0/test.svg', title='tester')
"""


#--------------------
# Numpy plots
# Vtk cell info
cell_bounds = np.array([np.array(vtk.cell_bounds(i))[[0,1,4,5]] for i in range(vtk.GetNumberOfCells())])
cell_center = np.column_stack((cell_bounds[:,[0,1]].mean(axis=1), cell_bounds[:,[2,3]].mean(axis=1)))




# Build a high denisty mesh
xs = np.arange(cell_bounds[:,[0,1]].min(), cell_bounds[:,[0,1]].max(), 0.5)
zs = np.arange(cell_bounds[:,[2,3]].min(), cell_bounds[:,[2,3]].max(), 0.1)
xx, zz = np.meshgrid(xs, zs)




# Map from vtk to high density grid
# Note, this takes a long time - only run once
RunIt = False


if RunIt:
    xg, zg = xx.ravel(), zz.ravel()
    grid  = np.column_stack((xg,zg))
    cgrid = np.ones(len(grid))*-9999.0
        
    for cell_id in range(len(cell_bounds)):
        #cell_id = 1
        xlo, xhi = cell_bounds[cell_id,[0,1]]
        zlo, zhi = cell_bounds[cell_id,[2,3]]
        inrect = np.where((xg >= xlo) & (xg <= xhi) & (zg >= zlo) & (zg <= zhi), True, False)
        cgrid[inrect] = cell_id #cell_data[cell_id]
    np.savetxt('cgrid.txt',cgrid)
cgrid = np.loadtxt('cgrid.txt')




# dummy index
dum_ind = len(cell_bounds)
cgrid[cgrid == -9999.0] = dum_ind

# Basic array
cc = cgrid.copy().reshape(len(zs),len(xs))
cc[cc == dum_ind] = np.NaN

# Velocity Angle
# Add a dummy value 
ang = ang_[:,np.newaxis,:].ravel()
dd = np.concatenate((ang, [dum_ind]))
vel_ang = dd[cgrid.astype(int)]
vel_ang[vel_ang==dum_ind] = np.NaN
vel_ang = vel_ang.reshape(len(zs),len(xs))

"""
fig, ax = plt.subplots(figsize=(8,6))
v = ax.imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Mangle Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
plt.show()
"""


#---------------------
# Read in saturation
sf = 'wy_2017_2021/wy_2017_2021.out.satur.01683.pfb'
sat = read_pfb(sf)[:,0,:]

dds = np.concatenate((sat.ravel(), [dum_ind]))
sat_mp = dds[cgrid.astype(int)]
sat_mp[sat_mp==dum_ind] = np.NaN
sat_mp = sat_mp.reshape(len(zs),len(xs))

"""
# Saturation
# Add a dummy value 
fig, ax = plt.subplots(figsize=(8,6))
v = ax.imshow(np.flip(sat_mp,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Mangle Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Saturation')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
plt.show()
"""

# Loop through columns, find row where saturation occurs
sat_layer = []
for i in range(sat.shape[1]):
    col = sat[:,i]
    indtokeep = (col < 1.0).argmax()
    sat_layer.append(indtokeep-1)
    
    
# What is velocity angle at these sat boundaries?
vel_wt  = []
vel_mag = []
for v in range(ang_.shape[1]):
    vel_wt.append(ang_[sat_layer[v],v])
    vel_mag.append(mag[sat_layer[v],v])



fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,6))
fig.subplots_adjust(right=0.85)
v = ax[1].imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Mangle Colorbar
#fig.colorbar(v, ax=ax)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
# Clean up
ax[1].set_xlabel('Distance (m)')
ax[1].set_ylabel('Elevation (m)')

# Add in velocity angle at water table
X = np.arange(cell_bounds[:,[0,1]].min(), cell_bounds[:,[0,1]].max(), 1.5125)
vv = ax[0].plot(X, vel_wt, color='black')
ax[0].axhline(0, color='black', linestyle='--')
ax[0].margins(x=0.0)
#ax2 = ax[0].twinx()
#ax2.plot(X, vel_mag)
# Mangle
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
#divider = make_axes_locatable(ax2)
#cax = divider.append_axes("right", size="1.5%", pad=0.05) 
#cax.remove()
# Cleanup
ax[0].set_ylabel('Velocity Angle (deg)')
#ax2.set_ylabel('Velocity Magnitude (m/hr)', color='C0')
plt.show()




#--------------------------------------------
# Ecoslim Age Distributions
#--------------------------------------------

def flux_wt_rtd(rtd_dict_unsort, model_time, well_name, nbins):
    '''Convert particle ages at a single model time and well location to a residence time distribution.
    Returns:
        - rtd_df:   Dataframe with mass weighted ages for all paricles (sorted by age)
        - rtd_dfs:  Dataframe similar to above but bins age distribution into discrete intervals  
    Inputs: 
        - rtd_dict_unsort: output from ecoslim_pnts_vtk.read_vtk() above
        - model_time: model time to consider. Must be in rtd_dict_unsort.keys()
        - well_name: observation well to consider. Must be in rtd_dict_unsort.keys()
        - nbins: Number of intervals to bin the ages into.'''
    # Flux Weighted RTD
    # Info regarding particles at a single timestep and single point (box) of the domain
    #pdb.set_trace()
    rtd    = rtd_dict_unsort[model_time][well_name] 
    rtd_df = pd.DataFrame(data=rtd,columns=['Time','Mass','Source','Xin'])
    rtd_df['wt'] = rtd_df['Mass']/rtd_df['Mass'].sum()
    rtd_df.sort_values('Time', inplace=True)
    rtd_df['Time'] /= 8760
    
    # Now some binning
    #nbins = 10
    gb =  rtd_df.groupby(pd.cut(rtd_df['Time'], nbins))
    rtd_dfs = gb.agg(dict(Time='mean',Mass='sum',Source='mean',Xin='mean',wt='sum'))
    rtd_dfs['count'] = gb.count()['Time']

    return rtd_df, rtd_dfs


#------------
# Read in forcing so I know what indices correspond to dates
def pf_2_dates(startdate, enddate, f):
    '''Assumes ParFlow outputs every 24 hours'''
    s = pd.to_datetime(startdate)
    e = pd.to_datetime(enddate)
    d_list = pd.date_range(start=s, end=e, freq=f)
    # Drop Leap years again
    d_list_ = d_list[~((d_list.month == 2) & (d_list.day == 29))]
    return d_list_

pf_17_21  =  pd.read_csv('./parflow_out/wy_2017_2021_wt_bls.csv') 
pf_17_21.index  =  pf_2_dates('2016-09-30', '2021-08-29', '24H')
date_map = pd.DataFrame(pf_17_21.index, columns=['Date'])


# Well Info
wells = pd.read_csv('../utils/wells_2_pf_v3b.csv', index_col='well') 


# Read in EcoSLIM particles
rtd_dict = pd.read_pickle('./ecoslim_rtd.pk')


# Dates for sampling
samp_date = '2021-05-11'
model_time = date_map[date_map['Date'] == samp_date].index[0]
model_time = list(rtd_dict.keys())[abs(list(rtd_dict.keys()) - model_time).argmin()]
#model_time = 1681


# Read in DEM
dem = pd.read_csv('Perm_Modeling/elevation.sa', skiprows=1, header=None, names=['Z'])
dem['X'] = dem.index * 1.5125

# Find location of particle inputs
rtd_df_dict = {}
for w in ['PLM1','PLM7','PLM6']:
    rtd_df_dict[w] = flux_wt_rtd(rtd_dict, model_time, w, 15)[0]

for i in list(rtd_df_dict.keys()):
    rtd_df = rtd_df_dict[i]
    dem[i] = 0.0
    for j in range(len(rtd_df)):
        ind = abs(dem['X'] - rtd_df.loc[j,'Xin']).idxmin()
        dem.loc[ind, i] += 1.0

# smooth out?
dem = dem.groupby(pd.cut(dem.index, 150)).agg(dict(Z='mean',X='mean',PLM1='sum',PLM7='sum',PLM6='sum'))

#
# Plots #1
fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [1,2]}, figsize=(7,4))
fig.subplots_adjust(right=0.82, top=0.98, hspace=0.05, left=0.15)
# Plot Velocity Angle
v = ax[1].imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Plot well Locations
for i in range(len(wells)):
    w = wells.index[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    rect = patches.Rectangle((xpos, zpos-5), 10, 10, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10)
    ax[1].add_patch(rect)
# Mangle Colorbar
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle (deg)')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
# Clean up
ax[1].set_xlabel('Distance (m)')
ax[1].set_ylabel('Elevation (m)')
# Add in Recharge Location
for w in ['PLM1','PLM7','PLM6']: 
    ax[0].plot(dem['X'], dem[w]/dem[w].max(), label=w)
# Cleanup
ax[0].margins(x=0.0)
ax[0].set_ylabel('Number of\nParticles')
ax[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.png', dpi=320)
plt.savefig('./figures/recharge_loc.svg', format='svg')
plt.show()




#---------------------------------------
# Extend to Include Temporal Dynamics
#---------------------------------------
dem = pd.read_csv('Perm_Modeling/elevation.sa', skiprows=1, header=None, names=['Z'])
dem['X'] = dem.index * 1.5125


def pull_particles(mod_time, rtd_dict, dem_df):
    rtd_df_dict = {}
    for w in ['PLM1','PLM7','PLM6']:
        rtd_df_dict[w] = flux_wt_rtd(rtd_dict, mod_time, w, 15)[0]
    
    for i in list(rtd_df_dict.keys()):
        rtd_df = rtd_df_dict[i]
        dem_df[i] = 0.0
        for j in range(len(rtd_df)):
            ind = abs(dem_df['X'] - rtd_df.loc[j,'Xin']).idxmin()
            dem_df.loc[ind, i] += 1.0
    return dem_df



time_list = list(rtd_dict.keys())[::5] + [model_time]
date_list = date_map.loc[time_list, 'Date']
dem_dict = {}
for t in range(len(time_list)):
    mod_time = time_list[t]
    dem_ = pull_particles(mod_time=mod_time, rtd_dict=rtd_dict.copy(), dem_df=dem.copy())
    # smooth with grouping?
    dem_ = dem_.groupby(pd.cut(dem_.index, 100)).agg(dict(Z='mean',X='mean',PLM1='sum',PLM7='sum',PLM6='sum'))
    dem_dict[mod_time] = dem_




# Plot through time
fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [1,2]}, figsize=(7,3.5))
fig.subplots_adjust(right=0.82, top=0.94, hspace=0.005, left=0.15)
# Plot Velocity Angle
v = ax[1].imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Plot well Locations
for i in range(len(wells)):
    w = wells.index[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    rect = patches.Rectangle((xpos-8, zpos-8), 16, 16, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10)
    ax[1].add_patch(rect)
# Mangle Colorbar
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle\n(degree)')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
# Clean up
ax[1].set_xlabel('Distance (m)')
ax[1].set_ylabel('Elevation (m)')

# Add in Recharge Location
wells_list = ['PLM1','PLM7','PLM6']
for mm in range(len(dem_dict.keys())):
    m = list(dem_dict.keys())[mm]
    dem_ = dem_dict[m]
    for ww in range(len(wells_list)):
        w = wells_list[ww]
        if m != 1681:
            ax[0].plot(dem_['X'], dem_[w]/dem_[w].max(), color='C{}'.format(ww), alpha=0.5, label=w if mm==0 else '')
        elif m == 1681:
            cc =  ['darkblue','darkorange','darkgreen']
            ax[0].plot(dem_['X'], dem_[w]/dem_[w].max(), color=cc[ww], linestyle=(0, (2, 2)), alpha=1.0, label=w if mm==0 else '', zorder=10) 
            ax[0].plot(dem_['X'], dem_[w]/dem_[w].max(), color='black', linestyle=(2, (2, 2)), alpha=0.5, label=w if mm==0 else '', zorder=10) 
# Cleanup
ax[0].margins(x=0.0)
ax[0].set_ylabel('Rel. Number\nof Particles')
# x-axis
[ax[i].xaxis.set_major_locator(ticker.MultipleLocator(200)) for i in [0,1]]
[ax[i].xaxis.set_minor_locator(ticker.MultipleLocator(50)) for i in [0,1]]
[ax[i].tick_params(axis='x', bottom=True, top=False, length=4, width=1.25) for i in [0,1]]
ax[0].set_xticklabels([])
# yaxis
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax[1].yaxis.set_major_locator(ticker.MultipleLocator(100))
ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(25))


# Legend
leg = ax[0].legend(loc='upper left', bbox_to_anchor=(0.98, 1.1), frameon=False)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc_ens.png', dpi=300)
plt.savefig('./figures/recharge_loc_ens.svg', format='svg')
plt.show()






