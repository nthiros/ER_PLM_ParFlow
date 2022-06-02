# Script to process the Parflow velocity fields


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

import matplotlib.colors as colors

import pdb



#-------------------------------------------
#
# Read in and parse .pfb velocity fields
#
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

Vel_mag = np.sqrt(Vx**2+Vz**2) # velocity vector magnitude
Vel_ang = np.arctan2(Vz,Vx) * 180/np.pi # velocity vectory angle
# These angles are measured as degrees form the line at (1,0) --> a horizontal line in plus x direction is the azimuth
# Positive means the vector is pointing above the horizontal line (1,0) (3 o'clock)
# Negative means the vector is pointing below the horizontal line (1,0)
# ang=90 is straight up
# ang=-90 is straight down
# abs(ang) <= 90 means the vector is pointing to right direction
# abs(ang) >= 90 means the vector is pointing to left  direction
vtk = dom.copy()
vtk.clear_arrays()
Vel_ang_ = Vel_ang.copy()
Vel_ang_[-1,0] = -180
Vel_ang_[-1,-1] = 180
vtk['Velocity Angle'] = Vel_ang_[:,np.newaxis,:].ravel()



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


#----------------------------
#
# Matplotlib Plotting
#
#----------------------------
# Extract vtk cell info
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


#
# Map from Parflow velocity grid to high density mesh
# Velocity Angle
#
# Add a dummy value 
ang = Vel_ang_[:,np.newaxis,:].ravel()
dd = np.concatenate((ang, [dum_ind]))
vel_ang = dd[cgrid.astype(int)]
vel_ang[vel_ang==dum_ind] = np.NaN
vel_ang = vel_ang.reshape(len(zs),len(xs))


fig, ax = plt.subplots(figsize=(8,6))
v = ax.imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Mangle Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
fig.tight_layout()
plt.show()



# 
# Mapper from higher density grid back to Parflow grid
#
# Parflow grid cell center cartesian coordinates
xx_ = np.flip(cell_center[:,0].reshape(32,559), axis=0)
zz_ = np.flip(cell_center[:,1].reshape(32,559), axis=0)





#-----------------------------
#
# Read in saturation field
#
#-----------------------------
sf = 'wy_2017_2021/wy_2017_2021.out.satur.01683.pfb'
sat = read_pfb(sf)[:,0,:]

dds = np.concatenate((sat.ravel(), [dum_ind]))
sat_mp = dds[cgrid.astype(int)]
sat_mp[sat_mp==dum_ind] = np.NaN
sat_mp = sat_mp.reshape(len(zs),len(xs))



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
fig.tight_layout()
plt.show()


# Find the Water Table
# Loop through columns, find row where saturation occurs
sat_layer = []
for i in range(sat.shape[1]):
    col = sat[:,i]
    indtokeep = (col < 1.0).argmax()
    sat_layer.append(indtokeep-1)
    
    
# What are the velocity components at the water table?
vel_ang_wt  = []
vel_mag_wt = []
for v in range(Vel_ang_.shape[1]):
    vel_ang_wt.append(Vel_ang_[sat_layer[v],v])
    vel_mag_wt.append(Vel_mag[sat_layer[v],v])



# Want to plot a line to delineate the water table depth
sat_ = sat.copy()

for i in range(len(sat_layer)):
    sat_[sat_layer[i],i] = 2.0

dds_ = np.concatenate((sat_.ravel(), [dum_ind]))
sat_mp_ = dds_[cgrid.astype(int)]
sat_mp_[sat_mp_==dum_ind] = np.NaN
sat_mp_ = sat_mp_.reshape(len(zs),len(xs))

sat_line = np.argwhere(sat_mp_==2.0)
sat_line_uni = []
for i in np.unique(sat_line[:,1]):
    sat_line_uni.append(sat_line[sat_line[:,1]==i][-1])
sat_line_uni = np.array(sat_line_uni)



#
# Land surface slope
#
slope = read_pfb('slope_x_v4.pfb')
slope = slope[0,0,:] * 180/np.pi




#
# Velocity at Fractured Bedrock Interface
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
por   = []
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
            if '.'.join(ll[i].split()[1].split('.')[:4]) == 'Geom.i{}.Porosity.Value'.format(j):
                #print (ll[i].split()[-1])
                por.append(float(ll[i].split()[-1]))

# Mangle Units
cell_multiplier = 10.0 
cells_ = dz
cells  = np.array(cells_) * cell_multiplier

Z  = np.flip(cells).cumsum() # depth below land surface for each layer
Z_ = cells.sum() - cells.cumsum() + cells/2 # cell-centered z value, starting at base of the domain then going up

K   = np.array(K_mhr + [K_mhr[-1]]) / 3600
por = np.array(por + [por[-1]])
K   = np.flip(K)
por = np.flip(por)

#----------------
# CHANGE ME
#----------------

fshale_bls = 9.0 # shale is 6 m bls

fs_ind_ = (por==por.min()).argmin()-1
fs_ind  = abs(Z_ - fshale_bls).argmin() # index of first bedrock layer
#fs_ind += 1 #+1 is layer right above the first fshale layer


#
# Clip the velocity field at the bedrock depths
#
vel_ang_bed0 = Vel_ang_[fs_ind,:] # first bedrock cell
vel_mag_bed0 = Vel_mag[fs_ind,:]

#
vel_ang_bed1 = Vel_ang_[fs_ind+1,:] # cell right about first bedrock
vel_mag_bed1 = Vel_mag[fs_ind+1,:]




# Plot a line to delineate the water fractured bedrock depth
bed_img = sat.copy()*0.0
bed_img[14,:]  = 2.0

dds_    = np.concatenate((bed_img.ravel(), [dum_ind]))
bed_mp = dds_[cgrid.astype(int)]
bed_mp[bed_mp==dum_ind] = np.NaN
bed_mp = bed_mp.reshape(len(zs),len(xs))

bed_line = np.argwhere(bed_mp==2.0)
bed_line_uni = []
for i in np.unique(bed_line[:,1]):
    bed_line_uni.append(bed_line[bed_line[:,1]==i][-1])
bed_line_uni = np.array(bed_line_uni)




#
# How does water table depth compare to bedrock depth?





# 
# Plot
#

fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(7,6))
fig.subplots_adjust(top=0.96, bottom=0.12, right=0.8, left=0.2, hspace=0.25)
#
# Velocity Field 
ax = axes[0]
v = ax.imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Line at water table
ax.plot(xs[sat_line_uni[:,1]], zs[sat_line_uni[:,0]], color='C0', alpha=0.75, linewidth=1.0)
# Line at bedrock
ax.plot(xs[bed_line_uni[:,1]], zs[bed_line_uni[:,0]], color='C6', alpha=0.75, linewidth=1.0)
# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05)  
cb = fig.colorbar(v, cax=cax, label='Velocity Angle\n(deg)')
cb.locator = ticker.MaxNLocator(nbins=7)
cb.update_ticks()
# Clean up
ax.set_ylabel('Elevation (m)')
ax.minorticks_on()
ax.grid()
#
# Velocity angle at water table
ax = axes[1]
X = np.arange(cell_bounds[:,[0,1]].min(), cell_bounds[:,[0,1]].max(), 1.5125)
# WT Angle
ax.plot(X, vel_ang_wt, color='C0', linewidth=1.5, alpha=0.75)
# WT Mag
#cc = np.array(vel_mag_wt)/3600
#vv = ax[0].scatter(X, vel_ang_wt, linestyle='-', s=5.0, c=cc, norm=colors.LogNorm(vmin=cc[:-1].min(), vmax=cc[:-1].max()), cmap='viridis', zorder=8)
#
# Velocity angle at bedrock
ax.plot(X, vel_ang_bed0, color='C2', linewidth=1.5, alpha=0.75)
ax.plot(X, vel_ang_bed1, color='C3', linewidth=1.5, alpha=0.75)
# bed Mag
#cc = np.array(vel_mag_bed0)/3600
#vv = ax[0].scatter(X, vel_ang_bed0, linestyle='-', s=5.0, c=cc, norm=colors.LogNorm(vmin=cc[:-1].min(), vmax=cc[:-1].max()), cmap='viridis', zorder=8)
#
# Topography slope
ax.plot(X, slope, color='C4', linestyle='--', linewidth=1.5, alpha=1.0, label='Topo. Slope (deg)')
#
# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
#cb = fig.colorbar(vv, cax=cax, label='Velocity Mag.\n(m/s)')
#cb.locator = ticker.MaxNLocator(nbins=7)
#cb.locator = ticker.LogLocator(base=10.0, subs=[0.1])
cb.update_ticks()
cax.remove()
#
# Cleanup
ax.axhline(0, color='black', linestyle='-')
ax.margins(x=0.0)
ax.grid()
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Velocity Angle\n(deg)')
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.set_ylim(-125, 125)
ax.minorticks_on()
#
# Velocity Magnitdues
#
ax = axes[2]
X = np.arange(cell_bounds[:,[0,1]].min(), cell_bounds[:,[0,1]].max(), 1.5125)
# WT mag
#ax.plot(X, np.array(vel_mag_wt)*8760, color='C0', linewidth=1.5, alpha=0.75)
# Velocity mag at bedrock
ax.plot(X, np.array(vel_mag_bed0)*8760, color='C2', linewidth=1.5, alpha=0.75)
#ax.plot(X, np.array(vel_mag_bed1)*8760, color='C3', linewidth=1.5, alpha=0.75)
# Testing 
ax.plot(X, np.array(Vz[fs_ind])*8760, color='C2', linestyle='--', linewidth=1.5, alpha=0.75)
#
# Colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
#cb.update_ticks()
cax.remove()
#
# Cleanup
#ax.set_yscale('log')
ax.margins(x=0.0)
ax.grid()
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Velocity Mag\n(m/s)')
#ax.set_ylim(1.e-10, 1.e-6)
#ax.minorticks_on()
#ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=8))
#ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.1,)))

#ax[0].legend()
plt.show()




















#--------------------------------------------
#
# Ecoslim Age Distributions
#
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
    if nbins:
        gb =  rtd_df.groupby(pd.cut(rtd_df['Time'], nbins))
        rtd_dfs = gb.agg(dict(Time='mean',Mass='sum',Source='mean',Xin='mean',wt='sum'))
        rtd_dfs['count'] = gb.count()['Time']
    else:
        rtd_dfs = rtd_df

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

yrs = [2017,2018,2019,2020,2021] # Calender years within timeseries
wy_inds_  = [np.where((date_map > '{}-09-30'.format(i-1)) & (date_map < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

date_map['wy'] = wy_inds.ravel()


# Well Info
wells = pd.read_csv('../utils/wells_2_pf_v4.dummy.csv', index_col='well') 


# Read in EcoSLIM particles
rtd_dict = pd.read_pickle('./parflow_out/ecoslim_rtd.pk')

# Dates for sampling
#samp_date = '2021-05-11'
#model_time = date_map[date_map['Date'] == samp_date].index[0]
#model_time = list(rtd_dict.keys())[abs(list(rtd_dict.keys()) - model_time).argmin()]
#model_time = 1681






# testing new way to do this faster
#---------------------------------------
#
# Extend to Include Temporal Dynamics
#
#---------------------------------------
dem = pd.read_csv('Perm_Modeling/elevation.sa', skiprows=1, header=None, names=['Z'])
dem['X'] = dem.index * 1.5125


def pull_particles(mod_time, rtd_dict):
    rtd_df_dict = {}
    for w in list(rtd_dict[mod_time].keys()):
        #pdb.set_trace()
        rtd_df = flux_wt_rtd(rtd_dict.copy(), mod_time, w, False)[0]
        #if w == 'PLM6':
        #    print (time_list.loc[mod_time,'Date'], len(rtd_df), rtd_df['Mass'].sum())
        rtd_df_locs_ = rtd_df.groupby(pd.cut(rtd_df['Xin'],bins=np.arange(0,559*1.5125,10.0)))
        rtd_df_locs = rtd_df_locs_.agg(dict(Time='mean',Mass='sum',Source='mean',Xin='mean',wt='sum'))
        rtd_df_locs['Xmu'] = np.arange(10.0/2,558*1.5125,10.0)
        
        rtd_df_dict[w] = rtd_df_locs
    return rtd_df_dict

#
# Pick a Year for plots
time_list_ = date_map[date_map['wy']==2019]
tinds_     = [i in list(rtd_dict.keys()) for i in time_list_.index]
time_list  = time_list_[tinds_]

inf_dict = {}
for t in range(len(time_list)):
    mod_time = time_list.index[t]
    # smooth with grouping?
    #dem_ = dem_.groupby(pd.cut(dem_.index, 100)).agg(dict(Z='mean',X='mean',PLM1='sum',PLM7='sum',PLM6='sum'))
    inf_dict[mod_time] = pull_particles(mod_time=mod_time, rtd_dict=rtd_dict.copy())

#
# First of month index
#first_month = [(time_list['Date'].dt.month==i).idxmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]
first_month = [(time_list['Date'].dt.month==i).idxmax() for i in [6]]
colors = plt.cm.twilight(np.linspace(0.25,0.75,len(first_month)+1))


wells_list = list(rtd_dict[1].keys())

#
# Maximum masses through all timesteps -- for normalization 
mass_max = {}
for w in wells_list:
    max_ = 0
    for t in list(inf_dict.keys()):
        if inf_dict[t][w]['Mass'].sum() >= max_:
            max_ = inf_dict[t][w]['Mass'].sum()
        else:
            pass
    mass_max[w] = max_
#
# Masses on June First -- for normalization 
mass_june = {}
for w in wells_list:
    mass_june[w] = inf_dict[(time_list['Date'].dt.month==6).idxmax()][w]['Mass'].sum()



#
# Plot of PLM1, PLM7, PLM6
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 2))
fig.subplots_adjust(right=0.78, left=0.2, top=0.94, bottom=0.3)
# Add in Recharge Location
wells_list = ['PLM1','PLM7','PLM6']
for mm in range(len(inf_dict.keys())):
    m = list(inf_dict.keys())[mm]
    inf_ = inf_dict[m]
    for ww in range(len(wells_list)):
        w = wells_list[ww]
        #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color='C{}'.format(ww), alpha=0.5, label=w if mm==0 else '')
        ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='C{}'.format(ww), alpha=0.5, label=w if mm==0 else '')
        if m in first_month:
            #cc =  ['darkblue','darkorange','darkgreen']
            #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color=cc[ww],  linestyle=(0, (2,2)), alpha=1.0, label=w if mm==0 else '', zorder=10) 
            #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color='black', linestyle=(2, (2, 2)), alpha=0.5, label=w if mm==0 else '', zorder=10) 
            ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='C{}'.format(ww), linestyle=(0, (2,2)), alpha=1.0, label=w if mm==0 else '', zorder=10)
            ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='black', linestyle=(2, (2,2)), alpha=0.75, label=w if mm==0 else '', zorder=10) 
        # testing
        if w == 'PLM6':
            print ('{}  {}/{}  {}'.format(time_list.loc[m,'Date'], inf_[w]['Mass'].sum(), mass_june[w], inf_[w]['Mass'].sum()/mass_june[w]))
# Cleanup
ax.set_xlabel('Distance (m)')
ax.margins(x=0.0)
ax.set_ylabel('Rel. Number\nof Particles')
# x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='x', bottom=True, top=False)
# yaxis
ax.set_ylim(0,1.05)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# Legend
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1.1), frameon=False, handlelength=1, labelspacing=0.25, handletextpad=0.25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.PLM1.PLM7.PLM6.png', dpi=300)
plt.savefig('./figures/recharge_loc.PLM1.PLM7.PLM6.svg', format='svg')
plt.show()


#
# Hillslope Plot
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 2))
fig.subplots_adjust(right=0.78, left=0.2, top=0.94, bottom=0.3)
ax.plot(dem['X'], dem['Z'], color='black', linewidth=2.0)
# Plot well Locations
wells_list = ['PLM1','PLM7','PLM6']
for i in range(len(wells_list)):
    w = wells_list[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    ax.scatter(xpos, zpos-10, marker='s', color='C{}'.format(i), label=wells_list[i])
    #rect = patches.Rectangle((xpos-12, zpos-16), 20, 16, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10)
    #ax.add_patch(rect)
# Cleanup
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (m)')
ax.margins(x=0.0)
# x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='x', bottom=True, top=False)
# yaxis
ax.set_ylim(dem['Z'].min()-50, dem['Z'].max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
# Legend
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1.1), frameon=False, handlelength=1, labelspacing=0.25, handletextpad=0.25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.hillslope.PLM1.PLM7.PLM7.png', dpi=300)
plt.savefig('./figures/recharge_loc.hillslope.PLM1.PLM7.PLM7.png', format='svg')
plt.show()












#
# Plot of Floodplain, PLM6_soil, PLM1_soil
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 2))
fig.subplots_adjust(right=0.78, left=0.2, top=0.94, bottom=0.3)
# Add in Recharge Location
wells_list = ['X404','X494','X540']
labs = ['PLM1 soil', 'PLM6 soil', 'Floodplain']
for mm in range(len(inf_dict.keys())):
    m = list(inf_dict.keys())[mm]
    inf_ = inf_dict[m]
    for ww in range(len(wells_list)):
        w = wells_list[ww]
        #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color='C{}'.format(ww+3), alpha=0.5, label=labs[ww] if mm==0 else '')
        ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='C{}'.format(ww+3), alpha=0.5, label=labs[ww] if mm==0 else '')
        if m in first_month:
            cc =  ['darkblue','darkorange','darkgreen']
            #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color=cc[ww],  linestyle=(0, (2,2)), alpha=1.0, label=w if mm==0 else '', zorder=10) 
            #ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/mass_june[w], color='black', linestyle=(2, (2, 2)), alpha=0.5, label=w if mm==0 else '', zorder=10) 
            ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='C{}'.format(ww+3), linestyle=(0, (2,2)), alpha=1.0, label=w if mm==0 else '', zorder=10) 
            ax.plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='black', linestyle=(2, (2,2)), alpha=0.75, label=w if mm==0 else '', zorder=10) 
        # testing
        if w == 'PLM6':
            print ('{}  {}/{}  {}'.format(time_list.loc[m,'Date'], inf_[w]['Mass'].sum(), mass_june[w], inf_[w]['Mass'].sum()/mass_june[w]))
# Cleanup
ax.set_xlabel('Distance (m)')
ax.margins(x=0.0)
ax.set_ylabel('Rel. Number\nof Particles')
# x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='x', bottom=True, top=False)
# yaxis
ax.set_ylim(0,1.05)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
# Legend
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1.1), frameon=False, handlelength=1, labelspacing=0.25, handletextpad=0.25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.soil.png', dpi=300)
plt.savefig('./figures/recharge_loc.soil.svg', format='svg')
plt.show()


#
# Hillslope Plot
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 2))
fig.subplots_adjust(right=0.78, left=0.2, top=0.94, bottom=0.3)
ax.plot(dem['X'], dem['Z'], color='black', linewidth=2.0)
# Plot well Locations
wells_list = ['X404','X494','X540']
labs = ['PLM1 soil', 'PLM6 soil', 'Floodplain']
for i in range(len(wells_list)):
    w = wells_list[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    ax.scatter(xpos, zpos-10, marker='s', color='C{}'.format(i+3), label=labs[i])
    #rect = patches.Rectangle((xpos-12, zpos-16), 20, 16, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10, label=labs[i])
    #ax.add_patch(rect)
# Cleanup
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (m)')
ax.margins(x=0.0)
# x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='x', bottom=True, top=False)
# yaxis
ax.set_ylim(dem['Z'].min()-50, dem['Z'].max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
# Legend
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1.1), frameon=False, handlelength=1, labelspacing=0.25, handletextpad=0.25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.hillslope.soil.png', dpi=300)
plt.savefig('./figures/recharge_loc.hillslope.soil.svg', format='svg')
plt.show()





#
# Hillslope Plot all
#
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 2))
fig.subplots_adjust(right=0.78, left=0.2, top=0.94, bottom=0.3)
ax.plot(dem['X'], dem['Z'], color='black', linewidth=2.0)
# Plot well Locations
wells_list = ['PLM1','PLM7','PLM6','X404','X494','X540']
labs = ['PLM1','PLM7','PLM6','PLM1 soil', 'PLM6 soil', 'Floodplain']
for i in range(len(wells_list)):
    w = wells_list[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    if w in ['PLM1','PLM7','PLM6']:
        ax.scatter(xpos, zpos-20, marker='s', color='C{}'.format(i), label=labs[i])
    else:
        ax.scatter(xpos, zpos-5, marker='s', color='C{}'.format(i), label=labs[i])
    #rect = patches.Rectangle((xpos-12, zpos-16), 20, 16, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10, label=labs[i])
    #ax.add_patch(rect)
# Cleanup
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Elevation (m)')
ax.margins(x=0.0)
# x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='x', bottom=True, top=False)
# yaxis
ax.set_ylim(dem['Z'].min()-50, dem['Z'].max()+10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
# Legend
leg = ax.legend(loc='upper left', bbox_to_anchor=(0.96, 1.15), frameon=False, handlelength=1, labelspacing=0.25, handletextpad=0.25)
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_linewidth(3.0)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1.5%", pad=0.05) 
cax.remove()
plt.savefig('./figures/recharge_loc.hillslope.comp.png', dpi=300)
plt.savefig('./figures/recharge_loc.hillslope.comp.svg', format='svg')
plt.show()














#
# Plot through time with velocity angle
#
fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [1,2]}, figsize=(7,3.5))
fig.subplots_adjust(right=0.82, top=0.94, hspace=0.005, left=0.15)
# Plot Velocity Angle
v = ax[1].imshow(np.flip(vel_ang,axis=0), extent=(xx.min(),xx.max(),zz.min(),zz.max()), cmap='coolwarm')
# Plot well Locations
for i in range(len(wells)):
    w = wells.index[i]
    xpos = wells.loc[w, 'X']
    zpos = wells.loc[w, 'land_surf_dem_m'] - wells.loc[w, 'smp_depth_m']
    rect = patches.Rectangle((xpos-10, zpos-16), 16, 16, linewidth=1, edgecolor='black', facecolor='C{}'.format(i), zorder=10)
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
for mm in range(len(inf_dict.keys())):
    m = list(inf_dict.keys())[mm]
    inf_ = inf_dict[m]
    for ww in range(len(wells_list)):
        w = wells_list[ww]
        ax[0].plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='C{}'.format(ww), alpha=0.5, label=w if mm==0 else '')
        if m in first_month:
            cc =  ['darkblue','darkorange','darkgreen']
            ax[0].plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color=cc[ww], linestyle=(0, (2, 2)), alpha=1.0, label=w if mm==0 else '', zorder=10) 
            ax[0].plot(inf_[w]['Xmu'], inf_[w]['Mass']/inf_[w]['Mass'].max(), color='black', linestyle=(2, (2, 2)), alpha=0.5, label=w if mm==0 else '', zorder=10) 
# Cleanup
ax[0].margins(x=0.0)
ax[0].set_ylabel('Rel. Number\nof Particles')
# x-axis
[ax[i].xaxis.set_major_locator(ticker.MultipleLocator(200)) for i in [0,1]]
[ax[i].xaxis.set_minor_locator(ticker.MultipleLocator(50)) for i in [0,1]]
[ax[i].tick_params(axis='x', bottom=True, top=False) for i in [0,1]]
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
#plt.savefig('./figures/recharge_loc_ens.png', dpi=300)
#plt.savefig('./figures/recharge_loc_ens.svg', format='svg')
plt.show()




