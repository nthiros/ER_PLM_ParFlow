# 11/27/2021
# Using Ecoslim that outputs particle input coordinates

import pandas as pd
import numpy as np
import pickle
import os


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
plt.rcParams['font.size']=14

#------------------------------
# Field Observations
#------------------------------
#----
# Well Info
wells = pd.read_csv('../utils/wells_2_pf_v3b.csv', index_col='well') 


#----
# Transducer Data
wl_field = pd.read_pickle('./Field_Data/plm_wl_obs.pk')

plm1_obs = wl_field['ER_PLM1']
plm6_obs = wl_field['ER_PLM6']

plm1_obs = plm1_obs[~((plm1_obs.index.month == 2) & (plm1_obs.index.day == 29))]
plm6_obs = plm6_obs[~((plm6_obs.index.month == 2) & (plm6_obs.index.day == 29))]

# 10/28/21 find depth below land surface
#plm1_obs.loc[:,'bls'] = wells.loc['PLM1','land_surf_dem_m'] - plm1_obs.loc[:, 'Elevation']
plm1_obs['bls'] = wells.loc['PLM1','land_surf_dem_m'] - plm1_obs.loc[:,'Elevation']
plm6_obs['bls'] = wells.loc['PLM6','land_surf_dem_m'] - plm6_obs.loc[:,'Elevation']



#-----
# Layer Depths (m to the bottom of layer)
l_depths = {}
l_depths['soil'] = 5.0
l_depths['sap']  = 9.0




#-----
# Forcing MET data
# spinup
met_sp    = pd.read_csv('./MET/met.avg.10yr.3hr.txt', delim_whitespace=True, names=['rad_s','rad_l','prcp','temp','wnd_u','wnd_v','press','vap'])
tstart    = pd.to_datetime('1989-10-01 00', format='%Y-%m-%d %H')
tend      = pd.to_datetime('1999-09-30 21', format='%Y-%m-%d %H') # Water year 21 is not over yet
hours     = pd.DatetimeIndex(pd.Series(pd.date_range(tstart, tend, freq='3H')))
hours     = hours[~((hours.month == 2) & (hours.day == 29))] # No leap years
met_sp.index = hours

# 2017-2021
met_17_21 = pd.read_csv('./MET/met.2017-2021.3hr.txt', delim_whitespace=True, names=['rad_s','rad_l','prcp','temp','wnd_u','wnd_v','press','vap'])
tstart    = pd.to_datetime('2016-10-01 00', format='%Y-%m-%d %H')
tend      = pd.to_datetime('2021-08-30 12', format='%Y-%m-%d %H') # Water year 21 is not over yet
hours     = pd.DatetimeIndex(pd.Series(pd.date_range(tstart, tend, freq='3H')))
hours     = hours[~((hours.month == 2) & (hours.day == 29))] # No leap years
met_17_21.index = hours

# 2000-2016
met_00_16 = pd.read_csv('./MET/met.2000-2016.3hr.txt', delim_whitespace=True, names=['rad_s','rad_l','prcp','temp','wnd_u','wnd_v','press','vap'])
tstart    = pd.to_datetime('1999-10-01 00', format='%Y-%m-%d %H')
tend      = pd.to_datetime('2016-09-30 23', format='%Y-%m-%d %H') # Water year 21 is not over yet
hours     = pd.DatetimeIndex(pd.Series(pd.date_range(tstart, tend, freq='3H')))
hours     = hours[~((hours.month == 2) & (hours.day == 29))] # No leap years
met_00_16.index = hours

# 1979-2021
met_comp = pd.concat((met_00_16, met_17_21))


#----
# summarize precipitation
# monthly amount (mm) of precipitation
# spinup
prcp0_summ       = (met_sp['prcp']*3600*3).groupby(pd.Grouper(freq='M')).sum() # mm/month
prcp0_summ.index = prcp0_summ.index.map(lambda x: x.replace(day=1))

# 2000 - 2016
prcp1_summ       = (met_00_16['prcp']*3600*3).groupby(pd.Grouper(freq='M')).sum() # mm/month
prcp1_summ.index = prcp1_summ.index.map(lambda x: x.replace(day=1))

# 2017-2021
prcp2_summ       = (met_17_21['prcp']*3600*3).groupby(pd.Grouper(freq='M')).sum() # mm/month
prcp2_summ.index = prcp2_summ.index.map(lambda x: x.replace(day=1))

# yearly accumulated precip
prcp0_sumy = (met_sp['prcp']*3600*3).groupby(pd.Grouper(freq='Y')).sum() # mm/yr
prcp1_sumy = (met_00_16['prcp']*3600*3).groupby(pd.Grouper(freq='Y')).sum() # mm/yr
prcp2_sumy = (met_17_21['prcp']*3600*3).groupby(pd.Grouper(freq='Y')).sum() # mm/yr

# Use the yearly one and clip; prevents skewed months because water year starts october
prcp_summ       = (met_comp['prcp']*3600*3).groupby(pd.Grouper(freq='M')).sum() # mm/month
prcp_summ.index = prcp_summ.index.map(lambda x: x.replace(day=1))

prcp_sumy       = (met_comp['prcp']*3600*3).groupby(pd.Grouper(freq='Y')).sum() # mm/yr






#------------------------------
# ParFlow Water Levels
#------------------------------
def pf_2_dates(startdate, enddate, f):
    '''Assumes ParFlow outputs every 24 hours'''
    s = pd.to_datetime(startdate)
    e = pd.to_datetime(enddate)
    d_list = pd.date_range(start=s, end=e, freq=f)
    # Drop Leap years again
    d_list_ = d_list[~((d_list.month == 2) & (d_list.day == 29))]
    return d_list_

# Read in files from pull_wl.py
pf_spinup =  pd.read_csv('./parflow_out/wy_spinup_wt_bls.csv')
pf_00_16  =  pd.read_csv('./parflow_out/wy_2000_2016_wt_bls.csv')
pf_17_21  =  pd.read_csv('./parflow_out/wy_2017_2021_wt_bls.csv') 

# Update index to dates
pf_spinup.index =  pf_2_dates('1969-09-30', '1979-09-30', '24H')
pf_00_16.index  =  pf_2_dates('1999-09-30', '2016-09-30', '24H')
pf_17_21.index  =  pf_2_dates('2016-09-30', '2021-08-29', '24H')



#
# Convert water level elevations to depth to water levels
#pf_spinup_  = pf_spinup.copy()
#pf_00_16_   = pf_00_16.copy()
#pf_17_21_   = pf_17_21.copy()
#for i in ['PLM1','PLM7','PLM6']:
#    pf_spinup_.loc[:,i] = wells.loc[i,'land_surf_cx'] - pf_spinup_.loc[:,i]
#    pf_00_16_.loc[:,i]  = wells.loc[i,'land_surf_cx'] - pf_00_16_.loc[:,i]
#    pf_17_21_.loc[:,i]  = wells.loc[i,'land_surf_cx'] - pf_17_21_.loc[:,i]



#--------------------------------------------
# Water Level Plots
#--------------------------------------------
# Create a new directory for figures
if os.path.exists('figures') and os.path.isdir('figures'):
    pass
else:
    os.makedirs('figures')



w = ['PLM1','PLM7','PLM6']



#--------------------------------------------
# spinup
#--------------------------------------------
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8,10))
fig.subplots_adjust(hspace=0.25, top=0.95, right=0.97)
w = ['PLM1','PLM7','PLM6']
w = ['blk']+w
# Plot water levels
for i in range(1,len(w)):
    ax[i].plot(pf_spinup[w[i]], label=w[i])
    ax[i].text(0.05, 0.05, str(w[i]), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
    ax[i].invert_yaxis()
    # Now add field observations
    #if w[i] == 'PLM1':
    #    ax[i].plot(plm1_obs['Elevation'], color='C3', alpha=0.5)
    #elif w[i] == 'PLM6':
    #    ax[i].plot(plm6_obs['Elevation'], color='C3', alpha=0.5)
    # Land Surface Elevation
    #ax[i].axhline(wells.loc[w[i],'land_surf_dem_m'], color='black', alpha=0.5, linestyle='--')
    #x[i].axhline(wells.loc[w[i],'land_surf_cx'], color='black', alpha=0.5, linestyle='--')
    # Plot vertical lines on May 1st
    df_may1 = ['{}-05-01'.format(j) for j in pf_spinup.index.year.unique()[1:]]
    [ax[i].axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]
    # Add in layers
    ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    ax[i].yaxis.set_major_locator(MultipleLocator(2))
    ax[i].yaxis.set_minor_locator(MultipleLocator(1))
ax[3].set_xlabel('Date')
ax[2].set_ylabel('Depth (m)')
# Add in precipitation rates
axp = ax[0]
axp.bar(prcp0_summ.index, prcp0_summ, width=25.0, color='black', alpha=0.7)
jan    = prcp0_summ[prcp0_summ.index.month == 1] # note, this is december 31
notjan = prcp0_summ[prcp0_summ.index.month != 1]
axp.bar(notjan.index, notjan, width=30.0, color='grey', alpha=1.0)
axp.bar(jan.index, jan, width=30.0, color='black', alpha=1.0) # Call out jan precip?
df_may1 = ['{}-05-01'.format(j) for j in prcp0_summ.index.year.unique()[1:]]
[axp.axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]
# Clean up
axp.set_ylabel('Precipitation\n(mm/month)')
#axp.invert_yaxis()
#axp.xaxis.tick_top()
# Clean up
[ax[i].xaxis.set_ticks_position('both') for i in [0,1,2,3]]
[ax[i].tick_params(axis='x', bottom=True, top=True, length=4, width=1.25) for i in [0,1,2,3]]
[ax[i].margins(x=0.01) for i in [0,1,2,3]]
#plt.savefig('./figures/waterlevels_17_21.jpg', dpi=320)
plt.show()




#--------------------------------------------
# 2000-2021
#--------------------------------------------
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8,10))
fig.subplots_adjust(hspace=0.25, top=0.95, right=0.97)
w = ['PLM1','PLM7','PLM6']
w = ['blk']+w
# Plot water levels
for i in range(1, len(w)):
    ax[i].plot(pf_00_16[w[i]], label=w[i])
    ax[i].plot(pf_17_21[w[i]], label=w[i])
    ax[i].text(0.05, 0.05, str(w[i]), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)   
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        ax[i].plot(plm1_obs['bls'], color='C3', alpha=0.5)
    elif w[i] == 'PLM6':
        ax[i].plot(plm6_obs['bls'], color='C3', alpha=0.5)
    # Land Surface Elevation
    #ax[i].axhline(wells.loc[w[i],'land_surf_dem_m'], color='black', alpha=0.5, linestyle='--')
    #x[i].axhline(wells.loc[w[i],'land_surf_cx'], color='black', alpha=0.5, linestyle='--')
    # Plot vertical lines on May 1st
    pf_may1 = ['{}-05-01'.format(j) for j in pf_00_16.index.year.unique()[1:]]
    [ax[i].axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in pf_may1]
    df_may1 = ['{}-05-01'.format(j) for j in pf_17_21.index.year.unique()[1:]]
    [ax[i].axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]    
    # Add in layers
    ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    ax[i].yaxis.set_major_locator(MultipleLocator(2))
    ax[i].yaxis.set_minor_locator(MultipleLocator(1))
ax[3].set_xlabel('Date')
ax[2].set_ylabel('Depth (m)')
# Add in precipitation rates
axp = ax[0]
axp.bar(prcp_summ.index, prcp_summ, width=25.0, color='black', alpha=0.7)
axp.plot(prcp_sumy/10, color='black', alpha=1.0)
jan    = prcp_summ[prcp_summ.index.month == 1] # note, this is december 31
notjan = prcp_summ[prcp_summ.index.month != 1]
axp.bar(notjan.index, notjan, width=30.0, color='grey', alpha=1.0)
axp.bar(jan.index, jan, width=30.0, color='black', alpha=1.0) # Call out jan precip?
df_may1 = ['{}-05-01'.format(j) for j in prcp_summ.index.year.unique()[1:]]
[axp.axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1] 
# Clean up
axp.set_ylabel('Precipitation\n(mm/month)')
[ax[i].tick_params(axis='x', bottom=True, top=True, length=4, width=1.25) for i in [0,1,2,3]]
[ax[i].margins(x=0.01) for i in [0,1,2,3]]
#plt.savefig('./figures/waterlevels_79_21.mo.jpg', dpi=320)
plt.savefig('./figures/waterlevels_00_21.jpg', dpi=320)
plt.show()




#--------------------------------------------
# 2017-2021
#--------------------------------------------
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8,10))
fig.subplots_adjust(hspace=0.25, top=0.95, right=0.97)
w = ['PLM1','PLM7','PLM6']
w = ['blk']+w
# Plot water levels
for i in range(1,len(w)):
    ax[i].plot(pf_17_21[w[i]], label=w[i])
    ax[i].text(0.05, 0.05, str(w[i]), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)   
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        ax[i].plot(plm1_obs['bls'], color='C3', alpha=0.5)
    elif w[i] == 'PLM6':
        ax[i].plot(plm6_obs['bls'], color='C3', alpha=0.5)
    # Land Surface Elevation
    #ax[i].axhline(wells.loc[w[i],'land_surf_dem_m'], color='black', alpha=0.5, linestyle='--')
    #x[i].axhline(wells.loc[w[i],'land_surf_cx'], color='black', alpha=0.5, linestyle='--')
    # Plot vertical lines on May 1st
    df_may1 = ['{}-05-01'.format(j) for j in pf_17_21.index.year.unique()[1:]]
    [ax[i].axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]    
    # Add in layers
    ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    ax[i].yaxis.set_major_locator(MultipleLocator(2))
    ax[i].yaxis.set_minor_locator(MultipleLocator(1))
ax[3].set_xlabel('Date')
ax[2].set_ylabel('Depth (m)')
# Add in precipitation rates
axp = ax[0]
axp.bar(prcp2_summ.index, prcp2_summ, width=25.0, color='black', alpha=0.7)
jan    = prcp2_summ[prcp2_summ.index.month == 1] # note, this is december 31
notjan = prcp2_summ[prcp2_summ.index.month != 1]
axp.bar(notjan.index, notjan, width=30.0, color='grey', alpha=1.0)
axp.bar(jan.index, jan, width=30.0, color='black', alpha=1.0) # Call out jan precip?
df_may1 = ['{}-05-01'.format(j) for j in prcp2_summ.index.year.unique()[1:]]
[axp.axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1] 
# Clean up
axp.set_ylabel('Precipitation\n(mm/month)')
#axp.invert_yaxis()
#axp.xaxis.tick_top()
# Clean up
[ax[i].xaxis.set_ticks_position('both') for i in [0,1,2,3]]
[ax[i].tick_params(axis='x', bottom=True, top=True, length=4, width=1.25) for i in [0,1,2,3]]
[ax[i].margins(x=0.01) for i in [0,1,2,3]]
plt.savefig('./figures/waterlevels_17_21.jpg', dpi=320)
plt.show()






"""
#--------------------
# Precipitation Plots
#--------------------
fig, ax = plt.subplots()
ax.bar(prcp_summ.index, prcp_summ, width=30.0, color='black', alpha=0.7)
jan    = prcp_summ[prcp_summ.index.month == 1] # note, this is december 31
notjan = prcp_summ[prcp_summ.index.month != 1]
ax.bar(notjan.index, notjan, width=30.0, color='grey', alpha=1.0)
ax.bar(jan.index, jan, width=30.0, color='black', alpha=1.0) # Call out jan precip?
# Add in spinup?
ax.bar(prcp0_summ.index, prcp0_summ, width=30.0, color='black', alpha=0.7)
jan    = prcp0_summ[prcp0_summ.index.month == 1] # note, this is december 31
notjan = prcp0_summ[prcp0_summ.index.month != 1]
ax.bar(notjan.index, notjan, width=30.0, color='grey', alpha=1.0)
ax.bar(jan.index, jan, width=30.0, color='black', alpha=1.0) # Call out jan precip?
plt.show()
"""




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



rtd_dict = pd.read_pickle('./ecoslim_rtd.pk')
date_map = pd.DataFrame(pf_17_21.index, columns=['Date'])
#print (date_map)


#------------
# Pick a well
well = 'PLM6'
samp_date = '2021-05-11'
model_time = date_map[date_map['Date'] == samp_date].index[0]
model_time = list(rtd_dict.keys())[abs(list(rtd_dict.keys()) - model_time).argmin()]
model_time_samp = model_time
print (model_time)

rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, well, 15)
tau = (rtd_df['Time'] * rtd_df['wt']).sum()
tau_med = np.median(rtd_df['Time'])

# not sure here, some NANs where there are zero particles
rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')


#--------------------------
# Plot the RTD
#--------------------------
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#fig.suptitle('{} {}'.format(well, samp_date))
##ax[0].plot(rtd_df['Time'],  rtd_df['wt'], marker='.')
#ax[0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='red')
#ax[0].axvline(tau, color='black', linestyle='--')
##ax[0].axvline(tau_med, color='black', linestyle=':')
#ax[0].set_ylabel('PDF (NP={})'.format(len(rtd_df)))
#ax[0].set_xlabel('Residence Time (years)')
#
#ax[1].plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
#ax[1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
#ax[1].axvline(tau, color='black', linestyle='--')
##ax[1].axvline(tau_med, color='black', linestyle=':')
#ax[1].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
#ax[1].set_xlabel('Residence Time (years)')
#fig.tight_layout()
#plt.savefig('./figures/ecoslim_rtd.jpg',dpi=320)
#plt.show()


#-----
# Combined RTD Plots
wells = ['PLM1','PLM7','PLM6']

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6,6))
fig.subplots_adjust(wspace=0.40, hspace=0.45, top=0.94, right=0.97, left=0.15)
#fig.suptitle('Sample Date: {}'.format(samp_date))
for i in range(3):
    # Get the data
    w = wells[i]
    rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
    tau = (rtd_df['Time'] * rtd_df['wt']).sum()
    tau_med = np.median(rtd_df['Time'])
    # not sure here, some NANs where there are zero particles
    rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    #fig.suptitle('{} {}'.format(well, samp_date))
    #ax[0].plot(rtd_df['Time'],  rtd_df['wt'], marker='.')
    ax[i,0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='black')
    ax[i,0].axvline(tau, color='black', linestyle='--')
    #ax[0].axvline(tau_med, color='black', linestyle=':')
    ax[i,0].set_ylabel('PDF (NP={})'.format(len(rtd_df)))
    #ax[i,0].text(0.9, 0.5, w, horizontalalignment='center',
    #          verticalalignment='center', transform=ax[i,0].transAxes)
    #ax[i,0].set_xlabel('Residence Time (years)')
    ax[i,0].set_title(w, fontsize=14)
    
    ax[i,1].plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
    #ax[i,1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
    ax[i,1].axvline(tau, color='black', linestyle='--')
    #ax[1].axvline(tau_med, color='black', linestyle=':')
    ax[i,1].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
    #ax[i,1].text(0.9, 0.5, w, horizontalalignment='center',
    #          verticalalignment='center', transform=ax[i,1].transAxes)
    ax[i,1].set_title(w, fontsize=14)
    # Clean up a bit
    matcks = ax[i,0].get_xticks()
    ax[i,0].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    matcks = ax[i,1].get_xticks()
    ax[i,1].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    # Tick sizes
    ax[i,0].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    ax[i,1].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    ax[i,0].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)
    ax[i,1].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)
    
ax[2,0].set_xlabel('Particle Ages (years)')    
ax[2,1].set_xlabel('Particle Ages (years)')

plt.savefig('./figures/ecoslim_rtd_comp.jpg',dpi=320)
plt.savefig('./figures/ecoslim_rtd_comp.svg',format='svg')
plt.show()






#----------------------------------
# Add in some observed ages
#----------------------------------
exp_age = pd.read_pickle('../utils/exp_age.pk')
he_age  = pd.read_pickle('../utils/he4_exp_age.pk')
he_age['PLM6'] = he_age.pop('PLM6_2021')

wells = ['PLM1','PLM7','PLM6']


# take observed age as marginal distributin of He4
he_age_ = {}
for w in wells:
    dd = [he_age[w][n].astype(float) for n in list(he_age[w].keys())]
    dd = np.concatenate((dd)).astype(float)
    he_age_[w] = dd



# Plots
fig, ax = plt.subplots(nrows=3, ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]},figsize=(3,6))
fig.subplots_adjust(wspace=0.05, hspace=0.45, top=0.94, right=0.92, left=0.25)
for i in range(3):
    # Get the data
    w = wells[i]
    rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
    tau = (rtd_df['Time'] * rtd_df['wt']).sum()
    tau_med = np.median(rtd_df['Time'])
    # not sure here, some NANs where there are zero particles
    rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    
    ax1 = ax[i,0]
    ax2 = ax[i,1]
    
    # Plot the Age CDF 
    ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
    #ax[i,1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
    ax[i,0].axvline(tau, color='black', linestyle='--')
    ax[i,0].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
    ##ax[i,1].text(0.9, 0.5, w, horizontalalignment='center',
    ##          verticalalignment='center', transform=ax[i,1].transAxes)
    ax[i,0].set_title(w, x=0.6, fontsize=14)
  
    # Add in observations
    obs = he_age_[w].mean()
    if obs - rtd_df['Time'].max() > 10: 
        
        # Again
        ax2.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
        ax2.axvline(int(obs), color='C{}'.format(i))
        
        # Zoom in for one with observations
        #ax1.set_xlim(0,60)
        ax2.set_xlim(int(obs)-2, int(obs)+2)
        ax2.set_xticks(np.array([obs]).astype(int))
    
        # hide the spines between ax1 and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.tick_params(axis='y', left=True, right=False)
        ax2.tick_params(axis='y', left=False, right=True)
           
    else:
        ax1.axvline(int(obs), color='C{}'.format(i))
    
        # Zoom in for one with observations
        #ax1.set_xlim(0,60)
        ax2.set_xlim(max(ax1.get_xlim()), max(ax1.get_xlim())+1)
        ax2.tick_params(axis='x', bottom=False, labelbottom=False)
     
        # hide the spines between ax1 and ax2
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.tick_params(axis='y', left=True, right=False)
        ax2.tick_params(axis='y', left=False, right=True)
        
    # add lines to split it up    
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)   
    
    # Clean up 
    matcks = ax[i,0].get_xticks()
    ax[i,0].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    #matcks = ax[i,1].get_xticks()
    #ax[i,1].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    ## Tick sizes
    ax[i,0].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    #ax[i,1].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    ax[i,0].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)
    #ax[i,1].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)
    
ax[2,0].set_xlabel('Particle Ages (years)', x=0.68)    
#ax[2,1].set_xlabel('Particle Ages (years)')

plt.savefig('./figures/ecoslim_rtd_comp_obs.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs.svg',format='svg')
plt.show()







#----------------------------------
# Transient Age Distributions
#----------------------------------
fig, ax = plt.subplots(nrows=3, ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 1]},figsize=(3,6))
fig.subplots_adjust(wspace=0.05, hspace=0.45, top=0.94, right=0.92, left=0.25)
for i in range(3):
    # Get the data
    w = wells[i]

    time_list = list(rtd_dict.keys())[::5] + [model_time]
    
    for t in range(len(time_list)):
        model_time = time_list[t]
    
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
        tau = (rtd_df['Time'] * rtd_df['wt']).sum()
        tau_med = np.median(rtd_df['Time'])
        # not sure here, some NANs where there are zero particles
        rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
           
        ax1 = ax[i,0]
        ax2 = ax[i,1]
        
        # Plot the Age CDF 
        #ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75, zorder=10)
        if model_time == model_time_samp:
            ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', alpha=0.75, zorder=6)
            ax1.axvline(tau, color='red', alpha=0.5, linestyle='--', zorder=5)
        else:
            ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75, zorder=4)
            ax1.axvline(tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
        ax1.set_ylabel('CDF (NP={})'.format(len(rtd_df)))
        ax1.set_title(w, x=0.6, fontsize=14)
            
        # Add in observations
        obs = he_age_[w].mean()
        if obs - rtd_df['Time'].max() > 10: 
            # Again
            if model_time == model_time_samp:
                ax2.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', alpha=0.75)
            else:
                ax2.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75)
            
            ax2.axvline(int(obs), color='C{}'.format(i), zorder=10)
            
            # Zoom in for one with observations
            ax2.set_xlim(int(obs)-2, int(obs)+2)
            ax2.set_xticks(np.array([obs]).astype(int))
        
            # hide the spines between ax1 and ax2
            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.tick_params(axis='y', left=True, right=False)
            ax2.tick_params(axis='y', left=False, right=True)
        
            
        else:
            ax1.axvline(int(obs), color='C{}'.format(i), zorder=10)
        
            # Zoom in for one with observations
            #ax1.set_xlim(0,60)
            ax2.set_xlim(max(ax1.get_xlim()), max(ax1.get_xlim())+1)
            ax2.tick_params(axis='x', bottom=False, labelbottom=False)
         
            # hide the spines between ax1 and ax2
            ax1.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax1.tick_params(axis='y', left=True, right=False)
            ax2.tick_params(axis='y', left=False, right=True)
            
    # add lines to split it up    
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d,1+d), (-d,+d), **kwargs)
    ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)   
    
    # Clean up 
    matcks = ax[i,0].get_xticks()
    ax[i,0].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    #matcks = ax[i,1].get_xticks()
    #ax[i,1].xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    ## Tick sizes
    ax[i,0].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    #ax[i,1].tick_params(which='major', axis='x', bottom=True, top=False, length=4, width=1.25)
    ax[i,0].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)
    #ax[i,1].tick_params(which='minor', axis='x', bottom=True, top=False, length=3, width=1.0)

ax[2,0].set_xlabel('Particle Ages (years)', x=0.68)    
#ax[2,1].set_xlabel('Particle Ages (years)')

plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.svg',format='svg')
plt.show()









"""
#------------------------
# Recharge Locations
dem = pd.read_csv('Perm_Modeling/elevation.sa', skiprows=1, header=None, names=['Z'])
dem['X'] = dem.index * 1.5125

rtd_df_r = rtd_df.copy().reset_index(drop=True)
for i in range(len(rtd_df_r)):
    ind = abs(dem['X'] - rtd_df_r.loc[i,'Xin']).idxmin()
    rtd_df_r.loc[i, 'Zin'] = dem.loc[ind,'Z']

# Plot it
fig, ax = plt.subplots()
ax.scatter(rtd_df_r['Xin'], rtd_df_r['Zin'])
plt.show()
"""



