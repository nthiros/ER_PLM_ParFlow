# 11/27/2021
# Using Ecoslim that outputs particle input coordinates

import pandas as pd
import numpy as np
import pickle
import os


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
plt.rcParams['font.size']=14

import matplotlib.dates as mdates



#------------------------------
#
# Field Observations
#
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
plm1_obs['bls'] = wells.copy().loc['PLM1','land_surf_dem_m'] - plm1_obs.copy().loc[:,'Elevation']
plm6_obs['bls'] = wells.copy().loc['PLM6','land_surf_dem_m'] - plm6_obs.copy().loc[:,'Elevation']



#----- 
# Layer Depths (m to the bottom of layer)
l_depths = {}
l_depths['soil'] = 5.0
l_depths['sap']  = 9.0

# Note -- should automate the above to pull from permeability script directly



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






#----------------------------------
#
# ParFlow Simulated Water Levels
#
#----------------------------------
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


#
# set ticks to beginning of water year (october 01)
def set_wy(df):
    dates     = df.copy().index
    yrs       = dates.year
    yrs_      = np.unique(yrs)[1:]
    wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs_]
    wy_inds   = np.array([wy_inds_[i]*yrs_[i] for i in range(len(yrs_))]).sum(axis=0)
    first_yrs = [(wy_inds==i).argmax() for i in yrs_]
    return list(yrs_), list(first_yrs)

yrs_spinup, first_yrs_spinup = set_wy(pf_spinup)
yrs_0021, first_yrs_0021 = set_wy(pd.concat((pf_00_16, pf_17_21)))
yrs_1721, first_yrs_1721 = set_wy(pf_17_21)




#--------------------------------------------
#
# Water Level Plots
#
#--------------------------------------------
# Create a new directory for figures
if os.path.exists('figures') and os.path.isdir('figures'):
    pass
else:
    os.makedirs('figures')



w = ['PLM1','PLM7','PLM6']



#--------------------------------------------
#
# Spinup
#
#--------------------------------------------
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,4.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.15, right=0.97, left=0.2)
w = ['PLM1','PLM7','PLM6']
ts = pf_spinup.copy()
# Plot water levels
for i in range(len(w)):
    ax[i].plot(np.arange(len(ts)), ts[w[i]], label=w[i])
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        xx = np.arange(len(ts))[np.isin(ts.index, plm1_obs.index)]
        ax[i].plot(xx, plm1_obs['bls'][np.isin(plm1_obs.index, ts.index)], color='C3', alpha=0.5)
    elif w[i] == 'PLM6':
        xx = np.arange(len(ts))[np.isin(ts.index, plm6_obs.index)]
        ax[i].plot(xx, plm6_obs['bls'][np.isin(plm6_obs.index, ts.index)], color='C3', alpha=0.5)
    # Plot vertical lines on May 1st
    df_may1 = ['{}-05-01'.format(j) for j in ts.index.year.unique()[1:]]
    [ax[i].axvline(np.where(ts.index==pd.to_datetime(z))[0][0], color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]
    # X-ticks
    ax[i].set_xticks(first_yrs_spinup)
    ax[i].set_xticklabels(labels=yrs_spinup)
    # Add in layers
    #ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    #ax[i].yaxis.set_major_locator(MultipleLocator(2))
    #ax[i].yaxis.set_minor_locator(MultipleLocator(1))
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.01)
    else:
        ax[i].tick_params(axis='x', labelbottom=False)
    if i == 1:
        ax[i].set_ylabel('{}\n{}'.format('Depth (m)', w[i]))
    else:
        ax[i].set_ylabel(w[i])
    ax[i].minorticks_on()
    ax[i].tick_params(axis='x', which='minor', bottom=False)
    ax[i].margins(x=0.01)
#ax[2].set_xlabel('Date')
plt.savefig('./figures/waterlevels_spinup.jpg', dpi=300)
plt.show()




#--------------------------------------------
#
# 2000-2021
#
#--------------------------------------------
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,4.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.15, right=0.97, left=0.2)
w = ['PLM1','PLM7','PLM6']
ts = pd.concat((pf_00_16,pf_17_21))
# Plot water levels
for i in range(len(w)):
    ax[i].plot(np.arange(len(ts)), ts[w[i]], label=w[i])
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        xx = np.arange(len(ts))[np.isin(ts.index, plm1_obs.index)]
        ax[i].plot(xx, plm1_obs['bls'][np.isin(plm1_obs.index, ts.index)], color='C3', alpha=0.5)
    elif w[i] == 'PLM6':
        xx = np.arange(len(ts))[np.isin(ts.index, plm6_obs.index)]
        ax[i].plot(xx, plm6_obs['bls'][np.isin(plm6_obs.index, ts.index)], color='C3', alpha=0.5)
    # Plot vertical lines on May 1st
    df_may1 = ['{}-05-01'.format(j) for j in ts.index.year.unique()[1:]]
    [ax[i].axvline(np.where(ts.index==pd.to_datetime(z))[0][0], color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]
    # X-ticks
    ax[i].set_xticks(first_yrs_0021[::3])
    ax[i].set_xticklabels(labels=yrs_0021[::3])
    # Add in layers
    #ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    #ax[i].yaxis.set_major_locator(MultipleLocator(2))
    #ax[i].yaxis.set_minor_locator(MultipleLocator(1))
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.01)
    else:
        ax[i].tick_params(axis='x', labelbottom=False)
    if i == 1:
        ax[i].set_ylabel('{}\n{}'.format('Depth (m)', w[i]))
    else:
        ax[i].set_ylabel(w[i])
    ax[i].minorticks_on()
    ax[i].tick_params(axis='x', which='minor', bottom=False)
    ax[i].margins(x=0.01)
#ax[2].set_xlabel('Date')
plt.savefig('./figures/waterlevels_00_21.jpg', dpi=300)
plt.show()





#--------------------------------------------
#
# 2017-2021
#
#--------------------------------------------
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4.5,3.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.25, right=0.97, left=0.25)
w = ['PLM1','PLM7','PLM6']
ts = pf_17_21.copy()
# Plot water levels
for i in range(len(w)):
    ax[i].plot(ts[w[i]], lw=1.5, label=w[i])
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        xx = np.arange(len(ts))[np.isin(ts.index, plm1_obs.index)]
        ax[i].plot(plm1_obs['bls'], color='black', ls='--', alpha=0.75)
    elif w[i] == 'PLM6':
        xx = np.arange(len(ts))[np.isin(ts.index, plm6_obs.index)]
        ax[i].plot(plm6_obs['bls'], color='black', ls='--', alpha=0.75)
    # Plot vertical lines on May 1st
    #df_may1 = ['{}-05-01'.format(j) for j in ts.index.year.unique()[1:]]
    #[ax[i].axvline(pd.to_datetime(z), color='grey', linestyle='--', linewidth=0.75, alpha=0.5) for z in df_may1]
    # X-ticks
    #ax[i].set_xticks(first_yrs_1721)
    #ax[i].set_xticklabels(labels=yrs_1721)
    loc = mdates.MonthLocator(bymonth=[10])
    loc_min = mdates.MonthLocator(interval=1)
    ax[i].xaxis.set_major_locator(loc)
    ax[i].xaxis.set_minor_locator(loc_min)
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    # Add in layers
    #ax[i].axhline(0.0, color='grey', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['soil'], color='olivedrab', linestyle='--', alpha=0.75)
    #ax[i].axhline(l_depths['sap'], color='saddlebrown', linestyle='--', alpha=0.75)
    # Clean up
    #ax[i].yaxis.set_major_locator(MultipleLocator(2))
    #ax[i].yaxis.set_minor_locator(MultipleLocator(1))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.005, length=4, width=1.1)
    else:
        ax[i].tick_params(axis='x', labelbottom=False, length=4, width=1.1)
    if i == 1:
        ax[i].set_ylabel('{}\n\n{}'.format('Depth (m)', w[i]))
    else:
        ax[i].set_ylabel(w[i])
    #ax[i].minorticks_on()
    #ax[i].tick_params(axis='x', which='minor', bottom=False)
    ax[i].margins(x=0.01)
    ax[i].grid()
#ax[2].set_xlabel('Date')
plt.savefig('./figures/waterlevels_17_21.jpg', dpi=300)
plt.show()




#--------------------
#
# Precipitation Plots
#
#--------------------
#
# Only modern times
#
prcp_summ_ = pd.DataFrame(prcp_summ.iloc[np.where(prcp_summ.index>pd.to_datetime('2016-09-30'))[0]])

dates     = prcp_summ_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

prcp_summ_['wy'] = wy_inds.T
prcp_summ_cs = prcp_summ_.groupby(by=['wy']).cumsum()

# use daily data for cumulative plots
prcp_sumd_ = (met_comp['prcp']*3600*3).groupby(pd.Grouper(freq='D')).sum()
prcp_sumd_ = pd.DataFrame(prcp_sumd_.iloc[np.where(prcp_sumd_.index>pd.to_datetime('2016-09-30'))[0]])

dates     = prcp_sumd_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

prcp_sumd_['wy'] = wy_inds.T
prcp_sumd_cs = prcp_sumd_.groupby(by=['wy']).cumsum()

###
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.5,3))
fig.subplots_adjust(top=0.98, bottom=0.12, left=0.20, right=0.94, hspace=0.2)
ax = axes[1]
notjan = prcp_summ_[prcp_summ_.index.month != 1]
#ax.bar(prcp_summ_.index, prcp_summ_['prcp'], width=20.0, color='black', alpha=0.9) 
ax.bar(notjan.index, notjan['prcp'], width=20.0, color='black', alpha=0.9) 
jan = prcp_summ_[prcp_summ_.index.month == 1]
ax.bar(jan.index, jan['prcp'], width=20.0, color='grey', alpha=1.0) 
ax.set_ylabel('Precipitation\n(mm/month)')
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(25))

ax = axes[0]
ax.plot(prcp_sumd_cs.index, prcp_sumd_cs, color='black', lw=2.0)
ax.set_ylabel('Cumulative\nPrecipitation\n(mm/year)')
ax.tick_params(axis='x', labelbottom=False)
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(100))

for i in [0,1]:
    ax = axes[i]
    #ax.minorticks_on()
    loc = mdates.MonthLocator(bymonth=[10])
    loc_min = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_minor_locator(loc_min)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax.margins(x=0.01)
    ax.tick_params(axis='both', length=4, width=1.1)
    ax.grid()
#axes[1].spines['top'].set_visible(False)
plt.savefig('./figures/precpipitation_cumsum.jpg', dpi=300)
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
    #nbins = 10
    gb =  rtd_df.groupby(pd.cut(rtd_df['Time'], nbins))
    rtd_dfs = gb.agg(dict(Time='mean',Mass='sum',Source='mean',Xin='mean',wt='sum'))
    rtd_dfs['count'] = gb.count()['Time']

    return rtd_df, rtd_dfs



rtd_dict = pd.read_pickle('./parflow_out/ecoslim_rtd.pk')
date_map = pd.DataFrame(pf_17_21.index, columns=['Date'])
#print (date_map)


#------------
# Pick a well
well = 'PLM6'
samp_date = '2021-05-11'
model_time_samp = date_map[date_map['Date'] == samp_date].index[0]
model_time_samp = list(rtd_dict.keys())[abs(list(rtd_dict.keys()) - model_time_samp).argmin()]

"""
rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time_samp, well, 15)
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
    rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time_samp, w, 30)
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

"""






#----------------------------------
#
# Transient Age Distributions
#
#----------------------------------

# pick a water years
#yr = [2017, 2018, 2019, 2020, 2021]
yr = [2019, 2020]
wy_inds_  = [np.where((date_map['Date'] > '{}-09-30'.format(i-1)) & (date_map['Date'] < '{}-10-01'.format(i)), True, False) for i in yr]
wy_inds   = [date_map.index[wy_inds_[i]] for i in range(len(yr))]
wy_inds   = np.concatenate(((wy_inds)))

#time_list = list(wy_inds[np.isin(wy_inds, list(rtd_dict.keys()))]) + [model_time_samp]
time_list = list(wy_inds[np.isin(wy_inds, list(rtd_dict.keys()))]) #+ [model_time_samp]

wells = ['PLM1','PLM7','PLM6']

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(2.8,4.2))
fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.98, bottom=0.15, right=0.87, left=0.32)
for i in range(3):
    # Get the data
    w    = wells[i]
    ax1  = ax[i]
    tau_ = []
    for t in range(len(time_list)):
        model_time = time_list[t]
        try:
            rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
            tau = (rtd_df['Time'] * rtd_df['wt']).sum()
            tau_med = np.median(rtd_df['Time'])
            tau_.append(tau)
            # not sure here, some NANs where there are zero particles
            rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
        except ValueError:
            pass
        else:
            if model_time == model_time_samp:
                pass
                #ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
                #ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
            else:
                ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75, zorder=4)
                ax1.axvline(tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
    #
    # Replot the middle one in red?
    mid_ind = np.where(tau_==np.sort(np.array(tau_))[len(tau_)//2])[0][0]
    model_time = time_list[mid_ind]
    try:
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
        tau = (rtd_df['Time'] * rtd_df['wt']).sum()
        tau_med = np.median(rtd_df['Time'])
        # not sure here, some NANs where there are zero particles
        rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    except ValueError:
        pass
    else:
        ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
        ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
    # Clean up 
    if i == 1:
        ax1.set_ylabel('CDF\n{}'.format(w))
    else:
        ax1.set_ylabel('{}'.format(w))
    #matcks = ax1.get_xticks()
    #ax1.xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.tick_params(which='major', axis='both', length=4, width=1.25)
ax[2].set_xlabel('Particle Ages (years)')    
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.svg',format='svg')
plt.show()




#
# dummy soil wells
#
wells = ['X404', 'X494', 'X528']
names = ['PLM1 Soil', 'PLM6 Soil', 'Floodplain']

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(2.8,4.2))
fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.98, bottom=0.15, right=0.87, left=0.32)
for i in range(3):
    # Get the data
    w    = wells[i]
    ax1  = ax[i]
    tau_ = []
    for t in range(len(time_list)):
        model_time = time_list[t]
        try:
            rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
            tau = (rtd_df['Time'] * rtd_df['wt']).sum()
            tau_med = np.median(rtd_df['Time'])
            tau_.append(tau)
            # not sure here, some NANs where there are zero particles
            rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
        except ValueError:
            pass
        else:
            if model_time == model_time_samp:
                pass
                #ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
                #ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
            else:
                ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75, zorder=4)
                ax1.axvline(tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
    #
    # Replot the middle one in red?
    mid_ind = np.where(tau_==np.sort(np.array(tau_))[len(tau_)//2])[0][0]
    model_time = time_list[mid_ind]
    try:
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
        tau = (rtd_df['Time'] * rtd_df['wt']).sum()
        tau_med = np.median(rtd_df['Time'])
        # not sure here, some NANs where there are zero particles
        rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    except ValueError:
        pass
    else:
        ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
        ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
    # Clean up 
    if i == 1:
        ax1.set_ylabel('CDF\n{}'.format(names[i]))
    else:
        ax1.set_ylabel('{}'.format(names[i]))
    #matcks = ax1.get_xticks()
    #ax1.xaxis.set_minor_locator(MultipleLocator((matcks[1]-matcks[0])/2))
    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.tick_params(which='major', axis='both', length=4, width=1.25)
ax[2].set_xlabel('Particle Ages (years)')    
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.svg',format='svg')
plt.show()


 

#---------------------------
#
# Fraction of young versus old water in the floodplain wells
#
#---------------------------
#w = 'X528'
w = 'X494'

younger = 20.0 


labs_ = {'X528':'Floodplain',
         'X494':'PLM6 Soil',
         'X404':'PLM1 Soil'}

# pick a water years
yr = [2017, 2018, 2019, 2020, 2021]
wy_inds_  = [np.where((date_map['Date'] > '{}-09-30'.format(i-1)) & (date_map['Date'] < '{}-10-01'.format(i)), True, False) for i in yr]
wy_inds   = [date_map.index[wy_inds_[i]] for i in range(len(yr))]
wy_inds   = np.concatenate(((wy_inds)))
time_list = list(wy_inds[np.isin(wy_inds, list(rtd_dict.keys()))])
#time_list_ = time_list[::5]
time_list_ = time_list[::1]


fig, ax = plt.subplots(figsize=(6.5,2.2))
fig.subplots_adjust(top=0.86, bottom=0.15, left=0.20, right=0.94, hspace=0.2)
ax.set_title(labs_[w], fontsize=14)
for t in range(len(time_list_)):
    model_time = time_list_[t]
    try:
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
        tau = (rtd_df['Time'] * rtd_df['wt']).sum()
        tau_med = np.median(rtd_df['Time'])
        # not sure here, some NANs where there are zero particles
        rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    except ValueError:
        pass
    else:
        # Fraction of sample that is less than 10 years
        frac_young = rtd_df[rtd_df['Time']<younger]['wt'].sum()
        #print (date_map.loc[model_time,'Date'])
        ax.scatter(date_map.loc[model_time,'Date'], frac_young, c='black') 
ax.set_ylabel('Fraction Younger\n{} years'.format(int(younger)))
loc = mdates.MonthLocator(bymonth=[10])
#loc = mdates.MonthLocator(interval=2)
ax.set_ylim(0.0,1.0)
loc_min = mdates.MonthLocator(interval=1)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_minor_locator(loc_min)
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
ax.tick_params(axis='both', length=4, width=1.1)
ax.margins(x=0.01)
ax.grid()
plt.savefig('./figures/rtd_younger_than.jpg', dpi=300)
plt.savefig('./figures/rtd_younger_than.svg', format='svg')
plt.show()





#---------------------------
#
# Infiltration Location versus Age
#
#---------------------------
fig, ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(top=0.88, bottom=0.2, left=0.25, right=0.9, hspace=0.2)
ax.set_title(labs_[w], fontsize=14)
for t in range(len(time_list_)):
    model_time = time_list_[t]
    try:
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 60)
    except ValueError:
        pass
    if len(rtd_df) > 1000:
        rtd_df_ = rtd_df.sample(n=1000)
        ax.scatter(rtd_df_['Xin'], rtd_df_['Time'], c='black', marker='.') 
    else:
        ax.scatter(rtd_df['Xin'], rtd_df['Time'], c='black', marker='.')
ax.set_xlabel('Infiltration Location (m)')
ax.set_ylabel('Age (years)')
ax.tick_params(axis='both', length=4, width=1.1)
ax.grid()
ax.minorticks_on()
ax.set_xlim(0,850)
plt.savefig('./figures/rtd_vs_inf.jpg', dpi=300)
plt.savefig('./figures/rtd_vs_inf.svg', format='svg')
plt.show()









#
# FIHM Plots
#

wells = ['X494']
names = ['PLM6 Soil']

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5,2.3))
fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.98, bottom=0.25, right=0.87, left=0.3)
for i in range(len(wells)):
    w    = wells[i]
    ax1  = ax
    tau_ = []
    for t in range(len(time_list)):
        model_time = time_list[t]
        try:
            rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
            tau = (rtd_df['Time'] * rtd_df['wt']).sum()
            tau_med = np.median(rtd_df['Time'])
            tau_.append(tau)
            # not sure here, some NANs where there are zero particles
            rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
        except ValueError:
            pass
        else:
            # Plot the Age CDF 
            if model_time == model_time_samp:
                pass
                #ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.75, zorder=6)
                #ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
            else:
                ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='black', alpha=0.75, zorder=4)
                ax1.axvline(tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
    # Replot the middle one in red?
    mid_ind = np.where(tau_==np.sort(np.array(tau_))[len(tau_)//2])[0][0]
    model_time = time_list[mid_ind]
    try:
        rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 30)
        tau = (rtd_df['Time'] * rtd_df['wt']).sum()
        tau_med = np.median(rtd_df['Time'])
        # not sure here, some NANs where there are zero particles
        rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    except ValueError:
        pass
    else:
        ax1.plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
        ax1.axvline(tau, color='red', alpha=0.65, linestyle='--', zorder=5)
    #
    # Clean up 
    if i == 0:
        ax1.set_ylabel('CDF\n{}'.format(names[i]))
    else:
        ax1.set_ylabel('{}'.format(names[i]))
    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.tick_params(which='major', axis='both', length=4, width=1.25)
ax1.set_xlabel('Particle Ages (years)')    
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.PLM6soil.png',dpi=300)
#plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.svg',format='svg')
plt.show()




"""
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(6,6))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.1, right=0.97, left=0.2)
w = ['PLM1','PLM7','PLM6']
w = ['blk']+w
# Plot water levels
for i in range(1,len(w)):
    ax[i].plot(pf_17_21[w[i]], label=w[i])
    #ax[i].text(0.05, 0.05, str(w[i]), horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)   
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
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.01)
    else:
        ax[i].tick_params(axis='x', labelbottom=False)
    if i == 2:
        ax[i].set_ylabel('{}\n{}'.format('Depth (m)', w[i]))
    else:
        ax[i].set_ylabel(w[i])
ax[3].set_xlabel('Date')
#ax[2].set_ylabel('Depth (m)')
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
# Clean up
axp.set_ylabel('Precipitation\n(mm/month)')
axp.minorticks_on()
axp.tick_params(axis='x', labelbottom=False)
axp.tick_params(axis='x', which='minor', bottom=False)
[ax[i].margins(x=0.01) for i in [0,1,2,3]]
plt.savefig('./figures/waterlevels_17_21.jpg', dpi=320)
plt.show()
"""
