# Plots ParFlow and observed Water Levels and EcoSLIM RTDS
#
# 11/27/2021
#   Using Ecoslim that outputs particle input coordinates
# 04/22/2022
#   Added soil wells for ecoslim
# 08/31/2022
#   Adding fracture/matrix diffusion RTD to ecoslim
# 10/26/2022
#   Plots EcoSLIM tracer concentrations with matrix diffusion RTD model
# 11/03/2022
#   Adding plots for mean age in soil, saprolite, and bedrock
#   Adding plots for depth resolved RTDs



import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import seaborn as sns
import pdb


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
plt.rcParams['font.size']=14
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.insert(0, '/Users/nicholasthiros/Documents/SCGSR/Age_Modeling')
import convolution_integral_utils as conv

sys.path.insert(0, '/Users/nicholasthiros/Documents/SCGSR/Age_Modeling/ng_interp')
import noble_gas_utils as ng_utils


# set a random seed for preporducability
np.random.seed(10)




#---------------------------------------------------------------
#
# Water Level Field Observations
#
#---------------------------------------------------------------
# Well Info
wells = pd.read_csv('../ER_PLM_ParFlow/utils/wells_2_pf_v3b.csv', index_col='well') 


#----
# Transducer Data
wl_field = pd.read_pickle('../ER_PLM_ParFLow/Field_Data/plm_wl_obs.pk')

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



#-------------------------------------------------------------------
#
# Tracer Observation Data Imports
#
#-------------------------------------------------------------------
#
# CFC -- this is molar concentration
cfc_obs = pd.read_excel('../ER_PLM_ParFLow/Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','CFC11','CFC12','CFC113'])
cfc_obs += 1.e-10 # avoids zeros for some numerical lubricant

# SF6 -- this is modlar concentration
sf6_obs = pd.read_excel('../ER_PLM_ParFLow/Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','SF6'])
sf6_obs += 1.e-10

# Tritium
h3_obs = pd.read_excel('../ER_PLM_ParFLow/Field_Data/PLM_tracers_2021.xlsx', index_col='Sample', usecols=['Sample','H3'])
h3_obs += 1.e-10

# Helium (both 4He and 3He)
he_obs_ = pd.read_excel('../ER_PLM_ParFLow/Field_Data/PLM_noblegas_2021.xlsx', skiprows=1, index_col='SiteID', nrows=9)
he_obs  = he_obs_.copy()[['4He','3He']]
he_obs.rename(columns={'4He':'He4','3He':'He3'}, inplace=True)
he_obs.dropna(inplace=True)

obs_dict = {'cfc12':cfc_obs['CFC12'],
            'sf6':sf6_obs['SF6'],
            'h3':h3_obs['H3'],
            'he4':he_obs['He4']}

#
# Use the recharge temp and excess air corrected mixing ratio ensembles
# These are output from Age_Modeling directory from age_modeling_mcmc.prep.py
# CFC and SF6 in pptv, 3H in TU, and 4He_ter in cm3/g
#
map_dict   = pd.read_pickle('../ER_PLM_ParFLow/Field_Data/map_dict.pk')
ens_dict   = pd.read_pickle('../ER_PLM_ParFLow/Field_Data/ens_dict.pk')





#-----------------------------------
#
# Forcing MET data
#
#-----------------------------------
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



#---------------------------------------------------------------
#
# ParFlow Simulated Water Levels
#
#---------------------------------------------------------------
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
pf_00_21  =  pd.concat((pf_00_16, pf_17_21))


# Update index to dates
pf_spinup.index =  pf_2_dates('1969-09-30', '1979-09-30', '24H')
pf_00_16.index  =  pf_2_dates('1999-09-30', '2016-09-30', '24H')
pf_17_21.index  =  pf_2_dates('2016-09-30', '2021-08-29', '24H')
pf_00_21.index  =  pf_2_dates('1999-09-30', '2021-08-30', '24H')




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
yrs_0021, first_yrs_0021 = set_wy(pf_00_21)













#---------------------------------------------------------------
#
# Water Level Plots
#
#---------------------------------------------------------------
# Create a new directory for figures
if os.path.exists('figures') and os.path.isdir('figures'):
    pass
else:
    os.makedirs('figures')




#
# Spinup
#
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,3.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.17, right=0.97, left=0.22)
w = ['PLM1','PLM6']
ts = pf_spinup.copy()
# Plot water levels
for i in range(len(w)):
    ax[i].plot(ts[w[i]], label=w[i])
    ax[i].invert_yaxis()
    # Now add field observations
    if w[i] == 'PLM1':
        xx = np.arange(len(ts))[np.isin(ts.index, plm1_obs.index)]
        ax[i].plot(plm1_obs['bls'][np.isin(plm1_obs.index, ts.index)], color='black', ls='--', alpha=0.75)
    elif w[i] == 'PLM6':
        xx = np.arange(len(ts))[np.isin(ts.index, plm6_obs.index)]
        ax[i].plot(plm6_obs['bls'][np.isin(plm6_obs.index, ts.index)], color='black', ls='--', alpha=0.75)
    # X-ticks
    ax[i].xaxis.set_major_locator(mdates.YearLocator(base=1, month=10, day=1))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for label in ax[i].get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.01)
    else:
        ax[i].tick_params(axis='x', labelbottom=False)
    ax[i].set_ylabel(w[i])
    #ax[i].minorticks_on()
    ax[i].margins(x=0.01)
    ax[i].grid()
fig.text(0.03, 0.6, 'Water Table Depth (m)', ha='center', va='center', rotation='vertical')
plt.savefig('./figures/waterlevels_spinup.jpg', dpi=300)
plt.show()



#
# WY 2000-2021
#
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,3.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.17, right=0.97, left=0.2)
w = ['PLM1','PLM6']
ts = pf_00_21.copy()
ts = ts[ts.index>'2000-09-30']
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
    # ticks
    ax[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    #ax[i].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    for label in ax[i].get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.005, length=4, width=1.1)
    else:
        ax[i].tick_params(axis='x', labelbottom=False, length=4, width=1.1)
    ax[i].set_ylabel(w[i])
    ax[i].margins(x=0.01)
    ax[i].grid()
fig.text(0.03, 0.6, 'Water Table Depth (m)', ha='center', va='center', rotation='vertical')
plt.savefig('./figures/waterlevels_00_21.jpg', dpi=300)
plt.show()



#
# WY 2017-2021
#
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,3.5))
fig.subplots_adjust(hspace=0.15, top=0.98, bottom=0.17, right=0.97, left=0.2)
w = ['PLM1','PLM6']
ts = pf_00_21.copy()
ts = ts[ts.index>'2016-09-30']
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
    # ticks
    ax[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[i].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    for label in ax[i].get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    if i == len(w)-1:
        ax[i].tick_params(axis='x', rotation=45, pad=0.005, length=4, width=1.1)
    else:
        ax[i].tick_params(axis='x', labelbottom=False, length=4, width=1.1)
    ax[i].set_ylabel(w[i])
    ax[i].margins(x=0.01)
    ax[i].grid()
fig.text(0.03, 0.6, 'Water Table Depth (m)', ha='center', va='center', rotation='vertical')
plt.savefig('./figures/waterlevels_07_21.jpg', dpi=300)
plt.show()







#--------------------------------------------------
#
# Precipitation Plots
#
#--------------------------------------------------
#
#-------
# WY 2017-2021
#-------
#
prcp_summ_ = pd.DataFrame(prcp_summ.iloc[np.where(prcp_summ.index>pd.to_datetime('2016-09-30'))[0]])

dates     = prcp_summ_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

#prcp_summ_['wy'] = wy_inds.T
#prcp_summ_cs     = prcp_summ_.groupby(by=['wy']).cumsum()

# use daily data for cumulative plots
prcp_sumd_ = (met_comp['prcp']*3600*3).groupby(pd.Grouper(freq='D')).sum()
prcp_sumd_ = pd.DataFrame(prcp_sumd_.iloc[np.where(prcp_sumd_.index>pd.to_datetime('2016-09-30'))[0]])

dates     = prcp_sumd_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
prcp_sumd_['wy'] = wy_inds.T
prcp_sumd_cs = prcp_sumd_.groupby(by=['wy']).cumsum()

#-------
# Plots
#------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.5,3))
fig.subplots_adjust(top=0.98, bottom=0.12, left=0.20, right=0.98, hspace=0.2)
# monthly on bottom
ax = axes[1]
notjan = prcp_summ_[prcp_summ_.index.month != 1]
#ax.bar(prcp_summ_.index, prcp_summ_['prcp'], width=20.0, color='black', alpha=0.9) 
ax.bar(notjan.index, notjan['prcp'], width=20.0, color='black', alpha=0.9) 
jan = prcp_summ_[prcp_summ_.index.month == 1]
ax.bar(jan.index, jan['prcp'], width=20.0, color='C0', alpha=1.0) 
ax.set_ylabel('Precipitation\n(mm/month)')
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(25))
# cumsum on top
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.margins(x=0.01)
    ax.tick_params(axis='both', length=4, width=1.1)
    ax.tick_params(axis='y', which='both', right=True)
    #ax.tick_params(axis='x', which='major', top=True)
    ax.grid()
#axes[1].spines['top'].set_visible(False)
plt.savefig('./figures/precpipitation_cumsum.jpg', dpi=300)
plt.show()




#---------------
# WY 2008-2021
#---------------
#
prcp_summ_ = pd.DataFrame(prcp_summ.iloc[np.where(prcp_summ.index>pd.to_datetime('2007-09-30'))[0]])

dates     = prcp_summ_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

#prcp_summ_['wy'] = wy_inds.T
#prcp_summ_cs     = prcp_summ_.groupby(by=['wy']).cumsum()

# use daily data for cumulative plots
prcp_sumd_ = (met_comp['prcp']*3600*3).groupby(pd.Grouper(freq='D')).sum()
prcp_sumd_ = pd.DataFrame(prcp_sumd_.iloc[np.where(prcp_sumd_.index>pd.to_datetime('2007-09-30'))[0]])

dates     = prcp_sumd_.index
yrs       = np.unique(dates.year)
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
prcp_sumd_['wy'] = wy_inds.T
prcp_sumd_cs = prcp_sumd_.groupby(by=['wy']).cumsum()

#-------
# Plots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.5,3))
fig.subplots_adjust(top=0.98, bottom=0.12, left=0.20, right=0.98, hspace=0.2)
# monthly on bottom
ax = axes[1]
notjan = prcp_summ_[prcp_summ_.index.month != 1]
#ax.bar(prcp_summ_.index, prcp_summ_['prcp'], width=20.0, color='black', alpha=0.9) 
ax.bar(notjan.index, notjan['prcp'], width=20.0, color='black', alpha=0.9) 
jan = prcp_summ_[prcp_summ_.index.month == 1]
ax.bar(jan.index, jan['prcp'], width=20.0, color='C0', alpha=1.0) 
ax.set_ylabel('Precipitation\n(mm/month)')
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(MultipleLocator(25))
# cumsum on top
ax = axes[0]
ax.plot(prcp_sumd_cs.index, prcp_sumd_cs, color='black', lw=2.0)
ax.set_ylabel('Cumulative\nPrecipitation\n(mm/year)')
ax.tick_params(axis='x', labelbottom=False)
ax.yaxis.set_major_locator(MultipleLocator(200))
ax.yaxis.set_minor_locator(MultipleLocator(100))
# add 20 year average and percentiales
ax.axhline(prcp_sumy.mean(), color='grey', linestyle='--')
ax.axhline(np.percentile(prcp_sumy.to_numpy(),10), color='grey', linestyle=':')
ax.axhline(np.percentile(prcp_sumy.to_numpy(),90), color='grey', linestyle=':')
for i in [0,1]:
    ax = axes[i]
    #ax.minorticks_on()
    loc = mdates.YearLocator(base=2, month=10, day=1)
    loc_min = mdates.YearLocator(base=1, month=10, day=1)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_minor_locator(loc_min)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.margins(x=0.01)
    ax.tick_params(axis='both', length=4, width=1.1)
    ax.tick_params(axis='y', which='both', right=True)
    #ax.tick_params(axis='x', which='major', top=True)
    ax.grid()
#axes[1].spines['top'].set_visible(False)
plt.savefig('./figures/precpipitation_cumsum.wy08_21.jpg', dpi=300)
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
        - rtd_dict_unsort: output from ecoslim_pnts_vtk.read_vtk()
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
    # not sure here, some NANs where there are zero particles
    rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
    
    return rtd_df, rtd_dfs


def find_tau(_rtd_df):
    '''Mean age from EcoSlim distribution'''
    return (_rtd_df['Time'] * _rtd_df['wt']).sum()


# Dictionary with RTDs for all wells and timesteps
# Generated by read_vtk.py
rtd_dict = pd.read_pickle('./parflow_out/ecoslim_rtd.1721.pk')



#----------------------------
# Plotting label helpers
date_map_     = [np.where((pf_17_21.index>'{}-09-30'.format(i-1)) & (pf_17_21.index<'{}-10-01'.format(i)), True, False) for i in np.unique(pf_17_21.index.year)]
date_map_yr   = np.array([date_map_[i]*np.unique(pf_17_21.index.year)[i] for i in range(len(np.unique(pf_17_21.index.year)))]).sum(axis=0)
date_map_day_ = [np.arange(len(pf_17_21.index[date_map_[i]])) for i in range(len(np.unique(pf_17_21.index.year)))]
date_map_day  = np.concatenate((date_map_day_))
date_map      = pd.DataFrame(pf_17_21.index, columns=['Date'])
date_map['wy']  = date_map_yr
date_map['day'] = date_map_day


#----------------------------------
# Pick a well and sample time
well = 'PLM6'
samp_date = '2021-05-11' # date to consider and extract RTD info for
model_time_samp = date_map[date_map['Date'] == samp_date].index[0] # convert the date to a model time
model_time_samp = list(rtd_dict.keys())[abs(list(rtd_dict.keys()) - model_time_samp).argmin()]

rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time_samp, well, 15)
tau = (rtd_df['Time'] * rtd_df['wt']).sum()
tau_med = np.median(rtd_df['Time'])




#-----------------------------------------
# Plot a Single RTD at single well
#-----------------------------------------
"""
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
fig.suptitle('{} {}'.format(well, samp_date))
ax[0].plot(rtd_df['Time'],  rtd_df['wt'], marker='.', color='red')
ax[0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='black')
ax[0].axvline(tau, color='black', linestyle='--')
#ax[0].axvline(tau_med, color='black', linestyle=':')
ax[0].set_ylabel('PDF (NP={})'.format(len(rtd_df)))
ax[0].set_xlabel('Residence Time (years)')
ax[0].minorticks_on()
#
ax[1].plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='red')
ax[1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='black')
ax[1].axvline(tau, color='black', linestyle='--')
#ax[1].axvline(tau_med, color='black', linestyle=':')
ax[1].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
ax[1].set_xlabel('Residence Time (years)')
ax[1].minorticks_on()
fig.tight_layout()
#plt.savefig('./figures/ecoslim_rtd.jpg',dpi=300)
plt.show()
"""



#---------------------------------------------------
#
# Matrix Diffusion Convolution with EcoSlim RTD
#
#---------------------------------------------------
#
# Testing with Dispersion RTD
#
# For now, using pieces that were generated in Age_modeling dir
C_in_dict  = pd.read_csv('../ER_PLM_ParFlow/C_in_df.csv', index_col=['tau']) # convolution tracer input series

tr_in = C_in_dict['CFC12']
# first generate a dispersion RTD for reference
conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
conv_.update_pars(tau=100, mod_type='dispersion', D=0.5) 
dm = conv_.gen_g_tp() 
print ('Dispersion RTD CFC12: ', conv_.convolve())
print ('Dispersion RTD CFC12 w/ external RTD:', conv_.convolve(g_tau=dm))
print ('Dispersion RTD tau: ', conv_.tau)
# Testing - does FM RTD match the above?
# These parameters such that the total RTD mean age is = to the adjective only mean age - diffusion has little impact
convo = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
convo.update_pars(tau=1, mod_type='frac_inf_diff', Phi_im=0.000001, bbar=0.1, f_tadv_ext=dm) # tau should not matter here
gg_ = convo.gen_g_tp()
print ('FM RTD CFC12: ', convo.convolve())  # re-generate the RTD for the convolution
print ('FM RTD CFC12 w/ external RTD:: ', convo.convolve(g_tau=gg_)) # re-use the RTD for the convolution - should be faster
print ('FM RTD tau: ', convo.FM_mu)



# Predicting Tritium
tr_in = C_in_dict['H3_tu']
# first generate a pure dispersion RTD, this is just for reference
conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
conv_.update_pars(tau=25, mod_type='dispersion', D=0.5, t_half=12.34) 
dm = conv_.gen_g_tp() 
print ('Dispersion RTD 3H: ', conv_.convolve())
print ('Dispersion RTD 3H w/ external RTD:', conv_.convolve(g_tau=dm))
print ('Dispersion RTD tau: ', conv_.tau)
# Testing - does FM RTD match the above?
# These parameters such that the total RTD mean age is = to the adjective only mean age - diffusion has little impact
convo = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
convo.update_pars(tau=1, mod_type='frac_inf_diff', Phi_im=0.000001, bbar=0.01, f_tadv_ext=dm, t_half=12.34) # tau should not matter here
gg_ = convo.gen_g_tp()
print ('FM RTD 3H: ', convo.convolve())  # re-generate the RTD for the convolution
print ('FM RTD 3H w/ external RTD:: ', convo.convolve(g_tau=gg_)) # re-use the RTD for the convolution - should be faster
print ('FM RTD tau: ', convo.FM_mu)


# Predicting Helium-4
tr_in = C_in_dict['He4_ter']
# first generate a pure dispersion RTD, this is just for reference
conv_.C_t = tr_in
conv_.update_pars(tau=500, mod_type='dispersion', D=0.5, rad_accum='4He', J=ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)) 
dm = conv_.gen_g_tp() 
print ('Dispersion RTD 4He: ', conv_.convolve())
print ('Dispersion RTD 4He w/ external RTD:', conv_.convolve(g_tau=dm))
print ('Dispersion RTD tau: ', conv_.tau)
# Testing - does FM RTD match the above?
# These parameters such that the total RTD mean age is = to the adjective only mean age - diffusion has little impact
convo = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
convo.update_pars(tau=1, mod_type='frac_inf_diff', Phi_im=0.000001, bbar=0.01, f_tadv_ext=dm, rad_accum='4He', J=ng_utils.J_flux(Del=1., rho_r=2700, rho_w=1000, U=3.0, Th=10.0, phi=0.05)) # tau should not matter here
gg_ = convo.gen_g_tp()
print ('FM RTD 4He: ', convo.convolve())  # re-generate the RTD for the convolution
print ('FM RTD 4He w/ external RTD:: ', convo.convolve(g_tau=gg_)) # re-use the RTD for the convolution - should be faster
print ('FM RTD tau: ', convo.FM_mu)





#----------------------------------
# EcoSLIM rtd needs to be
# 1 - same length as tr_in
# 2 - yearly timesteps
# 3 - sum to 1


# define needed inputs
tr_in = C_in_dict['CFC12']
rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time_samp, well, 15)
xx = np.arange(0, len(tr_in)) # input function times
xx[0] += 1.e-6
#yy = np.interp(xx, rtd_dfs['Time'].to_numpy(), rtd_dfs['wt'].to_numpy(), left=0.0, right=0.0) # input function concentrations
yy = np.interp(xx, rtd_dfs['Time'].to_numpy(), rtd_dfs['wt'].to_numpy(), left=0.0, right=0.0) # input function concentrations
# Need to make sure there is at least one value above zero to avoid nans
if (yy>0.0).sum() == 0:
    yy[int(rtd_df['Time'].mean()//1)] = 1.0
yy /= yy.sum() 





#------------------------------------------------------------------------------
# Plot comparing ecoslim with interpolated RTD needed for convolution
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3.5))
fig.suptitle('{} {}'.format(well, samp_date))
#ax[0].plot(rtd_df['Time'],  rtd_df['wt'], marker='.')
ax[0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='red', zorder=10)
ax[0].plot(xx,  yy, color='C4')
ax[0].axvline(tau, color='black', linestyle='--')
#ax[0].axvline(tau_med, color='black', linestyle=':')
ax[0].set_ylabel('PDF (NP={})'.format(len(rtd_df)))
ax[0].set_xlabel('Residence Time (years)')
ax[0].set_xlim(50, 150)
#
#ax[1].plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
ax[1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
ax[1].plot(xx, np.cumsum(yy), color='C4')
ax[1].axvline(tau, color='black', linestyle='--')
#ax[1].axvline(tau_med, color='black', linestyle=':')
ax[1].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
ax[1].set_xlabel('Residence Time (years)')
ax[1].set_xlim(50, 150)
fig.tight_layout()
#plt.savefig('./figures/ecoslim_rtd.jpg',dpi=300)
plt.show()



#
# EcoSLIM Advective RTD only
#
# With FM model - but minimize matrix impact - should equal ecoslim mean age
convo = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
convo.update_pars(mod_type='frac_inf_diff', Phi_im=0.000001, bbar=0.1, f_tadv_ext=yy.copy()) 
gg_ = convo.gen_g_tp()
print ('Minimizing Matrix Diffusion')
print ('Mean age EcoSlim: ', find_tau(rtd_dfs))
print ('Mean age FM RTD:  ', convo.FM_mu)
print ('CFC-12:', convo.convolve())
print ('CFC-12 w/ external RTD:', convo.convolve(g_tau=gg_))


#
# Now compare with matrix diffusion added (Fracture RTD model)
#
conv_ = conv.tracer_conv_integral(C_t=tr_in, t_samp=tr_in.index[-1])
bbar_list = np.logspace(-5,-1,5)
fm_list   = []
fm_mu     = []
for b in bbar_list:
    conv_.update_pars(mod_type='frac_inf_diff', Phi_im=0.01, bbar=b, f_tadv_ext=yy.copy())
    fm_list.append(conv_.gen_g_tp())
    fm_mu.append(conv_.FM_mu)
    #print (conv_.convolve())
 
Phim_list = np.logspace(-5,-1,5)
fm_list_   = []
fm_mu_     = []
for p in Phim_list:
    conv_.update_pars(mod_type='frac_inf_diff', Phi_im=p, bbar=0.001, f_tadv_ext=yy.copy())
    fm_list_.append(conv_.gen_g_tp())
    fm_mu_.append(conv_.FM_mu)   
    #print (conv_.convolve())
    
#
# Plot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,6))
fig.subplots_adjust(left=0.15, right=0.6, top=0.94, bottom=0.1, hspace=0.4)
ax = axes[0]
#ax.plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='red', zorder=10)
ax.plot(xx, yy, marker='.', color='grey', alpha=0.8, zorder=10, label=r'EcoSLIM, $\tau_{{adv}}$={:.0f} yrs'.format(find_tau(rtd_dfs)))
for b in reversed(range(len(bbar_list))):
    ax.plot(fm_list[b], label=r'$\bar{{b}}$={:.1E} m, $\tau_{{tot}}$={:.0f} yrs'.format(bbar_list[b], fm_mu[b]))
ax.set_xlim(50,150)
ax.minorticks_on()
#ax.set_xlabel(r'Residence Time, t$^{\prime}$ (years)')
ax.set_ylabel(r'Fraction of Sample, g(t$^{\prime}$)')
ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1.0), framealpha=0.9, fontsize=13, 
          handlelength=1.0, labelspacing=0.25, handletextpad=0.1)
ax.set_title(r'$\phi_{{im}}$={:.1E}'.format(0.01))

ax = axes[1]
#ax.plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='red', zorder=10)
ax.plot(xx, yy, marker='.', color='grey', zorder=10, alpha=0.8, label=r'EcoSLIM, $\tau_{{adv}}$={:.0f} yrs'.format(find_tau(rtd_dfs)))
for p in range(len(Phim_list)):
    ax.plot(fm_list_[p], label=r'$\phi_{{im}}$={:.1E}, $\tau_{{tot}}$={:.0f} yrs'.format(Phim_list[p], fm_mu_[p]))
ax.set_xlim(50,150)
ax.minorticks_on()
ax.set_xlabel(r'Residence Time, t$^{\prime}$ (years)')
ax.set_ylabel(r'Fraction of Sample, g(t$^{\prime}$)')
ax.legend(loc='upper left', bbox_to_anchor=(0.7, 1.0), fancybox=True, framealpha=0.9, fontsize=13,
          handlelength=1.0, labelspacing=0.25, handletextpad=0.1)
ax.set_title(r'$\bar{{b}}$={:.1E} m'.format(0.001))
plt.savefig('./figures/ecoslim_fm_pdf.jpg',dpi=300)
plt.show()




#------------------------------
#
# FM RTD Sensitivity Analysis
#
#------------------------------

# define needed inputs
tr_in = C_in_dict['CFC12']
rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time_samp, well, 15)
xx = np.arange(0, len(tr_in)) # input function times
xx[0] += 1.e-6
#yy = np.interp(xx, rtd_dfs['Time'].to_numpy(), rtd_dfs['wt'].to_numpy(), left=0.0, right=0.0) # input function concentrations
yy = np.interp(xx, rtd_df['Time'].to_numpy(), rtd_df['wt'].to_numpy(), left=0.0, right=0.0) # input function concentrations
# Need to make sure there is at least one value above zero to avoid nans
if (yy>0.0).sum() == 0:
    yy[int(rtd_df['Time'].mean()//1)] = 1.0
yy /= yy.sum() 


# Randomly sample over bbar and phim
#bbar_list_ = 10**(np.random.uniform(-6,-2, 250))
#Phim_list_ = 10**(np.random.uniform(-4,-1, 250))

# Improved parameters using effective rock physics
_bbar    = lambda Keff, Km, L: ((L*12.*1.e-3/9810.)*(Keff-Km))**(1/3.)
_phi_eff = lambda phim, b, L, phif: (L*phim+b*phif)/(L+b)
_Keff    = lambda Km, b, L: (L*Km + (b**3)*9810./(12*1.e-3)) / (L+b)


# Priors Based on Uhlemann 2022 Borehole NMR data
nsize = 100

Km_list_   = 10**(np.random.uniform(-8, -6, nsize)) # 1.e-8 to 1.e-6 matrix hydraulic conductivty
Keff_list_ = 10**(np.random.uniform(-6, -4, nsize)) # 1.e-6 to 1.e-4 matrix hydraulic conductivty, can also do Keff_list_ as mulitiplicative factor of Km
L_list_    = np.random.uniform(0.1, 0.5, nsize)     # fracture spacing (m), based on Gardner 2020
Phim_list_ = 10**(np.random.uniform(np.log10(0.05/100), np.log10(5/100), nsize)) # 0.05% to 5% matrix porosity (variable needs to be in decimal)
bbar_list_ = _bbar(Keff_list_, Km_list_, L_list_)


rerun = False
if rerun:
    # Dictionary to hold outputs
    fm_dict = {'tau_mu':[],
               'cfc12':[],
               'sf6':[],
               'h3':[],
               'he4':[]}
    for b,p in zip(bbar_list_, Phim_list_):
        # Initialize
        conv_ = conv.tracer_conv_integral(C_t=C_in_dict['CFC12'], t_samp=C_in_dict['CFC12'].index[-1])
        conv_.update_pars(C_t=C_in_dict['CFC12'].copy(), mod_type='frac_inf_diff', Phi_im=p, bbar=b, f_tadv_ext=yy.copy())
        g_tau = conv_.gen_g_tp().copy()
        fm_dict['tau_mu'].append(conv_.FM_mu)
        fm_dict['cfc12'].append(conv_.convolve(g_tau=g_tau.copy()))
        # sf6 prediciton
        conv_.C_t=C_in_dict['SF6'].copy()
        fm_dict['sf6'].append(conv_.convolve(g_tau=g_tau.copy()))
        # tritium prediction
        conv_.C_t = C_in_dict['H3_tu'].copy()
        conv_.update_pars(t_half=12.34)
        fm_dict['h3'].append(conv_.convolve(g_tau=g_tau.copy()))
        # Helium-4 prediction
        conv_.C_t = C_in_dict['He4_ter'].copy()
        conv_.update_pars(rad_accum='4He', t_half=False, lamba=False, J=ng_utils.J_flux(Del=1.,rho_r=2700,rho_w=1000,U=3.0,Th=10.0,phi=0.05))
        fm_dict['he4'].append(conv_.convolve(g_tau=g_tau.copy()))
    with open('fm_dm_dict.pk', 'wb') as handle:
        pickle.dump(fm_dict, handle)
else:
    fm_dict = pd.read_pickle('./fm_dm_dict.pk')


#
# Plotting
#
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    return r'$10^{{{}}}$'.format(float(a))

ylabs = {'cfc11':'CFC-11',
         'cfc12':'CFC-12',
         'cfc113':'CFC-113',
         'sf6':'SF$_{6}$',
         'he4':'$^{4}$He$\mathrm{_{terr}}$',
         'H3_He3':'$\mathrm{^{3}H/^{3}He}$',
         'h3':'$\mathrm{^{3}H}$',
         'He3':'$\mathrm{^{3}He}$',
         'tau_mu':r'$\it{\tau_{comp}}$'}

ylabs_units = {'cfc11':'pptv',
               'cfc12':'pptv',
               'cfc113':'pptv',
               'sf6':'pptv',
               'he4':r'cm$^{3}$STP/g',
               'H3_He3':'TU',
               'h3':'TU',
               'He3':'TU',
               'tau_mu':'yrs'}

if rerun:
    # Mean Age as function of bbar and immobile porosity
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    #fig.subplots_adjust(left=0.15, right=0.6, top=0.96, bottom=0.1, hspace=0.4)
    pts = ax.scatter(bbar_list_, Phim_list_, c=np.log10(fm_dict['tau_mu']))
    #pts = ax.scatter(bbar_list_, Phim_list_, c=fm_mu_)
    ax.set_title(r'$\tau_{{adv}}$={:.0f} years ({})'.format(find_tau(rtd_dfs), well))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\bar{b}$ (m)')
    ax.set_ylabel(r'$\phi_{im}$ (-)')
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax.set_xlim(10**(-4.8), 10**(-3.0))
    #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    #ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(pts, cax=cax, format=ticker.FuncFormatter(fmt), label='Mean Residence Time (years)')
    fig.tight_layout()
    plt.savefig('./figures/ecoslim_fm_mc.jpg',dpi=300)
    plt.show()
    
    
    
    # 
    # Density Distributions
    #
    fig, axes = plt.subplots(nrows=5,ncols=1,figsize=(3, 8))
    fig.subplots_adjust(left=0.15, right=0.9, top=0.98, bottom=0.08, hspace=0.55)
    #fig.subplots_adjust(hspace=0.4)
    fm_dict_ = list(fm_dict.keys())
    for i in range(len(fm_dict_)):
        ax = axes[i]
        dd  = np.array(fm_dict[fm_dict_[i]])
        dd_ = dd[~np.isnan(dd)]
        if fm_dict_[i]=='he4':
            ax.set_xscale('log')
            hist, bins = np.histogram(dd_, bins=15)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            ax.hist(dd_, bins=logbins, log=False, histtype='step', color='black', linewidth=1.5)   
        else:
            if dd_.max() < 1.e-9:
                ax.axvline(0.0, c='black')
                ax.set_xlim(-1,1)
            else:
                ax.hist(dd_, bins=15, log=False, histtype='step', color='black', linewidth=1.5)
        ax.minorticks_on()
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)
        ax.set_xlabel('{} ({})'.format(ylabs[fm_dict_[i]], ylabs_units[fm_dict_[i]]), labelpad=0.5)
    fig.text(0.02, 0.53, 'Density ({})'.format(well), va='center', rotation='vertical')
    plt.savefig('./figures/ecoslim_fm_conc.jpg',dpi=300)
    plt.show()














#----------------------------
#
# Combined RTD Plots
#
#----------------------------
wells = ['PLM1','PLM7','PLM6']

"""
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
    ax[i,0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='black')
    ax[i,0].axvline(tau, color='black', linestyle='--')
    #ax[0].axvline(tau_med, color='black', linestyle=':')
    ax[i,0].set_ylabel('PDF (NP={})'.format(len(rtd_df)))
    ax[i,0].set_title(w, fontsize=14)
    #
    ax[i,1].plot(rtd_df['Time'],  np.cumsum(rtd_df['wt']), marker='.', color='black')
    #ax[i,1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
    ax[i,1].axvline(tau, color='black', linestyle='--')
    #ax[1].axvline(tau_med, color='black', linestyle=':')
    ax[i,1].set_ylabel('CDF (NP={})'.format(len(rtd_df)))
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
#plt.savefig('./figures/ecoslim_rtd_comp.jpg',dpi=300)
#plt.savefig('./figures/ecoslim_rtd_comp.svg',format='svg')
plt.show()
"""








#-------------------------------------------------------
#
# Transient Age Distributions
#
#-------------------------------------------------------
yr = np.arange(2017,2022)

wy_inds_   = [np.where((date_map['Date'] > '{}-09-30'.format(i-1)) & (date_map['Date'] < '{}-10-01'.format(i)), True, False) for i in yr]
wy_inds    = [date_map.index[wy_inds_[i]] for i in range(len(yr))]
wy_map     = dict(zip(yr, wy_inds))
wy_inds    = np.concatenate(((wy_inds)))

time_list       = list(wy_inds[np.isin(wy_inds, list(rtd_dict.keys()))]) #+ [model_time_samp] # model times that I want
time_list_dates = date_map.loc[time_list,'Date'].to_list() # dates
time_list_da   = date_map.loc[time_list,'day'].to_list() # corresponding day of water year
time_list_mn   = [date_map.loc[i,'Date'].month for i in time_list] # corresponding month
time_list_yr   = date_map.loc[time_list,'wy'].to_list() # corresponding water year
time_list_map_ = [np.where(np.array(time_list_yr)==y)[0] for y in np.unique(np.array(time_list_yr))]
time_list_map  = dict(zip(np.unique(np.array(time_list_yr)), time_list_map_))


# Want first day of the month for x labels -- pick a full year like 2019
month_map  = {1:'O',2:'N',3:'D',4:'J',5:'F',6:'M',7:'A',8:'M',9:'J',10:'J',11:'A',12:'S'}
months     = list(month_map.values())
days_ = [date_map['Date'][date_map['wy']==2019].iloc[i].month for i in range(len(date_map['Date'][date_map['wy']==2019]))]
first_month  = [(np.array(days_)==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]


# Sample date to model time
samp_date   = '2021-05-11' # date to consider and extract RTD info for
samp_date_  = np.array([(i-pd.to_datetime(samp_date)).days for i in time_list_dates])
_model_time = abs(samp_date_).argmin()




#
# Generate a Dictionary with all the RTDS and the taus for the chosen years and wells
#
wells = ['PLM1','PLM7','PLM6']

rtd_trans = {}
for i in range(len(wells)):
    w       = wells[i]
    tau_mu  = []
    tau_med = []
    rtd_df_ = []
    keep_list = [] # needed for when there are no particles for a timestep
    rtd_trans[w] = {}
    for t in range(len(time_list)):
        model_time = time_list[t]
        try:
            rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 10)
            rtd_df_.append(rtd_df)
            tau_mu.append((rtd_df['Time'] * rtd_df['wt']).sum())
            tau_med.append(np.median(rtd_df['Time']))
            # not sure here, some NANs where there are zero particles
            #rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
            keep_list.append(t)
        except ValueError:
            print (w, model_time)
            pass
    rtd_trans[w]['tau_mu']    = tau_mu
    rtd_trans[w]['tau_med']   = tau_med
    rtd_trans[w]['rtd_df']    = rtd_df_
    rtd_trans[w]['keep_list'] = keep_list




#
# Plot CDF Curves for all Timesteps
#
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.8,4.2))
fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.98, bottom=0.15, right=0.87, left=0.32)
for i in range(3):
    w    = wells[i]
    ax   = axes[i]
    for t in range(len(time_list)):
        try:
            _rtd_df = rtd_trans[w]['rtd_df'][t]
            _tau    = rtd_trans[w]['tau_mu'][t]
            ax.plot(_rtd_df['Time'], np.cumsum(_rtd_df['wt']), color='black', alpha=0.75, zorder=4)
            ax.axvline(_tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
        except IndexError:
            print ('no {} {}'.format(w, time_list[t]))
            pass
    # Replot the middle one in red?
    mid_ind  = np.where(np.array(rtd_trans[w]['tau_mu'])==np.sort(rtd_trans[w]['tau_mu'])[len(rtd_trans[w]['tau_mu'])//2])[0][0]
    _rtd_df_ = rtd_trans[w]['rtd_df'][mid_ind]
    _tau_    = rtd_trans[w]['tau_mu'][mid_ind]
    ax.plot(_rtd_df_['Time'], np.cumsum(_rtd_df_['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
    ax.axvline(_tau_, color='red', alpha=0.65, linestyle='--', zorder=5)
    # Clean up 
    if i == 1:
        ax.set_ylabel('CDF\n{}'.format(w))
    else:
        ax.set_ylabel('{}'.format(w))
    ax.minorticks_on()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='major', axis='both', length=4, width=1.25)
axes[2].set_xlabel('Particle Ages (years)')    
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.svg',format='svg')
plt.show()




#
# Plot Temporal Dynamics of the Median Age
#
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7,4.5))
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=0.96, bottom=0.15, right=0.98, left=0.20)
for i in range(3):
    w    = wells[i]
    ax   = axes[i]
    ax.plot(np.array(time_list_dates)[rtd_trans[w]['keep_list']], rtd_trans[w]['tau_mu'], color='black')
    ax.set_ylabel('Mean Age\n(years)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', which='major', length=4.0, rotation=30, pad=0.1)
    #ax.tick_params(axis='both', which='both', top=True, right=True, labelbottom=False)
    ax.margins(x=0.01)
    ax.grid()
    # Clean up 
    ax.set_ylabel('{}'.format(w))
    ax.tick_params(which='both', axis='y', left=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for label in ax.get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    for i in np.arange(2017,2021).reshape(2,2):
        ax.axvspan(pd.to_datetime('{}-10-01'.format(i[0])), pd.to_datetime('{}-09-30'.format(i[1])), alpha=0.04, color='red')
fig.text(0.03, 0.55, 'Mean Age (years)', ha='center', va='center', rotation='vertical')
[axes[i].tick_params(axis='x', labelbottom=False) for i in [0,1]]
plt.savefig('./figures/ecoslim_tau_trans.png',dpi=300)
plt.savefig('./figures/ecoslim_tau_trans.svg',format='svg')
plt.show()






#
# Plot Tracer Concentrations for Sampling Time
#

tr_map = {'cfc12':'CFC12','sf6':'SF6','h3':'H3','he4':'He4_ter'}


def gen_eco_adv(_rtd_df):
    '''Generate ecoslim advective RTD to be used in FM RTD'''
    tr_in = C_in_dict['CFC12']    
    xx = np.arange(0, len(tr_in)) # input function times
    xx[0] += 1.e-6
    yy = np.interp(xx, _rtd_df['Time'].to_numpy(), _rtd_df['wt'].to_numpy(), left=0.0, right=0.0) # input function concentrations
    # Make sure there are no Nans
    if (yy>0.0).sum() == 0:
        yy[int(_rtd_df['Time'].mean()//1)] = 1.0
    yy /= yy.sum() 
    return yy

# Priors Based on Uhlemann 2022 Borehole NMR data
nsize = 100

Km_list_   = 10**(np.random.uniform(-8, -6, nsize)) # 1.e-8 to 1.e-6 matrix hydraulic conductivty
Keff_list_ = 10**(np.random.uniform(-6, -4, nsize)) # 1.e-6 to 1.e-4 matrix hydraulic conductivty, can also do Keff_list_ as mulitiplicative factor of Km
L_list_    = np.random.uniform(0.1, 0.5, nsize)     # fracture spacing (m), based on Gardner 2020
Phim_list_ = 10**(np.random.uniform(-3, -1, nsize)) # 0.1% to 10% matrix porosity
bbar_list_ = _bbar(Keff_list_, Km_list_, L_list_)
# Want first value to minimize matrix impacts # Phi_im=0.000001, bbar=0.1
Phim_list_ = np.array([1.e-6] + list(Phim_list_))
bbar_list_ = np.array([0.1] + list(bbar_list_))



def fm_conc(eco_adv):
    '''eco_adv from gen_eco_adv above'''
    # Dictionary to hold outputs
    fm_dict = {'tau_mu':[],
               'cfc12':[],
               'sf6':[],
               'h3':[],
               'he4':[]}
    for b,p in zip(bbar_list_, Phim_list_):
        # Initialize
        conv_ = conv.tracer_conv_integral(C_t=C_in_dict['CFC12'], t_samp=C_in_dict['CFC12'].index[-1])
        conv_.update_pars(C_t=C_in_dict['CFC12'], mod_type='frac_inf_diff', Phi_im=p, bbar=b, f_tadv_ext=eco_adv.copy())
        g_tau = conv_.gen_g_tp().copy()
        fm_dict['tau_mu'].append(conv_.FM_mu)
        fm_dict['cfc12'].append(conv_.convolve(g_tau=g_tau.copy()))
        # sf6 prediciton
        conv_.C_t=C_in_dict['SF6']
        fm_dict['sf6'].append(conv_.convolve(g_tau=g_tau.copy()))
        # tritium prediction
        conv_.C_t = C_in_dict['H3_tu']
        conv_.update_pars(t_half=12.34)
        h3 = conv_.convolve(g_tau=g_tau.copy())
        if h3 < 1.e-10:
            fm_dict['h3'].append(0.0)
        else:
            fm_dict['h3'].append(h3)
        # Helium-4 prediction
        conv_.C_t = C_in_dict['He4_ter']
        conv_.update_pars(rad_accum='4He', t_halfe=False, lamba=False, J=ng_utils.J_flux(Del=1.,rho_r=2700,rho_w=1000,U=3.0,Th=10.0,phi=0.05))
        fm_dict['he4'].append(conv_.convolve(g_tau=g_tau))
    return fm_dict


rerun = False
if rerun:
    fm_dict_wells = {}
    for r in range(3):
        w    = wells[r]
        fm_dict_wells[w] = {}
        _rtd_df = rtd_trans[w]['rtd_df'][_model_time]
        _tau    = rtd_trans[w]['tau_mu'][_model_time]
        _yy     = gen_eco_adv(_rtd_df)
        _conc_df = fm_conc(_yy)
        fm_dict_wells[w] = _conc_df
    
        with open('fm_dict_wells.pk', 'wb') as handle:
            pickle.dump(fm_dict_wells, handle)
else:
    fm_dict_wells = pd.read_pickle('./fm_dict_wells.pk')



# Histogram plot
# tracers as columns - wells as rows
tracers = ['tau_mu', 'h3', 'cfc12', 'he4']
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(8,5))
fig.subplots_adjust(wspace=0.1, hspace=0.3, top=0.98, bottom=0.15, left=0.05, right=0.96)
for r in range(3):
    w   = wells[r]
    _cc = fm_dict_wells[w]
    for c in range(4):
        ax  = axes[r,c]
        tr_ = tracers[c]
        dd_ = np.array(_cc[tr_])
        dd  = dd_[1:]
        #if tr_=='he4':
        if tr_ in ['tau_mu','he4']:
            ax.set_xscale('log')
            hist, bins = np.histogram(dd, bins=15)
            logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
            ax.hist(dd, bins=logbins, log=False, histtype='bar', color='grey', alpha=0.7, linewidth=1.5)   
            #if tr_ == 'tau_mu':
                #ax.set_xlim(10, ax.get_xlim()[1])
                #ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
                #ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
        else:
            if dd.max() < 1.e-7:
                ax.axvline(0.00, c='grey')
                if dd_.max() < 1:
                    ax.set_xlim(-0.1,0.1)
            else:
                ax.hist(dd, bins=15, log=False, histtype='bar', color='grey', alpha=0.7, linewidth=1.5)
        ax.axvline(np.median(dd), color='black', alpha=0.6)
        ax.minorticks_on()
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)
        # Plot the EcoSlim - no matrix diffusion as line
        dd = dd_[0]
        ax.axvline(dd, color='C0', linestyle='--', label='advective' if r==0 and c==1 else '')
        # Plot field observation
        if tr_ != 'tau_mu':
            #obs = obs_dict[tr_][w]
            obs = ens_dict[tr_map[tr_]][w].to_numpy().mean()
            ax.axvline(obs, color='C3', linestyle=':', linewidth=2.5)
        if c == 0:
            ax.set_ylabel(w)
        if r == 2:
            ax.set_xlabel('{} ({})'.format(ylabs[tr_], ylabs_units[tr_]), labelpad=1.0)
        else:
            #ax.tick_params(axis='x', labelbottom=False)
            pass
#plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.png',dpi=300)
#plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.svg',format='svg')
plt.show()





# Errorbar plot
# tracers as columns - wells as rows
tracers = ['tau_mu', 'h3', 'cfc12', 'he4']
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8,4))
fig.subplots_adjust(wspace=0.5, hspace=0.3, top=0.8, bottom=0.15, left=0.08, right=0.96)
for i in range(4):
    ax  = axes[i]
    for j in range(3):
        w   = wells[j]
        _cc = fm_dict_wells[w]
        tr_ = tracers[i]
        dd_ = np.array(_cc[tr_])
        dd_ = dd_[~np.isnan(dd_)] # drop any nans
        dd  = dd_[1:]
        if tr_ in ['tau_mu','he4']:
            ax.set_yscale('log')
            ax.scatter(j, dd.mean(), marker='o', facecolors='none', color='C{}'.format(j))   
            ax.scatter(j, np.median(dd), marker='_', color='C{}'.format(j))
            ax.vlines(j, dd.min(), dd.max(), color='C{}'.format(j))
        else:
            ax.scatter(j, dd.mean(), marker='o', facecolors='none', color='C{}'.format(j))
            ax.scatter(j, np.median(dd), marker='_', color='C{}'.format(j))
            ax.vlines(j, dd.min(), dd.max(), color='C{}'.format(j))
        ax.minorticks_on()
        #ax.tick_params(axis='y', which='both', labelleft=False, left=False)
        # Plot the EcoSlim - no matrix diffusion as line
        dd = dd_[0]
        ax.scatter(j+0.1, dd, marker='D', color='C{}'.format(j))
        # Plot field observation
        if tr_ != 'tau_mu':
            #obs = obs_dict[tr_][w]
            obs = ens_dict[tr_map[tr_]][w].to_numpy().mean()
            ax.scatter(j-0.1, obs, marker='*', s=55, color='C{}'.format(j))
    ax.set_title('{}\n({})'.format(ylabs[tr_], ylabs_units[tr_]))
    ax.set_xticks(ticks=[0,1,2], labels=wells)
    ax.tick_params(axis='x', rotation=30, pad=0.1)
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
plt.savefig('./figures/ecoslim_rtd_fm.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_fm.svg',format='svg')
plt.show()










#---------------------------------
#
# Dummy Soil Wells
#
#--------------------------------

#yr = [2017, 2018, 2019, 2020, 2021]
yr = np.arange(2017,2021)
wells = ['X404', 'X494', 'X508']
names = ['PLM1 Soil', 'PLM6 Soil', 'Floodplain']

#
# Generate a Dictionary with all the RTDS and the taus for the chosen years and wells
#
rtd_trans = {}
for i in range(len(wells)):
    w       = wells[i]
    tau_mu  = []
    tau_med = []
    rtd_df_ = []
    keep_list = [] # needed for when there are no particles for a timestep
    rtd_trans[w] = {}
    for t in range(len(time_list)):
        model_time = time_list[t]
        try:
            rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, model_time, w, 10)
            rtd_df_.append(rtd_df)
            tau_mu.append((rtd_df['Time'] * rtd_df['wt']).sum())
            tau_med.append(np.median(rtd_df['Time']))
            # not sure here, some NANs where there are zero particles
            #rtd_dfs['Time'] = rtd_dfs['Time'].interpolate(method='linear')
            keep_list.append(t)
        except ValueError:
            print (w, model_time)
            pass
    rtd_trans[w]['tau_mu']    = tau_mu
    rtd_trans[w]['tau_med']   = tau_med
    rtd_trans[w]['rtd_df']    = rtd_df_
    rtd_trans[w]['keep_list'] = keep_list

#
# Plot CDF Curves for all Timesteps
#
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(2.8,4.2))
fig.subplots_adjust(wspace=0.05, hspace=0.4, top=0.98, bottom=0.15, right=0.87, left=0.32)
for i in range(3):
    w    = wells[i]
    ax   = axes[i]
    for t in range(len(time_list)):
       try:
           _rtd_df = rtd_trans[w]['rtd_df'][t]
           _tau    = rtd_trans[w]['tau_mu'][t]
           ax.plot(_rtd_df['Time'], np.cumsum(_rtd_df['wt']), color='black', alpha=0.75, zorder=4)
           ax.axvline(_tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
       except IndexError:
           print ('no {}'.format(time_list[t]))
           pass
    # Replot the middle one in red?
    mid_ind  = np.where(np.array(rtd_trans[w]['tau_mu'])==np.sort(rtd_trans[w]['tau_mu'])[len(rtd_trans[w]['tau_mu'])//2])[0][0]
    _rtd_df_ = rtd_trans[w]['rtd_df'][mid_ind]
    _tau_    = rtd_trans[w]['tau_mu'][mid_ind]
    ax.plot(_rtd_df_['Time'], np.cumsum(_rtd_df_['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
    ax.axvline(_tau_, color='red', alpha=0.65, linestyle='--', zorder=5)
    # Clean up 
    if i == 1:
        ax.set_ylabel('CDF\n{}'.format(names[i]))
    else:
        ax.set_ylabel('{}'.format(names[i]))
    ax.minorticks_on()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='major', axis='both', length=4, width=1.25)
axes[2].set_xlabel('Particle Ages (years)')    
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.png',dpi=300)
plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.svg',format='svg')
plt.show()




#
# Plot Temporal Dynamics of the Median Age
#
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7,4.5))
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=0.96, bottom=0.15, right=0.98, left=0.20)
for i in range(3):
    w    = wells[i]
    ax   = axes[i]
    ax.plot(np.array(time_list_dates)[rtd_trans[w]['keep_list']], rtd_trans[w]['tau_mu'], color='black')
    ax.set_ylabel('Mean Age\n(years)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
    #ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', which='major', length=4.0, rotation=30, pad=0.1)
    #ax.tick_params(axis='both', which='both', top=True, right=True, labelbottom=False)
    ax.margins(x=0.01)
    ax.grid()
# Clean up 
    ax.set_ylabel('{}'.format(names[i]))
    ax.tick_params(which='both', axis='y', left=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for label in ax.get_xticklabels(which='major'):
        label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
    for i in np.arange(2017,2021).reshape(2,2):
        ax.axvspan(pd.to_datetime('{}-10-01'.format(i[0])), pd.to_datetime('{}-09-30'.format(i[1])), alpha=0.04, color='red')
fig.text(0.03, 0.55, 'Mean Age (years)', ha='center', va='center', rotation='vertical')
[axes[i].tick_params(axis='x', labelbottom=False) for i in [0,1]]
plt.savefig('./figures/ecoslim_tau_trans.soil.png',dpi=300)
plt.savefig('./figures/ecoslim_tau_trans.soil.svg',format='svg')
plt.show()




#
# Plot Fraction Younger
#
labs_ = {'X508':'Floodplain',
         'X494':'PLM6 Soil',
         'X404':'PLM1 Soil'}
w = 'X508'

# multiple fractions
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,4.5))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.20, right=0.96, hspace=0.15)
ax = axes[1]
#ax.set_title(labs_[w], fontsize=14)
frac_young_ = {1:[],10:[]}
dates_      = []
for t in range(len(time_list)):
    rtd_df_ = rtd_trans[w]['rtd_df'][t]
    dates_.append(date_map.loc[time_list[t],'Date'])
    for k in list(frac_young_.keys()):
        frac_young_[k].append(rtd_df_[rtd_df_['Time']<k]['wt'].sum())
fkeys = list(frac_young_.keys())
for i in range(len(fkeys)):
    ax.plot(dates_, 100-np.array(frac_young_[fkeys[i]])*100, c='C{}'.format(i), linestyle='-', label='{} year'.format(fkeys[i])) 
ax.set_ylabel('PLM6 Soil\nFraction Older\n(%)')
#ax.set_yscale('log')
#ax.set_ylim(0.11,110.0)
##ax.yaxis.set_minor_locator(ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)))
ax.yaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', which='major', length=4.0, rotation=30, pad=0.1)
ax.tick_params(axis='both', which='both', top=True, right=True)
# Cleanup
ax.margins(x=0.01)
ax.grid()
ax.legend(handlelength=0.75, labelspacing=0.25, handletextpad=0.1, fontsize=13, loc='lower left')
          #bbox_to_anchor=(-0.007, 1.18) , fancybox=True, framealpha=0.95)
for label in ax.get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
for i in np.arange(2017,2021).reshape(2,2):
    ax.axvspan(pd.to_datetime('{}-10-01'.format(i[0])), pd.to_datetime('{}-09-30'.format(i[1])), alpha=0.04, color='red')
#
# Add Mean Age
ax2 = axes[0]
ax2.plot(dates_, rtd_trans[w]['tau_mu'], color='black')
ax2.set_ylabel('{}\nMean Age\n(years)'.format(labs_[w]))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10]))
ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='x', which='major', length=4.0)
ax2.tick_params(axis='both', which='both', top=True, right=True, labelbottom=False)
ax2.margins(x=0.01)
ax2.grid()
for label in ax2.get_xticklabels(which='major'):
    label.set(horizontalalignment='right', rotation_mode="anchor")#, horizontalalignment='right')
for i in np.arange(2017,2021).reshape(2,2):
    ax2.axvspan(pd.to_datetime('{}-10-01'.format(i[0])), pd.to_datetime('{}-09-30'.format(i[1])), alpha=0.04, color='red')
plt.savefig('./figures/rtd_younger_than.jpg', dpi=300)
plt.savefig('./figures/rtd_younger_than.svg', format='svg')
plt.show()






#
# In-depth CDF plots for Soil Wells
#
# Better for full timeseries plots
yrs_comp = np.arange(2017, 2021).reshape(2,2)

# Plot PLM6 Soil
w = 'X494'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,4))
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=0.98, bottom=0.15, left=0.15, right=0.97)
for i in range(2):
    for j in range(2):
        r,c = i%2, j%2
        print(r,c)
        ax   = axes[r,c]
        wy = yrs_comp[i][j]
        time_list_ = time_list_map[wy]
        for t in time_list_:
            _rtd_df = rtd_trans[w]['rtd_df'][t]
            _tau    = rtd_trans[w]['tau_mu'][t]
            ax.plot(_rtd_df['Time'], np.cumsum(_rtd_df['wt']), color='black', alpha=0.75, zorder=4)
            ax.axvline(_tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
        # Replot the middle one in red?
        mid_ind  = np.where(np.array(rtd_trans[w]['tau_mu'])==np.sort(rtd_trans[w]['tau_mu'])[len(rtd_trans[w]['tau_mu'])//2])[0][0]
        _rtd_df_ = rtd_trans[w]['rtd_df'][mid_ind]
        _tau_    = rtd_trans[w]['tau_mu'][mid_ind]
        #ax.plot(_rtd_df_['Time'], np.cumsum(_rtd_df_['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
        #ax.axvline(_tau_, color='red', alpha=0.65, linestyle='--', zorder=5)
        # Clean up 
        #ax.set_ylabel(wy)
        ax.text(0.85, 0.12, wy, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_xlim(-1.0, 130.0)
        if c != 0:
            ax.tick_params(axis='y', labelleft=False)
        if r != 1:
            ax.tick_params(axis='x', labelbottom=False)
#axes[2].set_xlabel('Particle Ages (years)')    
fig.text(0.55, 0.02, 'Particle Age', ha='center')
fig.text(0.01, 0.55, '{} CDF'.format(labs_[w]), va='center', rotation='vertical')
mx =  max([rtd_trans[w]['rtd_df'][zz]['Time'].max() for zz in range(len(rtd_trans[w]['rtd_df']))])
print ('Max Age = ', mx)
plt.savefig('./figures/ecoslim_rtd.plm6_soil.grid.png',dpi=300)
#plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.svg',format='svg')
plt.show()



#
# In-depth CDF plots for Soil Wells
#
# Better for full timeseries plots
yrs_comp = np.arange(2017, 2021).reshape(2,2)

# Plot PLM6 Soil
w = 'X508'
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,4))
fig.subplots_adjust(wspace=0.05, hspace=0.1, top=0.98, bottom=0.15, left=0.15, right=0.97)
for i in range(2):
    for j in range(2):
        r,c = i%2, j%2
        print(r,c)
        ax   = axes[r,c]
        wy = yrs_comp[i][j]
        time_list_ = time_list_map[wy]
        for t in time_list_:
            _rtd_df = rtd_trans[w]['rtd_df'][t]
            _tau    = rtd_trans[w]['tau_mu'][t]
            ax.plot(_rtd_df['Time'], np.cumsum(_rtd_df['wt']), color='black', alpha=0.75, zorder=4)
            ax.axvline(_tau, color='grey', alpha=0.5, linestyle='-', zorder=2)
        # Replot the middle one in red?
        mid_ind  = np.where(np.array(rtd_trans[w]['tau_mu'])==np.sort(rtd_trans[w]['tau_mu'])[len(rtd_trans[w]['tau_mu'])//2])[0][0]
        _rtd_df_ = rtd_trans[w]['rtd_df'][mid_ind]
        _tau_    = rtd_trans[w]['tau_mu'][mid_ind]
        #ax.plot(_rtd_df_['Time'], np.cumsum(_rtd_df_['wt']), color='red', lw=2.0, alpha=0.90, zorder=6)
        #ax.axvline(_tau_, color='red', alpha=0.65, linestyle='--', zorder=5)
        # Clean up 
        #ax.set_ylabel(wy)
        ax.text(0.85, 0.12, wy, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_xlim(-1.0, 220.0)
        if c != 0:
            ax.tick_params(axis='y', labelleft=False)
        if r != 1:
            ax.tick_params(axis='x', labelbottom=False)
#axes[2].set_xlabel('Particle Ages (years)')    
fig.text(0.55, 0.02, 'Particle Age', ha='center')
fig.text(0.01, 0.55, '{} CDF'.format(labs_[w]), va='center', rotation='vertical')
mx =  max([rtd_trans[w]['rtd_df'][zz]['Time'].max() for zz in range(len(rtd_trans[w]['rtd_df']))])
print ('Max Age = ', mx)
plt.savefig('./figures/ecoslim_rtd.floodplain.grid.png',dpi=300)
#plt.savefig('./figures/ecoslim_rtd_comp_obs_ens.soil.svg',format='svg')
plt.show()





#---------------------------------------------------------
#
# Infiltration Location versus Age for Young Component
#
#---------------------------------------------------------

w = 'X508'

fig, ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(top=0.88, bottom=0.2, left=0.25, right=0.9, hspace=0.2)
ax.set_title(labs_[w], fontsize=14)
for t in range(len(time_list)):
    model_time = time_list[t]
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
#plt.savefig('./figures/rtd_vs_inf.jpg', dpi=300)
#plt.savefig('./figures/rtd_vs_inf.svg', format='svg')
plt.show()











"""
#------------------------------------
#
# Infiltration Location versus Age
#
#------------------------------------
fig, ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(top=0.88, bottom=0.2, left=0.25, right=0.9, hspace=0.2)
ax.set_title(labs_[w], fontsize=14)
for t in range(len(time_list)):
    model_time = time_list[t]
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


# Seperate by Month
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,8))
#fig.subplots_adjust(top=0.88, bottom=0.2, left=0.25, right=0.9, hspace=0.2)
w = 'X494'
yr = [2019]
mnths = [10,11,12,1,2,3,4,5,6,7,8,9]
for t in range(len(yr)):
    for m in range(len(mnths)):
        c = m%4
        r = m//4
        ax = axes[r,c]
        try:
            #if m==7:
            #    pdb.set_trace()
            msk = np.where((np.array(time_list_yr)==yr[t]) & (np.array(time_list_mn)==mnths[m]))[0]
            rtd_df_ = pd.concat((rtd_trans[w]['rtd_df'][msk[0]:msk[-1]+1]))
        except IndexError:
            pass
        else:
            print (mnths[m], rtd_df_['Xin'].mean())
            #print (np.array(rtd_trans[w]['tau_mu'])[msk].mean())
            print (np.array(rtd_trans[w]['tau_mu'])[msk].mean(), rtd_df_['Xin'].mean())
            if len(rtd_df_)>1000:
                rtd_df_.sample(n=1000)
                #ax.scatter(rtd_df_['Xin'], rtd_df_['Time'], c='C{}'.format(t), marker='.') 
                #sns.kdeplot(x=rtd_df_['Xin'], y=rtd_df_['Time'], ax=ax, fill=True)
                sns.kdeplot(x=rtd_df_['Xin'], ax=ax)
            else:
                ax.scatter(rtd_df_['Xin'], rtd_df_['Time'], c='C{}'.format(t), marker='.')
                #sns.kdeplot(x=rtd_df['Xin'], y=rtd_df['Time'], ax=ax, fill=True)
                #sns.kdeplot(x=rtd_df_['Xin'], ax=ax)
        if t==0:
            ax.text(0.92, 0.95, months[m], horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
plt.show()
"""
        
        









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
