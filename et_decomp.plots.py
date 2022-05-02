# Update: 05/01/2022 makes less plots
# Makes a bunch of plots of the ParFLOW-CLM land surface dynamics
# 
# Need to run these scripts first...
# Run these on the server where all the output files are located
# -- et_to_pickle.py
# -- sat_press_to_pickle.py
# -- pull_parflow_wtab_all.py



import numpy as np
import pandas as pd
import pickle as pk
import os

from parflowio.pyParflowio import PFData
#import pyvista as pv
import parflow.tools as pftools


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.patches as patches
import matplotlib
plt.rcParams['font.size'] = 14





#---------------------------
# Define some variables
#---------------------------

nsteps = 1794

yrs = [2017,2018,2019,2020,2021] # Calender years within timeseries

yr = 2019 # year for plotting

fpath = os.path.join('./','clm_figs_wy{}'.format(yr))
if not os.path.exists(fpath):
    os.makedirs(fpath)


#
# Read in pickles with all data 
#
# Generate using et_to_pickle.py
clm_out = pd.read_pickle('parflow_out/clm_out_dict.pk') # this is from the CLM output files
#et_out  = pd.read_pickle('parflow_out/et_out_dict.pk') # This one does not work as well -- not sure why
# Generate using vel_decomp_2_rech.py
pf = pd.read_pickle('parflow_out/pf_out_dict.pk') # this is from parflow files using pftools
et_out = pf['et']



#
# Read in the topography file
#
topo = np.loadtxt('./Perm_Modeling/elevation.sa', skiprows=1)


#
# Read in the vegetation index file
#
veg = pd.read_csv('./clm_inputs/drv_vegm.dat', skiprows=2, header=None, sep=' ', index_col=0)  
header = ['y','lat','long','sand','clay','color'] + list(np.arange(1,19))
veg.columns = header

# Array with the plant type index
v_type = veg[np.arange(1,19)]
vind = []
vnum = []
for i in range(1, len(v_type)+1):
    v_ind = v_type.columns.astype(float) * v_type.loc[i,:]
    # test how many indices there are
    vnum.append((v_type.loc[i,:] > 0.0).sum())
    vind.append(int(v_ind.max()))




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

dates = pf_2_dates('2016-10-01', '2021-08-29', '24H')
find_date_ind = lambda d: (dates==d).argmax()

# Water year indexes for et_out and clm_out
wy_inds_  = [np.where((dates > '{}-09-30'.format(i-1)) & (dates < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
wy_map_et = np.column_stack((list(et_out.keys()), wy_inds.T))

wy_17_inds = wy_map_et[wy_map_et[:,1] == 2017][:,0]
wy_18_inds = wy_map_et[wy_map_et[:,1] == 2018][:,0]
wy_19_inds = wy_map_et[wy_map_et[:,1] == 2019][:,0]
wy_20_inds = wy_map_et[wy_map_et[:,1] == 2020][:,0]
wy_21_inds = wy_map_et[wy_map_et[:,1] == 2021][:,0]




#
# MET Forcing Data -- note, this is at 3 hours
# Units in parflow manual pg. 136
#

fmet      = './MET/met.2017-2021.3hr.txt'

met       = pd.read_csv(fmet, delim_whitespace=True, names=['rad_s','rad_l','prcp','temp','wnd_u','wnd_v','press','vap'])
tstart    = pd.to_datetime('2016-10-01 00', format='%Y-%m-%d %H')
tend      = pd.to_datetime('2021-08-30 12', format='%Y-%m-%d %H') # Water year 21 is not over yet
hours     = pd.DatetimeIndex(pd.Series(pd.date_range(tstart, tend, freq='3H')))
hours     = hours[~((hours.month == 2) & (hours.day == 29))] # No leap years
met.index = hours

# Chunk into water years
wy_inds_  = [np.where((hours > '{}-09-30'.format(i-1)) & (hours < '{}-10-01'.format(i)), True, False) for i in yrs]
wy_inds   = np.array([wy_inds_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
met['wy'] = wy_inds

# summarize precipitation
# monthly amount (mm) of precipitation
prcp_summ  = pd.DataFrame((met['prcp']*3600*3).groupby(pd.Grouper(freq='M')).sum()) # mm/month
wy_  = [np.where((prcp_summ.index > '{}-09-30'.format(i-1)) & (prcp_summ.index < '{}-10-01'.format(i)), True, False) for i in yrs]  
wy   = np.array([wy_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
prcp_summ['wy'] = wy

# annual precip (mm)
prcp_sumy = prcp_summ.groupby(prcp_summ['wy']).sum()
# annual but cumulative sum of monthly
prcp_sumy_cs  = prcp_summ.groupby(prcp_summ['wy']).cumsum()
prcp_sumy_cs['wy'] = prcp_summ['wy']


# cumulative sum using daily precip sums
prcp_sumd = pd.DataFrame((met['prcp']*3600*3).groupby(pd.Grouper(freq='D')).sum())
wy_  = [np.where((prcp_sumd.index > '{}-09-30'.format(i-1)) & (prcp_sumd.index < '{}-10-01'.format(i)), True, False) for i in yrs]  
wy   = np.array([wy_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
prcp_sumd['wy'] = wy
# annual cumulative
prcp_sumy_cs_  = prcp_sumd.groupby(prcp_sumd['wy']).cumsum()
prcp_sumy_cs_['wy'] = prcp_sumd['wy']







#
# Calculate the sum of the ET over each domain column
# ET at land surface over the full hillslope
#
et_mhr   = np.array([et_out[i].sum(axis=0) for i in range(1,len(et_out)+1)])[:,0,:]
et_mmday = et_mhr*1000*24/3 

# convert to dict to make life easier
et_df    = pd.DataFrame(et_mmday)
et_df.index = dates

wy_daily_ = [np.where((et_df.index > '{}-09-30'.format(i-1)) & (et_df.index < '{}-10-01'.format(i)), True, False) for i in yrs]  
wy_daily  = np.array([wy_daily_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)

et_df.insert(loc=0, column='wy', value=wy_daily)






#----------------------------------
#
# Plotting
#
#----------------------------------


#
# Some helper functions
#
month_map  = {1:'Oct',2:'Nov',3:'Dec',4:'Jan',5:'Feb',6:'Mar',7:'Apr',8:'May',9:'Jun',10:'Jul',11:'Aug',12:'Sep'}
months     = list(month_map.values())
months_num = np.array(list(month_map.keys()))


days_ = prcp_sumd[prcp_sumd['wy']==yr].index
first_month = [(days_.month==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]


wy_inds_helper = {2017:wy_17_inds,2018:wy_18_inds,2019:wy_19_inds,2020:wy_20_inds,2021:wy_21_inds}


pull_wy     = lambda df, wy: pd.DataFrame(df[df['wy']==wy])
pull_prcp   = lambda df, wy: pd.DataFrame(df[df['wy']==wy]['prcp'])
pull_et     = lambda wy: pd.DataFrame([(clm_out[i]['qflx_evap_tot'] + clm_out[i]['qflx_tran_veg']) for i in wy_inds_helper[wy]]).T
pull_infil  = lambda wy: pd.DataFrame([(clm_out[i]['qflx_infl']) for i in wy_inds_helper[wy]]).T
pull_swe    = lambda wy: pd.DataFrame([(clm_out[i]['swe_out']) for i in wy_inds_helper[wy]]).T
pull_grndT  = lambda wy: pd.DataFrame([(clm_out[i]['t_grnd']-273.15) for i in wy_inds_helper[wy]]).T
pull_soilT  = lambda wy: pd.DataFrame([(clm_out[i]['t_soil_0']-273.15) for i in wy_inds_helper[wy]]).T





#
# MET Forcing -- precipitation
#

pp1 = pull_prcp(prcp_summ, 2019)
pp2 = pull_prcp(prcp_summ, 2020)
pp3 = pull_prcp(prcp_sumy_cs, 2019)
pp4 = pull_prcp(prcp_sumy_cs, 2020)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,4))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.2, right=0.96, hspace=0.45)
ax = axes[1]
ax.bar(months_num-0.1, pp1['prcp'].to_numpy(), width=0.18, color='C0', fill='C0', hatch=None,  label='WY2019')
ax.bar(months_num+0.1, pp2['prcp'].to_numpy(), width=0.18, color='C1', fill='C1', hatch='///', label='WY2020')

ax.set_xlim([0.5, 12.5])
ax.set_xticks(list(months_num))
ax.set_xticklabels(labels=months)
ax.tick_params(axis='x', top=False, labelrotation=45, pad=0.2)
ax.set_ylim(0,150)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis='y', which='both', right=True)
ax.grid()

ax.set_ylabel('Monthly\nPrecip. (mm)')

# cumulative precipitation on new axis
ax = axes[0]
#ax.plot(months_num, pull_wy(prcp_sumy_cs, 2019)['prcp'].to_numpy(), color='C0', ls='-',  lw=2.0, label='WY2019')
#ax.plot(months_num, pull_wy(prcp_sumy_cs, 2020)['prcp'].to_numpy(), color='C1', ls='--', lw=2.0,  label='WY2020')
ax.plot([0]+list(months_num), [0]+pp3['prcp'].to_list(), color='C0', ls='-',  lw=2.0,  label='WY2019')
ax.plot([0]+list(months_num), [0]+pp4['prcp'].to_list(), color='C1', ls='--', lw=2.0,  label='WY2020')

ax.set_xlim([0.0, 12.0])
ax.set_xticks([0]+list(months_num))
ax.set_xticklabels(labels=months+[''])
ax.tick_params(axis='x', top=False, labelrotation=45, pad=0.2)
#ax.set_ylim(0,150)
ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='y', which='both', right=True)
ax.grid()

ax.set_ylabel('Annual\nCumulative (mm)')
ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)

#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'precip.png'),dpi=300)
plt.show()







#--------------------------------------------
#
# CLM versus time plots
#
#--------------------------------------------

drng = np.arange(len(wy_inds_helper[yr]))

#
# CLM ET
#
et1 = pull_et(2019).mean(axis=0)*86400 / 3
et2 = pull_et(2020).mean(axis=0)*86400 / 3

et1 = pull_wy(et_df, 2019).iloc[:,1:].mean(axis=1)
et2 = pull_wy(et_df, 2020).iloc[:,1:].mean(axis=1)

ppd1 = pull_prcp(prcp_sumd,2019)['prcp'].to_numpy().cumsum()
ppd2 = pull_prcp(prcp_sumd,2020)['prcp'].to_numpy().cumsum()

### 
# Plot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,4))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.2, right=0.96, hspace=0.45)
# ET
ax = axes[1]
ax.plot(drng, et1, color='C0', label='WY2019')
ax.plot(drng, et2, color='C1', label='WY2020')
# Cleanup
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_ylabel('Daily ET\n(mm)')
ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# Cumulative Sums
ax = axes[0]
# Precip
ax.plot(np.arange(len(ppd1)), ppd1, color='C0', linestyle='--', lw=1.5, label='Precip.')
ax.plot(np.arange(len(ppd2)), ppd2, color='C1', linestyle='--', lw=1.5)
# ET
ax.plot(np.arange(len(et1)), (et1).cumsum(), color='C0', lw=2.0, label='ET')
ax.plot(np.arange(len(et2)), (et2).cumsum(), color='C1', lw=2.0)
# Cleanup
ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.set_ylabel('Annual\nCumulative (mm)')
ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# 
[axes[i].set_xlim(0,366) for i in [0,1]]
[axes[i].set_xticks(first_month) for i in [0,1]]
[axes[i].set_xticklabels(labels=months) for i in [0,1]]
[axes[i].tick_params(axis='x', top=True, labelrotation=45, pad=0.2) for i in [0,1]]
[axes[i].tick_params(axis='y', which='both', right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_et_temporal.png'),dpi=300)
plt.show()





#
# CLM infiltration
#
inf1 = pull_infil(2019).mean(axis=0)*86400 / 3
inf2 = pull_infil(2020).mean(axis=0)*86400 / 3

###
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,4))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.2, right=0.96, hspace=0.45)
# Infiltration
ax = axes[1]
ax.plot(np.arange(len(inf1)), inf1, color='C0', label='WY2019')
ax.plot(np.arange(len(inf2)), inf2, color='C1', label='WY2020')
# Cleanup
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_ylabel('Daily\nInfiltration (mm)')
ax.legend(loc='upper left',handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# Cumulative Sum
ax = axes[0]
# Precip
ax.plot(np.arange(len(ppd1)), ppd1, color='C0', linestyle='--', lw=1.5, label='Precip.')
ax.plot(np.arange(len(ppd2)), ppd2, color='C1', linestyle='--', lw=1.5)
# Infiltration
ax.plot(np.arange(len(inf1)), inf1.cumsum(), color='C0', lw=2.0, label='Infil.')
ax.plot(np.arange(len(inf2)), inf2.cumsum(), color='C1', lw=2.0)
# Cleanup
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.set_ylabel('Annual\nCumulative (mm)')
ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
#
[axes[i].set_xlim(0,366) for i in [0,1]]
[axes[i].set_xticks(first_month) for i in [0,1]]
[axes[i].set_xticklabels(labels=months) for i in [0,1]]
[axes[i].tick_params(axis='x', top=True, labelrotation=45, pad=0.2) for i in [0,1]]
[axes[i].tick_params(axis='y', which='both', right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_infil_temporal.png'),dpi=300)
plt.show()




#
# CLM rech = infil. - ET
#
rech1 = inf1.to_numpy() - et1.to_numpy()
rech2 = inf2.to_numpy() - et2.to_numpy()

###
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,4))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.2, right=0.96, hspace=0.45)
# Recharge
ax = axes[1]
ax.plot(np.arange(len(rech1)), rech1, color='C0', label='WY2019')
ax.plot(np.arange(len(rech2)), rech2, color='C1', label='WY2020')
# Cleanup
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_ylabel('Daily\nRecharge (mm)')
ax.legend(loc='upper left', handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# Cumulative Sum 
ax = axes[0]
# Precip
ax.plot(np.arange(len(ppd1)), ppd1, color='C0', linestyle='--', lw=1.5, label='Precip.')
ax.plot(np.arange(len(ppd2)), ppd2, color='C1', linestyle='--', lw=1.5)
# Recharge
ax.plot(np.arange(len(rech1)), rech1.cumsum(), color='C0', lw=2.0, label='Recharge')
ax.plot(np.arange(len(rech2)), rech2.cumsum(), color='C1', lw=2.0)
# Cleanup
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
ax.set_ylabel('Annual\nCumulative (mm)')
leg = ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# 
[axes[i].set_xlim(0,366) for i in [0,1]]
[axes[i].set_xticks(first_month) for i in [0,1]]
[axes[i].set_xticklabels(labels=months) for i in [0,1]]
[axes[i].tick_params(axis='x', top=True, labelrotation=45, pad=0.2) for i in [0,1]]
[axes[i].tick_params(axis='y', which='both', right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_rech_temporal.png'),dpi=300)
plt.show()




#
# CLM SWE
#
def accum_swe(wy):
    swe_accum = 0
    swe_accum_list = []
    swe = pull_swe(wy).mean(axis=0)
    for i in range(365-1):
        if (swe[i+1]-swe[i]) > 0.0:
            swe_accum += swe[i+1]-swe[i]
        swe_accum_list.append(swe_accum)
    return swe_accum_list
        


swe1 = pull_swe(2019).mean(axis=0)
swe2 = pull_swe(2020).mean(axis=0)

swe1a = accum_swe(2019)
swe2a = accum_swe(2020)

###
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,4))
fig.subplots_adjust(bottom=0.14, top=0.92, left=0.2, right=0.96, hspace=0.45)
ax = axes[1]
# SWE
ax.plot(np.arange(len(swe1)), swe1, color='C0', label='WY2019')
ax.plot(np.arange(len(swe2)), swe2, color='C1', label='WY2020')
# Precip
#ax.plot(pull_wy(prcp_sumd,2019)['prcp'].to_numpy())
# Cleanup
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.set_ylabel('SWE (mm)')
ax.legend(loc='upper left', handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
# Cumulative Sum 
ax = axes[0]
# Precip
ax.plot(np.arange(len(ppd1)), ppd1, color='C0', linestyle='--', lw=1.5, label='Precip.')
ax.plot(np.arange(len(ppd2)), ppd2, color='C1', linestyle='--', lw=1.5)
# SWE
ax.plot(np.arange(len(swe1a)), swe1a, color='C0', lw=2.0, label='SWE')
ax.plot(np.arange(len(swe2a)), swe2a, color='C1', lw=2.0)
# Cleanup
ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.set_ylabel('Annual\nCumulative (mm)')
leg = ax.legend(handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
#
[axes[i].set_xlim(0,366) for i in [0,1]]
[axes[i].set_xticks(first_month) for i in [0,1]]
[axes[i].set_xticklabels(labels=months) for i in [0,1]]
[axes[i].tick_params(axis='x', top=True, labelrotation=45, pad=0.2) for i in [0,1]]
[axes[i].tick_params(axis='y', which='both', right=True) for i in [0,1]]
[axes[i].grid() for i in [0,1]]
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_swe_temporal.png'),dpi=300)
plt.show()






#
# Stacked Daily Values
#

prcp_daily_2019 = pull_prcp(prcp_sumd, 2019)
prcp_daily_2020 = pull_prcp(prcp_sumd, 2020)


vlist1  = [prcp_daily_2019, swe1, et1, inf1, rech1]
vlist2  = [prcp_daily_2020, swe2, et2, inf2, rech2]
vlist1_ = ['Precipitation\n(mm/day)', 'SWE (mm)', 'ET\n(mm/day)', 'Infiltration\n(mm/day)', 'Recharge\n(mm/day)']


fig, axes = plt.subplots(nrows=len(vlist1), ncols=1, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.93, left=0.2, right=0.98, hspace=0.65)
# Plot each timeseries as own subuplot
for i in range(len(vlist1)):
    ax = axes[i]
    ax.plot(np.arange(len(vlist1[i])), vlist1[i], lw=1.5, color='C0', label='WY2019')
    ax.plot(np.arange(len(vlist2[i])), vlist2[i], lw=1.5, color='C1', label='WY2020')
    # Cleanup
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    ax.tick_params(axis='y', which='both', right=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylabel(vlist1_[i])
    ax.grid()
axes[0].legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.27), handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_daily_stack.png'),dpi=300)
plt.show()





#
# Stacked Annual Values
#

#vlist1  = [prcp_daily_2019.cumsum(), swe1a, et1.cumsum(), inf1.cumsum(), rech1.cumsum()]
#vlist2  = [prcp_daily_2020.cumsum(), swe2a, et2.cumsum(), inf2.cumsum(), rech2.cumsum()]
#vlist1_ = ['Precipitation\n(mm/year)', 'SWE (mm)', 'ET\n(mm/year)', 'Infiltration\n(mm/year)', 'Recharge\n(mm/year)']


vlist1  = [swe1a, et1.cumsum(), inf1.cumsum(), rech1.cumsum()]
vlist2  = [swe2a, et2.cumsum(), inf2.cumsum(), rech2.cumsum()]
vlist1_ = ['SWE (mm)', 'ET\n(mm/year)', 'Infiltration\n(mm/year)', 'Recharge\n(mm/year)']

fig, axes = plt.subplots(nrows=len(vlist1)+1, ncols=1, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.93, left=0.2, right=0.98, hspace=0.65)
# Monthly Precip on Top
ax = axes[0]
ax.bar(months_num-0.1, pp1['prcp'].to_numpy(), width=0.18, color='C0', fill='C0', hatch=None,  label='WY2019')
ax.bar(months_num+0.1, pp2['prcp'].to_numpy(), width=0.18, color='C1', fill='C1', hatch=None, label='WY2020')
ax.set_xlim([0.5, 12.5])
ax.set_xticks(list(months_num))
ax.set_xticklabels(labels=months)
ax.tick_params(axis='x', top=False, labelrotation=45, pad=0.001)
ax.set_ylim(0,150)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis='y', which='both', right=True)
ax.grid()
ax.set_ylabel('Monthly\nPrecip. (mm)')

# Plot each timeseries as own subuplot
for i in range(len(vlist1)):
    ax = axes[i+1]
    ax.plot(np.arange(len(vlist1[i])), vlist1[i], lw=2.0, color='C0')#, label='WY2019' if i==0 else '')
    ax.plot(np.arange(len(vlist2[i])), vlist2[i], lw=2.0, color='C1')#, label='WY2020' if i==0 else '')
    # Add Precipitation
    ax.plot(np.arange(len(prcp_daily_2019.cumsum())), prcp_daily_2019.cumsum(), lw=1.5, ls='--', color='C0', label='Precip.')
    ax.plot(np.arange(len(prcp_daily_2020.cumsum())), prcp_daily_2020.cumsum(), lw=1.5, ls='--', color='C1')
    # Cleanup
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    ax.tick_params(axis='y', which='both', right=True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.set_ylabel(vlist1_[i])
    ax.grid()
axes[0].legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.27), handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
axes[1].legend(loc='upper left', handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_annual_stack.png'),dpi=300)
plt.show()














#--------------------------------------------------------------------
#
# Hillslope Plots
#
#--------------------------------------------------------------------

xrng = np.arange(len(topo))*1.5125


# Colors for vegetation index
min_val, max_val = 0.4,0.8
n = 5
orig_cmap = plt.cm.gist_earth
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap_cust = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)


#    
# Timeseries at Discrete Points
#
x_pnts = [100, 300, 400, 500, 540]    
#cc = plt.cm.viridis(np.linspace(0,1,len(x_pnts)))
#cc = plt.cm.coolwarm(np.linspace(0,1,len(x_pnts)))
cc = plt.cm.turbo(np.linspace(0,1,len(x_pnts)))


# Plot the positions
fig, ax = plt.subplots(figsize=(6,2))
ax.scatter(xrng, topo, ls='-', marker=None, s=1.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
# Add points
for j in range(len(x_pnts)):
    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.minorticks_on()
ax.tick_params(axis='x', which='both', top=True, bottom=True)
ax.tick_params(axis='y', which='both', left=True, right=True)
ax.set_xlim(0, xrng.max())
ax.set_xlabel('Distance (m)')
fig.tight_layout()
plt.savefig(os.path.join(fpath,'X_points_locs.png'), dpi=300)
plt.show()



#--------------------------------------------------------
#
# ET
#
#--------------------------------------------------------
et = pull_et(yr)*86400 / 3
et.columns = np.arange(365)

dd = et.copy()

# Testing with parflow ET, rather than CLM ET
et_ = pull_wy(et_df, yr).iloc[:,1:]
et_.index = np.arange(365)
et_ = et_.T

dd = et_.copy()



"""
#
# Monthly Plots
#
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,10))
fig.subplots_adjust(top=0.96, bottom=0.10, right=0.98, left=0.2, hspace=0.2, wspace=0.40)
for i in range(len(first_month)):
    r,c = i%6,i//6
    ax = axes[r,c]
    ii = first_month[i]
    ax.plot(xrng, dd.loc[:,ii], lw=1.0, alpha=1.0, zorder=7, color='black', label=months[i])
    ax.text(0.05, 0.85, '{}'.format(months[i]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    # Plot the full ensemble also?
    ax.tick_params(axis='y', rotation=45, pad=0.1)
    if r != 5:
        ax.tick_params(axis='x', labelbottom=False)
    ax.minorticks_on()
fig.text(0.55, 0.04, 'Distance (m)', ha='center')
fig.text(0.04, 0.5, 'ET (mm/day)', va='center', rotation='vertical')
fig.text(0.55, 0.97, 'WY{}'.format(yr), ha='center')
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_et_spat_mn.png'),dpi=300)
plt.show()
"""
    
    




#--------------------------------------------------------
#
# Infiltration
#
#--------------------------------------------------------
inf = pull_infil(yr)*86400 / 3
inf.columns = np.arange(365)

dd = inf.copy()


"""
#
# Monthly Plots
#
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,10))
fig.subplots_adjust(top=0.96, bottom=0.10, right=0.98, left=0.2, hspace=0.2, wspace=0.40)
for i in range(len(first_month)):
    r,c = i%6,i//6
    ax = axes[r,c]
    ii = first_month[i]
    ax.plot(xrng, dd.loc[:,ii], lw=1.0, alpha=1.0, zorder=7, color='black', label=months[i])
    ax.text(0.05, 0.15, '{}-01'.format(months[i]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=45, pad=0.1)
    if r != 5:
        ax.tick_params(axis='x', labelbottom=False)
    ax.minorticks_on()
fig.text(0.55, 0.04, 'Distance (m)', ha='center')
fig.text(0.04, 0.5,  'Daily Infil (mm/day)', va='center', rotation='vertical')
fig.text(0.55, 0.97, 'WY{}'.format(yr), ha='center')
plt.savefig(os.path.join(fpath, 'clm_infil_spat_mn.png'),dpi=300)
#fig.tight_layout()
plt.show()
"""
  




#--------------------------------------------------------
#
# Rech = Infiltration - ET
#
#--------------------------------------------------------
inf = pull_infil(yr)*86400/3
inf.columns = np.arange(365)

et = pull_et(yr)*86400/3
et.columns = np.arange(365)

rech = inf - et

dd = rech.copy()





    
#--------------------------------------------------------
#
# SWE
#
#--------------------------------------------------------
swe = pull_swe(yr)
swe.columns = np.arange(365)

dd = swe.copy()

atemp     = pull_wy(met, yr)['temp'].groupby(pd.Grouper(freq='D')).mean()- 273.15
atemp_max = pull_wy(met, yr)['temp'].groupby(pd.Grouper(freq='D')).max() - 273.15


"""
#
# Monthly Plots
#
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,10))
fig.subplots_adjust(top=0.96, bottom=0.10, right=0.98, left=0.2, hspace=0.2, wspace=0.40)
for i in range(len(first_month)):
    r,c = i%6,i//6
    ax = axes[r,c]
    ii = first_month[i]
    ax.plot(xrng, dd.loc[:,ii], lw=1.0, alpha=1.0, zorder=7, color='black', label=months[i])
    ax.text(0.05, 0.30, '{}-01'.format(months[i]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=45, pad=0.1)
    if r != 5:
        ax.tick_params(axis='x', labelbottom=False)
    ax.minorticks_on()
fig.text(0.55, 0.04, 'Distance (m)', ha='center')
fig.text(0.04, 0.5, 'Daily SWE (mm)', va='center', rotation='vertical')
fig.text(0.55, 0.97, 'WY{}'.format(yr), ha='center')
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_swe_spat_mn.png'),dpi=300)
plt.show()
"""






#--------------------------------------------------------
#
# Ground Temperature
#
#--------------------------------------------------------

gtemp = pull_grndT(yr)
gtemp.columns = np.arange(365)

dd = gtemp.copy()


"""
#
# Monthly Plots
#
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,10))
fig.subplots_adjust(top=0.96, bottom=0.10, right=0.98, left=0.2, hspace=0.2, wspace=0.40)
for i in range(len(first_month)):
    r,c = i%6,i//6
    ax = axes[r,c]
    ii = first_month[i]
    ax.plot(xrng, dd.loc[:,ii], lw=1.0, alpha=1.0, zorder=7, color='black', label=months[i])
    ax.text(0.05, 0.30, '{}-01'.format(months[i]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=45, pad=0.1)
    if r != 5:
        ax.tick_params(axis='x', labelbottom=False)
    ax.minorticks_on()
fig.text(0.55, 0.04, 'Distance (m)', ha='center')
fig.text(0.04, 0.5, 'Daily Ground Temp (C)', va='center', rotation='vertical')
fig.text(0.55, 0.97, 'WY{}'.format(yr), ha='center')
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_gtemp_spat_mn.png'),dpi=300)
plt.show()
"""






#--------------------------------------------------------
#
# Soil Temperature
#
#--------------------------------------------------------
stemp = pull_soilT(yr)
stemp.columns = np.arange(365)

dd = stemp.copy()

"""
#
# Monthly Plots
#
fig, axes = plt.subplots(nrows=6,ncols=2, figsize=(6,10))
fig.subplots_adjust(top=0.96, bottom=0.10, right=0.98, left=0.2, hspace=0.2, wspace=0.40)
for i in range(len(first_month)):
    r,c = i%6,i//6
    ax = axes[r,c]
    ii = first_month[i]
    ax.plot(xrng, dd.loc[:,ii], lw=1.0, alpha=1.0, zorder=7, color='black', label=months[i])
    ax.text(0.05, 0.30, '{}-01'.format(months[i]), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.tick_params(axis='y', rotation=45, pad=0.1)
    if r != 5:
        ax.tick_params(axis='x', labelbottom=False)
    ax.minorticks_on()
fig.text(0.55, 0.04, 'Distance (m)', ha='center')
fig.text(0.04, 0.5, 'Daily Soil Temp (C)', va='center', rotation='vertical')
fig.text(0.55, 0.97, 'WY{}'.format(yr), ha='center')
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_stemp_spat_mn.png'),dpi=300)
plt.show()
"""











#
# Daily Stacked Spatial Plots -- Round 1
#
vlist  = [swe, et, inf, rech]
vlist_ = ['SWE (mm)', 'ET\n(mm/day)', 'Infiltration\n(mm/day)', 'Recharge\n(mm/day)']


fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,7))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.93, left=0.25, right=0.98, hspace=0.35)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    ax.plot(xrng, dd.mean(axis=1),      color='black', lw=2.0, zorder=8)
    ax.plot(xrng, np.median(dd,axis=1), color='black', lw=2.0, ls='--', zorder=8)
    # Plot the max and min
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmax()], lw=2.0, color='black', ls=':', zorder=8)
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmin()], lw=2.0, color='black', ls=':', zorder=8)
    # Plot full ensemble of timesteps
    for j in range(0,dd.shape[1],1):
        ax.plot(xrng, dd.iloc[:,j], color='grey', alpha=0.3, lw=0.5)
    ax.set_ylabel(vlist_[i])
    ax.minorticks_on()
    ax.set_xlim(0, xrng.max())
    ax.tick_params(axis='y', which='both', right=True)
    ax.tick_params(axis='x', which='both', top=True)
#
# Topography profile 
ax = axes[4]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)')
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=True)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_daily_stack_spatial.png'),dpi=300)
plt.show()


#
# Daily Stacked Timeseries -- Round 1
#
fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,7))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.96, left=0.2, right=0.98, hspace=0.65)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    for j in range(len(x_pnts)):
        ax.plot(np.arange(dd.shape[1]), dd.loc[x_pnts[j],:], color=cc[j], label='{:.0f}'.format(x_pnts[j]*1.5125))
    ax.set_ylabel(vlist_[i])
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    ax.tick_params(axis='y', which='both', right=True)
    ax.grid()
#
# Topography profile 
ax = axes[4]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
# Plot the positions
for j in range(len(x_pnts)):
    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)', labelpad=0.01)
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=False)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_daily_stack_temporal.png'),dpi=300)
plt.show()









#
# Daily Stacked Spatial Plots -- Round 2
#
atemp_ = pd.DataFrame(np.array(atemp) * np.ones_like(gtemp))

vlist  = [atemp_, gtemp, stemp]
vlist_ = ['Air Temp.\n(C)', 'Ground Temp.\n(C)', 'Soil Temp.\n(C)']


fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,6))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.1, top=0.93, left=0.25, right=0.98, hspace=0.35)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    ax.plot(xrng, dd.mean(axis=1),      color='black', lw=2.0, zorder=8)
    ax.plot(xrng, np.median(dd,axis=1), color='black', lw=2.0, ls='--', zorder=8)
    # Plot the max and min
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmax()], lw=2.0, color='black', ls=':', zorder=8)
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmin()], lw=2.0, color='black', ls=':', zorder=8)
    # Plot full ensemble of timesteps
    for j in range(0,dd.shape[1],1):
        ax.plot(xrng, dd.iloc[:,j], color='grey', alpha=0.3, lw=0.5)
    ax.set_ylabel(vlist_[i])
    ax.minorticks_on()
    ax.set_xlim(0, xrng.max())
    ax.tick_params(axis='y', which='both', right=True)
    ax.tick_params(axis='x', which='both', top=True)
#
# Topography profile 
ax = axes[3]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)')
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=True)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_daily_stack_spatial_temps.png'),dpi=300)
plt.show()



#
# Daily Stacked Timeseries -- Round 2
#
fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,6))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.1, top=0.96, left=0.2, right=0.98, hspace=0.65)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    for j in range(len(x_pnts)):
        ax.plot(np.arange(dd.shape[1]), dd.loc[x_pnts[j],:], color=cc[j], label='{:.0f}'.format(x_pnts[j]*1.5125))
    ax.set_ylabel(vlist_[i])
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    ax.tick_params(axis='y', which='both', right=True)
    ax.grid()
#
# Topography profile 
ax = axes[3]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
# Plot the positions
for j in range(len(x_pnts)):
    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)', labelpad=0.01)
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=False)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_daily_stack_temporal_temps.png'),dpi=300)
plt.show()








#
# Annual Stacked spatial -- Round 1
#
vlist  = [swe, et, inf, rech]
vlist_ = ['April 1st\nSWE (mm)', 'ET\n(mm/year)', 'Infiltration\n(mm/year)', 'Recharge\n(mm/year)']

diff_ = swe.diff(axis=1)
diff_[diff_ < 0.0] = 0.0
swe_acc = diff_.cumsum(axis=1)


fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,7))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.93, left=0.25, right=0.98, hspace=0.35)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    if i == 0:
        ax.plot(xrng, dd.loc[:,first_month[6]], color='black', lw=2.0) # for swe plot april 1st
        ax.axhline(dd.loc[:,first_month[6]].mean(), color='black', ls='--')
    else:
        ax.plot(xrng, dd.sum(axis=1), color='black', lw=2.0)
        ax.axhline((dd.sum(axis=1)).mean(), color='black', ls='--')
    ax.set_ylabel(vlist_[i])
    ax.minorticks_on()
    ax.set_xlim(0, xrng.max())
    ax.tick_params(axis='y', which='both', right=True)
    ax.tick_params(axis='x', which='both', top=True)
#
# Topography profile 
ax = axes[4]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)')
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=True)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_annual_stack_spatial.png'),dpi=300)
plt.show()




#
# Annual Stacked Timeseries -- Round 1
#
vlist_ = ['SWE\n(mm/year)', 'ET\n(mm/year)', 'Infiltration\n(mm/year)', 'Recharge\n(mm/year)']

fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, figsize=(5,7))
#fig, axes = plt.subplots(nrows=len(vlist)+1, ncols=1, gridspec_kw={'height_ratios': 4*[1]+[0.5]}, figsize=(5,7))
fig.subplots_adjust(bottom=0.08, top=0.96, left=0.2, right=0.98, hspace=0.65)
for i in range(len(vlist)):
    ax = axes[i]
    dd = vlist[i].copy()
    ax.plot(np.arange(len(ppd1)), ppd1, color='black', linestyle='--', lw=2.0, label='Precip.')
    for j in range(len(x_pnts)):
        if i == 0:
            dd_ = swe_acc.loc[x_pnts[j],:]
        else:
            dd_ = dd.loc[x_pnts[j],:].cumsum()
        ax.plot(np.arange(len(dd_)), dd_, color=cc[j], label='{:.0f}'.format(x_pnts[j]*1.5125))
    ax.set_ylabel(vlist_[i])
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    ax.tick_params(axis='y', which='both', right=True)
    ax.grid()
#
# Topography profile 
ax = axes[4]
ax.scatter(xrng, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
# Plot the positions
for j in range(len(x_pnts)):
    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)', labelpad=0.01)
ax.set_xlim(0, xrng.max())
ax.set_ylim(2750, 2950)
ax.tick_params(axis='y', which='both', right=True)
ax.tick_params(axis='x', which='both', top=False)
ax.minorticks_on()
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
#
# Cleanup
axes[0].set_title('WY{}'.format(yr), fontsize=14)
plt.savefig(os.path.join(fpath, 'clm_annual_stack_temporal.png'),dpi=300)
plt.show()











#------------------------------------------------------------
#
# Water Tables and Saturation and Storage
#
#------------------------------------------------------------

# Read in Raster Fields
prs = pd.read_pickle('press_out_dict.pk')
sat = pd.read_pickle('sat_out_dict.pk')

# Shape for pressure and saturatio field is 32 rows by 559 columns
#   sat[0,:] is the bottom layer of the domain, sat[31,:] is land surface
#   sat[31,0] is top left corner, sat[31,558] is top right corner (stream)


# Parflow Domain 
# these are cell centered values, Z_layer_num corresponds to rows in sat and prs
z_info = pd.read_csv('../utils/plm_grid_info_v3b.csv') 

#
# Using the Parflow pftools scipt
wtd = pd.DataFrame(pf['wtd']).T[1:]
wtd.index = dates
wtd.insert(loc=0, column='wy', value=wy_daily)




#
# Plot Water Table Depths
#
wt = pull_wy(wtd, yr).iloc[:,1:]

colors = plt.cm.twilight(np.linspace(0,1,len(months)+1))

fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2,1]}, figsize=(6,3.7))
fig.subplots_adjust(top=0.90, bottom=0.15, left=0.2, right=0.78, hspace=0.4)
ax = axes[0]
# Plot all timesteps
for i in range(len(wt)):
    ax.plot(np.arange(wt.shape[1])*1.5125, wt.iloc[i,:], color='grey', alpha=0.3)
# Plot the first of the months
for j in range(len(first_month)):
    jj = first_month[j]
    ax.plot(np.arange(wt.shape[1])*1.5125, wt.iloc[jj,:], color=colors[j], alpha=1.0, label='{}'.format(months[j]))
ax.set_ylabel('Water Table\nDepth (m)', labelpad=0.1)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), fontsize=12.5, handlelength=1, labelspacing=0.25)
ax.set_title('WY{}'.format(yr), fontsize=14)
ax.invert_yaxis()
#
ax = axes[1]
ax.scatter(np.arange(len(topo))*1.5125, topo, ls='-', marker=None, s=2.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
ax.set_ylabel('Z (m)')
ax.set_xlabel('Distance (m)')
# 
[axes[i].tick_params(axis='x', which='both', top=True, bottom=True) for i in [0,1]]
[axes[i].tick_params(axis='y', which='both', left=True, right=True) for i in [0,1]]
[axes[i].minorticks_on() for i in [0,1]]
[axes[i].set_xlim(0,xrng.max()) for i in [0,1]]
plt.savefig(os.path.join(fpath, 'wt_spatial.png'),dpi=300)
plt.show()




#
# Find saturation at various depths below land surface
#

# pick depths below land surface from z_info
zinds    = [31,  28,  24,  20]
zbls     = z_info['Depth_bls'][zinds].to_list()
zbls_map = dict(zip(['d0','d1','d2','d3'], zbls))

sat_dict = {}
for j in range(len(zinds)):
    sat_df   = pd.DataFrame()
    for i in list(sat.keys()):
        dd = sat[i].copy()
        # pull saturation values at constant depth below land surface
        dd_  = dd[zinds[j],:]
        sat_df[i] = dd_
    sat_df = sat_df.iloc[:,1:].T
    sat_df.index = dates
    sat_df.insert(loc=0, column='wy', value=wy_daily)
    sat_dict[list(zbls_map.keys())[j]] = sat_df


#
# Plot saturation at depths
# 

# Saturation Plots through space
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6,6))
fig.subplots_adjust(top=0.92, bottom=0.1, left=0.2, right=0.78, hspace=0.4)
for i in range(len(zinds)):
    d  = list(zbls_map.keys())[i]
    d_ = zbls_map[d]
    ss = pull_wy(sat_dict[d], yr).iloc[:,1:]
    #
    ax = axes[i]
    # Plot all timesteps
    for i in range(len(ss)):
        ax.plot(np.arange(ss.shape[1])*1.5125, ss.iloc[i,:], color='grey', alpha=0.3)
    # Plot the first of the months
    for j in range(len(first_month)):
        jj = first_month[j]
        ax.plot(np.arange(ss.shape[1])*1.5125, ss.iloc[jj,:], color=colors[j], alpha=1.0, label='{}'.format(months[j]))
    ax.text(0.92, 0.13, '{:.2f}'.format(d_), horizontalalignment='center', verticalalignment='center', 
            transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='none'), fontsize=12.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, xrng.max())
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='both', top=True, bottom=True)
    ax.tick_params(axis='y', which='both', left=True, right=True)
    
axes[0].legend(loc='upper left', bbox_to_anchor=(0.99, 1.05), fontsize=12.5, handlelength=1, labelspacing=0.25)
axes[3].set_xlabel('Distance (m)')
axes[0].set_title('WY{}'.format(yr), fontsize=14)
fig.text(0.1, 0.5, 'Saturation (-)', va='center', rotation='vertical')
plt.savefig(os.path.join(fpath, 'sat_spatial.png'),dpi=300)
plt.show()




#
# Saturation Plots through Time
#
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5,5))
fig.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.86, hspace=0.3)

for i in range(len(zinds)):
    d  = list(zbls_map.keys())[i]
    d_ = zbls_map[d]
    ss = pull_wy(sat_dict[d], yr).iloc[:,1:]
    
    ax = axes[i]                
    for j in range(len(x_pnts)):
        ax.plot(np.arange(len(ss[x_pnts[j]])), ss[x_pnts[j]], color=cc[j], lw=2.0, label='{:.0f}'.format(x_pnts[j]*1.5125))
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.5)
    ax.tick_params(axis='y', which='both', right=True)
    ax.text(1.11, 0.5, '{:.2f}'.format(d_), horizontalalignment='center', verticalalignment='center', 
            transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='grey', alpha=0.6), fontsize=12.5)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    #ax.minorticks_on()
    #ax.grid()
[axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator()) for i in range(4)]
[axes[i].tick_params(axis='x', labelbottom=False) for i in [0,1,2]]
#axes[0].legend(loc='upper left', bbox_to_anchor=(0.99, 1.05), fontsize=12.5, handlelength=1, labelspacing=0.25)
axes[0].set_title('WY{}'.format(yr), fontsize=14)
fig.text(0.01, 0.5, 'Saturation (-)', va='center', rotation='vertical')
plt.savefig(os.path.join(fpath, 'sat_temporal_pnt.png'),dpi=300)
plt.show()





#
# Spatial saturation versus ET
# 

wtd_ = pull_wy(wtd, yr).iloc[:,1:]
et  = pull_et(2019).T * 86400/3

fig, ax = plt.subplots()
#ax.scatter(et.iloc[0,:], wtd.iloc[0,:], marker='.', c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
ax.scatter(et.iloc[0,:], wtd_.iloc[0,:], marker='.', c=topo, cmap='coolwarm')
ax.set_xlabel('ET (mm/day)')
ax.set_ylabel('WTD (m)')
ax.set_yscale('log')
ax.set_xscale('log')
fig.tight_layout()
plt.show()







#---------------------------------------
#
# Debug using top right of domain
#
#---------------------------------------

et   = pull_et(2019) * 86400

inf  = pull_infil(2019) * 86400
inf.columns = np.arange(inf.shape[1])

prcp = pull_prcp(prcp_sumd, 2019)


#    
# Timeseries at Discrete Points
#
#x_pnts = [5, 15, 25, 50, 100]    
x_pnts = [5,25,50] 
cc = plt.cm.viridis(np.linspace(0, 0.8, len(x_pnts)))


# Plot the positions
fig, ax = plt.subplots(figsize=(6,2))
ax.scatter(xrng, topo, ls='-', marker=None, s=1.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
for j in range(len(x_pnts)):
    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.minorticks_on()
ax.tick_params(axis='x', which='both', top=True, bottom=True)
ax.set_xlabel('Distance (m)')
fig.tight_layout()
plt.savefig(os.path.join(fpath,'X_points_locs.debug.png'), dpi=300)
plt.show()



#
# Compare infiltration and ET and precipitation
#
dd  = inf.copy() / 3
dd2 = et.copy() / 3

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6,6))
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.18, right=0.78, hspace=0.5)
# Daily Values
ax  = axes[1]
for j in range(len(x_pnts)):
    ax.plot(dd.loc[x_pnts[j],:],  color=cc[j],  label=x_pnts[j]*1.5125)
ax.set_ylabel('Daily Infil.\n(mm/day)')
#
ax = axes[2]
ax.set_ylabel('Daily ET\n(mm/day)')
for j in range(len(x_pnts)):
    ax.plot(dd2.loc[x_pnts[j],:],  color=cc[j], ls='--', label=x_pnts[j]*1.5125)
for i in [1,2]:
    ax = axes[i]
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', top=True, labelrotation=45)
    ax.tick_params(axis='y', which='both', right=True)
    ax.grid()
#
# Annual cumulative sum 
ax = axes[0]
ax.set_title('WY{}'.format(yr), fontsize=14)
# Precip
ax.plot(np.arange(len(ppd1)), ppd1, color='black', linestyle='--', lw=2.0, label='Precip.')
# ET
for j in range(len(x_pnts)):
    dd_ = dd.loc[x_pnts[j],:].cumsum()
    ax.plot(np.arange(len(dd_)), dd_, color=cc[j], label='{:.0f} m '.format(x_pnts[j]*1.5125))
    #
    dd2_ = dd2.loc[x_pnts[j],:].cumsum()
    ax.plot(np.arange(len(dd2_)), dd2_, color=cc[j], ls='--')#, label='{:.0f}'.format(x_pnts[j]*1.5125))
    #
    dd_sum = dd_+dd2_
    ax.plot(np.arange(len(dd_sum)), dd_sum, color=cc[j], ls=':', lw=2.5, label='sum' if j==len(x_pnts)-1 else '')
# Cleanup
ax.set_xticks(first_month)
ax.set_xticklabels(labels=months)
ax.set_xlim(0,366)
ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.5)
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='y', which='both', right=True)
ax.set_ylabel('Annual\nCumulative (mm)')
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.1),fontsize=12.5, handlelength=1, labelspacing=0.25)
ax.grid()
[axes[i].yaxis.set_minor_locator(ticker.AutoMinorLocator()) for i in range(3)]
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_temporal_pnt.debug.png'),dpi=300)
plt.show()










