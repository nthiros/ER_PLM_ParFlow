# Cleanup on 09/29/2022
# Makes a bunch of plots of the ParFLOW-CLM land surface dynamics
# 
# Need to run these scripts first
# Run these on the server where all the output files are located
# -- et_to_pickle.py
# -- vel_decomp_2_rech.py



import numpy as np
import pandas as pd
import pickle as pk
import os
import pdb

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
#
# Define some variables
#
#---------------------------
nsteps = 1794  # Daily Outputs

yrs = [2017,2018,2019,2020,2021] # Calender years within timeseries

yr = 2020 # year for plotting

fpath = os.path.join('./','clm_figs_wy{}'.format(yr))
if not os.path.exists(fpath):
    os.makedirs(fpath)



#---------------------------
#
# Read in data
#
#---------------------------
#
# Read in pickles with all data 
#
# Generated using 'et_to_pickle.py'
clm_out_ = pd.read_pickle('parflow_out/clm_out_dict.pk') # this is from the CLM output files
# Reset timestep keys to start with zero index
clm_out = {i: v for i, v in enumerate(clm_out_.values())}
clm_keys = list(clm_out[0].keys())

pf = pd.read_pickle('parflow_out/pf_out_dict.pk') # this is from parflow files using pftools


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
wy_map_et = np.column_stack((list(clm_out.keys()), wy_inds.T))

wy_17_inds = wy_map_et[wy_map_et[:,1] == 2017][:,0]
wy_18_inds = wy_map_et[wy_map_et[:,1] == 2018][:,0]
wy_19_inds = wy_map_et[wy_map_et[:,1] == 2019][:,0]
wy_20_inds = wy_map_et[wy_map_et[:,1] == 2020][:,0]
wy_21_inds = wy_map_et[wy_map_et[:,1] == 2021][:,0]

wy_inds_helper = {2017:wy_17_inds,2018:wy_18_inds,2019:wy_19_inds,2020:wy_20_inds,2021:wy_21_inds}
drng = np.arange(len(wy_inds_helper[yr]))


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
prcp_sumd  = prcp_sumd[~((prcp_sumd.index.month == 2) & (prcp_sumd.index.day == 29))] # No leap years
wy_  = [np.where((prcp_sumd.index > '{}-09-30'.format(i-1)) & (prcp_sumd.index < '{}-10-01'.format(i)), True, False) for i in yrs]  
wy   = np.array([wy_[i]*yrs[i] for i in range(len(yrs))]).sum(axis=0)
prcp_sumd['wy'] = wy
# annual cumulative
prcp_sumy_cs_  = prcp_sumd.groupby(prcp_sumd['wy']).cumsum()
prcp_sumy_cs_['wy'] = prcp_sumd['wy']




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
months     = ['O','N','D','J','F','M','A','M','J','J','A','S']
months_num = np.array(list(month_map.keys()))

days_ = prcp_sumd[prcp_sumd['wy']==yr].index
first_month = [(days_.month==i).argmax() for i in [10,11,12,1,2,3,4,5,6,7,8,9]]



pull_prcp   = lambda df, wy: pd.DataFrame(df[df['wy']==wy]['prcp'])
def pull_wy(df, wy, var):
    df_    = pd.DataFrame([(clm_out[i][var]) for i in wy_inds_helper[wy]]).T
    head_  = dates[wy_inds_helper[wy]]
    df_.columns = head_
    return df_




#
# MET Forcing -- precipitation and SWE
#

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,4))
fig.subplots_adjust(bottom=0.14, top=0.9, left=0.18, right=0.9, hspace=0.3)
# Monthly Precip Bar Plots
ax = axes[1]
ofs = -0.25
for i in range(len(yrs)):
    pp = pull_prcp(prcp_summ, yrs[i])
    ax.bar(np.arange(len(pp))+ofs, pp['prcp'].to_numpy(), width=0.125, color='C{}'.format(i), fill='C{}'.format(i), hatch=None, label='WY{}'.format(yrs[i]))
    ofs += 0.125
    #ax.plot(np.arange(len(pp)+1), [0]+pp['prcp'].to_list(), color='C{}'.format(i), label='{}'.format(yrs[i]))
#ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
ax.set_ylabel('P (mm/month)')
ax.set_xlim([-0.5, 12.0])
# cumulative precipitation on new axis
ax = axes[0]
for i in range(len(yrs)):
    _pp = pull_prcp(prcp_sumy_cs, yrs[i])
    ax.plot(np.arange(len(_pp)+1), [0]+_pp['prcp'].to_list(), color='C{}'.format(i), ls='-',  label='{}'.format(yrs[i]))
#ax.set_ylim(0,600)
ax.yaxis.set_major_locator(ticker.MultipleLocator(150))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.tick_params(axis='y', which='both', right=True, labelright=True)
ax.set_ylabel('Cumulative P\n(mm/year)')
#ax.set_xlim([0.0, 12.0])
ax.set_xlim([-0.5, 12.0])
# Cleanup
for i in [0,1]:
    axes[i].set_xticks([0]+list(months_num))
    axes[i].set_xticklabels(labels=months+[''])
    axes[i].tick_params(axis='x', top=False, labelrotation=0, pad=0.5)
    axes[i].grid()
    axes[i].tick_params(axis='y', which='both', right=True)
axes[0].legend(ncol=len(yrs), handlelength=0.75, labelspacing=0.15, columnspacing=0.7, handletextpad=0.3, loc='upper center', bbox_to_anchor=(0.5, 1.35))
plt.savefig(os.path.join('./figures', 'CLM_precip.png'),dpi=300)
plt.show()




#
# CLM SWE and ET
#
def accum_swe(wy):
    swe_accum = 0
    swe_accum_list = []
    swe = pull_wy(clm_out, wy, 'swe_out').mean(axis=0)
    for i in range(len(swe)-1):
        if (swe[i+1]-swe[i]) > 0.0:
            swe_accum += swe[i+1]-swe[i]
        swe_accum_list.append(swe_accum)
    return swe_accum_list


# Plot with SWE and ET
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7,6))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.18, right=0.9, hspace=0.3)
# ET
for i in range(len(yrs)):
    et   = pull_wy(clm_out, yrs[i], 'qflx_evap_tot').mean(axis=0) * 60*60*24 
    _et  = et.groupby(by=pd.Grouper(freq='M')).sum()
    # ET Cumulative Annual Sum
    _et_cs = _et.cumsum()
    axes[1].plot(np.arange(len(_et_cs)+1), [0]+_et_cs.to_list(), color='C{}'.format(i))
    axes[1].set_ylabel('Cumulative ET\n(mm/year)')
    axes[1].tick_params(axis='y', which='both', labelright=True)
    #
    axes[2].plot(np.arange(len(_et)), _et, color='C{}'.format(i))
    axes[2].set_ylabel('ET (mm/month)')
    for i in [1,2]:
        axes[i].set_xticks([0]+list(months_num))
        axes[i].set_xticklabels(labels=months+[''])
        axes[i].set_xlim([-0.5, 12.0]) 
# SWE
for i in range(len(yrs)):
    _swe = accum_swe(yrs[i])
    swe  = pull_wy(clm_out, yrs[i], 'swe_out').mean(axis=0)
    axes[0].plot(np.arange(len(swe)), swe, color='C{}'.format(i), label='{}'.format(yrs[i]))
    axes[0].set_xlim(-10,365)
    axes[0].set_xticks(first_month+[365])
    axes[0].set_xticklabels(labels=months+[''])
    axes[0].set_ylabel('SWE (mm)')
# Cleanup
for i in [0,1,2]:
    axes[i].grid()
    axes[i].minorticks_on()
    axes[i].tick_params(axis='x', labelrotation=0, pad=0.5)
    axes[i].tick_params(axis='x', which='minor', top=False, bottom=False)
    axes[i].tick_params(axis='y', which='both', right=True)
axes[0].legend(ncol=len(yrs), handlelength=0.75, labelspacing=0.15, columnspacing=0.7, handletextpad=0.3, loc='upper center', bbox_to_anchor=(0.5, 1.35))
plt.savefig(os.path.join('./figures', 'CLM_SWE_ET.png'),dpi=300)
plt.show()
    
















#------------ 
# Not sure what to do with stuff below
# Makes a bunch of plots for two years only
#------------











#--------------------------------------------
#
# CLM versus time plots
#
#--------------------------------------------
#
# CLM ET
#
et_20 = pull_wy(clm_out, 2019, 'qflx_evap_tot').mean(axis=0) * 60*60*24 
et_21 = pull_wy(clm_out, 2020, 'qflx_evap_tot').mean(axis=0) * 60*60*24 

#
# Precipitation
#
ppd_20 = pull_prcp(prcp_sumd,2019)['prcp'].to_numpy().cumsum()
ppd_21 = pull_prcp(prcp_sumd,2020)['prcp'].to_numpy().cumsum()

#
# CLM infiltration
#
inf_20 = pull_wy(clm_out, 2019, 'qflx_infl').mean(axis=0) * 60*60*24 
inf_21 = pull_wy(clm_out, 2020, 'qflx_infl').mean(axis=0) * 60*60*24 

#
# CLM rech = infil. - ET
#
rech_20 = 1 * (inf_20.to_numpy() - et_20.to_numpy())
rech_21 = 1 * (inf_21.to_numpy() - et_21.to_numpy())


#
# CLM SWE
#
def accum_swe(wy):
    swe_accum = 0
    swe_accum_list = []
    swe = pull_wy(clm_out, wy, 'swe_out').mean(axis=0)
    for i in range(365-1):
        if (swe[i+1]-swe[i]) > 0.0:
            swe_accum += swe[i+1]-swe[i]
        swe_accum_list.append(swe_accum)
    return swe_accum_list
        
swe_20 = pull_wy(clm_out, 2019, 'swe_out').mean(axis=0)
swe_21 = pull_wy(clm_out, 2020, 'swe_out').mean(axis=0)

swe_20_ = accum_swe(2019)
swe_21_ = accum_swe(2020)



#
# Stacked Daily Values
#
prcp_daily_20 = pull_prcp(prcp_sumd, 2019)
prcp_daily_21 = pull_prcp(prcp_sumd, 2020)

vlist1  = [prcp_daily_20, swe_20, et_20, inf_20, rech_20]
vlist2  = [prcp_daily_21, swe_21, et_21, inf_21, rech_21]
vlist1_ = ['Precipitation\n(mm/day)', 'SWE\n(mm)', 'ET\n(mm/day)', 'Infiltration\n(mm/day)', 'Recharge\n(mm/day)']


fig, axes = plt.subplots(nrows=len(vlist1), ncols=1, figsize=(5,6))
fig.subplots_adjust(bottom=0.09, top=0.92, left=0.2, right=0.98, hspace=0.2)
# Plot each timeseries as own subuplot
for i in range(len(vlist1)):
    ax = axes[i]
    ax.plot(np.arange(len(vlist1[i])), vlist1[i], lw=1.5, color='C0', label='WY2020')
    ax.plot(np.arange(len(vlist2[i])), vlist2[i], lw=1.5, color='C1', label='WY2021')
    # Cleanup
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    if i == len(vlist1)-1:
        ax.set_xticklabels(labels=months)
        ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    else:
        ax.set_xticklabels(labels=months)
        ax.tick_params(axis='x', top=True, labelbottom=False, pad=0.001)
    axes[0].tick_params(axis='x', top=True, labelrotation=45, pad=0.001, labeltop=True)
    ax.tick_params(axis='y', which='both', right=True)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylabel(vlist1_[i])
    ax.grid()
#axes[0].legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.27), handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
axes[1].legend(loc='upper right', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_daily_stack.png'),dpi=300)
plt.show()





#
# Stacked Annual Values
#
vlist1  = [swe_20_, et_20.cumsum(), inf_20.cumsum(), rech_20.cumsum()]
vlist2  = [swe_21_, et_21.cumsum(), inf_21.cumsum(), rech_21.cumsum()]
vlist1_ = ['SWE\n(mm/year)', 'ET\n(mm/year)', 'Infiltration\n(mm/year)', 'Recharge\n(mm/year)']

fig, axes = plt.subplots(nrows=len(vlist1), ncols=1, figsize=(6.0,5.5))
#fig.subplots_adjust(bottom=0.08, top=0.93, left=0.2, right=0.98, hspace=0.65)
fig.subplots_adjust(bottom=0.09, top=0.90, left=0.2, right=0.98, hspace=0.2)
# Plot each timeseries as own subuplot
for i in range(len(vlist1)):
    ax = axes[i]
    ax.plot(np.arange(len(vlist1[i])), vlist1[i], lw=2.0, color='C0', label='WY2019' if i==0 else '')
    ax.plot(np.arange(len(vlist2[i])), vlist2[i], lw=2.0, color='C1', label='WY2020' if i==0 else '')
    # Add Precipitation
    ax.plot(np.arange(len(prcp_daily_20.cumsum())), prcp_daily_20.cumsum(), lw=1.5, ls='--', color='C0', label='Precip.' if i==1 else '')
    ax.plot(np.arange(len(prcp_daily_21.cumsum())), prcp_daily_21.cumsum(), lw=1.5, ls='--', color='C1')
    # Cleanup
    ax.set_xlim(0,366)
    ax.set_xticks(first_month)
    if i == len(vlist1)-1:
        ax.set_xticklabels(labels=months)
        ax.tick_params(axis='x', top=True, labelrotation=45, pad=0.001)
    else:
        ax.set_xticklabels(labels=months)
        ax.tick_params(axis='x', top=True, labelbottom=False, pad=0.001)
    axes[0].tick_params(axis='x', top=True, labelrotation=45, pad=0.001, labeltop=True)
    ax.tick_params(axis='y', which='both', right=True)
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylabel(vlist1_[i])
    ax.grid()
#axes[0].legend(loc='center', ncol=2, bbox_to_anchor=(0.5, 1.27), handlelength=1.5, labelspacing=0.25, handletextpad=0.5)
axes[0].legend(loc='upper left', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)
axes[1].legend(loc='upper left', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)
#fig.tight_layout()
plt.savefig(os.path.join(fpath, 'clm_annual_stack.png'),dpi=300)
plt.show()






#
# 
# Simpler Plots
#
#
fpath2 = os.path.join('./','parflow_et_results')
if not os.path.exists(fpath2):
    os.makedirs(fpath2)

# 
# Daily
#
prcp_20_day = pull_prcp(prcp_sumd, 2019)
prcp_21_day = pull_prcp(prcp_sumd, 2020)


fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(5,5))
fig.subplots_adjust(top=0.96, bottom=0.1, left=0.25, right=0.96, hspace=0.3)
ax = axes[0]
ax.plot(np.arange(len(prcp_20_day)), prcp_20_day, color='C0', label='WY2020')
ax.plot(np.arange(len(prcp_21_day)), prcp_21_day, color='C1', label='WY2021')
ax.set_ylabel('Precipitation\n(mm/day)')

ax = axes[1]
ax.plot(np.arange(len(et_20)), et_20, color='C0', label='WY2020')
ax.plot(np.arange(len(et_21)), et_21, color='C1', label='WY2021')
ax.set_ylabel('ET\n(mm/day)')

# clean-up
for ax in axes:
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', labelrotation=45, pad=0.001)
    ax.set_ylim(-1, 21)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))
    ax.tick_params(axis='y', which='both', right=True)
    ax.margins(x=0.01)
axes[1].legend(loc='best', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)  
plt.savefig(os.path.join(fpath2, 'pf_daily_et.png'),dpi=300)
plt.show()

# save dataframes
daily_df_20 = pd.concat((prcp_20_day, et_20), axis=1)
daily_df_20.columns = ['precip mm/day', 'et mm/day']
daily_df_20.index.name = 'date'
#daily_df_20.to_csv(os.path.join(fpath2, 'pf_daily_2020.csv'), index=True)

daily_df_21 = pd.concat((prcp_21_day, et_21), axis=1)
daily_df_21.columns = ['precip mm/day', 'et mm/day']
daily_df_21.index.name = 'date'

daily_df = pd.concat((daily_df_20, daily_df_21))

#daily_df_21.to_csv(os.path.join(fpath2, 'pf_daily_2021.csv'), index=True)



#
# Monthly 
#
et_20_month = et_20.groupby(pd.Grouper(freq='M')).sum()
et_21_month = et_21.groupby(pd.Grouper(freq='M')).sum()

prcp_20_month = prcp_summ.loc[prcp_summ['wy']==2020, 'prcp']
prcp_21_month = prcp_summ.loc[prcp_summ['wy']==2021, 'prcp']

fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(5,5))
fig.subplots_adjust(top=0.96, bottom=0.1, left=0.25, right=0.96, hspace=0.3)
ax = axes[0]
ax.plot(np.arange(len(prcp_20_month)), prcp_20_month, marker='o', color='C0', label='WY2020')
ax.plot(np.arange(len(prcp_21_month)), prcp_21_month, marker='o', color='C1', label='WY2021')
ax.set_ylabel('Precipitation\n(mm/month)')

ax = axes[1]
ax.plot(np.arange(len(et_20_month)), et_20_month, marker='o', color='C0', label='WY2020')
ax.plot(np.arange(len(et_21_month)), et_21_month, marker='o', color='C1', label='WY2021')
ax.set_ylabel('ET\n(mm/month)')

# clean-up
for ax in axes:
    ax.set_xticks(np.arange(len(prcp_20_month)))
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', labelrotation=45, pad=0.001)
    ax.set_ylim(-10, 110)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='y', which='both', right=True)
    ax.margins(x=0.05)
axes[1].legend(loc='best', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)    
plt.savefig(os.path.join(fpath2, 'pf_monthly_et.png'),dpi=300)
plt.show()


# save dataframes
monthly_df_20 = pd.concat((prcp_20_month, et_20_month), axis=1)
monthly_df_20.columns = ['precip mm/month', 'et mm/month']
monthly_df_20.index.name = 'date'
#monthly_df_20.to_csv(os.path.join(fpath2, 'pf_monthly_2020.csv'), index=True)

monthly_df_21 = pd.concat((prcp_21_month, et_21_month), axis=1)
monthly_df_21.columns = ['precip mm/month', 'et mm/month']
monthly_df_21.index.name = 'date'
#monthly_df_21.to_csv(os.path.join(fpath2, 'pf_monthly_2021.csv'), index=True)

monthly_df = pd.concat((monthly_df_20, monthly_df_21))


#
# Annual Cumulative 
#
et_20_annual = et_20.cumsum()
et_21_annual = et_21.cumsum()

prcp_20_annual = pull_prcp(prcp_sumd, 2020).cumsum()
prcp_21_annual = pull_prcp(prcp_sumd, 2021).cumsum()

fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(5,5))
fig.subplots_adjust(top=0.96, bottom=0.1, left=0.25, right=0.96, hspace=0.3)
ax = axes[0]
ax.plot(np.arange(len(prcp_20_annual)), prcp_20_annual, color='C0', lw=2.0, label='WY2020')
ax.plot(np.arange(len(prcp_21_annual)), prcp_21_annual, color='C1', lw=2.0, label='WY2021')
ax.set_ylabel('Precipitation\n(mm/year)')

ax = axes[1]
ax.plot(np.arange(len(et_20_annual)), et_20_annual, color='C0', lw=2.0, label='WY2020')
ax.plot(np.arange(len(et_21_annual)), et_21_annual, color='C1', lw=2.0, label='WY2021')
ax.set_ylabel('ET\n(mm/year)')

# clean-up
for ax in axes:
    ax.set_xticks(first_month)
    ax.set_xticklabels(labels=months)
    ax.tick_params(axis='x', labelrotation=45, pad=0.001)
    ax.set_ylim(-1, 550)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.tick_params(axis='y', which='both', right=True)
    ax.margins(x=0.01)
axes[1].legend(loc='best', handlelength=1.0, labelspacing=0.25, handletextpad=0.1)  
plt.savefig(os.path.join(fpath2, 'pf_annual_et.png'),dpi=300)
plt.show()


# save dataframes
annual_df_20 = pd.concat((prcp_20_annual, et_20_annual), axis=1)
annual_df_20.columns = ['precip_cumulative mm/year', 'et_cumalative mm/year']
annual_df_20.index.name = 'date'
#annual_df_20.to_csv(os.path.join(fpath2, 'pf_annual_2020.csv'), index=True)

annual_df_21 = pd.concat((prcp_21_annual, et_21_annual), axis=1)
annual_df_21.columns = ['precip_cumulative mm/year', 'et_cumalative mm/year']
annual_df_21.index.name = 'date'
#annual_df_21.to_csv(os.path.join(fpath2, 'pf_annual_2021.csv'), index=True)

annual_df = pd.concat((annual_df_20, annual_df_21))


# Write to file
with pd.ExcelWriter(os.path.join(fpath2, 'parflow_et_results.xlsx')) as writer:
    daily_df.to_excel(writer, sheet_name='daily', index=True)
    monthly_df.to_excel(writer, sheet_name='monthly', index=True)
    annual_df.to_excel(writer, sheet_name='annual', index=True)








#
# Plots that Integrate over both space and time
#
et_ = et_20 
fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.22, right=0.95, hspace=0.3)
ax.hist(x=et_, bins=50, density=True)
ax.set_ylabel('Density')
ax.set_xlabel('ET (mm/day)')
ax.minorticks_on()
ax.grid()
plt.savefig(os.path.join(fpath, 'et_distribution.png'),dpi=300)
plt.show()
#
#
inf_ = inf_20
fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.22, right=0.95, hspace=0.3)
ax.hist(x=inf_, bins=50, density=True)
ax.set_ylabel('Density')
ax.set_xlabel('Infiltration (mm/day)')
ax.minorticks_on()
ax.grid()
plt.savefig(os.path.join(fpath, 'inf_distribution.png'),dpi=300)
plt.show()
#
#
rech_ = inf_20 - et_20
fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(top=0.96, bottom=0.25, left=0.22, right=0.95, hspace=0.3)
ax.hist(x=rech_, bins=50, density=True)
ax.set_ylabel('Density')
ax.set_xlabel('Recharge (mm/day)')
ax.minorticks_on()
ax.grid()
plt.savefig(os.path.join(fpath, 'rech_distribution.png'),dpi=300)
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
x_pnts = [100, 300, 404, 494, 528]   
#x_pnts = [404, 494, 540]    
cc = plt.cm.viridis(np.linspace(0,1,len(x_pnts)))
#cc = ['C3','C4','C5']
cc = plt.cm.coolwarm(np.linspace(0,1,len(x_pnts)))
#cc = plt.cm.turbo(np.linspace(0,1,len(x_pnts)))


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


# CLM plant veg plots
fig, ax = plt.subplots(figsize=(6,2))
ax.scatter(xrng, topo, ls='-', marker=None, s=1.0, c=np.array(vind)/np.array(vind).max(), cmap=cmap_cust)
# Add points
#for j in range(len(x_pnts)):
#    ax.axvline(x_pnts[j]*1.5125, color=cc[j], ls='--', lw=2.0)
ax.set_ylabel('Z (m)')
ax.minorticks_on()
ax.tick_params(axis='x', which='both', top=True, bottom=True)
ax.tick_params(axis='y', which='both', left=True, right=True)
ax.set_xlim(0, xrng.max())
ax.set_xlabel('Distance (m)')
fig.tight_layout()
plt.savefig(os.path.join(fpath,'CLM_veg.png'), dpi=300)
plt.show()






#--------------------------------------------------------
#
# ET
#
#--------------------------------------------------------
et = pull_et(yr)*86400 
et.columns = np.arange(365)

# Testing with parflow ET, rather than CLM ET
#et = pull_wy(et_df, yr).iloc[:,1:]
#et.index = np.arange(365)
#et = et.T



#--------------------------------------------------------
#
# Infiltration
#
#--------------------------------------------------------
inf = pull_infil(yr)*86400 
inf.columns = np.arange(365)


#--------------------------------------------------------
#
# Rech = Infiltration - ET
#
#--------------------------------------------------------
inf = pull_infil(yr)*86400
inf.columns = np.arange(365)

rech = inf - et


#--------------------------------------------------------
#
# SWE
#
#--------------------------------------------------------
swe = pull_swe(yr)
swe.columns = np.arange(365)

atemp     = pull_wy(met, yr)['temp'].groupby(pd.Grouper(freq='D')).mean()- 273.15
atemp_max = pull_wy(met, yr)['temp'].groupby(pd.Grouper(freq='D')).max() - 273.15


#--------------------------------------------------------
#
# Ground Temperature
#
#--------------------------------------------------------
gtemp = pull_grndT(yr)
gtemp.columns = np.arange(365)


#--------------------------------------------------------
#
# Soil Temperature
#
#--------------------------------------------------------
stemp = pull_soilT(yr)
stemp.columns = np.arange(365)



#-----------------------------------------------------------
#-----------------------------------------------------------

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
    #ax.plot(xrng, np.median(dd,axis=1), color='black', lw=2.0, ls='--', zorder=8)
    # Plot the max and min
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmax()], lw=2.0, color='black', ls=':', zorder=8)
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmin()], lw=2.0, color='black', ls=':', zorder=8)
    # Plot full ensemble of timesteps
    #for j in range(0,dd.shape[1],1):
    #    ax.plot(xrng, dd.iloc[:,j], color='grey', alpha=0.3, lw=0.5)
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
if atemp.shape[0] != gtemp.shape[1]:
    atemp_ = pd.DataFrame(np.array(atemp)[:-1] * np.ones_like(gtemp))
else:
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
    #ax.plot(xrng, np.median(dd,axis=1), color='black', lw=2.0, ls='--', zorder=8)
    # Plot the max and min
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmax()], lw=2.0, color='black', ls=':', zorder=8)
    #ax.plot(xrng, dd.loc[:,dd.sum(axis=0).idxmin()], lw=2.0, color='black', ls=':', zorder=8)
    # Plot full ensemble of timesteps
    #for j in range(0,dd.shape[1],1):
    #    ax.plot(xrng, dd.iloc[:,j], color='grey', alpha=0.3, lw=0.5)
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

sat = pf['sat'].copy()

sat_dict = {}
for j in range(len(zinds)):
    sat_df   = pd.DataFrame()
    for i in list(sat.keys()):
        dd = sat[i].copy()
        # pull saturation values at constant depth below land surface
        dd_  = dd[zinds[j],0,:]
        sat_df[i] = dd_
    #sat_df = sat_df.iloc[:,1:].T
    sat_df = sat_df.T
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




"""
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
"""







