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


sys.path.insert(0, '/Users/nicholasthiros/Documents/SCGSR/Age_Modeling/utils')
import convolution_integral_utils as conv
import noble_gas_utils as ng_utils


# set a random seed for preporducability
np.random.seed(10)


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







#------------------------------
#
# FM RTD Sensitivity Analysis
#
#------------------------------

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


wells = ['PLM1','PLM7','PLM6']

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





#---------------------------------------------------
#
# Matrix Diffusion Convolution with EcoSlim RTD
#
#---------------------------------------------------
tr_map = {'cfc12':'CFC12','sf6':'SF6','h3':'H3','he4':'He4_ter'}

# For now, using pieces that were generated in Age_modeling dir
C_in_dict  = pd.read_csv('../ER_PLM_ParFlow/C_in_df.csv', index_col=['tau']) # convolution tracer input series


# Improved parameters using effective rock physics
_bbar    = lambda Keff, Km, L: ((L*12.*1.e-3/9810.)*(Keff-Km))**(1/3.)
_phi_eff = lambda phim, b, L, phif: (L*phim+b*phif)/(L+b)
_Keff    = lambda Km, b, L: (L*Km + (b**3)*9810./(12*1.e-3)) / (L+b)


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
nsize = 200


Km_list_   = 10**(np.random.uniform(-8, -6, nsize)) # 1.e-8 to 1.e-6 matrix hydraulic conductivty
Keff_list_ = 10**(np.random.uniform(-6, -4, nsize)) # 1.e-6 to 1.e-4 matrix hydraulic conductivty, can also do Keff_list_ as mulitiplicative factor of Km
L_list_    = np.random.uniform(0.1, 0.5, nsize)     # fracture spacing (m), based on Gardner 2020
Phim_list_ = 10**(np.random.uniform(np.log10(0.05/100), np.log10(5/100), nsize)) # 0.05% to 5% matrix porosity (variable needs to be in decimal)
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


rerun = True
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







