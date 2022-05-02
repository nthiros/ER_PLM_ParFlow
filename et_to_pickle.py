import numpy as np
import pandas as pd
import pickle as pk
import os

from parflowio.pyParflowio import PFData




def read_pfb(fname):
    pfdata = PFData(fname)
    pfdata.loadHeader()
    pfdata.loadData()
    return pfdata.copyDataArray()



#
# ET output files
#
nsteps = 1793 + 1


et_out_dict = {}
for i in range(nsteps):
    # Read in the file
    et_f    = './wy_2017_2021/wy_2017_2021.out.evaptrans.{0:05d}.pfb'.format(i)
    if os.path.exists(et_f):
        et_out  = read_pfb(et_f)[:,0,:]
        et_out_dict[i] = et_out
    else:
        print ('timstep {} does not exists'.format(i))

# Save it
with open('et_out_dict.pk', 'wb') as f:
    pk.dump(et_out_dict, f)        



#
# CLM output files
#

# pg. 147 of parflow manual
clm_var_base = ['eflx_lh_tot',  'eflx_lwrad_out','eflx_sh_tot',   'eflx_soil_grnd',
                'qflx_evap_tot','qflx_evap_grnd','qflx_evap_soil','qflx_evap_veg',
                'qflx_tran_veg','qflx_infl',     'swe_out',       't_grnd',        't_soil']

# have clm set to 4 soil layers
clm_var = clm_var_base + ['t_soil_{}'.format(i) for i in range(4)]



clm_out_dict = {}
for i in range(nsteps):
    # Read in the file
    clm_f    = './wy_2017_2021/wy_2017_2021.out.clm_output.{0:05d}.C.pfb'.format(i)
    if os.path.exists(clm_f):
        clm_out  = read_pfb(clm_f)
        clm_df   = pd.DataFrame(clm_out[:,0,:].T, columns=clm_var)
        clm_out_dict[i] = clm_df
    else:
        print ('timstep {} does not exists'.format(i))
        
# Save it
with open('parflow_out/clm_out_dict.pk', 'wb') as f:
    pk.dump(clm_out_dict, f)      

