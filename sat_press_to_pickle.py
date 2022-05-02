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



nsteps = 1793 + 1




#
# Saturation Output Fields
#
out_dict = {}
for i in range(nsteps):
    # Read in the file
    ff    = './wy_2017_2021/wy_2017_2021.out.satur.{0:05d}.pfb'.format(i)
    if os.path.exists(ff):
        dd    = read_pfb(ff)
        #dd_df = pd.DataFrame(dd[:,0,:].T, columns=clm_var)
        out_dict[i] = dd[:,0,:]
    else:
        print ('timstep {} does not exists'.format(i))
        
# Save it
with open('sat_out_dict.pk', 'wb') as f:
    pk.dump(out_dict, f)      




#
# Pressure Output Fields
#
out_dict = {}
for i in range(nsteps):
    # Read in the file
    ff    = './wy_2017_2021/wy_2017_2021.out.press.{0:05d}.pfb'.format(i)
    if os.path.exists(ff):
        dd    = read_pfb(ff)
        #dd_df = pd.DataFrame(dd[:,0,:].T, columns=clm_var)
        out_dict[i] = dd[:,0,:]
    else:
        print ('timstep {} does not exists'.format(i))
        
# Save it
with open('parflow_out/press_out_dict.pk', 'wb') as f:
    pk.dump(out_dict, f)      
