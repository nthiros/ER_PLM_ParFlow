# Pulls the water table depth along the full hillslope transect, not just well locations
# Does this for all timesteps
#
# Writes 'parflow_out/{}_wt_bls_full_trans.csv' to parflow_out dir
# This is used in et_decomp.py plotting scipt
#
# Note...
# Need to run on server where all the ParFlow output files are
#


from parflowio.pyParflowio import PFData

import numpy as np
import pandas as pd
import os

import pdb

import matplotlib.pyplot as plt



class pull_pfb_pressure():
    def __init__(self, pf_wells_df, pf_dz_cumsum_c, pf_zmax, dir_loc, run_name):
        self.pf_wells_df = pf_wells_df     # Where in ParFlow grid the wells are, above calculations and eventually own function
        self.dz_cumsum   = pf_dz_cumsum_c  # ParFlow variable z depths -- cell centered, above calculations
        self.zmax        = pf_zmax         # ParFlow total depth
        self.dir_loc     = dir_loc         # Location of parflow .pfb files
        self.run_name    = run_name        

        self.wt  = None
        self.wt_ = None
    
    def find_pfb(self):
        '''Gathers all press.pfb files in directory with path dir_loc into a list.
           Returns a lists'''
        #pdb.set_trace()
        ff = []
        for file in os.listdir(self.dir_loc):  
            if file.endswith(".pfb"):
                #if 'press' in file.split('.'):
                if all(x in file.split('.') for x in ['press',self.run_name]):
                    ff.append(os.path.join(self.dir_loc, file))
        ff.sort()
        return ff 

    def read_pfb(self, pfb_fname):
        '''parflowio magic to read the pfb files into numpy array.
           Returns 2D array with shape NZ,NX'''
        pfdata = PFData(pfb_fname)
        pfdata.loadHeader()
        pfdata.loadData()
        dd = pfdata.copyDataArray()
        return dd[:,0,:] # because our y-dim is zero
        

    def pf_pressure(self, pfb_fname):
        '''Pull simulated water table below land surface at well locations.
           This is referenced to the cell-centered grid.
           Returns array with distance below land surface in m'''
        #-----------------------------------------
        # Simulated Pressures at well locations
        #-----------------------------------------
        # mapping indices numbers to well names to be sure indexing done correctly
        #ind_map = {}
        #wells = self.pf_wells_df.index.to_list()
        #for i in range(len(wells)):
        #    ind_map[wells[i]] = i
        
        #pdb.set_trace()
        # ParFlow cell indices at X and sampling depth Z
        wellinds = self.pf_wells_df[['Cell_Z', 'Cell_X']].to_numpy().astype(int)
        
        #pdb.set_trace()
        
        # Pressure at all grid points 
        dd = self.read_pfb(pfb_fname)
        #dd_well = dd[wellinds[:,0], wellinds[:,1]]
        #self.pf_wells_df['Pressure_head'] = dd_well
        #
        # Update 11/23/21 calculate wl elev using first postive pressure head
        # Pressure head profile along column of located at well location X
        dd_col  = dd[:, wellinds[:,1]]
        # Z-cell that is closest to pressure head >= 0 (water table)
        dd_zero = (dd_col>0.0).argmin(axis=0) - 1
        # Pressure head at this cell
        pr_wt = dd_col[dd_zero, np.arange(len(dd_zero))]
        # Elevation head at this cell as depth below land surface
        h_wt = [self.dz_cumsum[i] for i in dd_zero]
        # Total Head (mbls)
        wt_bls = h_wt - pr_wt
        
        return wt_bls #wt_bls
        
    
    def get_water_table_ts(self):
        '''Loop through all times and pull water table depth (below top of model).
           Also calculates the water table elevation (meters above sea-level).
           Returns arrays of water table below land surface and water table elevation at well X locations'''
        wt_df  = pd.DataFrame(index=self.pf_wells_df.index)
        wt_df_ = pd.DataFrame(index=self.pf_wells_df.index)
        
        pfb_list = self.find_pfb()
        for i in pfb_list:
            wt_bls = self.pf_pressure(i)
            wt_ele = self.pf_wells_df['land_surf_dem_m'].to_numpy() - wt_bls
            t  = float(i.split('.')[-2])
            
            wt_df.loc[:, t] = wt_bls
            wt_df_.loc[:, t] = wt_ele
            
        self.wt  = wt_df.T
        self.wt_ = wt_df_.T
            
        return wt_df.T, wt_df_.T

    def mangle_dates(self, startdate):
        '''Put timeseries into acutal dates.
           Assumes 24 hour timesteps right now'''
        startd = pd.to_datetime(startdate)
        endd   = startd + pd.Timedelta(len(self.wt)-1, 'D')
        date_list = pd.date_range(start=startd, end=endd, freq='D')
        
        self.wt.index  = date_list
        self.wt_.index = date_list
        
        return self.wt, self.wt_
        

# Read in well info dataframe
# Produced by well_info_v2.py
#pf_wells  = pd.read_csv('../utils/wells_2_pf_v3b.csv', index_col='well')

z_info    = pd.read_csv('../utils/plm_grid_info_v3b.csv') # these are cell centered values
topo      = np.loadtxt('./Perm_Modeling/elevation.sa', skiprows=1)
pf_wells_ = pd.DataFrame(index=np.arange(len(topo)), data=np.column_stack((np.arange(len(topo)), 32*np.ones(len(topo)))), columns=['Cell_X','Cell_Z']) 
pf_wells_['land_surf_dem_m'] = topo.copy()



# Pull Simulated water levels

for ff in ['wy_2017_2021']:
    if ff in os.listdir():
        print ('working on {}'.format(ff))
        pf_wt = pull_pfb_pressure(pf_wells_, z_info['Depth_bls'].to_numpy(), z_info['Total_Depth'][0], ff, ff)
        pf_wt_bls, pf_wt_elev = pf_wt.get_water_table_ts()
        pf_wt_bls.index.name = 'Timestep'
        pf_wt_elev.index.name = 'Timestep'


        # save it
        if not os.path.exists('parflow_out'):
            os.makedirs('parflow_out')
        pf_wt_bls.to_csv('./parflow_out/{}_wt_bls_full_trans.csv'.format(ff))
        pf_wt_elev.to_csv('./parflow_out/{}_wt_elev_full_trans.csv'.format(ff))
    else:
        pass


# Plot Timeseries
#fig, ax = plt.subplots()
#for i in ['PLM1','PLM6','PLM7']:
#    ax.plot(pf_wt_elev[i], label=i)
#ax.set_xlabel('Time (days)')
#ax.set_ylabel('Water Table (m bls)')
#ax.legend()
#plt.show()










