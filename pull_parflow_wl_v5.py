# To-do: Pull pressure head for PLM1 and PLM6 at the actual peizometer depth
#        @ PLM7, pull pressure head at first fully saturated cell

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
        ind_map = {}
        wells = self.pf_wells_df.index.to_list()
        for i in range(len(wells)):
            ind_map[wells[i]] = i
        
        #pdb.set_trace()
        # ParFlow cell indices at X and sampling depth Z
        wellinds = self.pf_wells_df[['Cell_Z', 'Cell_X']].to_numpy().astype(int)
        
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
        # Depth below land surface to first fully saturated cell
        h_bls= self.dz_cumsum[dd_zero]
        #
        # Update 12/15/21 for PLM1 and PLM6 pull pressure at sampling location
        # Because PLM1 and PLM6 are peizometers
        # cell indices for sampling loc
        #z_samps = [abs(z_info['Depth_bls'] - pf_wells['screen_top_m'][i]).idxmin() for i in list(ind_map.keys())] # at top of screen elevation
        z_samps   = self.pf_wells_df['Cell_Z'].astype(int) # at sampling depth
        # Pressure head at sampling depth z
        pr_pz = dd_col[z_samps.to_numpy().T, [0,1,2]] 
        # Depth below land surface to the piezometer cell
        h_bls_pz  = self.dz_cumsum[z_samps]
        
        # combine for PLM1, PLM7, PLM6
        #h  = h_bls
        #pr = pr_wt
        h  = np.array([h_bls_pz[ind_map['PLM1']], h_bls[ind_map['PLM7']], h_bls_pz[ind_map['PLM6']]]) # depth to cell
        pr = np.array([pr_pz[ind_map['PLM1']], pr_wt[ind_map['PLM7']], pr_pz[ind_map['PLM6']]]) # pressure at the cell
        
        #print (h_bls)
        # Account for pressure head at this cell
        wt_bls = h - pr
        #print (wt_bls)
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
pf_wells = pd.read_csv('../utils/wells_2_pf_v3b.csv', index_col='well')
z_info   = pd.read_csv('../utils/plm_grid_info_v3b.csv') # these are cell centered values



# Pull Simulated water levels
# First Set up dirs
#rundir, runname = ['wy_spinup']*2
#rundir, runname = ['wy_1979_2014']*2
#rundir, runname = ['wy_2015_2021']*2

for ff in ['wy_spinup','wy_2000_2016','wy_2017_2021']:
#for ff in ['wy_2015_2021']:
    if ff in os.listdir():
        print ('working on {}'.format(ff))
        pf_wt = pull_pfb_pressure(pf_wells, z_info['Depth_bls'].to_numpy(), z_info['Total_Depth'][0], ff, ff)
        pf_wt_bls, pf_wt_elev = pf_wt.get_water_table_ts()
        pf_wt_bls.index.name = 'Timestep'
        pf_wt_elev.index.name = 'Timestep'
        #pf_wt_bls, pf_wt_elev = pf_wt.mangle_dates('2014-10-01',)

        # save it
        if not os.path.exists('parflow_out'):
            os.makedirs('parflow_out')
        pf_wt_bls.to_csv('./parflow_out/{}_wt_bls.csv'.format(ff))
        pf_wt_elev.to_csv('./parflow_out/{}_wt_elev.csv'.format(ff))
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










