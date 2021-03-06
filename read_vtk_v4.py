import pandas as pd
import numpy as np
import os
import pickle

import pyvista as pv

import matplotlib.pyplot as plt

import pdb



#------------------------------------------------------
# External info needed to parse EcoSlim vtk
#------------------------------------------------------   
# Read in well info -- created by well_info_v2.py
well_df = pd.read_csv('../utils/wells_2_pf_v3b.csv', index_col=('well'))



#----------------------------------------------------
# Class to pull from EcoSlim_cgrid.vtk files
#----------------------------------------------------
class ecoslim_grid_vtk():
    def __init__(self, wells_df):
        
        # List of vtk files to process
        self.vtk_files = None
        
        # Observation wells locations
        self.wells = wells_df   # output from well_info_v2.py
        
        # vtk info
        self.cell_xyz = None   # read_vtk_gird
        self.var = None        # variables in the vtk
        
        # Age df
        self.age_df = None     # final dataframe with ecoslim output at wells
               
    def find_cgrid_vtk(self, dir_loc):
        '''Gathers all *.vtk files in directory with path dir_loc into a list.
           Returns a list of lists'''
        ff = []
        for file in os.listdir(dir_loc):
            if file.endswith(".vtk"):
                if file.split('.')[0].split('_')[-1] == 'cgrid':
                    ff.append(os.path.join(dir_loc, file))
        ff.sort()
        self.vtk_files = ff
        return ff
        
    def read_vtk_grid(self, vtkfile):
        '''This generates centers for all cells.
           Only needs to be run once.
           Note - this is not needed anymore, use well_info_v2.py to find well cell indices'''
        #pdb.set_trace()
        mesh = pv.read(vtkfile) 
        # Pull the center coordinate (XYZ) for each cell
        cent_avg = lambda coords: [np.mean(coords[i:i+2]) for i in [0,2,4]]
        cell_xyz = np.array([cent_avg(mesh.cell_bounds(n)) for n in range(mesh.n_cells)])
        self.cell_xyz = cell_xyz
        self.vars = list(mesh.cell_arrays)
        return cell_xyz
       
    def update_df(self, mesh, df, time, varname):
        '''Utility function to store vtk output at points and times into a dataframe'''
        inds = self.wells['Cell_ind'].to_numpy().astype(int)
        df.loc[:,time] = mesh[varname][inds]
        return df
    
    def read_vtk(self):
        '''Loop through all vtk files and pull mean age at wells'''
        #pdb.set_trace()
        #inds = self.wells['Cell_ind'].to_numpy().astype(int)
        
        gen_df = lambda : pd.DataFrame(index=self.wells.index)
         
        df0,df1,df2,df3,df4,df5,df6,df7,df8 = [gen_df() for i in range(len(self.vars))]
        self.age_df = [df0,df1,df2,df3,df4,df5,df6,df7,df8]
        
        for i in range(len(self.vtk_files)):
            f = self.vtk_files[i]
            mesh_i = pv.read(f)
            ii = int(f.split('.')[-2])
            #print (i)
            for j in range(len(self.vars)):
                self.update_df(mesh_i, self.age_df[j], ii, self.vars[j])
        out_dict = {}
        for j in range(len(self.vars)):
            out_dict[self.vars[j]] = self.age_df[j].T.sort_index()
             
        return out_dict
    
# Mean Age Data
age           = ecoslim_grid_vtk(well_df)
vtk_files_c   = age.find_cgrid_vtk('./ecoslim_2017_2021')
age.vtk_files = vtk_files_c
cell_xyzm     = age.read_vtk_grid(vtk_files_c[0])
age_dict      = age.read_vtk()




#----------------------------------------------------------
# The pnts vtk files now
#----------------------------------------------------------
class ecoslim_pnts_vtk():
    def __init__(self, wells_df, eco_grid):
        
        # List of vtk files to process
        self.vtk_files = None
        
        # ParFlow-Ecoslim Grid Info
        self.cell_xyz = eco_grid # output from ecoslim_grid_vtk.read_vtk_grid()
        
        # Observation well locations
        self.wells_df = wells_df  # output from well_info_v2.py

        # Age Distribution
        self.rtd = {}

    def find_pnts_vtk(self, dir_loc):
        '''Gathers all *.vtk files in directory with path dir_loc into a list.
           Returns a list of all the vtk files'''
        ff = []
        for file in os.listdir(dir_loc):
            if file.endswith(".vtk"):
                if file.split('.')[0].split('_')[-1] == 'pnts':
                    ff.append(os.path.join(dir_loc, file))
        ff.sort()
        self.vtk_files = ff
        return ff
           
    def read_vtk(self):
        '''Loop through all vtk files and paticle Time, Mass, and Source around each well.
           Returns a dictionary where first key is the int(model time) and the second key is the str(well location) name.
               This then holds a px3 array where each row is a particle in the control volume for the well and columns are 'Time','Mass','Source'
               -- exmaple: rtd_dict[model_time][well_name] = px3 array'''
        #pdb.set_trace()
        
        # Coordinates of the wells in Parflow space
        #pf_well_coords =  self.cell_xyz[self.wells_df['Cell_ind'].astype(int).to_numpy()]
        x = self.wells_df['X']
        z = self.wells_df['land_surf_cx']  - self.wells_df['smp_depth_m']
        pf_well_coords = np.column_stack((x.to_numpy(), np.zeros(len(x.to_numpy())), z.to_numpy()))

        # Loop through each vtk file
        for i in range(len(self.vtk_files)):
            f = self.vtk_files[i]
            print ('{}/{}'.format(i+1, len(self.vtk_files)))
            
            dd = pv.read(f) 
            # xyz points of all particles -- big list
            pt_xyz = np.array(dd.points)
            
            ii = int(f.split('.')[-2])
            self.rtd[ii] = {}
            
            # Loop through each well
            for w in range(len(pf_well_coords)):
                # Draw a box around well points to collect particles in, meters
                xhigh = pt_xyz[:,0] <= pf_well_coords[w,0]+2.0 
                xlow  = pt_xyz[:,0] >= pf_well_coords[w,0]-2.0  
                zhigh = pt_xyz[:,2] <= pf_well_coords[w,2]+2.0 
                zlow  = pt_xyz[:,2] >= pf_well_coords[w,2]-2.0
                
                msk  = np.column_stack((xlow,xhigh,zlow,zhigh))
                msk_ = msk.sum(axis=1)==4

                # Now pull values in the define box
                time   = dd['Time'][msk_]
                mass   = dd['Mass'][msk_]
                source = dd['Source'][msk_]
                Xin    = dd['xInit'][msk_]
                
                rtd = np.column_stack((time,mass,source,Xin))
                self.rtd[ii][well_df.index[w]] = rtd
        return self.rtd    
            

# Particle Ages
get_rtd            = ecoslim_pnts_vtk(well_df, cell_xyzm)
vtk_files          = get_rtd.find_pnts_vtk('./ecoslim_2017_2021')
get_rtd.vtk_files  = vtk_files[::5]
rtd_dict           = get_rtd.read_vtk()



# Save the results to a dictionary
with open('ecoslim_MeanAge.pk', 'wb') as f:
    pickle.dump(age_dict, f) 
with open('ecoslim_rtd.pk', 'wb') as ff:
    pickle.dump(rtd_dict, ff) 







#def flux_wt_rtd(rtd_dict_unsort, model_time, well_name, nbins):
#    '''Convert particle ages at a single model time and well location to a residence time distribution.
#    Returns:
#        - rtd_df:   Dataframe with mass weighted ages for all paricles (sorted by age)
#        - rtd_dfs:  Dataframe similar to above but bins age distribution into discrete intervals  
#    Inputs: 
#        - rtd_dict_unsort: output from ecoslim_pnts_vtk.read_vtk() above
#        - model_time: model time to consider. Must be in rtd_dict_unsort.keys()
#        - well_name: observation well to consider. Must be in rtd_dict_unsort.keys()
#        - nbins: Number of intervals to bin the ages into.'''
#    # Flux Weighted RTD
#    # Info regarding particles at a single timestep and single point (box) of the domain
#    rtd    = rtd_dict_unsort[model_time][well_name] 
#    rtd_df = pd.DataFrame(data=rtd,columns=['Time','Mass','Source'])
#    rtd_df['wt'] = rtd_df['Mass']/rtd_df['Mass'].sum()
#    rtd_df.sort_values('Time', inplace=True)
#    rtd_df['Time'] /= 8760
#    
#    # Now some binning
#    #nbins = 10
#    gb =  rtd_df.groupby(pd.cut(rtd_df['Time'], nbins))
#    rtd_dfs = gb.agg(dict(Time='mean',Mass='sum',Source='mean',wt='sum'))
#    rtd_dfs['count'] = gb.count()['Time']
#
#    return rtd_df, rtd_dfs
#
#rtd_df, rtd_dfs = flux_wt_rtd(rtd_dict, 202, 'PLM6', 10)


## Plot the RTD
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
##ax[0].plot(rtd_df1['Time'], rtd_df1['wt'], marker='.')
#ax[0].plot(rtd_dfs['Time'], rtd_dfs['wt'], marker='.', color='red')
#ax[0].set_ylabel('Probability')
#ax[0].set_xlabel('Travel Time (years)')
#
#ax[1].plot(rtd_df['Time'], np.cumsum(rtd_df['wt']), marker='.')
#ax[1].plot(rtd_dfs['Time'], np.cumsum(rtd_dfs['wt']), marker='.', color='red')
#ax[1].set_ylabel('Probability CDF')
#ax[1].set_xlabel('Travel Time (years)')
#fig.tight_layout()
#plt.show()





    
 
