#!/bin/bash

cp ../ER_PLM_ParFlow/pull_parflow_wl_v5.py .
cp ../ER_PLM_ParFlow/read_vtk.041922.py .
cp ../ER_PLM_ParFlow/et_to_pickle.py .
cp ../ER_PLM_ParFlow/vel_decomp_2_rech.py .


python pull_parflow_wl_v5.py
python read_vtk.041922.py
python et_to_pickle.py
#python sat_press_to_pickle.py
python vel_decomp_2_rech.py
