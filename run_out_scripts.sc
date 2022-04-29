#!/bin/bash

python pull_parflow_wl_v5.py
python read_vtk.041922.py
python et_to_pickle.py
python sat_press_to_pickle.py
python vel_decomp_2_rech.py
