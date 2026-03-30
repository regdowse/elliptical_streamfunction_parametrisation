Elliptical Streamfunction Parametrisation (ESPresso) – Method Implementations

This repository contains Python implementations of the ESPresso framework, including the SOLO, DOPPIO, and LATTE methods. These methods provide a unified approach for reconstructing and analysing ocean eddy structure from velocity observations across a range of sampling configurations, including transects, gridded fields, and multi-platform datasets.

The repository is designed as a demonstration of how these methods can be applied, and the input datasets can be replaced with user-provided velocity data.

The core python functions are provided in functions.py. The SOLO, DOPPIO, and LATTE methods are used to estimate the ESP inner-core parameters from velocity observations. These estimates can then be passed to the out_core_param_fit function, which uses the velocity data to determine the outer-core ESP parameters.

Repository Structure
.
├── data/
├── functions.py
├── SOLO_transect_data.ipynb
├── DOPPIO_gridded_data.ipynb
├── LATTE_app1_multiplatform_data.ipynb
├── LATTE_app2_multiplatform_data.ipynb

functions.py:                           Core functions for ESP model evaluation and parameter estimation used across all methods.
SOLO_transect_data.ipynb:               Application of the SOLO method to single transect velocity data.
DOPPIO_gridded_data.ipynb:              Application of the DOPPIO method to two-dimensional gridded surface velocity data.
LATTE_app1_multiplatform_data.ipynb:    LATTE applied to multi-platform data combining S-ADCP and satellite-derived velocities.
LATTE_app2_multiplatform_data.ipynb:    LATTE applied to multi-platform data combining drifter velocity observations and satellite-derived velocities.
data/:                                  Directory containing input datasets required to run the notebooks.

Methods Overview:
SOLO – A single-transect method for estimating ESP parameters, assuming quasi-axisymmetric structure.
DOPPIO – A two-transect method for estimating ESP parameters by exploiting orthogonal velocity transects in gridded velocity fields.
LATTE – A generalised multi-platform method that combines sparse, irregular, and heterogeneous velocity observations to estimate ESP parameters.

Data:
The data/ directory contains example datasets used in the notebooks:
sadcp_data.mat:             Shipboard ADCP (S-ADCP) velocity observations along transects. Used in: SOLO_transect_data.ipynb, LATTE_app1_multiplatform_data.ipynb
gridded_numerical_data.nc:  Two-dimensional gridded surface velocity field from a numerical model. Used in: DOPPIO_gridded_data.ipynb
satellite_data.nc:          Gridded satellite-derived surface velocity data. Used in: LATTE_app1_multiplatform_data.ipynb, LATTE_app2_multiplatform_data.ipynb
drifter_data.nc:            Lagrangian drifter velocity observations at the ocean surface. Used in: LATTE_app2_multiplatform_data.ipynb

Requirements:
Python 3.9 or higher
numpy
scipy
pandas
xarray
matplotlib
netCDF4

Usage:
Each notebook is self-contained
Open a notebook
Run cells sequentially

Citation:
If you use this code, please cite:
Reg Dowse (2026) “regdowse/elliptical_streamfunction_parametrisation: ESP”. Zenodo. doi:10.5281/zenodo.19309976.

Author:
Reg Dowse
UNSW Sydney
