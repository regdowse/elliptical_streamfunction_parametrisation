Elliptical Streamfunction Parametrisation (ESPresso) – Method Implementations

This repository contains Python implementations of the ESPresso framework, including the SOLO, DOPPIO, and LATTE methods. These methods provide a unified approach for reconstructing and analysing ocean eddy structure from velocity observations across a range of sampling configurations, including transects, gridded fields, and multi-platform datasets.

The repository is designed as a demonstration of how these methods can be applied, and the input datasets can be replaced with user-provided velocity data.

Scope: these four notebooks demonstrate ESP methods on observational and numerical
datasets. Generators for the manuscript's idealised and sensitivity experiments
are outside the scope of this application repository.

The ESPresso python functions are provided in functions.py. The SOLO, DOPPIO, and LATTE methods are used to estimate the ESP inner-core parameters from velocity observations. These estimates can then be passed to the out_core_param_fit function, which uses the velocity data to determine the outer-core ESP parameters.

Repository Structure
.
├── data/
├── functions.py
├── reporting.py
├── SOLO_transect_data.ipynb
├── DOPPIO_gridded_data.ipynb
├── LATTE_app1_multiplatform_data.ipynb
├── LATTE_app2_multiplatform_data.ipynb

functions.py:                           Core functions for ESP model evaluation and parameter estimation used across all methods.
reporting.py:                           Shared printing utilities for the final manuscript results summaries.
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

CE23 timestamp correction (review revision, 10 November 2026):
The two ship-ADCP notebooks use `decode_ce23_mat_time` to apply a dataset-specific
12-hour subtraction validated against CSIRO Data Trawler UTC labels. Raw MAT
values are unchanged. SOLO selection anchors shift with the decoder to retain
the same observations. The ADCP–satellite LATTE example uses native satellite
times to interpolate at the corrected ADCP mean; other products' clocks are
unchanged. The final notebook summaries report the corrected times, source counts
and fitted parameters directly from each run.

Requirements:
Python 3.9 or higher
numpy
scipy
pandas
xarray
matplotlib
netCDF4

Usage:
Run notebooks from the repository root with functions.py and reporting.py available.
Open a notebook
Run cells sequentially

Each notebook ends with a **Manuscript results summary** cell. It prints the actual
data intervals and selection settings, observation counts at the fitting stages,
final parameters in manuscript units, velocity-reconstruction R², and additional
precision for comparisons. Tracking notebooks also print the actual snapshot date,
time-series ranges and their dates. The drifter-only comparison reports inner-core
results separately; unavailable outer-core parameters are not inferred.

Outer-fit diagnostics distinguish normalized tangential-fit R² from reconstructed
velocity R² and identify unsuccessful or unresolved fits. The summaries
also print package versions and file hashes. They report the current computation;
agreement with a manuscript or a reference publication run must be checked from
those values. Reporting does not change the fitting calculations.

Fitting and tracking:
The outer fit has no default upper limit on the fitted Gaussian radius `Rc`.
The 100-km observation-selection window is a separate sampling setting. An
optional `Rc_max` passed to `out_core_param_fit` is an explicit optimizer bound;
a solution reaching that bound is flagged as unavailable. Optimization uses
scaled variables and returns covariance in SI units. Failures, degenerate radial
sampling, and a numerically unresolved infinite-radius limit return NaN parameters,
not initial guesses. Pass `fit_details={}` to retain the status and diagnostic data.
The summaries include the fitted radius standard error; finite optimization does
not by itself establish adequate outer-core sampling or physical model validity.

LATTE rejects unsuccessful, rank-deficient or non-elliptic inner fits. Geometry
must be symmetric positive-definite with determinant one; absolute determinants
or eigenvalues are not used to disguise invalid ellipses. Invalid fits return NaN
parameters; an optional `fit_details` dictionary records the reason. The two-stage
selection uses the preliminary fitted centre and Q to refine the current sample.

Tracking advances only after a valid inner fit is within the configured distance
of the previous accepted centre and has the first accepted fit's rotation sign.
Either clockwise or anticlockwise eddies are supported. A failed outer fit leaves
its Gaussian reconstruction unavailable; separately stored `Omega_core` and
`w_core` retain valid inner-fit estimates for inspection.

DOPPIO retains its unshifted cubic basis. Its intercepts enforce the polynomial
roots exactly, while tangent coefficients are evaluated at those roots. This is
algebraically equivalent to shifting the polynomial to the root; the polynomial
intercept and the tangent-line intercept are different quantities.

Citation:
If you use this code, please cite:

Dowse, R. W., Keating, S. R., and Roughan, M.: A Geometric Framework for Mapping Ocean Eddies Using Elliptical Streamfunction Parameterisation, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2026-3599, 2026.

Reg Dowse (2026) “regdowse/elliptical_streamfunction_parametrisation: ESP”. Zenodo. doi:10.5281/zenodo.19309976.

Author:
Reg Dowse
UNSW Sydney
