# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: aiwqd
#     language: python
#     name: python3
# ---

# %%
"""
Climatology prediction
will always predict k/5 for the kth quintile

Example usages:
  python src/models/climatology/batch_predict.py era5-f1_tas 19 -t std_test
  for dates in std_future; do
  for var in tas pr mslp; do
    for f in f1 f2 f3 f4; do
      for horizon in 19 26; do
        python src/models/climatology/batch_predict.py era5-${f}_$var $horizon -t $dates
      done
    done
  done
  done

  python src/models/climatology/batch_predict.py era5-F10_tas 19 -t std_test
  for dates in std_test; do
  for var in tas pr mslp; do
    for f in F5 F95 F10 F90; do
      for horizon in 19 26; do
        src/batch/batch_python.sh -m 1 -c 1 -h 1 src/models/climatology/batch_predict.py era5-${f}_$var $horizon -t $dates
      done
    done
  done
  done


Positional args:
  gt_id: e.g. era5-f1_tas, era5-F95_tas
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction

The horizon isn't used in climatology prediction but is kept to ensure compatibility with the batch_metrics.py script.
"""
# %%
from utils.notebook import isnotebook
import os
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser
    
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from models.utils.data_utils import get_measurement_variable
from models.utils.general_util import printf, tic, toc
from models.utils.experiments_util import get_start_delta
from models.utils.eval_util import get_target_dates, mean_rmse_to_score
from models.utils.models_util import get_submodel_name
from models.utils.ecmwf_utils import geometric_median, ssm, mean
from utils.data_io import load_data, save_to_netcdf

# %%
#
# Specify model parameters
#
model_name = "climatology"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    args, opt = parser.parse_known_args()
    # Assign variables                                                                                                                          
    gt_id = args.pos_vars[0]
    horizon = args.pos_vars[1] 
    target_dates = args.target_dates
else:
    # Otherwise, specify arguments interactively 
    gt_id =  "era5-f4_tas" # e.g. "era5-pr", "era5-f1_tas"
    horizon = "19" # "19" or "26"
    target_dates = "std_test" #"std_test" or "std_tune"

# %%
""" 
Process model parameters
"""
task = f'{gt_id}_{horizon}'
measurement_variable = get_measurement_variable(gt_id) # e.g. "f1_tas" or "F95_pr"
if measurement_variable.startswith("f"):
    # Compute quantile from quintile number
    quantile = int(measurement_variable.split("_")[0].replace("f", "")) / 5.0 
elif measurement_variable.startswith("F"):
    # Compute quantile from percentile
    quantile = int(measurement_variable.split("_")[0].replace("F", "")) / 100.0
else: 
     raise ValueError(f"Measurement variable {measurement_variable} has no quintile or percentile")

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

# Set submodel name as "climatology" since there are not hyperparameters
submodel_name = "climatology"

# Construct preds dataset to fill in for each target date
# Prediction = quantile value for all locations
lats = np.arange(-90, 91.5, 1.5)
lons = np.arange(0, 360, 1.5)
preds_template = xr.DataArray(
    np.full((len(lats), len(lons)), quantile),
    dims=["latitude", "longitude"],
    coords={"latitude": lats, "longitude": lons}
)
# Convert from DataArray to Dataset
preds_template = preds_template.to_dataset(name=measurement_variable)


# %%
def prior_forecasts_exist(preds_f):
    return os.path.isfile(preds_f)


# %%
for target_date_obj in target_date_objs:
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')

    # Check if predictions netcdf store exists
    preds_f = Path(os.path.join('models', model_name, 'submodel_forecasts',
                                f'{submodel_name}', task, f'{task}-{target_date_str}.nc'))

    if prior_forecasts_exist(preds_f):
        printf(f"prior forecast exists for target {target_date_str}")
        #toc()
        continue
    else:
        printf(f"Getting climatology forecast for target {target_date_str}")
        # Add time dimension and save prediction to file
        save_to_netcdf(preds_template.expand_dims(time=[target_date_obj]), preds_f)
