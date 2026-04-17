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
Predicts outcomes using raw (undebiased) AIFS ensemble forecast

Example usages:
  python src/models/aifs/batch_predict.py era5-f1_tas 19 -t std_aifs_forecast

  for dates in std_aifs_forecast; do
  for var in tas mslp pr; do
    for f in f1 f2 f3 f4; do
      for horizon in 19 26; do
        src/batch/batch_python.sh --memory 10 --cores 1 --hours 1 src/models/aifs/batch_predict.py era5-${f}_${var} ${horizon} -t $dates   
      done
    done
  done
  done

Positional args:
  gt_id: e.g., era5-f1_tas
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction (default: std_aifs_forecast)
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

# %%
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from models.utils.data_utils import get_measurement_variable
from models.utils.general_util import printf, tic, toc
from models.utils.experiments_util import get_forecast_delta
from models.utils.eval_util import get_target_dates
from utils.data_io import load_data, save_to_netcdf

# %%
#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_aifs_forecast")
    args, opt = parser.parse_known_args()

    # Assign variables                                                                                                                          
    gt_id = args.pos_vars[0]
    horizon = args.pos_vars[1] 
    target_dates = args.target_dates
else:
    # Otherwise, specify arguments interactively 
    gt_id =  "era5-f1_mslp"
    horizon = "19"
    target_dates = "std_aifs_forecast"

# %%
measurement_variable = get_measurement_variable(gt_id)
task = f"{gt_id}_{horizon}"
target_date_objs = get_target_dates(target_dates, horizon=horizon)
aifs_delta = get_forecast_delta(horizon)

model_name = "aifs"
submodel_name = "aifs"

# %%
#
# Helper functions
#
def load_aifs_ds(forecast_type="forecast"):
    """Returns AIFS forecast or reforecast data
    
    Args:
        forecast_type: "forecast" or "reforecast"
    """
    # Load in AIFS data
    printf(f"Loading AIFS {forecast_type} data")
    tic()
    if gt_id.endswith("tas") or gt_id.endswith("pr"):
        aifs_forecast_ds = load_data(f"{model_name}-{forecast_type}-{measurement_variable}", lsmask=True).load()
    elif gt_id.endswith("mslp"):
        aifs_forecast_ds = load_data(f"{model_name}-{forecast_type}-{measurement_variable}", lsmask=False).load()
    toc()

    # Select the lead corresponding to the selected horizon
    aifs_forecast_ds = aifs_forecast_ds.sel(lead=np.timedelta64(int(horizon), 'D'), drop=True)

    return aifs_forecast_ds

# %%
# Load AIFS forecast data
forecast_data = load_aifs_ds("forecast")

tic()
forecast_data = forecast_data.rename({'issuance_date': 'time'})
printf(f'Shifting dataset by {aifs_delta} days')
forecast_data = forecast_data.assign_coords(time=forecast_data.time + pd.Timedelta(days=aifs_delta))
print(f"Dropping times with NA values")
forecast_data = forecast_data.dropna(dim='time', how='all')
toc()

# %%
# 
# Prepare forecast-based predictions when possible
# 
printf('Preparing forecast-based base predictions')
tic()
# Identify the forecast target dates
forecast_targets = forecast_data.get_index('time').intersection(target_date_objs) 
# Get base predictions for forecast targets
preds = forecast_data.sel(time=forecast_targets).sortby('time')
toc()

# %%
#
# Form predictions
#
# Predictions directory
preds_dir = os.path.join('models', model_name, 'submodel_forecasts',
                         f'{submodel_name}', task)
for target_date_obj in sorted(forecast_targets):
    # Skip if forecast already produced for this target
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    # Check if predictions exists
    preds_f = Path(os.path.join(preds_dir, f'{task}-{target_date_str}.nc'))
    if os.path.isfile(preds_f):
        printf(f"prior forecast exists for target {target_date_str}")
        continue
        
    printf(f'\nProcessing {model_name} forecast for {target_date_obj}')
    tic()
    # Extract base prediction using [] to preserve time dimension in the result
    pred = preds.sel(time=[target_date_obj])
    toc()

    # Save prediction to netcdf
    tic()   
    printf(f"Saving to {preds_f}")
    save_to_netcdf(pred, preds_f)
    toc()
