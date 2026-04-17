#!/usr/bin/env python
# coding: utf-8
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
#
# Predicts outcomes using MSN++
#
# Example usages:
#   python src/models/msnpp/batch_predict.py era5-f1_pr 26 -t std_test -i True -y 20 -m 28 -fl 15 -ll 15 
#
# Positional args:
#   gt_id: e.g., era5-tas, era5-pr, era5-mslp
#   horizon: 19 or 26
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --fit_intercept (-i): if "True" fits intercept to debias 
#     MSN predictions; if "False" does not fit intercept; (default: "False")
#   --years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: 20)
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --days (-d): number of daily MSN forecasts to average (default: 1)
#   --loss (-l): loss function in ["mse"] (default: "mse")
#   --first_lead (-fl): first MSN lead to average into forecast (0-29) (default: 0)
#   --last_lead (-ll): last MSN lead to average into forecast (0-29) (default: 29)
#   --overwrite (-o): overwrite existing prediction files (default: False)
import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser

import re
import pdb
import numpy as np
import pandas as pd
import xarray as xr
import calendar
from sklearn import linear_model
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
from models.utils.data_utils import get_measurement_variable, df_merge
from models.utils.experiments_util import (month_day_subset, get_start_delta, clim_merge, get_forecast_delta,
                                           get_forecast_variable, get_first_year)
from models.utils.eval_util import get_target_dates, mean_rmse_to_score, save_metric
from models.utils.models_util import (get_submodel_name, get_forecast_filename,
                                      save_forecasts)
from utils.data_io import load_data, save_to_netcdf
from utils.logging import printf
from utils.timing import tic, toc

#
# Specify model parameters
#
model_name = "msnpp"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon 
    parser.add_argument('--target_dates', '-t', default="std_test")
    # Fit intercept parameter if and only if this flag is specified
    parser.add_argument('--fit_intercept', '-i', default="False",
                        choices=['True', 'False'],
                        help="Fit intercept parameter if \"True\"; do not if \"False\"")    
    parser.add_argument('--years', '-y', default=20,
                       help="Number of years to use in debiasing ('all' or positive integer)")     
    parser.add_argument('--margin_in_days', '-m', default="None",
                       help="number of month-day combinations on either side of the target combination "
                            "to include when training; set to 0 include only target month-day combo; "
                            "set to None to include entire year")    
    parser.add_argument('--days', '-d', default=1,
                       help="number of daily MSN forecasts to average")
    parser.add_argument('--loss', '-l', default="mse", choices=["mse"],
                        help="loss function")
    parser.add_argument('--first_lead', '-fl', default=0, 
                        help="first MSN lead to average into forecast (0-29)")
    parser.add_argument('--last_lead', '-ll', default=29, 
                        help="last MSN lead to average into forecast (0-29)")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                       help="overwrite existing prediction files")
    args, opt = parser.parse_known_args()
    
    # Assign variables
    gt_id = args.pos_vars[0]
    horizon = args.pos_vars[1]
    target_dates = args.target_dates
    fit_intercept = args.fit_intercept   
    if fit_intercept == "False":
        fit_intercept = False
    elif fit_intercept == "True":
        fit_intercept = True
    else:
        raise ValueError(f"unrecognized value {fit_intercept} for fit_intercept")
    years = args.years
    if years != "all":
        years = int(years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)        
    days = int(args.days)    
    loss = args.loss
    first_lead = int(args.first_lead)
    last_lead = int(args.last_lead)
    overwrite = args.overwrite
else:
    # Otherwise, specify arguments interactively 
    gt_id = "era5-f1_tas" 
    horizon = "19" 
    target_dates = 'std_tune'
    fit_intercept = True    
    loss = "mse"
    years = 20
    margin_in_days = 35
    days = 7
    first_lead = 19
    last_lead = 26
    overwrite = True

""" 
Process model parameters
"""
task = f'{gt_id}_{horizon}'

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))
# Sort target_date_objs by day of week
# target_date_objs = target_date_objs[target_date_objs.dt.weekday.argsort(kind='stable')]

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id) 

# Identify column names
gt_col = measurement_variable
base_col = measurement_variable

# Store the number of days between a target date and the associated MSN model issuance date
msn_delta = get_forecast_delta(horizon)
# This is also the amount of shift we will apply to the (re)forecast dates
base_shift_delta = timedelta(days=msn_delta)
# Store delta between target date and last observable training date
# (for weekly aggregated targets, this will be equal to base_shift_delta + 7)
start_delta =  timedelta(days=get_start_delta(horizon, gt_id)) 

# Record model and submodel names
submodel_name = get_submodel_name(
    model_name, fit_intercept=fit_intercept, years=years, 
    margin_in_days=margin_in_days, days=days, loss=loss, 
    first_lead=first_lead, last_lead=last_lead)
print(submodel_name)


# %%
#
# Helper functions
#
def load_msn_df(forecast_type="forecast"):
    """Returns MSN forecast or reforecast data averaged over leads 
    [first_lead, last_lead] as an xarray dataset
    
    Args:
        forecast_type: "forecast" or "reforecast"
    """
    # Load in MSN data
    printf(f"Loading MSN {forecast_type} data averaged over leads [{first_lead},{last_lead}]")
    tic()
    if gt_id.endswith("pr"): 
        msn_forecast_ds = load_data(f"model_poet-ec-v4-{forecast_type}-{measurement_variable}", lsmask=True)
    elif gt_id.endswith("tas"): 
        msn_forecast_ds = load_data(f"model_poet-ec-v5-{forecast_type}-{measurement_variable}", lsmask=True)
    elif gt_id.endswith("mslp"): 
        msn_forecast_ds = load_data(f"model_poet-ec-v5-{forecast_type}-{measurement_variable}", lsmask=False)
    toc()

    # Average values over specified lead range
    printf(f"Averaging leads {first_lead} to {last_lead}")
    tic()
    lead_range = slice(np.timedelta64(first_lead, 'D'), np.timedelta64(last_lead, 'D'))
    msn_forecast_ds = msn_forecast_ds.sel(lead=lead_range).mean(dim='lead').load()
    toc()
    
    return msn_forecast_ds


# %%
#
# Load MSN reforecast data
#
print("Loading MSN reforecast data")
reforecast_data = load_msn_df(forecast_type="reforecast")

printf('Reindexing using target date start time')
tic()
# Stack model_issuance_date and start_year into a multi-index
reforecast_data = reforecast_data.stack(time=("issuance_date","year_delta"))
# Compute each start_date, the first day of the target period being predicted
# by this reforecast
index = reforecast_data.get_index('time')
model_issuance_dates = index.get_level_values(0)
# Note: start_years are negative
start_years = index.get_level_values(1)
start_dates = pd.to_datetime({
    'year': model_issuance_dates.year + start_years,
    'month': model_issuance_dates.month, 
    'day': model_issuance_dates.day
}) + base_shift_delta

reforecast_data = reforecast_data.drop_vars(['time', 'issuance_date', 'year_delta'])
reforecast_data = reforecast_data.assign_coords(time=start_dates)
toc()
print(f"Dropping times with NA values")
tic()
reforecast_data = reforecast_data.dropna(dim='time', how='all')
toc()
printf("Dropping duplicates and sorting by time")
tic()
# For duplicated start_dates, preserve the one with the earliest model_issuance_date (earliest reforecast)
reforecast_data = reforecast_data.drop_duplicates('time', keep='first')
# Sort by time
reforecast_data = reforecast_data.sortby('time')
toc()

if days > 1:
    printf(f"Computing rolling mean over {days} days")
    tic()
    # Extract dates on which reforecasts are available
    available_reforecast_dates = reforecast_data.time
    # Upsample reforecasts to daily frequency to enable rolling mean computation
    reforecast_data = reforecast_data.resample(time="1D").mean()
    # Compute rolling mean over days time indices
    reforecast_data = reforecast_data.rolling(time=int(days), min_periods=1, center=False).mean()
    # Restrict to dates on which reforecasts were originally available
    reforecast_data = reforecast_data.sel(time=available_reforecast_dates)
    toc()

# %%
#
# Load MSN forecast data
#
print("Loading MSN forecast data")
forecast_data = load_msn_df(forecast_type="forecast")

tic()
forecast_data = forecast_data.rename({'issuance_date': 'time'})
print(f"Dropping times with NA values")
forecast_data = forecast_data.dropna(dim='time', how='all')
printf(f'Shifting dataset by {msn_delta} days')
forecast_data = forecast_data.assign_coords(time=forecast_data.time + pd.Timedelta(days=msn_delta))
if days > 1:
    printf(f"Computing rolling mean over {days} days")
    # Extract dates on which forecasts are available
    available_forecast_dates = forecast_data.time
    # Upsample forecasts to daily frequency to enable rolling mean computation
    forecast_data = forecast_data.resample(time="1D").mean()
    # Compute rolling mean over days time indices
    forecast_data = forecast_data.rolling(time=int(days), min_periods=1, center=False).mean()
    # Restrict to dates on which forecasts were originally available
    forecast_data = forecast_data.sel(time=available_forecast_dates)
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

# 
# Prepare reforecast-based predictions for earlier target dates
# 
printf('Preparing reforecast-based base predictions')
tic()
# The first target date for which an MSN forecast exists
first_forecast_target = forecast_targets.min()
# Identify the reforecast target dates: those before first forecast target date
# with reforecast start_dates
reforecast_targets = reforecast_data.get_index('time').intersection(target_date_objs)
if not forecast_targets.empty: 
    reforecast_targets = reforecast_targets[reforecast_targets < first_forecast_target]
# Concatenate base predictions for reforecast targets
preds = xr.concat([reforecast_data.sel(time=reforecast_targets), preds], dim='time').sortby('time')
toc()

# 
# For debiasing, concatenate forecast and reforecasts, deduplicating in favor of forecasts
# 
printf('Concatenate forecast and reforecasts, deduplicate in favor of forecasts, and sort by time')
tic()
all_data = xr.concat([reforecast_data, forecast_data], dim='time')
del reforecast_data, forecast_data
# Drop duplicates before sorting in case sorting does not preserve original order
all_data = all_data.drop_duplicates(dim='time', keep='last')
all_data = all_data.sortby('time')
toc()

# %%
"""
Load and merge ground truth 
"""
printf("Loading ground truth data")
tic()
if gt_id.endswith("tas") or gt_id.endswith("pr"):
    gt_ds = load_data(gt_id, lsmask=True)
elif gt_id.endswith("mslp"): 
    gt_ds = load_data(gt_id, lsmask=False)
else:
    raise ValueError(f"Unknown ground truth id {gt_id}")
toc()

tic()
# Restrict to times in all_data
gt_times = all_data.get_index('time').intersection(gt_ds.time)
gt_ds = gt_ds.sel(time=gt_times).load()
toc()

# %%
# 
# Prepare data for bias correction
# 
printf('Preparing data for bias correction')
tic()
# Use ground truth - prediction as a basis for bias correction
debias_data = gt_ds - all_data
del all_data
toc()

# %%
#
# Form predictions for reforecast targets and evaluate and save all predictions
#
printf('Making reforecast-based predictions and saving all predictions')
days_per_year = 365.242199
valid_targets = forecast_targets.union(reforecast_targets)
# Compute cosine weights for each latitude
lats = gt_ds.latitude
weights = xr.ufuncs.cos(xr.ufuncs.deg2rad(xr.ufuncs.abs(lats))).astype("float32")
# Dataframe to store date relationships when computing reforecast predictions
X = pd.DataFrame(index=debias_data.time, 
                 columns = ["delta", "dividend", "remainder"], 
                 dtype=np.float64)
# Initialize weighted MSE sum and count for computing mean wtd_mse
wtd_mse_sum = 0.0
wtd_mse_count = 0
# Predictions directory
preds_dir = os.path.join('models', model_name, 'submodel_forecasts',
                         f'{submodel_name}', task)
for target_date_obj in sorted(valid_targets):
    # Skip if forecast already produced for this target
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    # Check if predictions exists
    preds_f = Path(os.path.join(preds_dir, f'{task}-{target_date_str}.nc'))
    if not overwrite and os.path.isfile(preds_f):
        printf(f"prior forecast exists for target {target_date_str}")
        continue
        
    printf(f'\nProcessing {model_name} forecast for {target_date_obj}')
    tic()
    printf(f"Preparing covariates for {target_date_str}")
    # Compute days from target date
    X['delta'] = (target_date_obj - pd.to_datetime(debias_data.time.values)).days
    # Extract the dividend and remainder when delta is divided by the number of days per year
    # The dividend is analogous to the year
    # (Negative values will ultimately be excluded)
    X['dividend'] = np.floor(X.delta / days_per_year)
    # The remainder is analogous to the day of the year
    X['remainder'] = np.floor(X.delta % days_per_year)
    # Find the last observable training date for this target
    last_train_date = target_date_obj - start_delta
    # Restrict data based on training date, dividend, and remainder
    indic = (X.index <= last_train_date)
    if margin_in_days is not None:
        indic &= ((X.remainder <= margin_in_days) | (X.remainder >= 365-margin_in_days))
    if years != "all":
        indic = indic & (X.dividend < years)
    toc()
    printf(f'Fitting {model_name} model with loss {loss} for {target_date_obj}')
    tic()
    # Extract base prediction using [] to preserve time dimension in the result
    pred = preds.sel(time=[target_date_obj])
    if fit_intercept: 
        if not indic.any():
            printf(f'-Warning: no training data for {target_date_str}; skipping')
            continue
        else:
            # Add bias correction
            pred += debias_data.isel(time=indic).mean(dim="time")
        
    # Clip probabilities to between 0 and 1
    pred = pred.clip(min=0, max=1)
    toc()

    # Save prediction to netcdf
    tic()   
    printf(f"Saving to {preds_f}")
    save_to_netcdf(pred, preds_f)
    toc()

    # Evaluate error if we have ground truth data
    tic()
    if target_date_obj in gt_ds.time:
        #tic()
        wtd_mse = (np.square(pred - gt_ds.sel(time=target_date_obj))).weighted(weights).mean()[measurement_variable].values
        print("-wtd_mse: {}".format(wtd_mse))
        wtd_mse_sum += wtd_mse
        wtd_mse_count += 1
        print("-mean wtd_mse: {}".format(wtd_mse_sum/wtd_mse_count))
        #toc()
    toc()
