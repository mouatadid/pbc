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
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#
# Persistence++ with ECMWF
#
# Regress onto ECMWF forecasts, climatology, and lagged measurements
#
# Example usage:
#   python src/models/perpp_ecmwf/batch_predict.py era5-f1_tas 19 -t std_test -y all -m None 
#   python src/models/perpp_ecmwf/batch_predict.py era5-F95_pr 19 -t std_test -y all -m None 
#
# Positional args:
#   gt_id: era5-f1_tas, era5-f1_pr, era5-f1_mslp, era5-F10_tas, era5-F90_pr, era5-F95_mslp, etc.
#   horizon: 19 or 26
#
# Named args:
#   --target_dates (-t): target dates for batch prediction 
#   --train_years (-y): number of years to use in training ("all" or integer)
#   --margin_in_days (-m): number of month-day combinations on either side of the target combination to include when training
#     Set to 0 to include only target month-day combo
#     Set to "None" to include entire year
#  --date_order_seed (-s): if None, sort target_dates in order (as usual), otherwise randomize order of target_dates
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

# Imports
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import linear_model
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
from models.utils.data_utils import get_measurement_variable
from models.utils.general_util import printf, tic, toc
from models.utils.experiments_util import get_start_delta, get_forecast_delta, get_forecast_delta
from models.utils.eval_util import get_target_dates
from models.utils.models_util import get_submodel_name
from utils.data_io import load_data, save_to_netcdf
from datetime import datetime, timedelta
from pathlib import Path
import re

# %%
#
# Specify model parameters
#
model_name = "perpp_ecmwf"
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--train_years', '-y', default="all",
                        help='number of years to use in debiasing ("all" or integer)')
    parser.add_argument('--margin_in_days', '-m', default="None",
                        help="number of month-day combinations on either side of the target combination "
                             "to include when training; set to 0 include only target month-day combo; "
                             "set to None to include entire year")
    parser.add_argument('--date_order_seed', '-s', default="None",
                        help="if None, sort target_dates in order (as usual), otherwise randomize order of target_dates")
    args, opt = parser.parse_known_args()

    # Assign variables
    gt_id = args.pos_vars[0] 
    horizon = args.pos_vars[1] 
    target_dates = args.target_dates
    train_years = args.train_years
    if train_years != "all":
        train_years = int(train_years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    if args.date_order_seed == "None": 
        date_order_seed = None
    else:
        date_order_seed = int(args.date_order_seed)
else:
    # Otherwise, specify arguments interactively
    gt_id = "era5-F10_tas"
    horizon = "19"
    target_dates = "std_future" 
    train_years = "all"
    margin_in_days = None
    date_order_seed = None

if margin_in_days is not None:
    raise ValueError("margin_in_days is not currently supported in this model. Set to None.")
if train_years != "all":
    raise ValueError("train_years is not currently supported in this model. Set to 'all'.")

# %%
#
# Process model parameters
#
task = f'{gt_id}_{horizon}'
agg_period = 7
clim_years = 20 # number of years in climatology period
clim_margin = 4 # number of days on either side of the target date to include in climatology

# Get list of target date objects
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

# Randomize target dates if date_order_seed is specified
if date_order_seed is not None:
    print("Shuffling order of target dates")
    target_date_objs = target_date_objs.sample(frac=1, random_state=date_order_seed)

# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_id)  # f1_tas, f1_pr, f1_mslp, etc.

# For a given target date, the last observable training date is target date - gt_delta
# as gt_delta is the gap between the start of the target date and the start of the
# last ground truth period that's fully observable at the time of forecast issuance
gt_delta = timedelta(days=get_start_delta(horizon, gt_id))

# The number of days between a target date and the associated ECMWF model issuance date
ecmwf_delta = get_forecast_delta(horizon)

# %%
#
# Choose regression parameters
#
gt_col = measurement_variable
clim_col = measurement_variable + "_clim"
ecmwf_col = f"iri_ecmwf_{measurement_variable}"
first_shift = int(gt_delta.days)
x_cols = [
    f"{measurement_variable}_shift{first_shift}",
    f"{measurement_variable}_shift{first_shift + int(horizon) - 1}",
    ecmwf_col,
    clim_col, 
    'ones'
]
group_by_cols = ['latitude', 'longitude']
pred_cols = x_cols
exclude_cols = set([clim_col, ecmwf_col])

# Record submodel names for perpp model
submodel_name = get_submodel_name(
    model_name, train_years=train_years, margin_in_days=margin_in_days, clim_years=clim_years)
printf(f"Submodel name {submodel_name}")

# %%
## Ground truth
#
# Load ground truth data
#
printf("Loading ground truth data")
tic()
if gt_id.endswith("tas") or gt_id.endswith("pr"):
    gt_ds = load_data(gt_id, lsmask=True)
elif gt_id.endswith("mslp"): 
    gt_ds = load_data(gt_id, lsmask=False)
else:
    raise ValueError(f"Unknown ground truth id {gt_id}")
# Explicitly load gt_ds into memory (note: occurs inplace)
# This prevents extreme chunking during the rolling climatology phase
gt_ds = gt_ds.load()
toc()

# Store all lat/lon coordinates for reindexing later
all_lats = gt_ds.latitude.values.copy()
all_lons = gt_ds.longitude.values.copy()

# Drop NA latitudes
gt_ds = gt_ds.dropna(dim='latitude', how='all')

# %%
#
# Add lagged ground truth measurements
#
printf("Adding lagged measurements")
shifts = [int(re.search(r'\d+$', col).group()) for col in x_cols if col.startswith(gt_col+"_shift")]
gt_times = gt_ds.time.get_index('time')
lld_data = gt_ds
for shift in shifts:
    tic()
    # Get copy of original gt data with renamed variable
    gt_shift = gt_ds.rename({measurement_variable: f"{measurement_variable}_shift{shift}"})
    # Shift dates forward so that lagged measurements from shift days in the 
    # past are associated with a given date
    gt_shift['time'] = gt_times.shift(freq = f"{shift}D")
    # Join right since lagged measurements are required for prediction
    lld_data = xr.merge([lld_data, gt_shift], join="right")
    del gt_shift
    toc()

# %%
#
# Adding rolling climatology
#
if clim_col in pred_cols:
    printf(f"Adding {clim_years}-year rolling climatology with {clim_margin}-day margin")
    # Check if measurement_variable is probabilistic quintile (fk_*) target
    match = re.match(r"f(\d+)", measurement_variable)
    if match:
        # For probabilistic (fk_*) targets, pad ground truth with clim_years-1 years of default values,
        # where default value is .2 times k (the nominal probability of belonging to the k-th quintile bin)
        fill_value = int(match.group(1)) * 0.2
        printf(f"Padding ground truth with {clim_years-1} years of default value {fill_value} for rolling climatology")
        tic()
        time_index = gt_ds.time.get_index('time')
        # Start on January 1st of the year clim_years-1 years before the first time in the dataset
        first_time = time_index[0]
        first_time = first_time.replace(year=first_time.year - (clim_years - 1), day=1, month=1)
        last_time = time_index[-1]
        new_time_index = pd.date_range(start=first_time, end=last_time, freq='D')
        rolling_clim = gt_ds.reindex(time=new_time_index, fill_value=fill_value)  
        toc()
    else:
        # Check if measurement_variable is a probabilistic percentile (Fp_*) target
        match = re.match(r"F(\d+)", measurement_variable)
        if match:
            # For probabilistic (Fp_*) targets, pad ground truth with clim_years-1 years of default values,
            # where default value is p / 100. (the nominal value of the p-th percentile bin)
            fill_value = int(match.group(1)) / 100.0
            printf(f"Padding ground truth with {clim_years-1} years of default value {fill_value} for rolling climatology")
            tic()
            time_index = gt_ds.time.get_index('time')
            # Start on January 1st of the year clim_years-1 years before the first time in the dataset
            first_time = time_index[0]
            first_time = first_time.replace(year=first_time.year - (clim_years - 1), day=1, month=1)
            last_time = time_index[-1]
            new_time_index = pd.date_range(start=first_time, end=last_time, freq='D')
            rolling_clim = gt_ds.reindex(time=new_time_index, fill_value=fill_value)  
            toc()
        else:
            # Otherwise, use ground truth data without padding
            rolling_clim = gt_ds
    # Ignore leap days
    printf(f"Dropping leap days")
    tic()
    rolling_clim = rolling_clim.sel(time=~((rolling_clim.time.dt.month == 2) & (rolling_clim.time.dt.day == 29)))
    toc()
    # Compute rolling climatology over clim_years years for each month-day combination;
    # each resulting entry will contain the mean of its value and the values from the
    # past clim_years-1 years
    printf(f"Rolling over {clim_years} years for each month-day combination")
    def groupby_rolling(sub_ds, periods=clim_years, min_periods=clim_years):
        return sub_ds.rolling(time=periods, min_periods=min_periods, 
                              center=False).mean()
    tic()
    month_day_str = rolling_clim.time.dt.strftime("%m-%d")
    # Drop dates with no available climatology
    rolling_clim = rolling_clim.groupby(month_day_str).map(groupby_rolling).dropna("time", how="all")
    toc()
    # Shift time forward one year to so that climatology for a date is based only 
    # on prior years; each resulting entry will contain the mean of past clim_years years' values
    tic()
    rolling_clim['time'] = rolling_clim.time.get_index('time').shift(freq = pd.DateOffset(years=1))
    toc()
    # Next compute rolling mean of rolling_clim over 2*clim_margin+1 days;
    # with center=True, each resulting entry will contain the mean of its value and the values
    # within clim_margin days on either side of the date
    printf(f"Rolling over margin of +/- {clim_margin} days")
    tic()
    # Drop dates with no available climatology
    rolling_clim = rolling_clim.rolling(time=2*clim_margin+1, center=True).mean().dropna("time", how="all")
    toc()
    # Identify variable as climatology
    rolling_clim = rolling_clim.rename({measurement_variable: clim_col})
    # Add climatology to dataset
    printf("Adding rolling climatology to dataset")
    tic()
    # Join right since climatology is required for prediction
    lld_data = xr.merge([lld_data, rolling_clim], join="right")
    del rolling_clim
    toc()

# %%
#
# Add ones (intercept) column
#
if 'ones' in pred_cols:
    printf("Adding ones")
    tic()
    lld_data.update({'ones': xr.ones_like(lld_data[measurement_variable])})
    toc()

# %%
#
# Convert to pandas dataframe
#
printf("Sorting data by time")
tic()
lld_data = lld_data.sortby('time')
toc()

printf("Converting to dataframe")
tic()
del gt_ds
lld_data = lld_data.to_dataframe()
toc()
#
# Drop rows with empty pred_cols
#
printf("Dropping empty predictions")
tic()
lld_data = lld_data.dropna(subset=set(pred_cols) - exclude_cols)
toc()

printf("Reordering index")
tic()
lld_data = lld_data.reorder_levels(['latitude', 'longitude', 'time'])
toc()


# %%
def load_ecmwf_df(forecast_type="forecast", lead=ecmwf_delta+1):
    """Returns ECMWF forecast or reforecast data for lead lead as a dataframe
    
    Args:
        forecast_type: "forecast" or "reforecast"
        lead: lead time in days (lead = ecmwf_delta+1 uses lead k to predict day k)
    """
    printf(f"Loading ECMWF {forecast_type} data with lead {lead}")
    tic()
    if gt_id.endswith("tas") or gt_id.endswith("pr"):
        ecmwf_forecast_ds = load_data(f"ecmwf-{forecast_type}-{measurement_variable}", lsmask=True)
    elif gt_id.endswith("mslp"):
        ecmwf_forecast_ds = load_data(f"ecmwf-{forecast_type}-{measurement_variable}", lsmask=False)
    toc()

    # Select single lead
    # drop=True drops the singleton lead coordinate
    return ecmwf_forecast_ds.sel(lead=f'{lead} days', drop=True)


# %%
#
# Load ECMWF reforecast data
#
reforecast_data = load_ecmwf_df(forecast_type="reforecast")
printf('Reindexing using start_date')
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
}) + timedelta(days=ecmwf_delta)
# Replace old indices with start_date
reforecast_data = reforecast_data.drop_vars(['time', 'issuance_date', 'year_delta'])
reforecast_data = reforecast_data.assign_coords(time=start_dates)
toc()
printf("Dropping duplicates and sorting by time")
tic()
# For duplicated start_dates, preserve the one with the latest model_issuance_date (most recent reforecast)
reforecast_data = reforecast_data.drop_duplicates('time', keep='last')
# Sort by time
reforecast_data = reforecast_data.sortby('time')
# Store reforecast coordinate names in order
reforecast_coords = list(reforecast_data.coords)
toc()
# Convert to Series
printf("Converting to Series")
tic()
reforecast_data = reforecast_data.to_array(name=ecmwf_col).squeeze().to_series()
reforecast_data.dropna(inplace=True)
toc()

# %%
#
# Load ECMWF forecast data
#
forecast_data = load_ecmwf_df(forecast_type="forecast")
# Shift issuance_date to match start of the target period being predicted
# rather than the issuance date
tic()
date_name = 'issuance_date'
forecast_data = forecast_data.rename({date_name: 'time'})
forecast_data = forecast_data.assign_coords(time=forecast_data.get_index("time").shift(freq = f"{ecmwf_delta}D"))
toc()
printf("Reordering coordinates")
tic()
# Reorder coordinates to match reforecast_data
forecast_data = forecast_data.transpose(*reforecast_coords)
toc()
# Convert to Series
printf("Converting to Series")
tic()
forecast_data = forecast_data.to_array(name=ecmwf_col).squeeze().to_series()
forecast_data.dropna(inplace=True)
toc()
# Ensure same index level order as reforecast_data
printf("Reordering index levels")
tic()
forecast_data = forecast_data.reorder_levels(reforecast_data.index.names)
toc()

# %%
# Concatenate ECMWF forecast and reforecast dataframes
printf(f"Combining forecast and reforecast data")
tic()
ecmwf = pd.concat([forecast_data, reforecast_data])
del forecast_data, reforecast_data
# Drop duplicated indices, preserving the first (prioritizes forecasts over reforecasts)
ecmwf = ecmwf.loc[~ecmwf.index.duplicated(keep="first")]
# Sort index
ecmwf.sort_index(inplace=True)
toc()

# %%
# Merge ecmwf features with lld_data
printf(f"Merging {ecmwf_col} with lld_data")
tic()
lld_data = lld_data.join(ecmwf, how="inner")
del ecmwf
toc()
tic()
lld_data = lld_data.sort_index()
toc()

# %%
# Restrict data to relevant columns and rows for which predictions can be made
relevant_cols = list(set([gt_col]+x_cols).intersection(lld_data.columns))
lld_data = lld_data[relevant_cols].dropna(subset=x_cols)


# %%
def apply_parallel(df_grouped, func, num_cores=os.process_cpu_count(), **kwargs):
    """Apply func to each group dataframe in df_grouped in parallel

    Args:
        df_grouped: output of grouby applied to pandas DataFrame
        func: function to apply to each group dataframe in df_grouped
        num_cores: number of CPU cores to use
        kwargs: additional keyword args to pass to func
    """
    # Associate only one OpenMP thread with each core
    os.environ['OMP_NUM_THREADS'] = str(1)
    pool = Pool(num_cores)
    # Pass additional keyword arguments to func using partial
    results = pool.map(partial(func, **kwargs), [group for name, group in df_grouped])
    pool.close()
    # Unset environment variable
    del os.environ['OMP_NUM_THREADS']

    # Separate predictions and coefficients
    ret_list = [result[0] for result in results]
    coef_list = [result[1] for result in results]

    return pd.concat(ret_list, ignore_index=False), pd.concat(coef_list, ignore_index=False)

def fit_and_predict(df, gt_col=None, x_cols=None, target_dates=None, gt_delta=None):
    """Fits model using rolling linear regression framework for a single gridpoint.

    Args:
        df: Dataframe with 'time' in index and gt_col, x_cols columns
        gt_col: Name of ground truth column in df
        x_cols: Names of columns used as input features
        target_dates: Nonempty list of target dates in ascending order for prediction
        gt_delta: Timedelta for computing last training date

    Returns DataFrame mapping target_date to prediction dataframe
    """
    # Initialize sufficient statistics with training data from first target date
    last_stored_date = target_dates[0] - gt_delta
    t = df.index.get_level_values('time')
    date_block = (t <= last_stored_date)
    train_data = df[date_block]
    n_train = len(train_data)

    if n_train > 0:
        X_train = train_data[x_cols].values
        y_val = train_data[gt_col].values
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_val
    else: 
        # If no training data, initialize sufficient statistics to zero
        n_features = len(x_cols)
        XtX = np.zeros((n_features, n_features))
        Xty = np.zeros(n_features)

    # Store predictions for all target dates
    all_predictions = []
    all_coefficients = []
    for target_date in target_dates:
        # Find the last observable training date for this target
        last_train_date = target_date - gt_delta        
        date_block = ((t <= last_train_date) & (t > last_stored_date))
        new_data = df[date_block]

        last_stored_date = last_train_date
        n_train += len(new_data)

        if len(new_data) > 0:
            X_train = new_data[x_cols].values
            y_val = new_data[gt_col].values

            # Update sufficient statistics
            XtX += X_train.T @ X_train
            Xty += X_train.T @ y_val

        # If there's at least one training date
        if n_train > 0:
            try:
                # Solve linear system when XtX full rank
                coef = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Otherwise, find minimum norm solution
                coef = np.linalg.lstsq(XtX, Xty)[0]
            
            # Store prediction in dataframe
            X_test = df.loc[t == target_date, x_cols]
            pred_df = pd.DataFrame(
                {gt_col: [np.dot(X_test.values[0], coef)]},
                index=X_test.index
            )
            all_predictions.append(pred_df)

            # Store coefficients with feature names
            coef_data = {f'coef_{x_cols[i]}': coef[i] for i in range(len(x_cols))}
            coef_df = pd.DataFrame([coef_data], index=X_test.index)
            all_coefficients.append(coef_df)
    
    # return single dataframe with all predictions
    return pd.concat(all_predictions, ignore_index=False), pd.concat(all_coefficients, ignore_index=False)


# %%
# Predictions directory
preds_dir = os.path.join('models', model_name, 'submodel_forecasts',
                         submodel_name, task)

# Sort target dates in ascending order
sorted_targets = sorted(target_date_objs.tolist())

# Check if target dates already have predictions or if some features
# are unavailable for prediction
printf("Identifying viable target dates for prediction")
tic()
dates_with_features = lld_data[x_cols].index.unique(level='time')
targets_to_process = []
for target_date_obj in sorted_targets:
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
    if os.path.exists(preds_f):
        printf(f"prior forecast exists for target {target_date_str}")
        continue
    if target_date_obj not in dates_with_features:
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        continue
    targets_to_process.append(target_date_obj)
toc()
        
if not targets_to_process:
    printf("No target dates available for prediction; exiting")
else:
    printf(f"Processing {len(targets_to_process)} target dates with rolling regression")
    # Apply rolling regression to each grid point in parallel
    prediction_func = partial(fit_and_predict, 
                            gt_col=gt_col, 
                            x_cols=x_cols, 
                            target_dates=targets_to_process,
                            gt_delta=gt_delta)
    num_cores = os.process_cpu_count()
    tic()
    all_preds, all_coefs = apply_parallel(
        lld_data.groupby(group_by_cols),
        prediction_func,
        num_cores=num_cores
    )
    toc()
    printf(f"Clipping probabilistic forecasts to [0, 1]")
    tic()
    all_preds = all_preds.clip(0,1)
    toc()

    # Save predictions for each target date to disk
    t = all_preds.index.get_level_values('time')
    for target_date_obj in targets_to_process:
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        printf(f'Saving predictions for target {target_date_str}')
        
        tic()
        # Get predictions for this date
        date_preds = all_preds[t == target_date_obj]
        # Convert to xarray and reindex to full lat-lon grid
        preds_i = date_preds.to_xarray()
        preds_i = preds_i.reindex(
            time=[target_date_obj], 
            latitude=all_lats, 
            longitude=all_lons, 
            fill_value=np.nan
        )
        toc()

        # Save prediction 
        tic()
        preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
        printf(f"Saving to {preds_f}")
        save_to_netcdf(preds_i, Path(preds_f))
        toc()


# %%
