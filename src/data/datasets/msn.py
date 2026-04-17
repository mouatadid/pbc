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
# Download MSN forecasts and reforecasts
#
# Example usages: 
#  src/batch/batch_python.sh src/data/datasets/msn.py -wv tas -d 20250811

import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# Imports
import argparse
import numpy as np
import pandas as pd

import xarray as xr
from utils.data_io import (
    load_data, 
    save_data, 
    DATA_DIR
)
from utils.timing import tic, toc
from utils.logging import printf

from data.helpers.general_utils import print_ok

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tas",
    choices=["tas", "pr", "mslp"],
    help="Name of weather variable to process",
)
parser.add_argument(
    "--reforecast",
    "-ref",
    action="store_true",
    help="If true, process reforecast (i.e., the hindcasts for the last 20 years)",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print verbose output",
)
parser.add_argument(
    "--date",
    "-d",
    default=None,
    help="Date to process (format: YYYYMMDD). If None, process all issuances from 2024.",
)

if isnotebook():
    args = parser.parse_known_args(["-v", "-wv", "tas", "-d", "20250731"])[0]
else:
    args = parser.parse_args()


#
# Process arguments
#
weather_variable = args.weather_variable
reforecast = args.reforecast
forecast_type = "reforecast" if reforecast else "forecast"
if args.date == "None": 
    date = None
else: 
    date = args.date
# Use v4 for pr, v5 for tas and mslp
if weather_variable == 'pr': 
    model = 'poet-ec-v4'
    model_subfolder = 'poet-ec-weekly-v4-pr'
elif weather_variable in ['tas', 'mslp']: 
    model = 'poet-ec-v5'
    model_subfolder = 'poet-ec-weekly-v5'
print_ok(model, bold=True, verbose=args.verbose)
if (date is not None) and reforecast:
    raise ValueError("Single date processing is not currently supported for reforecasts.")


# %%
def issuance_to_target(issuance_dates, leads):   
    """
    Returns the corresponding target dates (the start dates of the target periods being forecasted)
    for each (issuance date, lead pair).
    
    Args:
        issuance_dates: datetime array of issuance dates.
        leads: timedelta array of lead times.
        
    Returns:
        datetime array of target dates
    """
    # Lead 1 forecasts day 1 = issuance_date 
    return np.add.outer(issuance_dates, leads - np.timedelta64(1, "D"))


# %%
# Load quintile boundaries 
printf(f"Loading ERA5 quintiles for measurement: {weather_variable}")
tic()
quintiles = load_data(f"era5-quintiles-{weather_variable}")
# Load data explicitly into memory
quintiles.load()
quantiles = quintiles["quantile"].values
toc()

# Load in MSN forecasts/reforecasts
if date == "2025": 
    printf(f"Loading MSN {forecast_type} on {date} for measurement: {weather_variable}")
    tic()
    if weather_variable == 'pr': 
        filename = os.path.join(DATA_DIR, "msn", "mai-forecasts", f"{model_subfolder}-2025", f"model_{model}-{weather_variable}_forecast.zarr")
        ds = xr.open_zarr(filename, decode_timedelta=True)
    elif weather_variable in ['tas', 'mslp']: 
        filename = os.path.join(DATA_DIR, "msn", "mai-forecasts", f"{model_subfolder}-2025", f"model_{model}_forecast.zarr")
        ds = xr.open_zarr(filename, decode_timedelta=True)[[weather_variable]] # Open relevant variable forecasts as dataset
    toc()
elif date is not None:
    printf(f"Loading MSN {forecast_type} on {date} for measurement: {weather_variable}")
    tic()
    filename = os.path.join(DATA_DIR, "msn", "mai-forecasts", date + "00", "forecast.zarr")
    if weather_variable == "pr": 
        ds = xr.open_zarr(filename, decode_timedelta=True)[['scaled_pr']] # Open relevant variable reforecasts as dataset
    elif weather_variable in ["tas", "mslp"]: 
        ds = xr.open_zarr(filename, decode_timedelta=True)[[weather_variable]] # Open relevant variable reforecasts as dataset
    toc()
else: 
    printf(f"Loading MSN {forecast_type}s for measurement: {weather_variable}")
    tic()
    if reforecast: 
        if weather_variable == "pr": 
            filename = os.path.join(DATA_DIR, "msn", "mai-hindcasts", model_subfolder, "hindcasts.zarr")
            ds = xr.open_zarr(filename, decode_timedelta=True)
        elif weather_variable in ['tas', 'mslp']: 
            filename = os.path.join(DATA_DIR, "msn", "mai-hindcasts", model, "hindcasts.zarr")
            ds = xr.open_zarr(filename, decode_timedelta=True)[[weather_variable]] # Open relevant variable reforecasts as dataset
    else: 
        if weather_variable == 'pr': 
            filename = os.path.join(DATA_DIR, "msn", "mai-forecasts", model_subfolder, f"model_{model}-{weather_variable}_forecast.zarr")
            ds = xr.open_zarr(filename, decode_timedelta=True)
        elif weather_variable in ['tas', 'mslp']: 
            filename = os.path.join(DATA_DIR, "msn", "mai-forecasts", model_subfolder, f"model_{model}_forecast.zarr")
            ds = xr.open_zarr(filename, decode_timedelta=True)[[weather_variable]] # Open relevant variable forecasts as dataset
    toc()

# Clear all encoding to avoid zarr v2/v3 compatibility issues
for var in ds.data_vars:
    if hasattr(ds[var], 'encoding'):
        ds[var].encoding.clear()
for coord in ds.coords:
    if hasattr(ds[coord], 'encoding'):
        ds[coord].encoding.clear()

# Precip unscaling conversion
if weather_variable == 'pr': 
    printf("Converting scaled precip into raw forecasts")
    ds['pr'] = np.expm1(ds['scaled_pr'])
    ds = ds.drop_vars('scaled_pr')

# Rename 'step' to 'lead' and increment all leads by 1 day to match ECMWF
ds = ds.rename({"step": "lead"})
ds = ds.assign_coords(lead=ds.lead + np.timedelta64(1, 'D'))

printf(f"Calculating {forecast_type} start dates")
tic()
if not reforecast:
    # Rename 'time' dimension to 'issuance_date'for clarity
    ds = ds.rename({"time": "issuance_date"})
    # Map issuance date and lead to start date of period being forecasted
    times = issuance_to_target(ds.issuance_date.values, ds.lead.values)
    ds = ds.assign_coords({
        "time": (("issuance_date", "lead"), times)
    }).stack(combo=("issuance_date", "lead")).swap_dims({"combo": "time"})
else: 
    # Rename 'time' dimension to 'hindcast_date' for clarity
    ds = ds.rename({"time": "hindcast_date"})
    # Store leads for possible reindexing later
    leads = ds.lead.values
    num_leads = leads.size
    # Map hindcast date and lead to start date of period being hindcasted
    times = issuance_to_target(ds.hindcast_date.values, ds.lead.values)
    ds = ds.assign_coords({
        "time": (("hindcast_date", "lead"), times)
    }).stack(combo=("hindcast_date", "lead")).swap_dims({"combo": "time"})
toc()

# %%
num_ensemble_members = 10 if reforecast else 100

# Select dataset names for each quantile
dataset_names = {quantile: f"model_{model}-{forecast_type}-f{ii+1}_{weather_variable}" for (ii, quantile) in enumerate(quantiles)}

# Chunk zarr data by date to enable appending
chunk = ({"lead": -1, "latitude": -1, "longitude": -1, "year_delta": -1, "issuance_date": 1} 
    if reforecast else {"lead": -1, "latitude": -1, "longitude": -1, "issuance_date": 1})
if reforecast:
    # Specify standard year_delta range for reforecasts
    full_year_deltas = np.arange(-20, 0, 1)
    num_year_deltas = full_year_deltas.size

# %%
# Get unique (month, day) combinations
if reforecast:
    hindcast_dates = pd.to_datetime(ds.hindcast_date.values)
    month_day_tuples = list(zip(hindcast_dates.month, hindcast_dates.day))
    unique_month_days = np.unique(month_day_tuples, axis=0).tolist()
else:
    issuance_dates = pd.to_datetime(ds.issuance_date.values)
    month_day_tuples = list(zip(issuance_dates.month, issuance_dates.day))
    unique_month_days = np.unique(month_day_tuples, axis=0).tolist()

printf(f"Processing {len(unique_month_days)} unique month-day combinations")

for month, day in unique_month_days:
    printf(f"Processing {month:02d}-{day:02d}")
    if reforecast:
        mask = (ds.hindcast_date.dt.month == month) & (ds.hindcast_date.dt.day == day)
        ds_day = ds.sel(time=mask)
    else:
        mask = (ds.issuance_date.dt.month == month) & (ds.issuance_date.dt.day == day)
        ds_day = ds.sel(time=mask)

    # Explicitly load predictions
    printf("Explicitly load predictions")
    tic()
    ds_day.load()
    toc()
    
    # Check whether any ds values are NaN
    ds_notnull = ds_day[weather_variable].notnull()
    ds_has_nans = not ds_notnull.all()
    if ds_has_nans:
        printf(f"Warning: {model} contains NaNs for day-of-year {day}")

    # Initialize dictionary to hold ensemble predictions for each quantile
    ensemble_pred = {}

    for quantile in quantiles: 
        printf(f"Processing quantile {quantile}")
        tic()
        quintile_boundaries = quintiles.sel(quantile=quantile, time=ds_day.time, drop=True)
        # Compute cumulative probabilities
        cdf_pred = (ds_day[weather_variable] < quintile_boundaries).astype("float32")
        if ds_has_nans: 
            cdf_pred = cdf_pred.where(ds_notnull, np.nan)
        
        # Reformat to match ECMWF forecasts
        if not reforecast: 
            # For forecasts, reformat to have issuance_date and lead dims
            contribution = cdf_pred.swap_dims({"time": "combo"}).drop_vars("time")
            contribution = contribution.set_index(combo=["issuance_date", "lead"]).unstack("combo")
        else: 
            printf("Reformatting")
            tic()
            contribution = cdf_pred.swap_dims({"time": "combo"}).drop_vars("time")
            contribution = contribution.set_index(combo=["hindcast_date", "lead"]).unstack("combo")
            
            # For reforecasts, replace hindcast_date with issuance_date and year_delta
            hindcasts = pd.to_datetime(contribution.hindcast_date.values)
            issuance_dates = hindcasts.map(lambda x: x.replace(year=2024))
            year_deltas = hindcasts.year - 2024 # year_deltas are negative
            
            contribution = contribution.assign_coords({
                "issuance_date": ("hindcast_date", issuance_dates), 
                "year_delta": ("hindcast_date", year_deltas)
            })
            contribution = contribution.set_index(hindcast_date=["issuance_date", "year_delta"])
            contribution = contribution.unstack("hindcast_date")
            
            # Reindex if necessary
            if contribution.sizes["lead"] < num_leads: 
                contribution = contribution.reindex(lead=leads)
            if contribution.sizes["year_delta"] < num_year_deltas:
                contribution = contribution.reindex(year_delta=full_year_deltas)
                printf(f"Warning: Missing year deltas {year_deltas}")
            toc()
        
        # Sum over forecast ensemble members
        contribution = contribution.sum(dim="number", skipna=False)
        # Add to ensemble
        ensemble_pred[quantile] = contribution
        toc()

    # Save each quantile for this month-day
    for ii, quantile in enumerate(quantiles): 
        printf(f"Saving to {dataset_names[quantile]}")
        tic()
        # Normalize by number of ensemble members
        ensemble_pred[quantile] /= num_ensemble_members
        # Sort by ascending latitude and rename to probabilistic variable
        ensemble_pred[quantile] = ensemble_pred[quantile].sortby("latitude").rename({weather_variable: f"f{ii+1}_{weather_variable}"})
        # Save with appropriate chunking
        save_data(ensemble_pred[quantile].chunk(chunk), dataset_names[quantile], mode='a', append_dim='issuance_date')
        toc()
    
    # Clean up memory
    del ds_day, ds_notnull, ensemble_pred
