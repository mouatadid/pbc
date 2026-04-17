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
Create probabilistic AIFS forecasts and reforecasts

Example usages: 
  python src/data/datasets/aifs.py -v -wv tas
  python src/data/datasets/aifs.py -v -ref -wv tas
  python src/data/datasets/aifs.py -v -wv pr
  python src/data/datasets/aifs.py -v -ref -wv pr
  python src/data/datasets/aifs.py -v -wv mslp
  python src/data/datasets/aifs.py -v -ref -wv mslp
  python src/data/datasets/aifs.py -v -ref -wv tas; python src/data/datasets/aifs.py -v -ref -wv pr; python src/data/datasets/aifs.py -v -ref -wv mslp
  python src/data/datasets/aifs.py -v -wv tas; python src/data/datasets/aifs.py -v -wv pr; python src/data/datasets/aifs.py -v -wv mslp
"""
import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
import argparse
import os
import numpy as np
import pandas as pd

from utils.data_io import (
    load_data, 
    save_data, 
    get_last_date_in_zarr, 
    get_zarr_store_path,
    DATA_DIR
)
from utils.timing import tic, toc
from utils.logging import printf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tas",
    choices=["tas", "pr", "mslp"],
    help="Name of weather variable to download",
)
parser.add_argument(
    "--reforecast",
    "-ref",
    action="store_true",
    help="If true, download reforecast (i.e., the hindcasts for the last 20 years)",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print verbose output",
)
if isnotebook():
    args = parser.parse_known_args(["-v", "-wv", "tas", "-ref"])[0]
else:
    args = parser.parse_args()


#
# Process arguments
#

weather_variable = args.weather_variable

reforecast = args.reforecast
forecast_type = "reforecast" if reforecast else "forecast"

model = "aifs"

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
# Load quintile boundaries explicitly into memory
printf(f"Loading ERA5 quintiles for measurement: {weather_variable}"); tic()
quintiles = load_data(f"era5-quintiles-{weather_variable}").load()
quantiles = quintiles["quantile"].values
toc()

# %%
# Chunk zarr data by date to enable appending
chunk = ({"lead": -1, "latitude": -1, "longitude": -1, "year_delta": -1, "issuance_date": 1} 
    if reforecast else {"lead": -1, "latitude": -1, "longitude": -1, "issuance_date": 1})
if reforecast:
    # Specify standard year_delta range for reforecasts
    year_deltas = np.arange(-20, 0, 1)
    num_year_deltas = year_deltas.size

#
# Load deterministic data
#
printf(f"Reformatting deterministic data..."); tic()
year = 2025
weather_variable_names_on_server = {
    "tas": "2t",
    "pr": "tp",
    "mslp": "msl"
}

server_weather_variable = weather_variable_names_on_server[weather_variable]
input_name = (f"hindcast_{server_weather_variable}" if reforecast 
              else f"{forecast_type}_{server_weather_variable}_{year}")
ds = load_data(os.path.join(model, input_name))
# Use our conventional names for weather variable, lead dimension, 
# and ensemble member number
ds = ds.rename({server_weather_variable: weather_variable, 
                "step": "lead", "number": "M"})
# Shift leads to match the start of the forecasted period instead of the end
# Steps were initially set to 25 and 32 so shifting to 19 and 26
ds["lead"] = ds["lead"] - np.timedelta64(6, "D")
# Identify the number of ensemble members
num_ensemble_members = ds.sizes["M"]
# Rename 'time' dimension to match our convention
date_name = "hindcast_date" if reforecast else "issuance_date"
ds = ds.rename({"time": date_name})
# Map date_name and lead to start date of period being forecasted
times = issuance_to_target(ds[date_name].values, ds.lead.values)
ds = ds.assign_coords({
    "time": ((date_name, "lead"), times)
}).stack(combo=(date_name, "lead")).swap_dims({"combo": "time"})
toc()

# print("Loading deterministic data..."); tic()
# ds.load()
# toc()

# %%
print("Computing unique (month, day) combinations..."); tic()
all_dates = pd.to_datetime(ds[date_name].values)
month_day_tuples = list(zip(all_dates.month, all_dates.day))
unique_month_days = np.unique(month_day_tuples, axis=0).tolist()
toc()

# %%
# Select dataset names for each quantile
dataset_names = {quantile: f"{model}-{forecast_type}-f{ii+1}_{weather_variable}" for (ii, quantile) in enumerate(quantiles)}

# Identify the last date stored in the output zarr files
store_path = get_zarr_store_path(DATA_DIR, dataset_names[quantiles[0]])
last_saved_date = get_last_date_in_zarr(
    store_path, 
    time_coord = "issuance_date")
if last_saved_date:
    # Convert to naive datetime (without timezone info)
    last_saved_date = last_saved_date.replace(tzinfo=None)
printf(f"\nLast saved issuance date in zarr: {last_saved_date}\n")

# Process each available issuance date
printf(f"\nProcessing {model} {forecast_type}s\n")
###last_saved_date = None###
all_months = ds[date_name].dt.month
all_days = ds[date_name].dt.day
for month, day in unique_month_days:
    issuance_date = np.datetime64(f"2025-{month:02d}-{day:02d}")
    if last_saved_date and issuance_date <= last_saved_date:
        printf(
            f"Skipping {issuance_date}: already saved.",
            verbose=args.verbose,
        )
        continue

    print(f"Loading deterministic data for {issuance_date.astype('datetime64[D]')}..."); tic()
    # Restrict to issuance date being processed
    mask = (all_months == month) & (all_days == day)
    ds_day = ds.sel(time=mask).load()
    quintiles_day = quintiles.sel(time=ds_day.time)
    toc()

    print("Computing probabilistic forecast..."); tic()
    # Check whether any ds values are NaN
    ds_notnull = ds_day[weather_variable].notnull()
    ds_has_nans = not ds_notnull.all()
    if ds_has_nans:
        print(f"Warning: {model} {issuance_date} contains NaNs")
    # Initialize dictionary to hold ensemble predictions for each quantile
    ensemble_pred = {}
    for quantile in quantiles:
        # Compute cumulative probabilistic prediction for this quantile bin
        cdf_pred = (ds_day < quintiles_day.sel(quantile = quantile, drop=True)).astype("float32")
        if ds_has_nans:
            # Keep cdf_pred values if ds value is not null; otherwise, replace with NaN
            cdf_pred = cdf_pred.where(ds_notnull, np.nan)
        # Reformat prediction to have lead and date_name dimensions instead of time
        contribution = cdf_pred.swap_dims({"time": "combo"}).drop_vars("time")
        contribution = contribution.set_index(combo=[date_name, "lead"]).unstack("combo")
        if reforecast:
            # For reforecasts, replace hindcast_date with issuance_date and year_delta
            hindcasts = pd.to_datetime(contribution.hindcast_date.values)
            issuance_dates = hindcasts.map(lambda x: x.replace(year=year))
            year_deltas = hindcasts.year - year # year_deltas are negative
            
            contribution = contribution.assign_coords({
                "issuance_date": ("hindcast_date", issuance_dates), 
                "year_delta": ("hindcast_date", year_deltas)
            })
            contribution = contribution.set_index(hindcast_date=["issuance_date", "year_delta"])
            contribution = contribution.unstack("hindcast_date")

        # Sum over the member dimension ("M") of perturbed forecasts
        # while ensuring that nans are preserved
        contribution = contribution.sum("M", skipna=False)
        # Add prediction to the ensemble
        ensemble_pred[quantile] = contribution
    toc()
    
    for ii, quantile in enumerate(quantiles):
        print(f"Saving to {dataset_names[quantile]}..."); tic()
        # Normalize ensemble predictions by number of ensemble members
        ensemble_pred[quantile] /= num_ensemble_members
        # Append ensemble forecast, sorted by ascending latitude,
        # to appropriate zarr with quantile name and appropriate chunking
        save_data(ensemble_pred[quantile].sortby("latitude").rename(
            {weather_variable: f"f{ii+1}_{weather_variable}"}).chunk(
            chunk), dataset_names[quantile], mode='a', append_dim='issuance_date')
        toc()
