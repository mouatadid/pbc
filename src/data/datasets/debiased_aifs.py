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
Compute "debiased" probabilistic (re)forecasts from deterministic AIFS (re)forecasts 
using AIFS model-based quintiles

This script takes existing deterministic AIFS operational (re)forecasts and converts them
to probabilistic forecasts using AIFS reforecast-based quintiles (computed by aifs-quintiles.py).
The logic follows the second half of aifs.py but uses AIFS quintiles instead of ERA5 quintiles
and reuses the AIFS quintiles for a given forecast date as the quintiles for each associated
reforecast.

Example usages: 
  python src/data/datasets/debiased_aifs.py -v -wv tas
  python src/data/datasets/debiased_aifs.py -v -ref -wv tas
  python src/data/datasets/debiased_aifs.py -v -wv pr
  python src/data/datasets/debiased_aifs.py -v -ref -wv pr
  python src/data/datasets/debiased_aifs.py -v -wv mslp
  python src/data/datasets/debiased_aifs.py -v -ref -wv mslp
  python src/data/datasets/debiased_aifs.py -v -ref -wv tas; python src/data/datasets/debiased_aifs.py -v -ref -wv pr; python src/data/datasets/debiased_aifs.py -v -ref -wv mslp
  python src/data/datasets/debiased_aifs.py -v -wv tas; python src/data/datasets/debiased_aifs.py -v -wv pr; python src/data/datasets/debiased_aifs.py -v -wv mslp
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

# %%
# Load quintile boundaries explicitly into memory
printf(f"Loading AIFS quintiles for measurement: {weather_variable}"); tic()
quintiles = load_data(os.path.join(f"{model}-quintiles", f"{model}-quintiles-{weather_variable}")).load()
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

if reforecast:
    # For reforecasts, replace hindcast_date with associated
    # forecast issuance_date and year_delta
    hindcasts = pd.to_datetime(ds.hindcast_date.values)
    issuance_dates = hindcasts.map(lambda x: x.replace(year=year))
    year_deltas = hindcasts.year - year # year_deltas are negative
    
    ds = ds.assign_coords({
        "issuance_date": ("hindcast_date", issuance_dates), 
        "year_delta": ("hindcast_date", year_deltas)
    })
    ds = ds.set_index(hindcast_date=["issuance_date", "year_delta"])
    ds = ds.unstack("hindcast_date")
toc()

# print("Loading deterministic data..."); tic()
# ds.load()
# toc()

# %%
# Select dataset names for each quantile
dataset_names = {quantile: os.path.join(f"debiased_{model}", f"debiased_{model}-{forecast_type}-f{ii+1}_{weather_variable}") for (ii, quantile) in enumerate(quantiles)}

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
for issuance_date in ds.issuance_date.values:
    if last_saved_date and issuance_date <= last_saved_date:
        printf(
            f"Skipping {issuance_date}: already saved.",
            verbose=args.verbose,
        )
        continue

    print(f"Loading deterministic data for {issuance_date.astype('datetime64[D]')}..."); tic()
    # Restrict to issuance date being processed
    # Wrap issuance_date in a list to ensure issuance_date dimension is preserved
    ds_day = ds.sel(issuance_date=[issuance_date]).load()
    # For both forecasts and reforecasts, use the quintiles corresponding to the 
    # associated forecast issuance date
    quintiles_day = quintiles.sel(issuance_date=issuance_date)
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

        # Sum over the member dimension ("M") of perturbed forecasts
        # while ensuring that nans are preserved
        cdf_pred = cdf_pred.sum("M", skipna=False)
        # Add prediction to the ensemble
        ensemble_pred[quantile] = cdf_pred
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
