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
Compute debiased probabilistic forecasts from deterministic FuXi forecasts 
using FuXi model-based quintiles 

This script takes existing deterministic FuXi forecasts and converts them
to probabilistic forecasts using FuXi reforecast-based quintiles (computed by fuxi-quintiles.py).

Example usages: 
  python src/data/datasets/debiased_fuxi.py -v -wv tas
  python src/data/datasets/debiased_fuxi.py -v -wv pr
  python src/data/datasets/debiased_fuxi.py -v -wv mslp
  python src/data/datasets/debiased_fuxi.py -v -wv tas; python src/data/datasets/debiased_fuxi.py -v -wv pr; python src/data/datasets/debiased_fuxi.py -v -wv mslp
  src/batch/batch_python.sh -m 10 -c 8 -h 2 src/data/datasets/debiased_fuxi.py -wv tas 
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
import glob
import gc
import numpy as np
import pandas as pd
import xarray as xr
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
    help="Name of weather variable to process",
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="Print verbose output",
)
if isnotebook():
    args = parser.parse_known_args(["-v", "-wv", "tas"])[0]
else:
    args = parser.parse_args()

#
# Process arguments
#
weather_variable = args.weather_variable
verbose = args.verbose
model = "fuxi"
forecast_type = "forecast"

# Config
FUXI_DIR = os.path.join(DATA_DIR, "fuxi")
FORECAST_START = 2017
FORECAST_END = 2021
FUXI_MAP = {
    "t2m": "tas",
    "tp": "pr",
    "msl": "mslp",
    "lat": "latitude",
    "lon": "longitude",
    "lead_time": "lead",
}


# Load quintile boundaries explicitly into memory
printf(f"Loading FuXi quintiles for measurement: {weather_variable}"); tic()
quintiles = load_data(os.path.join(f"{model}-quintiles", f"{model}-quintiles-{weather_variable}")).load()
quantiles = quintiles["quantile"].values
toc()

# Select dataset names for each quantile
dataset_names = {quantile: os.path.join(f"debiased_{model}", f"debiased_{model}-{forecast_type}-f{ii+1}_{weather_variable}") for (ii, quantile) in enumerate(quantiles)}

# Precompute day-of-year index
tic()
quintiles = quintiles.assign_coords(
    dayofyear=quintiles.issuance_date.dt.dayofyear
).swap_dims({"issuance_date": "dayofyear"}).drop_vars("issuance_date")
toc()


# Preprocess
def preprocess(ds):
    # Rename only if needed
    ds = ds.rename({k: v for k, v in FUXI_MAP.items() if k in ds})
    # Ensure latitude is increasing
    ds = ds.sortby("latitude")
    return ds


# Load forecast data
printf("Loading FuXi deterministic data..."); tic()
files = sorted([
    f for f in glob.glob(os.path.join(FUXI_DIR, "*.nc"))
    if FORECAST_START <= int(os.path.basename(f)[:4]) <= FORECAST_END
])
ds = xr.open_mfdataset(
    files,
    preprocess=preprocess,
    combine="nested",
    concat_dim="time",
    chunks={
        "time": 1,     
        "member": -1
    }
)
ds = ds.rename({
    "time": "issuance_date",
    "member": "M"
})
ds = ds.sel(lead=[19, 26])
toc()

# Align quintiles with forecast dates
printf("Aligning climatology with FuXi forecast dates..."); tic()
# Map each forecast date → dayofyear
ds = ds.assign_coords(dayofyear = ds.issuance_date.dt.dayofyear)
# Broadcast quintiles to forecast dates
quintiles_aligned = quintiles.sel(dayofyear = ds.dayofyear)
toc()

# Compute probabilistic forecasts (vectorized)
printf("Computing probabilistic FuXi forecasts..."); tic()
ds_var = ds[weather_variable]
# Check whether any ds values are NaN
ds_notnull = ds_var.notnull()
ds_has_nans = not ds_notnull.all()
if ds_has_nans:
    print(f"Warning: {model} {issuance_date} contains NaNs")
# Expand dims for broadcasting:
# (time, M, lead, lat, lon) vs (time, quantile, lead, lat, lon)
cdf_pred = (ds_var.expand_dims(quantile=quantiles) <
       quintiles_aligned[weather_variable]).astype("float32")
if ds_has_nans:
    # Keep cdf_pred values if ds value is not null; otherwise, replace with NaN
    cdf_pred = cdf_pred.where(ds_notnull.expand_dims(quantile=quantiles), np.nan)
# Sum over the member dimension ("M") of FuXi forecasts
# while ensuring that nans are preserved
cdf_pred = cdf_pred.sum("M", skipna=False)
# Normalize ensemble predictions by number of ensemble members
cdf_pred = cdf_pred / ds.sizes["M"]
toc()


# Save outputs (split by quantile)
printf("Saving outputs..."); tic()
chunk = {
    "lead": -1,
    "latitude": -1,
    "longitude": -1,
    "issuance_date": 1
}


for ii, quantile in enumerate(quantiles):
    print(f"Saving to {dataset_names[quantile]}..."); tic()
    cdf_q = cdf_pred.sel(quantile=quantile).drop_vars(["dayofyear", "quantile"])
    # Save forecast, sorted by ascending latitude,
    # to appropriate zarr with quantile name and appropriate chunking
    save_data(cdf_q.sortby("latitude").rename(
        f"f{ii+1}_{weather_variable}").chunk(
        chunk), dataset_names[quantile], mode='w')
    toc()
