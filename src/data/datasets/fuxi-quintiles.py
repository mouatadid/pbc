# -*- coding: utf-8 -*-
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
# Generate and save FuXi model-based quintiles from forecast data
#
# This script implements the recipe to derive model climatology
# from FuXi forecasts. For each issuance date, it:
#   1. Pools the reforecasts from the years 2002-2016 with the same month and day
#   2. Computes quintile thresholds (20th, 40th, 60th, 80th percentiles)
#   3. Only calculates quintiles for leads 19 and 26
#
# Args:
#     --weather_variables (-wv): List of the measurement variable to process (e.g., "tas", "pr", "mslp")
#
# Example usage:
#   # Process dates for a specific model with optional date range filtering
#   python src/data/datasets/fuxi-quintiles.py -wv tas 
#   python src/data/datasets/fuxi-quintiles.py -wv tas pr 
#   src/batch/batch_python.sh -m 50 -c 8 -h 12 src/data/datasets/fuxi-quintiles.py -wv tas pr mslp
#
# Recommended to run with at least 50GB of memory

# %%
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
import sys
import gc
import os
import glob
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import List, Optional
from utils.timing import tic, toc
from utils.data_io import save_data, DATA_DIR
from utils.logging import printf
from utils.data_io import load_data
from data.helpers.data_utils import load_s2s_reforecast_dataset, IRI_GRID_NAMES


# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variables",
    "-wv",
    nargs="+",
    default=["tas"],
    choices=["tas", "pr", "mslp"],
    help="List of weather variables to process (e.g., -wv tas pr)"
)

if isnotebook():
    args = parser.parse_known_args(['-wv', 'tas', 'pr', 'mslp'])[0]
else:
    args = parser.parse_args()

measurement_variables = args.weather_variables


# config
FUXI_DIR = os.path.join(DATA_DIR, "fuxi")
CLIM_START = 2002
CLIM_END = 2016
FUXI_MAP = {"t2m": "tas", 
            "tp": "pr", 
            "msl": "mslp",
            "lat": "latitude", 
            "lon": "longitude", 
            "lead_time": "lead"}


# 1. Load Hindcast Files (2002-2016)
printf(f"Scanning FuXi hindcast files...")
# Filter for years 2002-2016 only to establish climatology
hindcast_files = sorted([
    f for f in glob.glob(os.path.join(FUXI_DIR, "*.nc"))
    if CLIM_START <= int(os.path.basename(f)[:4]) <= CLIM_END
])

# Open multiple files, and rename variables
def preprocess(ds):
    # Rename only if needed
    ds = ds.rename({k: v for k, v in FUXI_MAP.items() if k in ds})
    # Ensure latitude is increasing
    ds = ds.sortby("latitude")
    return ds

tic()
ds_hindcast = xr.open_mfdataset(hindcast_files, preprocess=preprocess, combine='nested', concat_dim='time')
toc()


# %%
# Loop over variables
for measurement_variable in measurement_variables:

    printf(f"\n===== Processing variable: {measurement_variable} =====")
    tic()

    if measurement_variable not in ds_hindcast:
        printf(f"Skipping {measurement_variable} (not found in dataset)")
        continue

    ds_var = ds_hindcast[[measurement_variable]]

    # Group by Day-of-Year and compute quintiles
    # Fixed 15-year window, pooled by month/day
    printf(f"Computing {CLIM_START}-{CLIM_END} pooled quintiles...")
    quantiles = [0.2, 0.4, 0.6, 0.8]

    # Create a grouping key (MM-DD)
    day_groups = ds_var.groupby("time.dayofyear")
    # We compute quintiles across the 'time' (the years) and 'member' (ensemble members)
    quintiles_climatology = day_groups.quantile(
        quantiles,
        dim=['time', 'member']
    )

    # 3. Assign issuance date with default year 2020 (leap year safe)
    printf("Formatting dataset with default 2020 issuance dates...")
    # Map day-of-year back to actual dates in 2020
    # This handles the requirement that Feb 29 (if exists) or Leap Year logic is consistent
    base_dates = pd.date_range("2020-01-01", "2020-12-31")
    all_quintile_blocks = []


    printf(f"Generating FuXi quintiles for {measurement_variable}")
    for date in base_dates:
        doy = date.dayofyear
        # Select the pre-computed quintile for this day of year
        if doy in quintiles_climatology.dayofyear:
            day_q = quintiles_climatology.sel(dayofyear=doy).drop_vars("dayofyear")
            day_q = day_q.assign_coords(issuance_date=date).expand_dims("issuance_date")
            all_quintile_blocks.append(day_q)
        else:
            printf(f"Skipping {doy}, not found in quintiles_climatology dayofyear")
            continue

    quintiles_ds = xr.concat(all_quintile_blocks, dim="issuance_date")

    # Format dataset
    quintiles_ds = quintiles_ds.sortby("latitude").astype('float32')
    quintiles_ds = quintiles_ds.chunk({
        "quantile": 1,
        "issuance_date": 1,
        "lead": -1,
        "latitude": -1,
        "longitude": -1
    })   
    toc()

    # Save to Zarr
    tic()
    dataset_name = f"fuxi-quintiles-{measurement_variable}"
    dataset_relpath = os.path.join("fuxi-quintiles", dataset_name)
    printf(f"Saving to {dataset_relpath}...")
    save_data(quintiles_ds, dataset_relpath, mode='w')
    toc()

    # Cleanup per variable
    del ds_var, quintiles_climatology, quintiles_ds, all_quintile_blocks
    gc.collect()
