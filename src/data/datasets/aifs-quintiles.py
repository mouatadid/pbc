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
# Generate and save AIFS model-based quintiles from reforecast data
#
# This script implements the recipe to derive model climatology
# from AIFS reforecasts. For each issuance date, it:
#   1. Pools the reforecasts from the prior 20 years with the same month and day
#   2. Computes quintile thresholds (20th, 40th, 60th, 80th percentiles)
#   3. Only calculates quintiles for leads 19 and 26
#
# Args:
#     --weather_variable (-wv): The measurement variable to process (e.g., "tas", "pr", "mslp")
#     --start_date: Optional start date for processing (YYYYMMDD)
#     --end_date: Optional end date for processing (YYYYMMDD)
#     --force: Overwrite any existing AIFS model-based quintiles
#
# Example usage:
#   # Process dates for a specific model with optional date range filtering
#   python src/data/datasets/aifs-quintiles.py -wv tas -v
#   python src/data/datasets/aifs-quintiles.py -wv pr --start_date 20230101 --force
#   python src/data/datasets/aifs-quintiles.py -wv tas --end_date 20151224 
#   src/batch/batch_python.sh -m 50 -c 8 -h 12 src/data/datasets/aifs-quintiles.py -wv pr -v
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
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import List, Optional
import pathlib

from utils.timing import tic, toc
from utils.data_io import (
    save_data,
    DATA_DIR
)
from utils.logging import printf
from utils.data_io import load_data
from data.helpers.data_utils import (
    load_s2s_reforecast_dataset,
    IRI_GRID_NAMES
)

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    choices=["tas", "pr", "mslp"],
    help="Name of weather variable to process"
)
parser.add_argument(
    '--start_date',
    type=str,
    help='Start date for processing (YYYYMMDD)'
)
parser.add_argument(
    '--end_date',
    type=str,
    help='End date for processing (YYYYMMDD)'
)
parser.add_argument(
    '--verbose',
    '-v',
    action='store_true',
    help='Print verbose output'
)
parser.add_argument(
    '--force',
    action='store_true',
    help='Overwrite any existing AIFS model-based quintiles'
)

if isnotebook():
    args = parser.parse_known_args(['-wv', 'mslp', '-v', '--start_date', '20250101', '--end_date', '20251229'])[0]
else:
    args = parser.parse_args()

measurement_variable = args.weather_variable
verbose = args.verbose

# %%
printf(f"Loading issuance dates for {measurement_variable}..."); tic()
# Load forecast issuance dates from corresponding forecast file
year = 2025
model = "aifs"
weather_variable_names_on_server = {
    "tas": "2t",
    "pr": "tp",
    "mslp": "msl"
}
server_weather_variable = weather_variable_names_on_server[measurement_variable]
all_issuance_dates = pd.to_datetime(
    load_data(os.path.join(model, f"forecast_{server_weather_variable}_{year}")).time)

# Filter by date range if specified (applies to both --model and non-model cases)
if args.start_date and args.end_date:
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if start_date <= d <= end_date]
    printf(f"Filtered to date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
elif args.start_date:
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if d >= start_date]
    printf(f"Filtered to dates from: {start_date.strftime('%Y-%m-%d')}")
elif args.end_date:
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if d <= end_date]
    printf(f"Filtered to dates until: {end_date.strftime('%Y-%m-%d')}")
else:
    # Process all available issuance dates
    issuance_dates = all_issuance_dates
toc()

# %%
# Set generic dataset name
dataset_name = f"aifs-quintiles-{measurement_variable}"

# Check if quintiles dataset already exists and load existing dates
quintiles_subdir = os.path.join(DATA_DIR, "aifs-quintiles")
os.makedirs(quintiles_subdir, exist_ok=True)
dataset_path = os.path.join(quintiles_subdir, f"{dataset_name}.zarr")
existing_quintiles = None
existing_dates = []

if not args.force:
    try:
        existing_quintiles = xr.open_dataset(dataset_path)
        existing_dates = existing_quintiles.issuance_date.values
        existing_dates = [pd.Timestamp(d).to_pydatetime() for d in existing_dates]
        printf(f"Found existing dataset with {len(existing_dates)} dates")
        printf(f"Date range: {min(existing_dates).strftime('%Y-%m-%d')} to {max(existing_dates).strftime('%Y-%m-%d')}")

        # Filter out dates that already exist
        dates_to_process = [d for d in issuance_dates if d not in existing_dates]
        printf(f"Will process {len(dates_to_process)} new dates (skipping {len(issuance_dates) - len(dates_to_process)} existing)")
        issuance_dates = dates_to_process
    except Exception as e:
        printf(f"No existing dataset found or error loading: {e}")
        printf("Will create new dataset")
else:
    printf("Force mode - will reprocess all dates")

if len(issuance_dates) == 0:
    printf("No new dates to process. Exiting...")
    sys.exit(0)

num_issuance_dates = len(issuance_dates)
printf(f"Processing {num_issuance_dates} issuance dates")


# %%
print(f"Loading reforecast data for {measurement_variable}..."); tic()
reforecasts = load_data(os.path.join(model, f"hindcast_{server_weather_variable}"))
# Convert step into lead and rename variable
reforecasts = reforecasts.rename({'step': 'lead', server_weather_variable: measurement_variable})
# Shift leads to match the start of the forecasted period instead of the end
# Steps were initially set to 25 and 32 so shifting to 19 and 26
reforecasts["lead"] = reforecasts["lead"] - np.timedelta64(6, "D")
# Sort latitudes in increasing order
reforecasts = reforecasts.sortby("latitude")
toc()

# %%
# Main processing loop
printf(f"Generating AIFS quintiles for {measurement_variable}")

# Process each issuance date and compute quintiles
# Save each date incrementally to avoid memory issues and enable resumption
dates_saved = 0
first_save = args.force or existing_quintiles is None
dataset_relpath = os.path.join("aifs-quintiles", dataset_name)

reforecast_months = reforecasts.time.dt.month
reforecast_days = reforecasts.time.dt.day
quantiles = [0.2, 0.4, 0.6, 0.8]
for i, issuance_date in enumerate(issuance_dates):
    printf(f"\nProcessing issuance date {i+1}/{num_issuance_dates}: {issuance_date.strftime('%Y-%m-%d')}")

    printf("Extracting corresponding reforecasts..."); tic()
    mask = ((reforecast_months == issuance_date.month) &
            (reforecast_days == issuance_date.day))
    my_reforecasts = reforecasts.sel(time=mask).load()
    toc()

    printf(f"Computing quintiles..."); tic()
    quintiles_ds = my_reforecasts.quantile(quantiles, dim=['time', 'number'])
    # Assign issuance date coordinate and dimension
    quintiles_ds = quintiles_ds.assign_coords(
            issuance_date=issuance_date).expand_dims("issuance_date")
    # Convert to float32 and chunk appropriately
    quintiles_ds = quintiles_ds.astype('float32').chunk({
        "quantile": 1,
        "issuance_date": 1,
        "lead": -1,
        "latitude": -1,
        "longitude": -1
    }).load()
    toc()

    # Save immediately after processing each date
    tic()
    if first_save and dates_saved == 0:
        # First save: overwrite or create new dataset
        printf(f"Creating new dataset and saving {issuance_date.strftime('%Y-%m-%d')}...")
        save_data(quintiles_ds, dataset_relpath, mode='w')
    else:
        # Subsequent saves: append to existing dataset
        printf(f"Appending {issuance_date.strftime('%Y-%m-%d')} to dataset...")
        save_data(quintiles_ds, dataset_relpath, mode='a', append_dim="issuance_date")
    toc()

    dates_saved += 1
    printf(f"Saved {dates_saved} dates so far")

    # printf("Cleaning up memory..."); tic()
    # del my_reforecasts, quintiles_ds
    # gc.collect()
    # toc()

# %%
# Summary
if dates_saved == 0:
    printf("No quintile data was generated. Exiting...")
    sys.exit(1)

printf(f"\nSuccessfully saved {dates_saved} new dates to aifs-quintiles/{dataset_name}")
