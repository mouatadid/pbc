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
"""
Generate and save ECMWF model-based percentiles from reforecast data

This script implements the SUBS-M-climate recipe to derive model climatology
from ECMWF reforecasts. For each  issuance date, it:
  1. Selects 9 reforecast base dates centered on the closest preceding date
  2. Pools reforecast values from 20 years × 9 base dates × 11 ensemble members = 1980 values
  3. Computes quintile thresholds (20th, 40th, 60th, 80th percentiles)
  4. Only calculates percentiles for leads 19 and 26

Args:
    --weather_variable (-wv): The measurement variable to process (e.g., "tas", "pr", "mslp")
    --model: Optional model to process (ecmwf_cy41-47, ecmwf_cy48, ecmwf_cy49)
             If specified, saves to {model}-percentiles-{variable}.zarr
             If not specified, processes all models and saves to ecmwf-percentiles-{variable}.zarr
    --start_date: Optional start date for processing (YYYYMMDD)
    --end_date: Optional end date for processing (YYYYMMDD)
    --force: Overwrite any existing ECMWF model-based percentiles
    -ps: Comma-separated list of percentile levels for probabilistic forecasts (e.g., "10,90,5,95")

Example usage:
  # Process dates for a specific model with optional date range filtering
  python src/data/datasets/ecmwf-percentiles.py -wv pr --model ecmwf_cy49 -v
  python src/data/datasets/ecmwf-percentiles.py -wv pr --model ecmwf_cy48 --start_date 20230101 --force
  python src/data/datasets/ecmwf-percentiles.py -wv tas --model ecmwf_cy41-47 --end_date 20151224 
  src/batch/batch_python.sh -m 50 -c 8 -h 12 src/data/datasets/ecmwf-percentiles.py -wv pr --model ecmwf_cy49 -v

  for model in ecmwf_cy49 ecmwf_cy48 ecmwf_cy41-47; do
  for var in pr tas mslp; do 
  for p in 10 90 5 95; do
    src/batch/batch_python.sh -m 50 -c 8 -h 12 src/data/datasets/ecmwf-percentiles.py -wv ${var} --model ${model} -v -ps ${p}
  done; done

Recommended to run with at least 50GB of memory
"""

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
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from typing import List, Optional
import pathlib

from utils.timing import tic, toc
from utils.data_io import (
    save_data,
    DATA_DIR
)
from utils.logging import printf
from data.helpers.data_utils import (
    get_valid_forecast_dates,
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
    "--model",
    choices=["ecmwf_cy41-47", "ecmwf_cy48", "ecmwf_cy49"],
    help="Process forecast dates for a specific model"
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
    help='Overwrite any existing ECMWF model-based percentiles'
)
parser.add_argument(
    "-ps",
    type=lambda s: [int(x) for x in s.split(",")], 
    help='Comma-separated list of percentile levels for probabilistic forecasts (e.g., "10,90,5,95").',
    default=[10, 90, 5, 95]
)

if isnotebook():
    args = parser.parse_known_args(['-wv', 'mslp', '-v', '--start_date', '20201026', '--end_date', '20201123'])[0]
else:
    args = parser.parse_args()

measurement_variable = args.weather_variable
# Compute quantile levels from percentiles
ps = args.ps
quantiles = [p / 100. for p in ps]
verbose = args.verbose

# %%
def get_model_for_date(issuance_date: datetime) -> str:
    """
    Determine the ECMWF model version based on the reforecast issuance date.

    Model versions by reforecast issuance date:
    - cy41-47: 2015-05-14 to 2022-12-31 
    - cy48: 2023-01-01 to 2024-11-11 
    - cy49: 2024-11-12 onwards

    Args:
        issuance_date: The reforecast initialization date

    Returns:
        Model version string (e.g., "ecmwf_cy49")
    """
    if issuance_date < datetime(2023, 1, 1):
        return "ecmwf_cy41-47"
    elif issuance_date <= datetime(2024, 11, 11):
        return "ecmwf_cy48"
    else:
        return "ecmwf_cy49"


# %%
def should_process_date(date: datetime) -> bool:
    """
    Determine if a date should be processed based on day-of-week rules.

    Args:
        date: The issuance date to check

    Returns:
        True if date should be processed, False otherwise
    """
    if date <= cutoff_date:
        # Before June 27, 2023 (inclusive): only Mondays (0) and Thursdays (3)
        return date.weekday() in [0, 3]
    else:
        # Starting June 28, 2023: all days
        return True

# %%
def get_subs_m_climate_base_dates(
    issuance_date: datetime,
    valid_reforecast_dates: List[datetime]
) -> Optional[List[datetime]]:
    """
    Get the 9 base dates for SUBS-M-climate recipe.

    According to SUBS-M-climate recipe:
    - The middle reforecast corresponds to the closest base-date preceding the issuance date
    - Select 4 dates before and 4 dates after this middle date
    - Example: if issuance date is 20th, middle date is 19th,
      and the 9 dates are 11, 13, 15, 17, 19, 21, 25, 27, 29

    Args:
        issuance_date: The operational forecast initialization date
        valid_reforecast_dates: List of valid reforecast dates for the model

    Returns:
        List of 9 datetime objects representing the selected base dates,
        or None if fewer than 9 dates are available
    """
    # Find the closest reforecast date preceding the issuance date
    preceding_dates = [d for d in valid_reforecast_dates if d <= issuance_date]

    if not preceding_dates:
        return None

    middle_date = max(preceding_dates)

    # Find index of middle date in the sorted valid dates list
    valid_dates_sorted = sorted(valid_reforecast_dates)
    middle_idx = valid_dates_sorted.index(middle_date)

    # Select 9 dates centered on middle_date
    n_before = 4
    n_after = 4

    start_idx = max(0, middle_idx - n_before)
    end_idx = min(len(valid_dates_sorted), middle_idx + n_after + 1)

    selected_dates = valid_dates_sorted[start_idx:end_idx]

    # If we don't have enough dates, return None to skip this date
    if len(selected_dates) < 9:
        return None

    return selected_dates

def open_s2s_reforecast_dataset(file_path: pathlib.Path) -> xr.Dataset:
    """Returns an opened s2s reforecast dataset with hdate coordinates replaced by
    their integer indices and latitudes sorted in an increasing order.
    """

    # Note: if decode_timedelta = False, leads will be floats rather than timedelta64[ns]
    ds = xr.open_dataset(file_path, decode_times=False, decode_timedelta=True)
    ds[IRI_GRID_NAMES["S"]] = pd.to_datetime(
        ds[IRI_GRID_NAMES["S"]].values, unit="D", origin=pd.Timestamp("1960-01-01")
    )
    # Replace hdate values with integer indices
    ds[IRI_GRID_NAMES["hdate"]] = range(len(ds.coords[IRI_GRID_NAMES["hdate"]]))
    # Sort latitudes in increasing order
    ds = ds.sortby(ds[IRI_GRID_NAMES["Y"]])

    return ds

# %%
def compute_ecmwf_quantiles_for_date(
    issuance_date: datetime,
    measurement_variable: str,
    base_dates: List[datetime],
    data_dir: str = DATA_DIR,
    verbose: bool = False,
    pooled_data: Optional[xr.Dataset] = None,
    quantiles: List[float] = quantiles
) -> Optional[xr.Dataset]:
    """
    Compute ECMWF model-based quantiles for a single issuance date based on SUBS-M-climate recipe.

    Args:
        issuance_date: The operational forecast initialization date
        measurement_variable: Weather variable (e.g., "tas", "pr", "mslp")
        base_dates: List of 9 base dates for SUBS-M-climate recipe
        data_dir: Data directory path
        verbose: Whether to print verbose output
        pooled_data: if not None, use this dataset as a starting point for reforecasts,  
          delete any irrelevant base dates from it, and add only the missing base dates
        quantiles: List of quantiles to compute (e.g., [0.1, 0.9, 0.05, 0.95])

    Returns:
        xarray.Dataset with quantile thresholds indexed by (quantile, lead, latitude, longitude),
        with issuance_date as a coordinate, or None if data not available
    """
    if pooled_data is not None:
        # Convert base_dates to DatetimeIndex for easier comparison with pooled_data issuance_date index
        base_dates = pd.DatetimeIndex(pd.to_datetime(base_dates))
        # Keep only pooled data issuance dates that are also in base dates
        pooled_data = pooled_data.sel(issuance_date=pooled_data.get_index("issuance_date").intersection(base_dates))
        # Remove any base dates that are already in pooled_data
        base_dates = base_dates.difference(pooled_data.issuance_date)
        printf(f"Using prior pooled data: only adding {len(base_dates)} base dates")

    # Load reforecast data for each of the base dates
    reforecast_datasets = []

    printf(f"Loading base data")
    # Filter to only keep leads 19 and 26
    if measurement_variable == "tas": 
        target_leads = [pd.Timedelta(days=19, hours=12), pd.Timedelta(days=26, hours=12)]    
    else: 
        target_leads = [pd.Timedelta(days=19), pd.Timedelta(days=26)] 
    # Get the weather variable name from the dataset
    weather_var_names = {
        'tas': '2t',
        'pr': 'tp',
        'mslp': 'msl'
    }
    var_name = weather_var_names.get(measurement_variable, measurement_variable)   
    tic()
    for base_date in base_dates:
        printf(f"Processing base date: {base_date.strftime('%Y-%m-%d')}")
        # Determine which model and model path this base date belongs to
        base_date_model = get_model_for_date(base_date)
        ecmwf_folder = "ecmwf-clim" if base_date_model == "ecmwf_cy41-47" else "ecmwf"
        data_folder = os.path.join(data_dir, ecmwf_folder, f"{base_date_model}-reforecast-{measurement_variable}")

        # Look for control and perturbed forecast files for this base date
        date_str = base_date.strftime("%Y%m%d")
        control_file = os.path.join(data_folder, f"{date_str}-control.nc")
        perturbed_file = os.path.join(data_folder,  f"{date_str}-perturbed.nc")

        # Specify target leads to select from the reforecasts
        # target_leads = [pd.Timedelta(days=19, hours=12), pd.Timedelta(days=26, hours=12)]  

        try:
            printf("Loading control and perturbed reforecasts")
            # Load control forecast (1 member)
            ds_control = open_s2s_reforecast_dataset(control_file).sel(lead=target_leads).load()
            # Load perturbed forecast (10 members)
            ds_perturbed = open_s2s_reforecast_dataset(perturbed_file).sel(lead=target_leads).load()

            if var_name not in ds_control or var_name not in ds_perturbed:
                printf(f"Warning: Variable {var_name} not found in reforecast data for {base_date}")
                continue

            # Extract only the weather variable and necessary coordinates
            ds_control = ds_control[[var_name]]
            ds_perturbed = ds_perturbed[[var_name]]

            # Combine control and perturbed along M dimension (total 11 members)
            # Control has dims: (issuance_date, hindcast_date, lead, latitude, longitude)
            # Perturbed has dims: (issuance_date, hindcast_date, lead, latitude, longitude, M)
            # Add M dimension to control, then concatenate
            printf("Combining control and perturbed forecasts")
            ds_control = ds_control.expand_dims(M=[0])
            ds_combined = xr.concat([ds_control, ds_perturbed], dim='M')
            # # Replace hindcast_date with years between hindcast date and issuance date
            # ds_combined["hindcast_date"] = (ds_combined["hindcast_date"].astype('M8[Y]')
            #     - ds_combined["issuance_date"].values.astype('M8[Y]')).astype('int')
            # ds_combined = ds_combined.rename({"hindcast_date": "year_delta"})

            # Add to dataset list
            reforecast_datasets.append(ds_combined)
            # Clean up memory
            del ds_control, ds_perturbed, ds_combined
        except Exception as e:
            printf(f"Error loading reforecast data for {base_date.strftime('%Y-%m-%d')}: {e}")
            printf(f"Not enough reforecast data available for {issuance_date}. Terminating.")
            sys.exit(0)
    toc()
    if not reforecast_datasets:
        if len(base_dates) > 0:
            printf(f"No reforecast data available for {issuance_date}")
            return None, pooled_data
    else:
        # Stack all datasets together
        # Each dataset has dimensions: (issuance_date=1, hindcast_date=20, lead, lat, lon, M=11)
        # We want to pool across issuance_date (base dates), hindcast_date (years), and M (members)
        printf("Stacking all datasets for 9 base dates")
        tic()
        combined_data = xr.concat(reforecast_datasets, dim='issuance_date')
        toc()
        del reforecast_datasets

        # Check if we have 20 hindcast years
        n_hindcast_years = combined_data.sizes.get('hindcast_date', 0)
        if n_hindcast_years < 20:
            printf(f"Warning: Only {n_hindcast_years} hindcast years available for {issuance_date.strftime('%Y-%m-%d')} (expected 20)")

        # Get variable name
        weather_var_names = {
            'tas': '2t',
            'pr': 'tp',
            'mslp': 'msl'
        }
        var_name = weather_var_names.get(measurement_variable, measurement_variable)

        if pooled_data is None:
            pooled_data = combined_data[var_name]
        else:
            # Combine with existing pooled_data if provided
            printf("Concatenating with existing pooled data")
            tic()
            pooled_data = xr.concat([pooled_data, combined_data[var_name]], dim='issuance_date')
            toc()
        del combined_data

    # Compute quantile thresholds over issuance_date, hindcast_date, and M dimensions 
    printf(f"Computing {quantiles} quantiles...")
    tic()
    quantile_thresholds = pooled_data.quantile(quantiles, dim=['issuance_date', 'hindcast_date', 'M'])
    toc()

    # Create output dataset with proper variable name and coordinates
    output = xr.Dataset(
        {measurement_variable: quantile_thresholds},
        coords={
            'quantile': quantiles,
            'lead': pooled_data['lead'],
            'latitude': pooled_data['latitude'],
            'longitude': pooled_data['longitude'],
            'issuance_date': issuance_date
        }
    )

    # Clean up reforecast data to save memory
    del quantile_thresholds
    gc.collect()

    return output, pooled_data

# %%
# Main processing loop
printf(f"Generating ECMWF {ps}-th percentiles for {measurement_variable}")

# Get issuance dates based on model argument
all_models = ['ecmwf_cy41-47', 'ecmwf_cy48', 'ecmwf_cy49']

if args.model:
    # Get dates for a specific model
    printf(f"Processing forecast dates for model: {args.model}")
    all_issuance_dates = []

    try:
        dates = get_valid_forecast_dates(args.model, forecast_type="forecast")
        all_issuance_dates.extend(dates)
    except Exception as e:
        printf(f"Error: Could not get dates for {args.model}: {e}")
        sys.exit(1)

    # Remove duplicates and sort
    all_issuance_dates = sorted(set(all_issuance_dates))

    # Set model-specific dataset name
    dataset_names = {p: f"{args.model}-percentile{p}-{measurement_variable}" for p in ps}
else:
    # Get issuance dates from all models
    all_issuance_dates = []

    for model_name in all_models:
        try:
            dates = get_valid_forecast_dates(model_name, forecast_type="forecast")
            all_issuance_dates.extend(dates)
        except Exception as e:
            printf(f"Warning: Could not get dates for {model_name}: {e}")

    # Remove duplicates and sort
    all_issuance_dates = sorted(set(all_issuance_dates))

    # Set generic dataset name
    dataset_names = {p: f"ecmwf-percentile-{p}-{measurement_variable}" for p in ps}

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

# Check if percentiles dataset already exists and load existing dates
percentiles_subdir = os.path.join(DATA_DIR, "ecmwf-percentiles")
os.makedirs(percentiles_subdir, exist_ok=True)
existing_percentiles = None
existing_dates = []

if not args.force:
    try:
        dataset_path = os.path.join(percentiles_subdir, f"{dataset_names[ps[0]]}.zarr")
        existing_percentiles = xr.open_dataset(dataset_path)
        existing_dates = existing_percentiles.issuance_date.values
        existing_dates = [pd.Timestamp(d).to_pydatetime() for d in existing_dates]
        printf(f"Found existing dataset with {len(existing_dates)} dates")
        printf(f"Date range: {min(existing_dates).strftime('%Y-%m-%d')} to {max(existing_dates).strftime('%Y-%m-%d')}")
        # Check that all other dataset paths have the same existing dates
        for p in ps[1:]:
            dataset_path_p = os.path.join(percentiles_subdir, f"{dataset_names[p]}.zarr")
            existing_percentiles = xr.open_dataset(dataset_path_p)
            existing_dates_p = existing_percentiles.issuance_date.values
            existing_dates_p = [pd.Timestamp(d).to_pydatetime() for d in existing_dates_p]
            if set(existing_dates) != set(existing_dates_p):
                printf(f"Error: Existing dates in {dataset_path_p} do not match those in {dataset_path}.")
                printf("Cannot process all percentiles jointly. Exiting.")
                sys.exit(1)

        # Filter out dates that already exist
        dates_to_process = [d for d in issuance_dates if d not in existing_dates]
        printf(f"Will process {len(dates_to_process)} new dates (skipping {len(issuance_dates) - len(dates_to_process)} existing)")
        issuance_dates = dates_to_process
    except Exception as e:
        printf(f"No existing dataset found or error loading: {e}")
        printf("Will create new dataset")
else:
    printf("Force mode - will reprocess all dates")

# Filter by day of week to match ECMWF forecast schedule
# cy41-47: before June 26, 2023 (inclusive), forecast only Mondays and Thursdays
# cy48 and cy49: starting June 28, 2023, forecast all days
cutoff_date = datetime(2023, 6, 27)
dates_before_filter = len(issuance_dates)
issuance_dates = [d for d in issuance_dates if should_process_date(d)]

if not issuance_dates:
    printf("No new dates to process. Exiting...")
    sys.exit(0)

# Get valid reforecast dates from models
all_reforecast_dates = []
for model_name in all_models:
    try:
        # Restrict to subset of reforecasts sufficient to generate climatology for 2016 or later
        start_on = None #"20151001"
        dates = get_valid_forecast_dates(model_name, forecast_type="reforecast", start_on=start_on)
        all_reforecast_dates.extend(dates)
    except Exception as e:
        printf(f"Warning: Could not get reforecast dates for {model_name}: {e}")

# Remove duplicates and sort
all_reforecast_dates = sorted(set(all_reforecast_dates))

printf(f"Processing {len(issuance_dates)} issuance dates")

# %%
# Process each issuance date and compute percentiles
# Save each date incrementally to avoid memory issues and enable resumption
dates_saved = 0
first_save = args.force or existing_percentiles is None
dataset_relpaths = {p: os.path.join("ecmwf-percentiles", dataset_names[p]) for p in ps}

pooled_data = None  # Initialize outside loop to enable incremental saving
for i, issuance_date in enumerate(issuance_dates):
    # Determine which model to use based on the issuance date
    model = get_model_for_date(issuance_date)

    printf(f"\nProcessing issuance date {i+1}/{len(issuance_dates)}: {issuance_date.strftime('%Y-%m-%d')}")

    # Get the 9 base dates for SUBS-M-climate recipe
    base_dates = get_subs_m_climate_base_dates(issuance_date, all_reforecast_dates)

    # Skip this date if we don't have enough base dates
    if base_dates is None:
        printf(f"Skipping {issuance_date.strftime('%Y-%m-%d')}: insufficient base dates (need 9)")
        continue

    # Log base dates with their corresponding models
    base_dates_info = [f"{d.strftime('%Y-%m-%d')}" for d in base_dates]
    printf(f"Selected {len(base_dates)} base dates: {', '.join(base_dates_info)}")

    # Compute percentiles for this issuance date
    # Save pooled_data across iterations to avoid reloading and recomputing for overlapping base dates
    percentiles_ds, pooled_data = compute_ecmwf_quantiles_for_date(
        issuance_date=issuance_date,
        measurement_variable=measurement_variable,
        base_dates=base_dates,
        data_dir=DATA_DIR,
        verbose=verbose,
        pooled_data=pooled_data,
        quantiles=quantiles
    )

    if percentiles_ds is not None:
        # Expand issuance_date from scalar coord to dimension for proper concatenation
        percentiles_ds = percentiles_ds.expand_dims('issuance_date')

        # Convert to float32 and chunk appropriately
        percentiles_ds = percentiles_ds.astype('float32').chunk({
            "quantile": 1,
            "issuance_date": 1,
            "lead": -1,
            "latitude": 1,
            "longitude": -1
        }).load()

        # Save the dataset for each percentile separately
        for p in ps:
            dataset_relpath = dataset_relpaths[p]
            # Select quantile using list to preserve quantile dimension in output
            percentiles_ds_p = percentiles_ds.sel(quantile=[p/100.0])
            tic()
            if first_save and dates_saved == 0:
                # First save: overwrite or create new dataset
                printf(f"Creating new dataset and saving {issuance_date.strftime('%Y-%m-%d')} for percentile {p}...")
                save_data(percentiles_ds_p, dataset_relpath, mode='w')
            else:
                # Subsequent saves: append to existing dataset
                printf(f"Appending {issuance_date.strftime('%Y-%m-%d')} for percentile {p} to dataset...")
                save_data(percentiles_ds_p, dataset_relpath, mode='a', append_dim="issuance_date")
            toc()

        dates_saved += 1
        printf(f"Saved {dates_saved} dates so far")

        # Clean up to free memory
        del percentiles_ds
        gc.collect()

# %%
# Summary
if dates_saved == 0:
    printf("No quintile data was generated. Exiting...")
    sys.exit(1)

printf(f"\nSuccessfully saved {dates_saved} new dates to ecmwf-percentiles/{dataset_names}")
