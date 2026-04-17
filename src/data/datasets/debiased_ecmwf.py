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
# Compute "debiased" probabilistic (re)forecasts from deterministic ECMWF (re)forecasts 
# using ECMWF model-based quintiles
#
# This script takes existing deterministic ECMWF operational (re)forecasts and converts them
# to probabilistic forecasts using ECMWF reforecast-based quintiles (computed by ecmwf-quintiles.py).
# The logic follows the second half of ecmwf.py but uses ECMWF quintiles instead of ERA5 quintiles
# and reuses the ECMWF quintiles for a given forecast date as the quintiles for each associated
# reforecast.
#
# Args:
#     --weather_variable (-wv): The measurement variable to process (e.g., "tas", "pr", "mslp")
#     --start_date: Optional start date for processing (YYYYMMDD)
#     --end_date: Optional end date for processing (YYYYMMDD)
#     --model: Which ECMWF model cycle to use for the quintiles (e.g., "ecmwf_cy41-47", "ecmwf_cy48", "ecmwf_cy49")
#     --reforecast: Generate probabilistic reforecasts (instead of forecasts)
#     --verbose (-v): Print verbose output
#     --force: Overwrite any existing probabilistic forecasts
#
# Example usage:
#   python src/data/datasets/debiased_ecmwf.py -wv tas -v -ref --model ecmwf_cy41-47
#   src/batch/batch_python.sh -m 80 -c 8 -h 12 src/data/datasets/debiased_ecmwf.py -wv mslp -v --model ecmwf_cy49 -ref
#   for var in pr mslp; do for model in ecmwf_cy41-47 ecmwf_cy48 ecmwf_cy49; do src/batch/batch_python.sh -m 25 -c 2 -h 12 src/data/datasets/debiased_ecmwf.py -wv $var -v -ref --model $model; done; done
#   for var in tas pr mslp; do for model in ecmwf_cy48 ecmwf_cy49; do src/batch/batch_python.sh -m 25 -c 2 -h 12 src/data/datasets/debiased_ecmwf.py -wv $var -v --model $model; done; done

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
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

from utils.timing import tic, toc
from utils.data_io import (
    save_data,
    get_last_date_in_zarr,
    get_zarr_store_path,
    DATA_DIR
)
from utils.logging import printf
from data.helpers.data_utils import (
    get_valid_forecast_dates,
    load_s2s_reforecast_dataset,
)
from data.helpers.general_utils import (
    print_error,
)

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tas",
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
    "--model",
    "-m",
    default="ecmwf_cy49",
    choices=["ecmwf_cy49", "ecmwf_cy48", "ecmwf_cy41-47"],
    # nargs="+",
    help="Name of model to download",
)
parser.add_argument(
    "--reforecast",
    "-ref",
    action="store_true",
    help="If true, download reforecast (i.e., the hindcasts for the last 20 years)",
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
    help='Overwrite any existing probabilistic forecasts'
)

if isnotebook():
    args = parser.parse_known_args(['-wv', 'tas', '-v', '--reforecast', '--model', 'ecmwf_cy41-47'])[0]
else:
    args = parser.parse_args()

measurement_variable = args.weather_variable
verbose = args.verbose

# %%
def get_model_for_date(issuance_date: datetime) -> str:
    """
    Determine the ECMWF model version based on the forecast issuance date.

    Model versions by forecast issuance date:
    - cy41-47: before 2023-06-28
    - cy48: 2023-06-28 to 2024-11-11
    - cy49: 2024-11-12 onwards

    Args:
        issuance_date: The forecast initialization date

    Returns:
        Model version string (e.g., "ecmwf_cy49")
    """
    if issuance_date < datetime(2023, 6, 28):
        return "ecmwf_cy41-47"
    elif issuance_date <= datetime(2024, 11, 11):
        return "ecmwf_cy48"
    else:
        return "ecmwf_cy49"

# %%
# Open ECMWF-based quintiles dataset (lazy loading - data loaded per issuance date in the loop)
printf(f"Opening ECMWF-based quintiles dataset for {measurement_variable}")
try:
    ###NOTE: the time to .sel() a single issuance_date with this open_mfdataset
    ### is 10-30x the .sel() time on a single-model dataset; perhaps could
    ### be made more efficient by (a) selecting only those models needed for
    ### the issuance date range; (b) adaptively loading the right quintile
    ### file for a given issuance date; and/or (c) pre-loading all needed
    ### issuance dates into memory
    model_suffixes = ["cy41-47", "cy48", "cy49"]
    filenames = [os.path.join(DATA_DIR, "ecmwf-quintiles", f"ecmwf_{model_suffix}-quintiles-{measurement_variable}.zarr") for model_suffix in model_suffixes]
    quintiles_dataset = xr.open_mfdataset(filenames)
    printf(f"Opened quintiles dataset successfully!")
    printf(f"Quintiles cover issuance dates from {quintiles_dataset.issuance_date.min().values} to {quintiles_dataset.issuance_date.max().values}")
except Exception as e:
    raise Exception(f"Error opening ECMWF quintiles: {e}")

if measurement_variable == "tas":
    # Map all half day leads to integers
    quintiles_dataset = quintiles_dataset.assign_coords(lead=quintiles_dataset.lead - np.timedelta64(12, "h"))

quantiles = quintiles_dataset["quantile"].values
available_quintile_dates = set(quintiles_dataset.issuance_date.values)

# %%
# Map server variable names to our conventional names
weather_variable_names_on_server = {
    "tas": "2t",
    "pr": "tp",
    "mslp": "msl"
}
server_weather_variable = weather_variable_names_on_server[measurement_variable]

# Select output dataset names for each quantile
model = args.model
reforecast = args.reforecast
forecast_type = "reforecast" if reforecast else "forecast"
dataset_names = {quantile: os.path.join("debiased_ecmwf", f"debiased_{model}-{forecast_type}-f{ii+1}_{measurement_variable}") for (ii, quantile) in enumerate(quantiles)}

# Identify the last date stored in the output zarr files
if not args.force:
    store_path = get_zarr_store_path(DATA_DIR, dataset_names[quantiles[0]])
    last_saved_date = get_last_date_in_zarr(store_path, time_coord="issuance_date")
    printf(f"Last saved issuance date in zarr: {last_saved_date}")
else:
    last_saved_date = None
    printf("Force mode - will reprocess all dates")

# %%
# Get list of issuance dates to process from model
all_issuance_dates = []

# Limit earliest download date for reforecasts
start_on = "20151211" if reforecast else None

try:
    dates = get_valid_forecast_dates(model, forecast_type, start_on=start_on)
    all_issuance_dates.extend(dates)
except Exception as e:
    printf(f"Warning: Could not get dates for {model}: {e}")

# Remove duplicates and sort
all_issuance_dates = sorted(set(all_issuance_dates))

# Filter by date range if specified
if args.start_date and args.end_date:
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if start_date <= d <= end_date]
elif args.start_date:
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if d >= start_date]
elif args.end_date:
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    issuance_dates = [d for d in all_issuance_dates if d <= end_date]
else:
    issuance_dates = all_issuance_dates

printf(f"\nProcessing {len(issuance_dates)} issuance dates")

# Chunk zarr data by date to enable appending
chunk = ({"lead": -1, "latitude": -1, "longitude": -1, "year_delta": -1, "issuance_date": 1} 
    if reforecast else {"lead": -1, "latitude": -1, "longitude": -1, "issuance_date": 1})
if reforecast:
    # Specify standard year_delta range for reforecasts
    year_deltas = np.arange(-20, 0, 1)
    num_year_deltas = year_deltas.size

# %%
# Process each issuance date
for issuance_date in issuance_dates:
    if last_saved_date and issuance_date <= last_saved_date:
        printf(f"Skipping {issuance_date.strftime('%Y-%m-%d')}: already saved.")
        continue

    printf(f"\nProcessing issuance date: {issuance_date.strftime('%Y-%m-%d')}")

    # Check if quintiles exist for this issuance date
    if np.datetime64(issuance_date) not in available_quintile_dates:
        printf(f"Warning: No quintiles available for {issuance_date.strftime('%Y-%m-%d')}, skipping.", verbose=True)
        continue

    # Set input folder based on the model
    ###model = get_model_for_date(issuance_date)
    input_folder = os.path.join(DATA_DIR, "ecmwf", f"{model}-{forecast_type}-{measurement_variable}")

    # Load quintiles for this issuance date 
    # Note: only leads 19 and 26 have quintiles computed
    printf(f"Loading quintiles for {issuance_date.strftime('%Y-%m-%d')}")
    tic()
    quintiles_for_date = quintiles_dataset.sel(issuance_date=issuance_date)
    quintiles_for_date.load()
    toc()

    # Load forecast runs for this date
    forecast_runs = ["control", "perturbed"]
    # Identify the number of ensemble members, including the control forecast
    num_ensemble_members = 11 if reforecast else (51 if model == "ecmwf_cy41-47" else 101)

    # Initialize dictionary to hold ensemble predictions for each quantile
    ensemble_pred = {}

    # Track if data is available
    available = True

    for forecast_run in forecast_runs:
        # Construct file path
        date_str = issuance_date.strftime("%Y%m%d")
        file_path = os.path.join(input_folder, f"{date_str}-{forecast_run}.nc")

        if not os.path.exists(file_path):
            printf(f"Warning: File not found: {file_path}", verbose=True)
            available = False
            break

        printf(f"Loading {forecast_run} deterministic data...")
        tic()
        # Load content as xarray dataset
        # Note: if decode_timedelta = False, leads will be floats rather than timedelta64[ns]
        ds = (load_s2s_reforecast_dataset(file_path) 
            if reforecast else xr.load_dataset(file_path, decode_timedelta=True))

        # Filter to only keep leads 19 and 26
        if measurement_variable == "tas": 
            target_leads = [pd.Timedelta(days=19, hours=12), pd.Timedelta(days=26, hours=12)]    
        else: 
            target_leads = [pd.Timedelta(days=19), pd.Timedelta(days=26)]
        ds = ds.sel(lead=target_leads)

        # Check if the loaded dataset contains all NaN values for the weather variable
        if ds[server_weather_variable].isnull().all():
            print_error(f"Error: All deterministic values are NaN for {file_path}")
            available = False
            break

        # Use our conventional name for weather variable
        ds = ds.rename({server_weather_variable: measurement_variable})

        if measurement_variable == "tas":
            # Map all half day leads to integers
            ds = ds.assign_coords(lead=ds.lead - np.timedelta64(12, "h"))

        if reforecast:
            # Replace hindcast_date with years between hindcast date and issuance date
            ds["hindcast_date"] = (ds.hindcast_date.values.astype('M8[Y]')
                - ds.issuance_date.values.astype('M8[Y]')).astype('int')
            ds = ds.rename({"hindcast_date": "year_delta"})

        toc()

        printf("Computing probabilistic forecast...")
        tic()

        # Check whether any ds values are NaN
        ds_notnull = ds[measurement_variable].notnull()
        ds_has_nans = not ds_notnull.all()
        if ds_has_nans:
            printf(f"Warning: {model} {issuance_date.strftime('%Y-%m-%d')} {forecast_run} contains NaNs", verbose=True)

        for quantile in quantiles:
            # Get quintile thresholds for this quantile
            quintile_threshold = quintiles_for_date.sel(quantile=quantile, drop=True)

            # Compute cumulative probabilistic prediction for this quantile bin
            # Only compute for times where quintiles are available
            contribution = (ds < quintile_threshold).astype("float32")

            if ds_has_nans:
                # Keep cdf_pred values if ds value is not null; otherwise, replace with NaN
                contribution = contribution.where(ds_notnull, np.nan)

            if reforecast:
                # Reindex year_delta dimension in case some deltas are missing
                if contribution.sizes["year_delta"] < num_year_deltas:
                    contribution = contribution.reindex(year_delta=year_deltas)


            # Sum over the member dimension ("M") of perturbed forecasts
            # while ensuring that nans are preserved
            if forecast_run == "perturbed":
                contribution = contribution.sum("M", skipna=False)

            # Add prediction to the ensemble
            if quantile not in ensemble_pred:
                ensemble_pred[quantile] = contribution
            else:
                ensemble_pred[quantile] += contribution

            del quintile_threshold
            del contribution

        del ds
        del ds_notnull

        toc()

    if not available:
        printf(f"Skipping {issuance_date.strftime('%Y-%m-%d')}: data not available", verbose=True)
        continue

    # Normalize and save ensemble predictions
    for ii, quantile in enumerate(quantiles):
        printf(f"Saving to {dataset_names[quantile]}...")
        tic()

        # Normalize ensemble predictions by number of ensemble members
        ensemble_pred[quantile] /= num_ensemble_members

        # Append ensemble forecast, sorted by ascending latitude,
        # to appropriate zarr with quantile name and appropriate chunking
        save_data(
            ensemble_pred[quantile].sortby("latitude").rename(
                {measurement_variable: f"f{ii+1}_{measurement_variable}"}
            ).chunk(chunk),
            dataset_names[quantile],
            mode='a',
            append_dim='issuance_date'
        )

        toc()

    del quintiles_for_date
    del ensemble_pred

printf("\nProcessing complete!")
