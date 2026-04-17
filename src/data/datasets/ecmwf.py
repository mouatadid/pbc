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
Download ECMWF forecasts and reforecasts

Example usages: 
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv tas
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy48 -wv tas
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy41-47 -wv tas
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy49 -wv tas
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy48 -wv tas
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy41-47 -wv tas
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv pr
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy48 -wv pr
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy41-47 -wv pr
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy49 -wv pr
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy48 -wv pr
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy41-47 -wv pr
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv mslp
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy48 -wv mslp
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy41-47 -wv mslp
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy49 -wv mslp
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy48 -wv mslp
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy41-47 -wv mslp
  python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy49 -wv mslp; python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy48 -wv mslp; python src/data/datasets/ecmwf.py -v -ref -m ecmwf_cy41-47 -wv mslp
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy41-47 -wv mslp; python src/data/datasets/ecmwf.py -v -m ecmwf_cy48 -wv mslp; python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv mslp
  python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv mslp; python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv pr; python src/data/datasets/ecmwf.py -v -m ecmwf_cy49 -wv tas
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
import json
import pathlib
import sys
import time
import os
from datetime import datetime
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
from utils.file_io import set_file_permissions, make_directories, symlink
from pathlib import Path

from data.helpers.data_utils import ( 
    IRI_GRID_NAMES,
    get_valid_forecast_dates,
    load_s2s_reforecast_dataset,
)
from data.helpers.general_utils import (
    download_url,
    dt_to_string,
    print_error,
    print_ok,
    print_warning,
)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tas",
    choices=["tas", "pr", "mslp"],
    help="Name of weather variable to download",
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
    "--verbose",
    "-v",
    action="store_true",
    help="Print verbose output",
)
parser.add_argument(
    "--force_download",
    "-fd",
    action="store_true",
    help="If true, will download data if resulting file already exists; "
        "otherwise, will skip download for files that already exist",
)
parser.add_argument(
    "--process",
    "-p",
    action="store_true",
    help="If true, will process downloaded data into probabilistic forecasts; "
        "otherwise, will only download the data",
)
if isnotebook():
    args = parser.parse_known_args(["-p", "-v", "-m", "ecmwf_cy41-47", "-wv", "mslp", "-ref"])[0]
else:
    args = parser.parse_args()


#
# Process arguments
#
force_download = args.force_download
process = args.process

# For each lead start time, average or sum over this number of days.
ACCUMULATE_LEADS_PERIOD = 7  

weather_variable = args.weather_variable
model_names_on_server = {
    "ecmwf_cy49": "ECMF/.CY49",
    "ecmwf_cy48": "ECMF/.CY48",
    "ecmwf_cy41-47": "ECMF/.CY41-47",
}
weather_variable_names_on_server = {
    "tas": "2m_above_ground/.2t",
    "pr": "sfc_precip/.tp",
    "mslp": "sfc_pressure/.msl"
}
weather_variable_name_on_server = weather_variable_names_on_server[weather_variable]
leads_id = "LA" if weather_variable == "tas" else "L"

# Preserve "M" as the coordinate for ensemble members, and
# use interpretable names for other dimensions
rename_grids_url = (
    f"X/({IRI_GRID_NAMES['X']})/renameGRID/"
    f"Y/({IRI_GRID_NAMES['Y']})/renameGRID/"
    f"S/({IRI_GRID_NAMES['S']})/renameGRID/"
    f"{leads_id}/({IRI_GRID_NAMES[leads_id]})/renameGRID/"
)
# rename_grids_url += f"M/({IRI_GRID_NAMES['M']})/renameGRID/" if not args.control_forecast else ""

if weather_variable == "pr":
    # Precipitation is given as accumulated over lead time, so the data must be
    # subtracted from a lead-shifted version; this is done directly in the URL instead of here.
    accumulate_leads_url = ""
else:
    # Compute running average of variable over the leads period
    # After running average, each lead will be associated with its midpoint,
    # so shift lead index backward by ACCUMULATE_LEADS_PERIOD//2
    accumulate_leads_url = (
        f"{leads_id}/{ACCUMULATE_LEADS_PERIOD}/runningAverage/"
        f"{leads_id}/-{ACCUMULATE_LEADS_PERIOD//2}/shiftGRID/"
    )
    
# No need to convert units
convert_units_url = ""

# Restrict lead range to those needed for subseasonal forecast horizons
lead_offset = 0.5 if weather_variable == "tas" else 0.
FIRST_LEAD = 0. + lead_offset
LAST_LEAD = 26. + lead_offset
restrict_leads_url = f"{leads_id}/%28{FIRST_LEAD}%29%28{LAST_LEAD}%29RANGEEDGES/"

# Load IRI authentication key for ECMWF data
with open(pathlib.Path(os.path.join(os.path.expanduser("~"),".pycpt_dlauth")), "r") as credentials_file:
    credentials = json.load(credentials_file)
    ecmwf_key = credentials["key"]

model = args.model
print_ok(model, bold=True, verbose=args.verbose)
reforecast = args.reforecast
forecast_type = "reforecast" if reforecast else "forecast"
forecast_type_on_server = forecast_type

if reforecast:
    rename_grids_model_url = rename_grids_url + f"hdate/({IRI_GRID_NAMES['hdate']})/renameGRID/"
else:
    rename_grids_model_url = rename_grids_url

# Create output directory for storing downloaded NetCDF files
output_folder = (
    Path(DATA_DIR) / "ecmwf" /
    f"{model}-{forecast_type}"
    f"-{weather_variable}"
)
make_directories(output_folder)
printf(
    f"NetCDF storage folder: {output_folder}\n",
    verbose=args.verbose,
)
# Folder for ecmwf-backup data
backup_folder = (
    Path(DATA_DIR) / "ecmwf-backup" /
    f"{model}-{forecast_type}"
    f"-{weather_variable}"
)
# Folder for ecmwf-backup reforecast data (uses different naming convention)
reforecast_backup_folder = (
    Path(DATA_DIR) / "ecmwf-backup" /
    f"ecmwf-reforecast-{weather_variable}"
)
printf(
    f"NetCDF backup storage folder: {backup_folder}\n",
    verbose=args.verbose,
)

# Create template URL for downloading data
# Do not format forecast_run, restrict_issuance_date_url, or restrict_hindcast_date_url
# as they will be inputted later
base_url = "https://iridl.ldeo.columbia.edu/"
model_url = f"SOURCES/.ECMWF/.S2S/.{model_names_on_server[model]}/"
modifiers_url = (
    f".{forecast_type_on_server}/"
    ".{forecast_run}/"
    f".{weather_variable_name_on_server}/"
    "{restrict_issuance_date_url}"
    "{restrict_hindcast_date_url}"
)
if weather_variable == "pr":
    # Since precipitation is accumulated over leads, shift the data and subtract from it the original data.
    # Drop the extra L_lag column (of constant value, since the shift is always of same size).
    template_URL = (
        f"{base_url}{model_url}{modifiers_url}"
        f"{leads_id}/{ACCUMULATE_LEADS_PERIOD}/shiftdata/"
        f"{model_url}{modifiers_url}sub/"
        f"{leads_id}_lag/removeGRID/{restrict_leads_url}"
        f"{rename_grids_model_url}"
        f"data.nc"
    )
else:
    template_URL = (
        f"{base_url}{model_url}{modifiers_url}"
        f"{accumulate_leads_url}"
        f"{restrict_leads_url}"
        f"{convert_units_url}"
        f"{rename_grids_model_url}"
        f"data.nc"
    )    

# We will specify hours in reforecast_issuance_date_url to ensure a single download date
hours = "0000"


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

# Load quintile boundaries 
printf(f"Loading ERA5 quintiles for measurement: {weather_variable}")
quintiles = load_data(f"era5-quintiles-{weather_variable}")
if process:
    # Load data explicitly into memory
    quintiles.load()
quantiles = quintiles["quantile"].values

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

# Chunk zarr data by date to enable appending
chunk = ({"lead": -1, "latitude": -1, "longitude": -1, "year_delta": -1, "issuance_date": 1} 
    if reforecast else {"lead": -1, "latitude": -1, "longitude": -1, "issuance_date": 1})
if reforecast:
    # Specify standard year_delta range for reforecasts
    year_deltas = np.arange(-20, 0, 1)
    num_year_deltas = year_deltas.size

# Process each issuance date
printf(f"\nDownloading and processing {model} {forecast_type}s using template \n{template_URL}\n")
# Limit earliest download date for reforecasts
start_on = "20151211" if reforecast else None
###last_saved_date = None###
for issuance_date in get_valid_forecast_dates(model, forecast_type, start_on=start_on): 
    if last_saved_date and issuance_date <= last_saved_date:
        printf(
            f"Skipping {issuance_date}: already saved.",
            verbose=args.verbose,
        )
        continue
    # Construct the download URL for this issuance date
    day, month, year = datetime.strftime(issuance_date, "%d,%b,%Y").split(",")
    restrict_issuance_date_url = f"S/%28{hours}%20{day}%20{month}%20{year}%29VALUES/"
    # Restrict hindcast dates based on the model and issuance year
    if reforecast:
        year = int(year)
        if model == "ecmwf_cy41-47":
            if year == 2022:
                first_year, last_year = 2002, 2015
            elif year == 2021:
                first_year, last_year = 2001, 2004
            elif year == 2020:
                first_year, last_year = 2000, 2004
            elif year == 2019:
                first_year, last_year = 1999, 2002
            else:
                first_year, last_year = 1998, year-17
        else:
            # Restrict hindcast dates to the prior 20 years
            first_year, last_year = year - 20, year - 1
        restrict_hindcast_date_url = f"hdate/%28{first_year}%29%28{last_year}%29RANGEEDGES/" 
    else:
        # Hindcast dates are not used in forecasts
        restrict_hindcast_date_url = ""
 
    # Identify the number of ensemble members, including the control forecast
    forecast_runs = ["control", "perturbed"]
    num_ensemble_members = 11 if reforecast else (51 if model == "ecmwf_cy41-47" else 101)

    # Initialize dictionary to hold ensemble predictions for each quantile
    ensemble_pred = {}

    # For each forecast run
    available = True
    for forecast_run in forecast_runs:
        # Select path for individual file download
        file_path = output_folder / f"{dt_to_string(issuance_date)}-{forecast_run}.nc"

        # IRI reforecast data from April 2026 onwards is NA-filled;
        # pre-link backup files so the existing file-exists check skips the IRI download
        if reforecast and issuance_date >= datetime(2026, 4, 1) and not file_path.exists():
            reforecast_backup_file_path = reforecast_backup_folder / f"{dt_to_string(issuance_date)}-{forecast_run}.nc"
            if reforecast_backup_file_path.exists():
                printf(
                    f"Using reforecast backup for {issuance_date} {forecast_run}.",
                    verbose=args.verbose,
                )
                symlink(reforecast_backup_file_path, file_path, use_abs_path=True)
            else:
                printf(
                    f"No reforecast backup available for {issuance_date} {forecast_run}.",
                    verbose=args.verbose,
                )
                available = False
                # Disable processing for this and all future issuance dates to ensure that all dates
                # are eventually processed in order
                process = False
                continue

        if not force_download and file_path.exists():
            # Skip download if file already exists
            printf(
                f"Skipping {file_path} download: (file already exists).",
                verbose=args.verbose,
            )
        else:
            # Download deterministic forecast data 
            URL = template_URL.format(
                forecast_run=forecast_run, 
                restrict_issuance_date_url=restrict_issuance_date_url,
                restrict_hindcast_date_url=restrict_hindcast_date_url)
            t = time.time()
            printf(f"\nDownloading: {day} {month} {year} {forecast_run} from {URL}.", verbose=args.verbose)
            r = download_url(URL, cookies={"__dlauth_id": ecmwf_key})
            if r.status_code == 200 and r.headers["Content-Type"] == "application/x-netcdf":
                # Write content to nc file
                with open(file_path, "wb") as f:
                    f.write(r.content)
                set_file_permissions(file_path)
                printf(
                    f"-done (downloaded {sys.getsizeof(r.content)/1024:.2f} KB in {time.time() - t:.2f}s).\n",
                    verbose=args.verbose,
                )
            elif r.status_code == 404:
                print_warning(bold=True, verbose=args.verbose)
                printf(
                    f"Data for {day} {month} {year} is not available for model {model}.\n",
                    verbose=args.verbose,
                )
                # For forecasts, use ecmwf-backup data instead if available
                backup_file_path = backup_folder / f"{dt_to_string(issuance_date)}-{forecast_run}.nc"
                # For reforecasts, use ecmwf-reforecast backup data instead if available
                reforecast_backup_file_path = reforecast_backup_folder / f"{dt_to_string(issuance_date)}-{forecast_run}.nc"
                if (not reforecast) and backup_file_path.exists():
                    printf(
                        f"Using backup data from {backup_file_path}.\n",
                        verbose=args.verbose,
                    )
                    # Soft link backup file to output folder
                    symlink(backup_file_path, file_path, use_abs_path=True)
                elif reforecast and reforecast_backup_file_path.exists():
                    printf(
                        f"Using reforecast backup data from {reforecast_backup_file_path}.\n",
                        verbose=args.verbose,
                    )
                    # Soft link backup file to output folder
                    symlink(reforecast_backup_file_path, file_path, use_abs_path=True)
                else:
                    available = False
                    printf("Disabling processing for this and all future issuance dates to ensure that all dates are eventually processed in order.\n",
                        verbose=args.verbose)
                    process = False
                    continue
            else:
                print_error(bold=True)
                printf(
                    f"Unknown error occured when trying to download data for {day} {month} {year} for model {model} from {URL}.\n"
                )
                available = False
                printf("Disabling processing for this and all future issuance dates to ensure that all dates are eventually processed in order.\n",
                       verbose=args.verbose)
                process = False
                continue
        
        if not process:
            # Skip probabilistic processing
            continue 

        # Load content as xarray dataset
        # Note: if decode_timedelta = False, leads will be floats rather than timedelta64[ns]
        print("Loading deterministic data..."); tic()
        ds = (load_s2s_reforecast_dataset(file_path) 
            if reforecast else xr.load_dataset(file_path, decode_timedelta=True))

        # Check if the loaded dataset contains all NaN values for the weather variable
        server_weather_variable = weather_variable_name_on_server.split("/.")[1]
        if ds[server_weather_variable].isnull().all():
            print_error(f"Error: All deterministic values are NaN for {file_path}")
            os.remove(file_path)
            print(f"File deleted: {file_path}")
            if reforecast:  
                # For reforecasts, use ecmwf-reforecast backup data instead if available
                reforecast_backup_file_path = reforecast_backup_folder / f"{dt_to_string(issuance_date)}-{forecast_run}.nc"
                if reforecast_backup_file_path.exists():
                    printf(
                        f"Using reforecast backup data from {reforecast_backup_file_path}.\n",
                        verbose=args.verbose,
                    )
                    # Soft link backup file to output folder
                    symlink(reforecast_backup_file_path, file_path, use_abs_path=True)
                    # Reload data
                    print("Loading deterministic data..."); tic()
                    ds = (load_s2s_reforecast_dataset(file_path) 
                        if reforecast else xr.load_dataset(file_path, decode_timedelta=True))
                    # Check if the loaded dataset contains all NaN values for the weather variable
                    if ds[server_weather_variable].isnull().all():
                        print_error(f"Error: All deterministic values are NaN for {file_path}")
                        os.remove(file_path)
                        print(f"File deleted: {file_path}")
                        sys.exit(1)  # Exit with status code 1
                else:
                    sys.exit(1)  # Exit with status code 1
            else:
                sys.exit(1)  # Exit with status code 1

        # Use our conventional name for weather variable
        ds = ds.rename({server_weather_variable: weather_variable})
        if weather_variable == "tas":
            # Map all half day leads to integers
            ds["lead"] = ds["lead"] - np.timedelta64(12, "h")
        if not reforecast:
            # Map forecast issuance date and lead to start date of period being forecasted
            times = issuance_to_target(ds.issuance_date.values, ds.lead.values).squeeze()
            ds = ds.assign_coords({"time":('lead', times)}).swap_dims({"lead": "time"})
        else:
            # Map hindcast date and lead to start date of period being forecasted
            leads = ds.lead.values
            num_leads = leads.size
            hindcast_dates = ds.hindcast_date.values
            times = issuance_to_target(hindcast_dates, leads)
            # Replace hindcast_date with years between hindcast date and issuance date
            ds["hindcast_date"] = (hindcast_dates.astype('M8[Y]')
                - ds.issuance_date.values.astype('M8[Y]')).astype('int')
            ds = ds.rename({"hindcast_date": "year_delta"})
            # Replace (year_delta, lead) coordinates with time
            ds = ds.assign_coords(
                    {"time":(('year_delta', 'lead'), times)}
                 ).stack(combo=("year_delta", "lead")).swap_dims({"combo": "time"})
        toc()
        
        print("Computing probabilistic forecast..."); tic()
        # Check whether any ds values are NaN
        ds_notnull = ds[weather_variable].notnull()
        ds_has_nans = not ds_notnull.all()
        if ds_has_nans:
            print(f"Warning: {model} {issuance_date} {forecast_run} contains NaNs")
        for quantile in quantiles:
            # Compute cumulative probabilistic prediction for this quantile bin
            cdf_pred = (ds < quintiles.sel(quantile = quantile, drop=True)).astype("float32")
            if ds_has_nans:
                # Keep cdf_pred values if ds value is not null; otherwise, replace with NaN
                cdf_pred = cdf_pred.where(ds_notnull, np.nan)
            # Reformat prediction to have lead (and possibly year_delta) dimension instead of time
            if not reforecast:
                contribution = cdf_pred.swap_dims({"time":"lead"}).drop_vars("time")
            else:
                contribution = cdf_pred.swap_dims({"time":"combo"}).drop_vars("time")
                contribution = contribution.set_index(combo=["year_delta","lead"]).unstack("combo")
                # In case some leads were dropped due to the corresponding time not existing in 
                # quintiles, reindex lead dimension
                if contribution.sizes["lead"] < num_leads:
                    contribution = contribution.reindex(lead=leads)
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
        toc()
    
    if not process or not available:
        # Skip probabilistic processing
        continue 
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
