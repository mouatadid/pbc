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
Download ECMWF reforecasts to form ECMWF climatology

Example usages: 

  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv tas
  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv pr
  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv mslp
  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy49 -wv mslp; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy48 -wv mslp; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv mslp
  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy49 -wv pr; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy48 -wv pr; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv pr
  python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy49 -wv tas; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy48 -wv tas; python src/data/datasets/ecmwf-clim.py -v -m ecmwf_cy41-47 -wv tas
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

from utils.data_io import (
    DATA_DIR
)
from utils.logging import printf
from utils.file_io import set_file_permissions, make_directories
from pathlib import Path

from data.helpers.data_utils import ( 
    IRI_GRID_NAMES,
    get_valid_forecast_dates,
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
if isnotebook():
    args = parser.parse_known_args(["-v", "-m", "ecmwf_cy41-47", "-wv", "mslp"])[0]
else:
    args = parser.parse_args()


#
# Process arguments
#
force_download = args.force_download

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

# Restrict to specific leads needed for subseasonal forecast horizons
lead_offset = 0.5 if weather_variable == "tas" else 0.
FIRST_LEAD = 19. + lead_offset
LAST_LEAD = 26. + lead_offset
# This first selects the range FIRST_LEAD to LAST_LEAD; then requesting LAST_LEAD alone 
# leads to FIRST_LEAD also being included
restrict_leads_url = f"{leads_id}/%28{FIRST_LEAD}%29%28{LAST_LEAD}%29RANGEEDGES/{leads_id}/%28%2C{LAST_LEAD}%29VALUES/"

# Load IRI authentication key for ECMWF data
with open(pathlib.Path(os.path.join(os.path.expanduser("~"),".pycpt_dlauth")), "r") as credentials_file:
    credentials = json.load(credentials_file)
    ecmwf_key = credentials["key"]

model = args.model
print_ok(model, bold=True, verbose=args.verbose)
reforecast = True
forecast_type = "reforecast" if reforecast else "forecast"
forecast_type_on_server = forecast_type

if reforecast:
    rename_grids_model_url = rename_grids_url + f"hdate/({IRI_GRID_NAMES['hdate']})/renameGRID/"
else:
    rename_grids_model_url = rename_grids_url

# Create output directory for storing downloaded NetCDF files
output_folder = (
    Path(DATA_DIR) / "ecmwf-clim" /
    f"{model}-{forecast_type}"
    f"-{weather_variable}"
)
make_directories(output_folder)
printf(
    f"NetCDF storage folder: {output_folder}\n",
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

# Download each issuance date
printf(f"\nDownloading {model} {forecast_type}s using template \n{template_URL}\n")
# Restrict earliest downloaded date
# For reference, the earliest available reforecast issuance date is string_to_dt("20150514")
start_on = None #"20151001"
valid_forecast_dates = get_valid_forecast_dates(model, forecast_type, start_on=start_on)
for issuance_date in valid_forecast_dates: 
    # Construct the download URL for this issuance date
    day, month, year = datetime.strftime(issuance_date, "%d,%b,%Y").split(",")
    restrict_issuance_date_url = f"S/%28{hours}%20{day}%20{month}%20{year}%29VALUES/"
    # Do not restrict hindcast dates
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
                available = False
                continue
            else:
                print_error(bold=True)
                printf(
                    f"Unknown error occured when trying to download data for {day} {month} {year} for model {model} from {URL}.\n"
                )
                available = False
                continue



