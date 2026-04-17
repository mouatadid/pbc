#!/usr/bin/env python
# Identify new MSN forecast data to be processed through msn.py
# 
# Named args:
#   --weather_variable (-wv): weather variable to process

import os
import pandas as pd
from models.utils.data_utils import printf
import xarray as xr
import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--weather_variable", "-wv", choices=["tas", "pr", "mslp"])
args = parser.parse_args()

var = args.weather_variable

def get_last_issuance_date(var):
    """Get the latest issuance date from the zarr file for a given variable"""
    if var == "pr":
        zarr_file = "data/model_poet-ec-v4-forecast-f1_pr.zarr"
    else:
        zarr_file = f"data/model_poet-ec-v5-forecast-f1_{var}.zarr"
    
    if Path(zarr_file).exists():
        ds = xr.open_zarr(zarr_file)
        if 'issuance_date' in ds.coords and ds.issuance_date.size > 0:
            latest_date = pd.Timestamp(ds.issuance_date.max().values)
            return latest_date.strftime('%Y%m%d')
        else:
            printf(f"No issuance_date dimension found in {zarr_file}")
    else:
        printf(f"Forecasts zarr not found: {zarr_file}")
    
    return None

def get_available_dates():
    """Get available issuance dates from mai-forecasts folder"""
    mai_forecasts_dir = "data/msn/mai-forecasts"
    
    if not os.path.exists(mai_forecasts_dir):
        printf("mai-forecasts directory does not exist")
        return []
    
    # Find all directories labeled with issuance dates (issuance date format: YYYYMMDD00)
    date_dirs = []
    for item in os.listdir(mai_forecasts_dir):
        if item.endswith("00"):
            date_str = item[:-2] # Extract date part
            date_dirs.append(date_str)
    
    return sorted(date_dirs)

# Get available dates from mai-forecasts
available_dates = get_available_dates()

if not available_dates:
    printf("Warning: no dates found in mai-forecasts folder")

# Get latest date in zarr
last_issuance_date = get_last_issuance_date(var)
printf(f"Latest issuance date already stored: {last_issuance_date}")

# Find dates newer than latest zarr date
if last_issuance_date and available_dates:
    new_dates = [date for date in available_dates if date > last_issuance_date]
else: 
    new_dates = []

if not new_dates:
    printf(f"No new dates to process for {var}")
else: 
    printf(f"Found {len(new_dates)} new dates for {var}: {new_dates}")
    # Process new dates
    processed_count = 0
    for date in new_dates:
        printf(f"Processing date {date}")
        cmd = f"python src/data/datasets/msn.py -wv {var} -d {date} -v"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            processed_count += 1
            printf(f"Successfully processed date {date}")
        else:
            printf(f"Error processing date {date}: {result.stderr}")
    printf(f"Successfully processed {processed_count}/{len(new_dates)} new dates for {var}")

printf(f"\nFinished processing new MSN {var} data")