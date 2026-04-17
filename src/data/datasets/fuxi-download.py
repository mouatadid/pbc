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
# Fuxi pipeline: download, extract and pre-process Fuxi forecasts.
#
# Args:
#     --target_dates (-t): List of target dates to download and process (e.g., "20020103", "200201", "2002")
#     --skip_process (-sp): When enabled, the pipeline runs in download-only mode. It will download the .7z archives from the Hugging Face dataset but skip extraction, processing, and NetCDF generation.
#
# Example usage:
#     python src/data/datasets/fuxi-download.py -t 2002
#     python src/data/datasets/fuxi-download.py -t 200201
#     python src/data/datasets/fuxi-download.py -t 20020103
#     python src/data/datasets/fuxi-download.py -t 20020103 -sp
#     src/batch/batch_python.sh -h 11 -m 30 -e aiwqd src/data/datasets/fuxi-download.py -t 2002 
    


import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
import sys
import shutil
import pickle
import gc
import time
import numcodecs
import subprocess
import py7zr
import zarr
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from IPython import get_ipython
from glob import glob
from huggingface_hub import hf_hub_download, HfApi
from datetime import datetime
from utils.file_io import make_directories, set_file_permissions, make_parent_directories
from utils.timing import tic, toc
from utils.notebook import isnotebook
from models.utils.general_util import printf
from models.utils.eval_util import get_target_dates
from utils.data_io import DATA_DIR, save_to_netcdf

# --- Notebook setup ---
if isnotebook():
    os.chdir(os.path.join(os.path.expanduser("~"), "aiwq"))
    ip = get_ipython()
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")


# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--target_dates",
    default="2017",
    help="e.g. std_fuxi, 2017, 201701, 20170103"
)
parser.add_argument(
    "-sp", "--skip_process",
    action="store_true",
    help="If True, only download .7z files and skip extraction/processing"
)
if isnotebook():
    args = parser.parse_known_args(["-t", "20020103"])[0]
else:
    args = parser.parse_args()

target_dates = args.target_dates
skip_process = args.skip_process

# --- Config ---
TMP = os.path.join(DATA_DIR, "_fuxi_tmp")     # temporary download/extract
OUT = os.path.join(DATA_DIR, "fuxi")          # final zarr stores


HF_DATASET = "FudanFuXi/FuXi-S2S"

# Variable mapping
CHANNEL_TO_VAR = {"tp": "pr", "t2m": "tas", "msl": "mslp"}

# Only keep these leads after aggregation
KEEP_LEADS = [19, 26]

# Dataset dimensions
MEMBERS = 51
LAT = 121
LON = 240
LEADS = [19, 26]


# --- Helpers ---
def load_hf_token(path=None):
    """Read Hugging Face token from file."""
    if path is None:
        path = os.path.join(os.path.expanduser("~"), "hf_token")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hugging Face token file not found: {path}")
    with open(path, "r") as f:
        return f.read().strip()


def discover_dates():
    """Query HuggingFace and return available YYYYMMDD strings."""
    printf("Discovering dates from HuggingFace..."); tic()

    files = api.list_repo_files(HF_DATASET, repo_type="dataset")

    dates = sorted([
        f[:8] for f in files if f.endswith(".7z") and f[:8].isdigit()
    ])

    printf(f"Found {len(dates)} dates")
    toc()
    return dates


def filter_dates(date, available_dates):
    """Filter dates based on YYYY / YYYYMM / YYYYMMDD."""
    if date is None:
        return available_dates

    if len(date) == 4:
        return [d for d in available_dates if d.startswith(date)]
    elif len(date) == 6:
        return [d for d in available_dates if d.startswith(date)]
    elif len(date) == 8:
        return [d for d in available_dates if d == date]
    else:
        raise ValueError("Invalid target_dates format")


def download(date):
    """Download archive if not present."""
    path = os.path.join(TMP, f"{date}.7z")
    if not os.path.exists(path):
        printf(f"Downloading {date}..."); tic()
        hf_hub_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            filename=f"{date}.7z",
            local_dir=TMP,
            token=hf_token,
            etag_timeout=120,
            resume_download=True
        )
        toc()

    return path


def accumulate_leads(da, window=7):
    """Applies 7-day rolling aggregation with left-alignment.
    For precipitation (pr) we multiply by 24 then sum; for tas and mslp we use mean.
    """
    if da.name == "pr":
        agg = (da * 24).rolling(lead=window).sum()
    else:
        agg = da.rolling(lead=window).mean()

    agg["lead"] = agg["lead"] - (window - 1)
    return agg.where(agg["lead"] >= 1, drop=True)

    
def process_member(mdir, var_map=CHANNEL_TO_VAR, keep_leads=KEEP_LEADS):
    """
    Load a single member's .nc files, preprocess, and return xarray.Dataset
    """
    member_id = int(os.path.basename(mdir).split("_")[-1])
    files = sorted(glob(os.path.join(mdir, "*.nc")))

    ds_m = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 1})
    ds_m = ds_m.expand_dims(member=[member_id])

    # Select channels and rename
    ds_m = ds_m.sel(channel=list(var_map.keys()))
    da = ds_m["__xarray_dataarray_variable__"]
    ds_m = da.to_dataset(dim="channel").rename(var_map)

    ds_m = ds_m.rename({
        "lead_time": "lead",
        "lat": "latitude",
        "lon": "longitude"
    }).sortby("latitude")

    # Rolling aggregation
    for v in var_map.values():
        ds_m[v] = accumulate_leads(ds_m[v])

    # Select desired leads
    ds_m = ds_m.sel(lead=keep_leads)

    # Rechunk for Zarr
    ds_m = ds_m.chunk({
        "time": 1,
        "member": -1,
        "lead": -1,
        "latitude": -1,
        "longitude": -1
    })

    return ds_m

def process_archive(date, archive_path=None):
    """Extract, load, and process one archive into xarray dataset.
    
    Restart-safe:
    - Skips already extracted files
    - Extracts only missing files
    - Uses .complete marker to skip re-extraction
    """
    printf(f"Processing {date}..."); tic()

    extracted_path = os.path.join(TMP, date)
    complete_flag = os.path.join(extracted_path, ".complete")

    # leads 18–33 inclusive 
    leads = {f"{l:02d}.nc" for l in range(18, 34)}

    # --- Step 0: ensure directory exists ---
    os.makedirs(extracted_path, exist_ok=True)

    # --- Step 1: skip extraction if already complete ---
    if os.path.exists(complete_flag):
        printf(f"Skipping extraction (already extracted): {target_date}")
    else:
        printf(f"Checking existing extracted files..."); tic()

        # --- Step 2: collect existing relative paths ---
        existing_files = set()
        for root, _, files in os.walk(extracted_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, extracted_path)
                existing_files.add(rel_path)

        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            all_files = z.getnames()

            # --- Step 3: filter desired files ---
            files_to_extract = [
                f for f in all_files
                if f.split("/")[-1] in leads
            ]

            # --- Step 4: determine missing files ---
            missing_files = [
                f for f in files_to_extract
                if f not in existing_files
            ]

            if not missing_files:
                printf(f"All files already extracted for {date}.")
            else:
                printf(
                    f"Extracting {len(missing_files)} missing files "
                    f"(out of {len(files_to_extract)})..."
                ); tic()

                z.extract(path=extracted_path, targets=missing_files)

                toc()

        # --- Step 5: mark extraction complete ---
        # Only mark complete if nothing is missing anymore
        if len(missing_files) == 0:
            open(complete_flag, "w").close()
        else:
            # Re-check completeness after extraction
            extracted_now = set()
            for root, _, files in os.walk(extracted_path):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, extracted_path)
                    extracted_now.add(rel_path)

            still_missing = [
                f for f in files_to_extract
                if f not in extracted_now
            ]

            if not still_missing:
                open(complete_flag, "w").close()

        toc()

    # --- Step 6: locate base directory ---
    base_dir = os.path.join(TMP, date[:4], date)
    if not os.path.isdir(base_dir):
        base_dir = extracted_path

    # --- Step 7: load members ---
    printf("Loading members..."); tic()

    member_dirs = sorted(
        glob(os.path.join(extracted_path, "**", "member", "*"), recursive=True)
    )

    if not member_dirs:
        raise RuntimeError(f"No member directories found in {extracted_path}")

    datasets = []
    for mdir in member_dirs:
        member_id = int(os.path.basename(mdir))
        printf(f"-----> member {member_id}...")

        files = sorted(glob(os.path.join(mdir, "*.nc")))
        if not files:
            print(f"WARNING: no .nc files in {mdir}")
            continue

        ds_m = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 1})
        ds_m = ds_m.expand_dims(member=[member_id])
        datasets.append(ds_m)

    if not datasets:
        raise RuntimeError(f"No datasets loaded for {date}")

    ds = xr.concat(datasets, dim="member").sortby("member")
    toc()

    # --- Step 8: processing ---
    printf("Accumulating over 7 days..."); tic()
    ds = ds.sel(channel=list(CHANNEL_TO_VAR.keys()))
    ds = ds["__xarray_dataarray_variable__"].to_dataset(dim="channel")
    ds = ds.rename(CHANNEL_TO_VAR)
    ds = ds.rename({
        "lead_time": "lead",
        "lat": "latitude",
        "lon": "longitude"
    }).sortby("latitude")

    for v in CHANNEL_TO_VAR.values():
        ds[v] = accumulate_leads(ds[v])

    ds = ds.sel(lead=KEEP_LEADS)
    ds = ds.chunk({"time": 1, "member": -1})

    toc()

    return ds, extracted_path
# def process_archive(date, archive_path=None):
#     """Extract, load, and process one archive into xarray dataset."""
#     printf(f"Processing {date}..."); tic()
#     # --- Reuse extracted data if available ---
#     printf(f"Extracting {date}..."); tic()
#     # leads 18–33 inclusive 
#     leads = {f"{l:02d}.nc" for l in range(18, 34)}
#     extracted_path = os.path.join(TMP, date)
#     with py7zr.SevenZipFile(archive_path, mode='r') as z:
#         all_files = z.getnames()
#         # keep only files with desired leads
#         files_to_extract = [f for f in all_files if f.split("/")[-1] in leads]
#         # print(files_to_extract)
#         tic()
#         print(f"Extracting {len(files_to_extract)} files...")
#         z.extract(path=extracted_path, targets=files_to_extract) 
#     toc()

#     base_dir = os.path.join(TMP, date[:4], date)
#     if not os.path.isdir(base_dir):
#         base_dir = os.path.join(TMP, date)

#     printf("Loading members..."); tic()
#     # search recursively instead of assuming structure
#     member_dirs = sorted(
#         glob(os.path.join(extracted_path, "**", "member", "*"), recursive=True)
#     )
#     if not member_dirs:
#         raise RuntimeError(f"No member directories found in {extracted_path}")
#     datasets = []
#     for mdir in member_dirs:
#         member_id = int(os.path.basename(mdir))
#         printf(f"-----> member {member_id}...")
    
#         files = sorted(glob(os.path.join(mdir, "*.nc")))
#         if not files:
#             print(f"WARNING: no .nc files in {mdir}")
#             continue
#         ds_m = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 1})
#         ds_m = ds_m.expand_dims(member=[member_id])
#         datasets.append(ds_m)
#     if not datasets:
#         raise RuntimeError(f"No datasets loaded for {date}")
#     ds = xr.concat(datasets, dim="member").sortby("member")
#     toc()

#     printf("Accumulating over 7 days..."); tic()
#     ds = ds.sel(channel=list(CHANNEL_TO_VAR.keys()))
#     ds = ds["__xarray_dataarray_variable__"].to_dataset(dim="channel")
#     ds = ds.rename(CHANNEL_TO_VAR)
#     ds = ds.rename({
#         "lead_time": "lead",
#         "lat": "latitude",
#         "lon": "longitude"
#     }).sortby("latitude")
#     for v in CHANNEL_TO_VAR.values():
#         ds[v] = accumulate_leads(ds[v])
#     ds = ds.sel(lead=KEEP_LEADS)
#     ds = ds.chunk({"time": 1, "member": -1})
#     toc()
#     return ds, base_dir


def cleanup(archive, extracted):
    """Remove archive and extracted files."""
    printf("Cleaning up..."); tic()
    if archive and os.path.exists(archive):
        # os.remove(archive)
        subprocess.call(f"rm -rf {archive}", shell=True)
    if extracted and os.path.exists(extracted):
        # shutil.rmtree(extracted)
        subprocess.call(f"rm -rf {extracted}", shell=True)
    toc()



# Hugging Face API client used to query the dataset repo (e.g., list available files/dates)
api = HfApi()
hf_token = load_hf_token()

tic()
printf(f"=== RUN FuXi PIPELINE target={target_dates} ===")

# Discover dates 
available = discover_dates()
selected = filter_dates(target_dates, available)
    

print(f"Processing {len(selected)} dates...")

for target_date in selected:
    tic()
    print(f"\n\n------>{target_date}")

    archive = os.path.join(TMP, f"{target_date}.7z")
    extracted = os.path.join(TMP, f"{target_date}")
    out_file = os.path.join(OUT, f"{target_date}.nc")

    success = False

    # --- Skip if already processed ---
    if os.path.exists(out_file):
        printf(f"Skipping (already processed): {target_date}")
        continue

    try:
        # --- Download ---
        if os.path.exists(archive):
            printf(f"Skipping download (already downloaded): {target_date}")
        else:
            archive = download(target_date)

        # --- Early exit if skip_process flag is set ---
        if skip_process:
            printf(f"Skip processing enabled → download only for {target_date}")
            success = True   # mark success so we don't retry
            continue

        # --- Process archive ---
        ds, extracted = process_archive(target_date, archive)

        
        # --- Save to NetCDF ---
        printf(f"Saving to {out_file}"); tic()
        ds.to_netcdf(out_file)
        set_file_permissions(out_file)
        printf(f"File saved: {out_file}")
        toc()
        
        # --- Cleanup memory ---
        ds.close()
        del ds
        gc.collect()
        
        if os.path.isfile(out_file):
            success = True

    except Exception as e:
        print(f"ERROR {target_date}: {e}")

    finally:
        # --- Cleanup only if full processing succeeded ---
        if success and not skip_process:
            time.sleep(5)
            cleanup(archive, extracted)
        elif skip_process:
            printf(f"Preserving archive (download-only mode): {target_date}")
        else:
            printf(f"Preserving files for retry: {target_date}")

    printf(f"=====> Completed {target_date}")
    toc()

# print(f"Processing {len(selected)} dates...")
# for target_date in selected:
#     tic()
#     print(f"\n\n------>{target_date}")
#     archive = os.path.join(TMP, f"{target_date}.7z")
#     extracted = os.path.join(TMP, f"{target_date}")
            
#     # --- Check if already processed ---
#     if os.path.exists(os.path.join(OUT, f"{target_date}.nc")):
#         printf(f"Skipping (already processed): {target_date}")
#         continue

#     try:
#         # --- Download ---
#         if os.path.exists(archive):
#             printf(f"Skipping download (already downloaded): {target_date}")
#         else:
#             archive = download(target_date)

#         # --- Process archive ---
#         ds, extracted = process_archive(target_date, archive)

#         # --- Save to NetCDF ---
#         out_file = os.path.join(OUT, f"{target_date}.nc")
#         tic()
#         printf(f"Saving to {out_file}")
#         save_to_netcdf(ds, out_file)
#         toc()

#         # --- Explicitly close datasets ---
#         ds.close()
#         del ds
#         gc.collect()           

#     except Exception as e:
#         print(f"ERROR {target_date}: {e}")

#     finally:
#         # --- Cleanup ---
#         time.sleep(10)
#         cleanup(archive, extracted)
#     printf(f"=====> Completed {target_date}")
#     toc()  # Total cmd_run time

