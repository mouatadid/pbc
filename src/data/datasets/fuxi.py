"""
Processing Fuxi NetCDFs to probabilistic Zarr stores

Example:
    python src/data/datasets/fuxi.py

"""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import os
from models.utils.general_util import printf
from utils.timing import tic, toc
from utils.data_io import save_data, load_data
from datetime import datetime


# set parameters 
INPUT_PATTERN = os.path.join("data", "fuxi", "*.nc")   
ACCUMULATE_LEADS_PERIOD = 7         
FIRST_LEAD = 1
LAST_LEAD = 26

# map channel names in files -> conventional names
CHANNEL_TO_VAR = {"tp": "pr", "t2m": "tas", "msl": "mslp"}

# Chunk zarr data by date 
chunk = ({"lead": -1, "latitude": -1, "longitude": -1, "time": 1})


def open_fuxi(pattern, chunks=None):
    """Open all input NetCDFs as a single xarray Dataset (dask-backed).
    Returns the dataset and the single dataarray inside it.
    """
    print(f"Opening files with pattern: {pattern}")
    ds = xr.open_mfdataset(pattern, combine="by_coords", chunks=chunks)#, parallel=True)

    # Find the single data variable
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise RuntimeError(f"Expected a single data variable in input files, found: {data_vars}")
    data_var_name = data_vars[0]
    da = ds[data_var_name]

    # rename dims to interpretable names if present
    rename_map = {"lead_time": "lead",
                  "lat": "latitude",
                  "lon": "longitude"}
    da = da.rename(rename_map)
    return da

def channel_to_dataset(da):
    """Split channel dim into separate dataset variables and drop the 'channel' coord
    Returns xarray.Dataset with variables named 'pr','tas','mslp' (depending on channels found).
    """
    if "channel" not in da.dims:
        raise RuntimeError("Input DataArray has no 'channel' dimension")

    ds_vars = {}
    channels = [str(c) for c in da["channel"].values]
    print("Found channels:", channels)
    for ch in channels:
        if ch not in CHANNEL_TO_VAR:
            print(f"Warning: unexpected channel '{ch}' found; skipping")
            continue
        varname = CHANNEL_TO_VAR[ch]
        # select the channel and drop the channel coordinate to avoid MergeError when building Dataset
        arr = da.sel(channel=ch).drop_vars("channel")
        # ensure the array has dims: (member, time, lead, latitude, longitude)
        ds_vars[varname] = arr

    ds = xr.Dataset(ds_vars)
    return ds

def accumulate_leads(var_da, window=ACCUMULATE_LEADS_PERIOD):
    """Applies daily averaging and 7-day rolling aggregation with left-alignment.
    For precipitation (pr) we multiply by 24 then sum; for others we take mean.
    """
    
    var_name = var_da.name
    printf(f"Applying 7-day rolling aggregation for {var_name} ({'sum' if var_name == 'pr' else 'mean'})")
    if var_name == "pr":
        # convert hourly-average to daily accumulated (multiply by 24)
        arr = var_da * 24
        # Accumulate precipitation by summing over hours (x 24)
        agg = arr.rolling(lead=window).sum()
    else:
        # Accumulate other variables by averaging over the week
        arr = var_da
        agg = arr.rolling(lead=window).mean()
        
    # Make it left-aligned: lead axis labels indicating the start of each seven-day period.
    agg["lead"] = agg["lead"] - (window-1) #6
    printf(f"Applied left-alignment to time axis for {var_name}")
    return agg

def restrict_leads(ds, first=FIRST_LEAD, last=LAST_LEAD):
    if "lead" not in ds.dims:
        print("No 'lead' dimension present; skipping lead restriction")
        return ds
    lead_vals = ds["lead"].values
    # handle timedelta or integer lead values
    if np.issubdtype(lead_vals.dtype, np.timedelta64):
        lead_days = (lead_vals.astype('timedelta64[D]').astype(int))
        mask = (lead_days >= first) & (lead_days <= last)
        sel = ds["lead"].values[mask]
        return ds.sel(lead=sel)
    else:
        try:
            lead_ints = lead_vals.astype(int)
            mask = (lead_ints >= first) & (lead_ints <= last)
            sel = ds["lead"].values[mask]
            return ds.sel(lead=sel)
        except Exception:
            return ds.sel(lead=slice(first, last))


# Load data
da = open_fuxi(pattern=INPUT_PATTERN, chunks={"member": 51, "time": 1, "lead": -1, "channel": -1})

# split channels into separate variables
ds_all = channel_to_dataset(da)


# For each variable compute accumulated/averaged fields over lead window
for weather_variable in ['pr', 'tas', 'mslp']:
    print(f"Processing variable: {weather_variable}")
    tic()
    arr = ds_all[weather_variable]
    arr.name = weather_variable
    agg = accumulate_leads(arr, window=ACCUMULATE_LEADS_PERIOD)
    
    # restrict to subseasonal leads
    ds = restrict_leads(agg, FIRST_LEAD, LAST_LEAD)
    
    # Load quintile boundaries 
    printf(f"Loading ERA5 quintiles for measurement: {weather_variable}")
    quintiles = load_data(f"era5-quintiles-{weather_variable}")
    # Load data explicitly into memory
    quintiles.load()
    quantiles = quintiles["quantile"].values
    
    # Select dataset names for each quantile
    dataset_names = {quantile: f"fuxi-f{ii+1}_{weather_variable}" for (ii, quantile) in enumerate(quantiles)}
    
    print("Computing probabilistic forecast...")
    tic() 
    ds_notnull = ds.notnull()
    ds_has_nans = not ds_notnull.all()
    if ds_has_nans:
        print("Warning: forecast data contains NaNs") 
    ensemble_pred = {}    
    # Match quintile thresholds to forecast time (nearest day)
    qnt = quintiles[weather_variable].sel(time=ds.time)#, method="nearest")
    
    for quantile in quantiles:
        print(f"Processing quantile {quantile}...")
    
        # Select corresponding threshold field
        thr = qnt.sel(quantile=quantile, drop=True)
    
        # Compute probabilistic exceedance (CDF): fraction of members below threshold
        cdf_pred = (ds < thr).astype("float32")
    
        if ds_has_nans:
            cdf_pred = cdf_pred.where(ds_notnull, np.nan)
    
        # Compute probability (mean over members)
        prob = cdf_pred.mean(dim="member", skipna=False)
    
        # Store in dict using quantile as key
        ensemble_pred[quantile] = prob
    
    # --- Save results ---
    for ii, quantile in enumerate(quantiles):
        out_name = dataset_names[quantile]
        print(f"Saving to {out_name}..."); tic()
    
        da = ensemble_pred[quantile].sortby("latitude")
        da = da.rename(f"f{ii+1}_{weather_variable}")
       
        save_data(da.to_dataset().chunk(chunk), dataset_names[quantile])
    toc()
    toc()

