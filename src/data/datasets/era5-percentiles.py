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
# Generate and save climatological percentiles from ERA5 data
#
# Args:
#     measurement (str): The measurement variable to process (e.g., "tas", "pr", "mslp");
#       defaults to "tas" if not provided.
#     ps (comma-separated list): The percentile levels to generate; defaults to 10,90,5,95.
#
# Example usage: python src/data/datasets/era5-percentiles.py tas 90
# Note: top reported 31.4% memory usage for tas when run on a node with 128GB of RAM.
# Note: ran to completion for tas with 120GB and 8 cores and for mslp with 132GB and 8 cores
import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
from utils.data_io import load_data, save_data
import sys
import xarray as xr
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# %%
# Select measurement variable
if len(sys.argv) > 1:
    measurement = sys.argv[1]
else:
    # Default measurement if not provided
    measurement = "tas"

if len(sys.argv) > 2:
    # Parse comma-separated list
    ps = [int(x) for x in sys.argv[2].split(",")]
else:
    # Default percentile levels if not provided
    ps = [10,90,5,95]
if isnotebook():
    # Override for notebook testing
    measurement = "pr"
    ps = [5]
print(f"Generating ERA5 percentiles for {measurement} at percentile levels {ps}")

def complete_20yr_percentiles(da,initial_rolling_window=7,date_window=[-4,-2,0,2,4],rolling_operation='mean', p=10):
    ''' Overarching function which calculates 20-year percentiles of rolling function for every year. 
    Adapted from AI_WQ_package.compute_20yr_percentile_climatology.complete_20yr_percentiles

    Args: 
        da - DataArray to be processed.
        Initial_rolling_window (int) - The initial rolling-average taken, essentially set to seven for weekly-means.
        Date window (int) - the days to sample. Similar to taking five hindcast sets.
        p (int) - The percentile level to generate (e.g., 5, 10, 90, 95).

    Return: A complete record of 20-year percentiles of seven-day rolling means'''

    # find first timestep that you can commit 20-year climatology
    first_ts = pd.Timestamp(da['time'][0].values)
    first_20_clim_ts = first_ts + relativedelta(years=20) + relativedelta(days=float(date_window[-1])) # needs to be 20 years + 4 days from first timestep
    # find end 20 clim ts
    end_ts = pd.Timestamp(da['time'][-1].values)
    end_20_clim_ts = end_ts + relativedelta(years=1) - relativedelta(days=float(date_window[-1])) # add a year to the last date - two days.

    # create array with years to compute
    years_to_compute = range(first_20_clim_ts.year,end_20_clim_ts.year+1)

    # first compute weekly rolling mean or rolling sum for precip.
    if rolling_operation == 'mean':
        weekly_rolling = da.rolling(time=7,center=False).mean().shift(time=-6)
    elif rolling_operation == 'sum':
        weekly_rolling = da.rolling(time=7,center=False).sum().shift(time=-6)
    elif rolling_operation == 'none': # also given the option for none rolling, if it has already been performed. 
        weekly_rolling = da

    # set-up empty array
    doy_rolling_avgs = []

    for year in years_to_compute:
        print (year)
        # set the start and end dates dependent on whether in final year or not
        if year == first_20_clim_ts.year:
            start_date = first_20_clim_ts
        else:
            start_date = pd.Timestamp(f'{year}-01-01')
        if year == end_20_clim_ts.year:
            end_date = end_20_clim_ts
        else:
            end_date = pd.Timestamp(f'{year}-12-31')

        # computes 20-year average of 7-day rolling mean
        doy_avg = compute_20yr_avg(weekly_rolling, year,start_date,end_date,date_window=date_window,p=p)
        # append the empty array
        doy_rolling_avgs.append(doy_avg)

    # Combine results into a single DataArray
    return xr.concat(doy_rolling_avgs, dim='time')

# Function to compute the 20-year average for a specific year
def compute_20yr_avg(weekly_means, current_year,start_date,end_date,date_window=[-4,-2,0,2,4], p=10):
    ''' Function that computes 20-year percentiles of DataArray (should have already been given altered to a weekly-mean). 
    Will treat observational climatology in a similar manner to hindcast climatology. 
    After taking 7-day rolling window, take a five day rolling window to average across multiple weeks. 
    Seven-day rolling mean, and then five-day rolling-mean, is taken before computing average across the previous 20 years. 
    Adapted from AI_WQ_package.compute_20yr_percentile_climatology.complete_20yr_avg.

    Args:
        weekly_means - DataArray of weekly means to be processed.
        current_year - the year for which to compute the 20-year average.
        start_date - the start date for which to compute the 20-year average.
        end_date - the end date for which to compute the 20-year average.
        date_window - the days to sample. Similar to taking five hindcast sets.
        p - The percentile level to generate (e.g., 5, 10, 90, 95).

    Returns: A full year of 20-year mean of rolling-mean values.
    '''

    # Convert percentile level to quantile level
    q = p / 100.0

    percentiles = []

    for date in pd.date_range(start=start_date, end=end_date, freq='D'):
        print (date)
        clim_data = []
        # go through all days in date window.
        for year_change in np.arange(-20,0): # go through past 20 years
            for day_change in date_window: # +/- 2 and 4 days - determined by date_rolling_window. 
                new_date = date + relativedelta(years=year_change) + relativedelta(days=float(day_change))
                clim_data.append(weekly_means.sel(time=new_date))
        full_clim_set = xr.concat(clim_data,dim='time')
        # use full clim set (100 days), work out percentiles. need to rechunk due to dask handling
        percentile_clim = full_clim_set.chunk(dict(time=-1)).quantile(q=[q],dim='time')
        # add a time metric to percentile clim
        percentile_clim = percentile_clim.assign_coords(time=date)
        percentiles.append(percentile_clim)
    # after going through a full year, concate into a final xarray for that year.
    full_year_percentiles = xr.concat(percentiles,dim='time')

    return full_year_percentiles

# %%
# Load ERA5 data
print(f"Loading ERA5 data for measurement: {measurement}")
era5 = load_data(f"era5-{measurement}").load()
print(f"Loading AIWQ data for measurement: {measurement}")
aiwq = load_data(f"aiwq-{measurement}").load()
print(f"Using AIWQ data when available and ERA5 otherwise.")
era5 = aiwq.combine_first(era5)
del aiwq

# %%
for p in ps:
    # Compute climatological percentiles
    # Use 'none' for rolling_operation as data is already aggregated
    print(f"Computing climatological {p}th percentile for {measurement}...")
    era5_percentiles = complete_20yr_percentiles(era5, rolling_operation='none', p=p)

    # Make measurement variable float32 to save space
    # Chunk data by quantile and latitude; leave other dimensions unchunked
    print(f"Converting {measurement} data to float32 and chunking...")
    era5_percentiles = era5_percentiles.astype('float32').chunk({"quantile": 1, "latitude": 1, "longitude": -1, "time": -1})

    # Save the percentiles dataset
    print(f"Saving {p}th percentile dataset for {measurement}...")
    save_data(era5_percentiles, f"era5-percentile{p}-{measurement}")
    del era5_percentiles
