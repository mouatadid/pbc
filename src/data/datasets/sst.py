import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import tempfile
import os
from typing import Optional, Tuple
import urllib.request

from data.helpers.config import load_config
from utils.data_io import get_zarr_store_path, save_to_zarr, check_zarr_exists
from data.helpers.data_processing import daily_average

logger = logging.getLogger(__name__)

CONFIG = load_config()
BASE_DIR = Path(CONFIG['data_dir'])
TARGET_RESOLUTION = CONFIG['target_grid']['resolution_deg']
SOURCE_CONFIG = CONFIG['sources']['sst']
BASE_URL = SOURCE_CONFIG['base_url']

STD_VARIABLES = CONFIG['variables']
SRC_VARIABLES = SOURCE_CONFIG['variables']
NATIVE_RESOLUTION = SOURCE_CONFIG['native_res']

LATENCY = CONFIG['latency']['sst']
SST_CHUNK_CONFIG = {'time': -1, 'latitude': 1, 'longitude': -1}

STD_VARIABLE = STD_VARIABLES[list(SRC_VARIABLES.keys())[0]]  # 'sst'
ZARR_PATH = get_zarr_store_path(BASE_DIR, STD_VARIABLE)


def initial_download(start_date_str: str, end_date_str: str, force: bool = False) -> None:
    """
    Performs an initial download for SST data.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
    """
    logger.info(f"===== Starting SST Initial Download ({start_date_str} to {end_date_str}) =====")

    try:
        # Check if dataset already exists
        if check_zarr_exists(ZARR_PATH) and not force:
            logger.error(f"Dataset {ZARR_PATH} already exists. Use --force to overwrite.")
            return
            
        download_sst_range(
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            initial_download=True
        )
    except Exception as e:
        logger.error(f"Failed initial download for SST. Error: {e}", exc_info=True)
    
    logger.info("===== Finished SST Initial Download =====")


def update(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, force: bool = False) -> None:
    """
    Checks for new SST data and updates the Zarr store if needed.
    
    Args:
        start_date_str: Optional start date (YYYY-MM-DD). If provided, downloads from this date.
        end_date_str: Optional end date (YYYY-MM-DD). If provided, downloads until this date.
        force: If True, overwrite data for dates that already exist. If False, skip existing dates.
    """
    logger.info("===== Starting SST Update Process =====")
    
    # Determine what dates to download based on arguments
    dates_to_download = determine_update_dates(
        data_latency_days=LATENCY,
        start_date_str=start_date_str, 
        end_date_str=end_date_str
    )
    
    if not dates_to_download:
        logger.info("No new data to download based on provided parameters")
        logger.info("===== Finished SST Update Process =====")
        return
        
    start_date, end_date = dates_to_download

    if data_for_update_already_exists(start_date, end_date, force):
        logger.info(f"Skipping update for {STD_VARIABLE} due to existing data or check error. See previous logs.")
        logger.info("===== Finished SST Update Process =====")
        return

    logger.info(f"Updating SST from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        download_sst_range(
            start_date_str=start_date.strftime('%Y-%m-%d'),
            end_date_str=end_date.strftime('%Y-%m-%d'),
            initial_download=False,  # This is an update
            force=force
        )
    except Exception as e:
        logger.error(f"Failed to update SST. Error: {e}", exc_info=True)
    
    logger.info("===== Finished SST Update Process =====")


def determine_update_dates(
    data_latency_days: int = 5,
    start_date_str: Optional[str] = None, 
    end_date_str: Optional[str] = None
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determines what date range to download based on existing data and provided arguments.
    
    Args:
        data_latency_days: Number of days to lag behind current date
        start_date_str: Optional user-specified start date
        end_date_str: Optional user-specified end date
        
    Returns:
        Tuple of (start_date, end_date) timestamps to download, or None if no download needed
    """
    # Calculate today minus latency (the latest date we can download)
    today_lagged = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=data_latency_days)
    
    # Case 1: Both dates provided - use them directly
    if start_date_str and end_date_str:
        start_date = pd.Timestamp(start_date_str, tz='UTC')
        end_date = pd.Timestamp(end_date_str, tz='UTC')
        
        # Make sure end_date isn't in the future beyond data availability
        end_date = min(end_date, today_lagged)
        
        if end_date < start_date:
            logger.warning(f"End date {end_date.strftime('%Y-%m-%d')} is before start date {start_date.strftime('%Y-%m-%d')}")
            return None
            
        # Return dates and we'll append & sort later
        return start_date, end_date
    
    # Check if the Zarr store exists
    zarr_exists = check_zarr_exists(ZARR_PATH)
    
    # Case 2: Only start_date provided - download from start_date to today_lagged
    if start_date_str and not end_date_str:
        start_date = pd.Timestamp(start_date_str, tz='UTC')
        end_date = today_lagged
        
        if end_date < start_date:
            logger.warning(f"Data not yet available for period starting on {start_date.strftime('%Y-%m-%d')}")
            return None
            
        return start_date, end_date
    
    # For remaining cases, we need to know the last date in the dataset
    last_date = None
    if zarr_exists:
        try:
            ds = xr.open_zarr(ZARR_PATH)
            if "time" in ds.coords:
                last_date = pd.Timestamp(ds["time"].max().values).normalize()
                logger.debug(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
            else:
                logger.error(f"Time coordinate 'time' not found in dataset")
                return None
        except Exception as e:
            logger.error(f"Error opening Zarr store: {e}")
            return None
        finally:
            ds.close()
    else:
        logger.warning(f"Zarr store {ZARR_PATH} does not exist, cannot determine last date")
        return None
    
    # Case 3: Only end_date provided - download from last_date+1 to end_date
    if end_date_str and not start_date_str:
        end_date = min(pd.Timestamp(end_date_str, tz='UTC'), today_lagged)
        start_date = last_date + pd.Timedelta(days=1)
        
        if end_date < start_date:
            logger.warning(f"No new dates to download between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
            return None
            
        return start_date, end_date
    
    # Case 4: No dates provided - use standard update procedure (last_date+1 to today_lagged)
    start_date = last_date + pd.Timedelta(days=1)
    end_date = today_lagged
    
    if end_date < start_date:
        logger.info(f"Dataset already up-to-date, last date is {last_date.strftime('%Y-%m-%d')}")
        return None
        
    return start_date, end_date


def data_for_update_already_exists(
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp, 
    force: bool
) -> bool:
    """
    Checks for existing data in the store_path for the given date range.
    If overlapping data exists and force is False, logs an error and returns True (skip).
    If force is True, removes overlapping dates from the Zarr store.
    Returns True if the update for this variable should be skipped, False otherwise.
    """
    if not check_zarr_exists(ZARR_PATH):
        logger.debug(f"Zarr store {ZARR_PATH} does not exist. No existing data to check for overlap.")
        return False # No existing data, proceed with update

    existing_ds = None
    try:
        existing_ds = xr.open_zarr(ZARR_PATH)
        existing_times_utc = pd.DatetimeIndex(existing_ds.time.values).normalize().tz_localize('UTC')

        update_period_range_utc = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
        overlapping_dates_utc = existing_times_utc.intersection(update_period_range_utc)
        
        overlapping_dates_naive = overlapping_dates_utc.tz_convert(None)

        if not overlapping_dates_naive.empty:
            if force:
                logger.info(f"Force update: Removing overlapping dates from existing dataset {ZARR_PATH}: {overlapping_dates_naive.strftime('%Y-%m-%d').tolist()}")

                modified_ds = existing_ds.drop_sel({time_dim_name: overlapping_dates_naive})
                
                logger.info(f"Loading data (excluding overlaps) into memory before overwriting Zarr store {ZARR_PATH}...")
                modified_ds = modified_ds.load() # This can be memory intensive
                logger.info(f"Data loaded. Now saving to Zarr store {ZARR_PATH}.")
                save_to_zarr(modified_ds, ZARR_PATH, mode='w') 
                del modified_ds
                logger.info(f"Successfully removed overlapping dates from {ZARR_PATH}.")
                return False # Overlapping dates have been erased, proceed with update
            else:
                logger.error(
                    f"Update for {STD_VARIABLE} aborted. " 
                    f"Dates {overlapping_dates_naive.strftime('%Y-%m-%d').tolist()} already exist in {ZARR_PATH}. "
                    f"Use --force to overwrite or ensure the update range is distinct."
                )
                return True # Skip update
        else:
            logger.debug(f"No overlapping dates found in {ZARR_PATH} for the period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return False # No overlap, proceed with update
    except Exception as e:
        logger.error(
            f"Error during overlap check or data modification for {STD_VARIABLE} in {ZARR_PATH}: {e}. "
            f"Skipping update for this variable to be safe.", exc_info=True
        )
        return True # Skip update
    finally:
        if existing_ds is not None:
            existing_ds.close()
            logger.debug(f"Closed existing_ds for {ZARR_PATH}")


def download_sst_range(
    start_date_str: str,
    end_date_str: str,
    initial_download: bool = False,
    sort_by_time: bool = False,
    force: bool = False
) -> None:
    """
    Downloads SST data for a range of dates.
    
    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        initial_download: Whether this is an initial download (vs. an update)
        sort_by_time: Whether to sort the dataset by time before saving
    """
    start_date = pd.Timestamp(start_date_str, tz='UTC')
    end_date = pd.Timestamp(end_date_str, tz='UTC')
    end_date_buffered = end_date + pd.Timedelta(days=6)
    
    logger.info(f"--- Processing SST data from {start_date.strftime('%Y-%m-%d')} to {end_date_buffered.strftime('%Y-%m-%d')} ---")
    logger.info(f"Target Zarr store: {ZARR_PATH}")
    
    dates = pd.date_range(start=start_date, end=end_date_buffered, freq='D').normalize()
    
    # Check if we need to skip existing dates
    if not initial_download and not force and check_zarr_exists(ZARR_PATH):
        try:
            existing_ds = xr.open_zarr(ZARR_PATH)
            existing_dates = pd.DatetimeIndex(existing_ds.time.values).normalize().tz_localize('UTC')
            
            duplicate_dates = pd.DatetimeIndex([d for d in dates if d in existing_dates])
            
            if not duplicate_dates.empty:
                logger.warning(f"Found {len(duplicate_dates)} dates already in the dataset.")
                for date in duplicate_dates:
                    logger.warning(f"Skipping existing date: {date.strftime('%Y-%m-%d')}")
                
                # Filter out existing dates
                dates = pd.DatetimeIndex([d for d in dates if d not in existing_dates])
                
                if len(dates) == 0:
                    logger.info("All requested dates already exist. Nothing to download.")
                    return
                
                logger.info(f"Downloading only {len(dates)} new dates")
        except Exception as e:
            logger.error(f"Error checking existing dates: {e}")
            # Continue with original date range if there's an error
    
    combined_ds = None
    
    for date in dates:
        try:
            ds = download_sst_date(date)
            if ds is not None:
                if combined_ds is None:
                    combined_ds = ds
                else:
                    combined_ds = xr.concat([combined_ds, ds], dim='time')
            else:
                logger.warning(f"Skipping date {date.strftime('%Y-%m-%d')} due to download failure")
        except Exception as e:
            logger.error(f"Error processing date {date.strftime('%Y-%m-%d')}: {e}")
    
    if combined_ds is None:
        logger.error("Failed to download any data for the specified date range")
        return
    
    logger.info("Sorting dataset by time dimension")
    combined_ds = combined_ds.sortby('time')

    logger.info(f"Applying 7-day rolling aggregation for {STD_VARIABLE} ('mean')")
    combined_ds = combined_ds.rolling(time=7).mean()
    combined_ds['time'] = combined_ds['time'] - pd.Timedelta(days=6)
    combined_ds = combined_ds.isel(time=slice(6, None))

    combined_ds = combined_ds.chunk(SST_CHUNK_CONFIG)
    
    mode = 'w' if initial_download or not check_zarr_exists(ZARR_PATH) else 'a'
    append_dim = 'time' if mode == 'a' else None
    
    save_to_zarr(combined_ds, ZARR_PATH, mode=mode, append_dim=append_dim)
    
    logger.info("--- Finished processing SST data ---")


def download_sst_date(date: pd.Timestamp) -> Optional[xr.Dataset]:
    """
    Downloads SST data for a specific date.
    
    Args:
        date: The date to download data for
        
    Returns:
        xarray Dataset with the loaded data, or None if download failed
    """
    url = construct_sst_url(date, preliminary=False)
    logger.info(f"Downloading SST data for {date.strftime('%Y-%m-%d')} from {url}")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmpfile:
            target_path = tmpfile.name
        
        try:
            urllib.request.urlretrieve(url, target_path)
        except urllib.error.HTTPError as e:
            # If standard file not found, try the preliminary version
            if e.code == 404:
                logger.info("Standard file not found, trying preliminary version")
                url = construct_sst_url(date, preliminary=True)
                logger.info(f"Attempting download from: {url}")
                urllib.request.urlretrieve(url, target_path)
            else:
                # Re-raise if it's not a 404 error
                raise
        
        ds = xr.open_dataset(target_path)
        ds = process_sst_date(ds)
        
        return ds
    
    except Exception as e:
        logger.error(f"Failed to download SST data for {date.strftime('%Y-%m-%d')}: {e}")
        return None
    
    finally:
        # Clean up temporary file
        if 'target_path' in locals() and os.path.exists(target_path):
            os.remove(target_path)
            logger.debug(f"Removed temporary file: {target_path}")


def construct_sst_url(date: pd.Timestamp, preliminary: bool = False) -> str:
    """
    Constructs the URL to download SST data for a specific date.
    
    Args:
        date: The date to download data for
        preliminary: Whether to use the preliminary file naming pattern
    """
    # OISST files are typically named like: oisst-avhrr-v02r01.YYYYMMDD.nc
    # Recent files may have _preliminary suffix: oisst-avhrr-v02r01.YYYYMMDD_preliminary.nc
    date_str = date.strftime('%Y%m%d')
    year_str = date.strftime('%Y')
    month_str = date.strftime('%m')
    
    # Add _preliminary suffix if requested
    file_suffix = "_preliminary.nc" if preliminary else ".nc"
    
    url = f"{BASE_URL}/{year_str}{month_str}/oisst-avhrr-v02r01.{date_str}{file_suffix}"
    logger.debug(f"Constructed SST URL: {url}")
    return url


def process_sst_date(ds: xr.Dataset) -> xr.Dataset:
    """
    Process the raw SST dataset to match our standard format.
    
    Args:
        ds: Raw SST dataset
        
    Returns:
        Processed dataset with standardized variable names and coordinates
    """
    # if 'sst' in ds and STD_VARIABLE != 'sst':
    #     ds = ds.rename({'sst': STD_VARIABLE})
    #
    rename_dict = {}
    if 'latitude' not in ds.coords and 'lat' in ds.coords:
        rename_dict['lat'] = 'latitude'
    if 'longitude' not in ds.coords and 'lon' in ds.coords:
        rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    # Remove the zlev dimension if it exists (it's always 0 for sea surface)
    for zlev_name in ['zlev', 'depth', 'z', 'level']:
        if zlev_name in ds.dims:
            logger.debug(f"Removing {zlev_name} dimension from dataset")
            # If zlev dimension has length 1, we can just squeeze it out
            ds = ds.squeeze(dim=zlev_name, drop=True)
            break
    
    # Ensure data is daily - SST is typically already daily, but just in case
    if 'time' in ds.dims and ds.time.size > 1:
        unique_days = len(np.unique(ds.time.dt.floor('D')))
        if unique_days < ds.time.size:
            logger.info("Multiple time steps per day detected, performing daily averaging")
            ds = daily_average(ds)
    
    # Remove time component from timestamps, keeping only the date
    if 'time' in ds.coords:
        logger.debug("Converting timestamps to date-only format")
        dates_only = ds.time.dt.floor('D').values
        ds = ds.assign_coords(time=dates_only)
    
    # SST typically uses -180:180 longitude, convert to 0:360
    if ds['longitude'].min() < 0:
        logger.debug("Converting longitude from -180:180 to 0:360")
        ds['longitude'] = (ds['longitude'] + 360) % 360
        ds = ds.sortby('longitude')
    
    # Ensure the dataset only contains the SST variable
    if len(ds.data_vars) > 1:
        logger.debug(f"Selecting only the {STD_VARIABLE} variable from dataset")
        ds = ds[[STD_VARIABLE]]
    
    target_lat = np.arange(-90, 90 + TARGET_RESOLUTION, TARGET_RESOLUTION)
    target_lon = np.arange(0, 360, TARGET_RESOLUTION)

    logger.info(f"Interpolating {STD_VARIABLE} to {TARGET_RESOLUTION} degree grid...")
    ds = ds.interp(latitude=target_lat, longitude=target_lon, method="linear")
    
    ds.attrs['history'] = f"Downloaded from NOAA OISST v2, interpolated to {TARGET_RESOLUTION}deg."
    
    return ds

