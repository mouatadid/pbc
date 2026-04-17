from AI_WQ_package import retrieve_evaluation_data
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import os
from typing import Optional, Tuple

from data.helpers.config import load_config
from utils.data_io import get_zarr_store_path, save_to_zarr, check_zarr_exists, delete_contents

logger = logging.getLogger(__name__)

CONFIG = load_config()
BASE_DIR = Path(CONFIG['data_dir'])
SOURCE_CONFIG = CONFIG['sources']['aiwq']
PASSWORD = CONFIG['sources']['aiwq']['password']

STD_VARIABLES = CONFIG['variables']
SRC_VARIABLES = SOURCE_CONFIG['variables']
SRC_VARIABLES_SHORTCODE = SOURCE_CONFIG['variables_shortcode']

LAST_DATE_AVAILABLE = "2026-12-31"


def initial_download(start_date: str, end_date: str, force: bool = False) -> None:
    """Performs an initial download for all configured AIWQ quintile variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
    """

    logger.info(f"===== Starting AIWQ Quintiles Initial Download ({start_date} to {end_date}) =====")

    for weather_variable in SRC_VARIABLES.keys():

        try:
            store_path = get_zarr_store_path(BASE_DIR, f"aiwq-quintiles-{STD_VARIABLES[weather_variable]}")
            
            # Check if dataset already exists
            if check_zarr_exists(store_path) and not force:
                logger.error(f"Dataset {store_path} already exists. Use --force to overwrite.")
                continue
                
            download_aiwq_quintiles(
                variable_name=weather_variable,
                start_date_str=start_date,
                end_date_str=end_date,
                password=PASSWORD,
                output_zarr=store_path,
                force=force,
                initial_download=True
            )
        except Exception as e:
            logger.error(f"Failed initial download for {weather_variable}. Error: {e}", exc_info=True)

    if os.path.exists('data/_aiwq_quintiles'):
        logger.info("Deleting temporary directory for AIWQ data.")
        os.rmdir('data/_aiwq_quintiles')

    logger.info("===== Finished AIWQ Quintiles Initial Download =====")


def update(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, force: bool = False) -> None:

    """
    Checks each configured AIWQ quintiles variable and updates it if new data is available.
    
    Args:
        start_date_str: Optional start date (YYYY-MM-DD). If provided, downloads from this date.
        end_date_str: Optional end date (YYYY-MM-DD). If provided, downloads until this date.
        force: If True, overwrite data for dates that already exist. If False, skip existing dates.
    """

    logger.info("===== Starting AIWQ Quintiles Update Process =====")
    
    for weather_variable in SRC_VARIABLES.keys():

        store_path = get_zarr_store_path(BASE_DIR, f"aiwq-quintiles-{STD_VARIABLES[weather_variable]}")
        
        # Determine what dates to download based on arguments
        dates_to_download = determine_update_dates(
            store_path, 
            time_coord="time", 
            start_date_str=start_date_str, 
            end_date_str=end_date_str
        )
        
        if not dates_to_download:
            continue
            
        start_date, end_date = dates_to_download

        # Check if the dates to be downloaded are already present in the Zarr store
        if check_zarr_exists(store_path):
            existing_ds = None  # Initialize
            try:
                existing_ds = xr.open_zarr(store_path)
                existing_times = pd.DatetimeIndex(existing_ds.time.values).normalize()
                existing_times = existing_times.tz_localize('UTC')

                update_period_range = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
                overlapping_dates = existing_times.intersection(update_period_range).tz_convert(None)

                if not overlapping_dates.empty:
                    if force:
                        logger.info(f"Removing dates from existing dataset: {overlapping_dates.strftime('%Y-%m-%d').tolist()}")
                        modified_ds = existing_ds.drop_sel(time=overlapping_dates)
                        logger.info("Loading selected data into memory before overwriting Zarr store...")
                        modified_ds = modified_ds.load() # This is memory intensive
                        logger.info("Data loaded. Now saving to Zarr.")
                        save_to_zarr(modified_ds, store_path, mode='w')
                        del modified_ds
                    else:
                        logger.error(
                            f"Update for {weather_variable} aborted. "
                            f"Dates {overlapping_dates.strftime('%Y-%m-%d').tolist()} already exist in {store_path}."
                            f"Use --force to overwrite or ensure the update range is distinct."
                        )
                        continue # Skip to the next weather variable
            except Exception as e:
                logger.error(
                    f"Error during overlap check for {weather_variable} in {store_path}: {e}. "
                    f"Skipping update for this variable to be safe.", exc_info=True
                )
                continue # Skip to the next weather variable
            finally:
                if existing_ds is not None:
                    existing_ds.close()
                    logger.debug(f"Closed existing_ds for {store_path}")

        logger.info(f"Updating {weather_variable} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        try:
            download_aiwq_quintiles(
                variable_name=weather_variable,
                start_date_str=start_date.strftime('%Y-%m-%d'),
                end_date_str=end_date.strftime('%Y-%m-%d'),
                password=PASSWORD,
                output_zarr=store_path,
                initial_download=False, # This is an update
                force=force
            )
        except Exception as e:
            logger.error(f"Failed to update {weather_variable}. Error: {e}", exc_info=True)

    if os.path.exists('data/_aiwq_quintiles'):
        logger.info("Deleting temporary directory for AIWQ data.")
        os.rmdir('data/_aiwq_quintiles')
    
    logger.info("===== Finished AIWQ Quintiles Update Process =====")



def download_aiwq_quintiles(variable_name, start_date_str, end_date_str, password=None, output_zarr=None, force=False, initial_download=False):
    """ Download and concatenate annual AIWQ quintiles files for a given variable. """

    cds_var_name = SRC_VARIABLES.get(variable_name)
    var_short_name = SRC_VARIABLES_SHORTCODE.get(variable_name)

    if not cds_var_name:
        logger.error(f"Variable '{variable_name}' not configured in config.yaml sources.aiwq.variables")
        return

    store_path = get_zarr_store_path(BASE_DIR, f"aiwq-quintiles-{STD_VARIABLES[variable_name]}")
    logger.info(f"--- Processing AIWQ quintiles: {variable_name} ({cds_var_name}) ---")
    logger.info(f"Target Zarr store: {store_path}")

    if not os.path.exists('data/_aiwq_quintiles'):
        logger.info("Creating temporary directory for AIWQ quintiles data.")
        os.makedirs('data/_aiwq_quintiles')
    datasets = []

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    num_days = (end_date - start_date).days + 1

    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        if current_date.weekday() == 0:  # Only process Mondays
            logging.info(f"Downloading {variable_name} for date {current_date}...")
            ds = retrieve_evaluation_data.retrieve_20yr_quintile_clim(current_date.strftime('%Y%m%d'), var_short_name, password=PASSWORD, local_destination="data/_aiwq_quintiles")
            datasets.append(ds)

    logging.info("Concatenating datasets...")
    combined = xr.concat(datasets, dim='time')

    combined = combined.chunk({'time': 30, 'latitude': 121, 'longitude': 240})
    save_to_zarr(combined, store_path, mode='w')

    logger.info("Deleting temporary directory for AIWQ quintiles data.")
    delete_contents('data/_aiwq_quintiles')

    logger.info(f"--- Finished processing AIWQ quintiles for variable: {variable_name} ---")


def determine_update_dates(
    store_path: str, 
    time_coord: str = "time", 
    start_date_str: Optional[str] = None, 
    end_date_str: Optional[str] = None
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:

    """
    Determines what date range to download based on existing data and provided arguments.
    
    Args:
        store_path: Path to the Zarr store
        time_coord: Name of the time coordinate
        data_latency_days: Number of days to lag behind current date
        start_date_str: Optional user-specified start date
        end_date_str: Optional user-specified end date
        
    Returns:
        Tuple of (start_date, end_date) timestamps to download, or None if no download needed
    """

    # Calculate today minus latency (i.e., today_lagged, the latest date we can download)
    today_lagged = pd.Timestamp(LAST_DATE_AVAILABLE, tz='UTC').normalize()
    
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
    zarr_exists = check_zarr_exists(store_path)

    if zarr_exists:
        ds_check = None
        try:
            ds_check = xr.open_zarr(store_path)
            last_date = pd.Timestamp(ds_check[time_coord].max().values).normalize()
            logger.debug(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Error opening Zarr store: {e}")
            return None # Propagate error
        finally:
            if ds_check is not None:
                ds_check.close()
                logger.debug(f"Closed ds_check for {store_path} in determine_update_dates")
    else:
        logger.warning(f"Zarr store {store_path} does not exist, cannot determine last date")
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
