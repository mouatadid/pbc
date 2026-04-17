from AI_WQ_package import retrieve_training_data
import xarray as xr
import pandas as pd
from pathlib import Path
import logging
import os

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

AIWQ_TEMP_DIR = Path('data/aiwq')
AIWQ_CHUNK_CONFIG = {'time': -1, 'latitude': 1, 'longitude': -1}


def initial_download(start_date: str, end_date: str, force: bool = False) -> None:
    """Performs an initial download for all configured AIWQ variables."""

    start_year=int(start_date[:4])
    end_year=int(end_date[:4])

    logger.info(f"===== Starting AIWQ Initial Download ({start_year} to {end_year}) =====")
    for weather_variable in SRC_VARIABLES.keys():
        try:
            store_path = get_zarr_store_path(BASE_DIR, f"aiwq-{STD_VARIABLES[weather_variable]}")
            
            if check_zarr_exists(store_path) and not force:
                logger.error(f"Dataset {store_path} already exists; skipping download. Use --force to overwrite.")
                continue
                
            download_aiwq_variable(
                variable_name=weather_variable,
                start_year=start_year,
                end_year=end_year,
                password=PASSWORD
            )
        except Exception as e:
            logger.error(f"Failed initial download for {weather_variable}. Error: {e}", exc_info=True)

    logger.info("===== Finished AIWQ Initial Download =====")


def update(start_date: str, end_date: str, force: bool = False) -> None:
    """
    Downloads AIWQ data for the specified years and merges it with any existing datasets.

    If overlapping timestamps are found and force is False, existing data is kept for those
    timestamps and only new timestamps are appended. When force is True, overlapping timestamps
    in the existing store are removed so the newly downloaded data replaces them.
    """

    if not start_date or not end_date:
        logger.error("Both start_date and end_date must be provided for the AIWQ update.")
        return

    try:
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
    except ValueError:
        logger.error(f"Invalid start_date ({start_date}) or end_date ({end_date}); expected YYYY-MM-DD format.")
        return

    if end_year < start_year:
        logger.error(f"End year {end_year} is before start year {start_year}. Aborting AIWQ update.")
        return

    logger.info(f"===== Starting AIWQ Update ({start_year} to {end_year}) =====")

    for weather_variable in SRC_VARIABLES.keys():
        new_data = None
        new_datasets = []
        existing_ds = None

        try:
            std_var_name, _, src_var_short_name = get_config_variables(weather_variable)
            if not std_var_name:
                continue

            store_path = get_zarr_store_path(BASE_DIR, f"aiwq-{std_var_name}")
            logger.info(f"--- Updating AIWQ variable: {weather_variable} (Zarr: {store_path}) ---")

            ensure_temp_data_dir_exists()

            # Download requested years
            for year in range(start_year, end_year + 1):
                output_path = AIWQ_TEMP_DIR / f"{src_var_short_name}_sevenday_{'WEEKLYSUM' if src_var_short_name == 'pr' else 'WEEKLYMEAN'}_{year}.nc"

                if output_path.exists() and not force:
                    logger.info(f"Using existing file {output_path} for year {year}.")
                    ds_year = None 
                else:
                    ds_year = download_yearly_data(year, weather_variable, src_var_short_name, PASSWORD)

                if ds_year is None:
                    # logger.warning(f"No data for {weather_variable} in {year}. Skipping this year.")
                    continue

                ds_year = ds_year.sortby(['time', 'latitude', 'longitude'])
                new_datasets.append(ds_year.chunk(AIWQ_CHUNK_CONFIG))

            if not new_datasets:
                logger.warning(f"No data downloaded for {weather_variable} for {start_year}-{end_year}. Skipping.")
                continue

            new_data = xr.concat(new_datasets, dim='time')
            new_data = new_data.sortby(['time', 'latitude', 'longitude'])

            # Handle overlaps with existing data
            if check_zarr_exists(store_path):
                try:
                    existing_ds = xr.open_zarr(store_path)
                except Exception as e:
                    logger.error(f"Failed to open existing dataset {store_path}: {e}", exc_info=True)
                    continue

                existing_times = pd.DatetimeIndex(existing_ds.time.values)
                new_times = pd.DatetimeIndex(new_data.time.values)
                overlapping_times = existing_times.intersection(new_times)

                if not overlapping_times.empty:
                    if force:
                        logger.info(
                            f"Force update enabled. Removing {len(overlapping_times)} overlapping timestamps "
                            f"from {store_path}."
                        )
                        try:
                            existing_without_overlap = existing_ds.drop_sel(time=overlapping_times)
                            logger.info("Loading filtered existing data into memory before overwriting store...")
                            existing_without_overlap = existing_without_overlap.load()
                            save_to_zarr(existing_without_overlap, store_path, mode='w')
                            del existing_without_overlap
                        except Exception as e:
                            logger.error(
                                f"Failed to remove overlapping timestamps for {weather_variable}: {e}", exc_info=True
                            )
                            continue
                    else:
                        logger.info(
                            f"Found {len(overlapping_times)} overlapping timestamps for {weather_variable}; "
                            "keeping existing data for these times."
                        )
                        new_data = new_data.drop_sel(time=overlapping_times)
                        if new_data.time.size == 0:
                            logger.warning(
                                f"No new timestamps to append for {weather_variable} after removing overlaps."
                            )
                            continue

            save_mode = 'a' if check_zarr_exists(store_path) else 'w'
            append_dim = 'time' if save_mode == 'a' else None

            save_to_zarr(
                new_data.chunk(AIWQ_CHUNK_CONFIG),
                store_path,
                mode=save_mode,
                append_dim=append_dim,
                consolidated=True
            )
            logger.info(f"Successfully updated {weather_variable} for years {start_year}-{end_year}.")

        except Exception as e:
            logger.error(f"Failed update for {weather_variable}. Error: {e}", exc_info=True)
        finally:
            if existing_ds is not None:
                existing_ds.close()
                logger.debug(f"Closed existing dataset for {store_path}")
            if new_data is not None:
                try:
                    new_data.close()
                except Exception:
                    pass
            for ds in new_datasets:
                try:
                    ds.close()
                except Exception:
                    pass
            # Do not delete downloaded data; preserve for potential future use
            # cleanup_temp_data_dir()

    logger.info("===== Finished AIWQ Update =====")


def download_aiwq_variable(variable_name: str, start_year: int, end_year: int, password: str) -> None:
    """ Downloads, processes, and stores a single AIWQ variable for a given period. """

    std_var_name, src_var_name, src_var_short_name = get_config_variables(variable_name)
    if not std_var_name: # Implies other names might also be None, error already logged by helper
        return

    store_path = get_zarr_store_path(BASE_DIR, f"aiwq-{std_var_name}")
    logger.info(f"--- Processing AIWQ variable: {variable_name} (Zarr: {store_path}) ---")

    ensure_temp_data_dir_exists()

    first_year_successfully_saved = False
    try:
        for year in range(start_year, end_year + 1):
            ds_year = download_yearly_data(year, variable_name, src_var_short_name, password)
            
            if ds_year is None:
                logger.warning(f"No data for {variable_name} for year {year}. Skipping.")
                continue

            logger.info(f"Processing {variable_name} for year {year}...")
            try:
                ds_year = ds_year.sortby(['time', 'latitude', 'longitude'])
                ds_year_chunked = ds_year.chunk(AIWQ_CHUNK_CONFIG)

                current_mode = 'w'
                current_append_dim = None

                if first_year_successfully_saved:
                    current_mode = 'a'
                    current_append_dim = 'time'
                
                save_to_zarr(
                    ds_year_chunked,
                    store_path, 
                    mode=current_mode,
                    append_dim=current_append_dim,
                    consolidated=True 
                )
                first_year_successfully_saved = True
                logger.info(f"Successfully saved/appended {variable_name} for year {year} to {store_path}.")

            except Exception as e:
                logger.error(f"Error processing or saving {variable_name} for year {year}: {e}", exc_info=True)
            finally:
                if ds_year is not None:
                    ds_year.close() # Close the yearly dataset to free memory
        
        if not first_year_successfully_saved:
            logger.error(f"No data was successfully saved for {variable_name} for the period {start_year}-{end_year}.")

    finally:
        pass
    #     cleanup_temp_data_dir()

    logger.info(f"--- Finished processing AIWQ variable: {variable_name} ---")


def get_config_variables(variable_name: str) -> tuple[str | None, str | None, str | None]:
    """Retrieves AIWQ specific variable names and standard name from config."""
    std_var_name = STD_VARIABLES.get(variable_name)
    src_var_name = SRC_VARIABLES.get(variable_name)
    src_var_short_name = SRC_VARIABLES_SHORTCODE.get(variable_name)

    if not src_var_name or not src_var_short_name or not std_var_name:
        logger.error(f"Variable '{variable_name}' not fully configured in config.yaml for AIWQ (std_name, cds_name, or short_code missing).")
        return None, None, None
    return std_var_name, src_var_name, src_var_short_name


def ensure_temp_data_dir_exists() -> None:
    """Ensures the temporary directory for AIWQ data exists."""
    if not AIWQ_TEMP_DIR.exists():
        logger.info(f"Creating temporary directory: {AIWQ_TEMP_DIR}")
        os.makedirs(AIWQ_TEMP_DIR)


def download_yearly_data(
    year: int,
    variable_name: str, # For logging
    src_var_short_name: str,
    password: str
) -> xr.Dataset | None:
    """Downloads AIWQ data for a single year."""
    logger.info(f"Downloading {variable_name} for year {year}...")
    try:
        ds = retrieve_training_data.retrieve_annual_training_data(
            year, src_var_short_name, password=password, local_destination=str(AIWQ_TEMP_DIR)
        )
        return ds
    except Exception as e:
        logger.error(f"Failed to download {variable_name} for year {year}. Error: {e}", exc_info=True)
        return None


def cleanup_temp_data_dir() -> None:
    """Cleans up the temporary AIWQ data directory."""
    if AIWQ_TEMP_DIR.exists():
        logger.info(f"Deleting contents of temporary directory: {AIWQ_TEMP_DIR}")
        delete_contents(str(AIWQ_TEMP_DIR))
        try:
            if not os.listdir(AIWQ_TEMP_DIR): # Check if directory is empty
                logger.info(f"Removing empty temporary directory: {AIWQ_TEMP_DIR}")
                os.rmdir(AIWQ_TEMP_DIR)
            else:
                logger.warning(f"Temporary directory {AIWQ_TEMP_DIR} is not empty after attempting to delete contents. Manual check may be needed.")
        except FileNotFoundError:
            logger.info(f"Temporary directory {AIWQ_TEMP_DIR} was already removed.")
        except OSError as e:
            logger.error(f"Error removing temporary directory {AIWQ_TEMP_DIR}: {e}", exc_info=True)
