import cdsapi
import xarray as xr
import pandas as pd
from pathlib import Path
import logging
import tempfile
import os
from typing import Dict, Any, Optional, Tuple

from data.helpers.config import load_config
from utils.data_io import get_zarr_store_path, save_to_zarr, check_zarr_exists
from data.helpers.data_processing import daily_average

logger = logging.getLogger(__name__)

CONFIG = load_config()
BASE_DIR = Path(CONFIG["data_dir"])
TARGET_RESOLUTION = CONFIG["target_grid"]["resolution_deg"]
SOURCE_CONFIG = CONFIG["sources"]["era5"]

STD_VARIABLES = CONFIG["variables"]
SRC_VARIABLES = SOURCE_CONFIG["variables"]
SRC_VARIABLES_SHORTCODE = SOURCE_CONFIG["variables_shortcode"]

LATENCY = CONFIG["latency"]["era5"]
ERA5_CHUNK_CONFIG = {"time": -1, "latitude": 1, "longitude": -1}


def _resolve_weather_variables(weather_variables):
    """Resolve which variables to process based on the weather_variables argument.

    Args:
        weather_variables: Optional list of variable names (e.g., ["temperature"]).
            If None, returns all configured variable names.

    Returns:
        List of variable names to process.
    """
    if weather_variables is None:
        return list(SRC_VARIABLES.keys())

    resolved = []
    for var in weather_variables:
        if var in SRC_VARIABLES:
            resolved.append(var)
        else:
            logger.warning(
                f"Weather variable '{var}' not found in ERA5 config. "
                f"Available variables: {list(SRC_VARIABLES.keys())}"
            )
    return resolved


def initial_download(start_date_str: str, end_date_str: str, force: bool = False, weather_variables=None) -> None:
    """
    Performs an initial download for all configured ERA5 variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
        weather_variables: Optional list of variable names to process (e.g., ["temperature"]).
            If None, all configured variables are processed.
    """

    logger.info(f"===== Starting ERA5 Initial Download ({start_date_str} to {end_date_str}) =====")

    variables_to_process = _resolve_weather_variables(weather_variables)
    for weather_variable in variables_to_process:
        try:
            store_path = get_zarr_store_path(BASE_DIR, f"era5-{STD_VARIABLES[weather_variable]}")

            if check_zarr_exists(store_path) and not force:
                logger.error(f"Dataset {store_path} already exists; skipping download. Use --force to overwrite.")
                continue

            download_era5_variable(
                full_var_name=weather_variable,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
                store_path=store_path,
                initial_download=True,
            )

        except Exception as e:
            logger.error(f"Failed initial download for {weather_variable}. Error: {e}", exc_info=True)

    logger.info("===== Finished ERA5 Initial Download =====")


def update(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, force: bool = False, weather_variables=None) -> None:
    """
    Checks each configured ERA5 variable and updates it if new data is available.

    Args:
        start_date_str: Optional start date (YYYY-MM-DD). If provided, downloads from this date.
        end_date_str: Optional end date (YYYY-MM-DD). If provided, downloads until this date.
        force: If True, overwrite data for dates that already exist. If False, skip existing dates.
        weather_variables: Optional list of variable names to process (e.g., ["temperature"]).
            If None, all configured variables are processed.
    """

    logger.info("===== Starting ERA5 Update Process =====")

    variables_to_process = _resolve_weather_variables(weather_variables)
    for weather_variable in variables_to_process:
        store_path = get_zarr_store_path(BASE_DIR, f"era5-{STD_VARIABLES[weather_variable]}")

        # Determine what dates to download based on arguments
        dates_to_download = determine_update_dates(
            store_path,
            time_coord="time",
            data_latency_days=LATENCY,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )

        if not dates_to_download:
            logger.warning("No new dates to download. Skipping update.")
            continue

        start_date, end_date = dates_to_download

        # Check if data for the period already exists in the zarr store; if force is True, remove overlapping dates and continue
        if data_for_update_already_exists(store_path, start_date, end_date, weather_variable, force):
            continue

        logger.info(
            f"Proceeding with update for {weather_variable} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        try:
            download_era5_variable(
                full_var_name=weather_variable,
                start_date_str=start_date.strftime("%Y-%m-%d"),
                end_date_str=end_date.strftime("%Y-%m-%d"),
                store_path=store_path,
                initial_download=False,  # This is an update
                force=force,
            )
        except Exception as e:
            logger.error(f"Failed to update {weather_variable}. Error: {e}", exc_info=True)

    logger.info("===== Finished ERA5 Update Process =====")


def determine_update_dates(
    store_path: str,
    time_coord: str = "time",
    data_latency_days: int = 5,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
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
    today_lagged = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=data_latency_days)

    # Case 1: Both dates provided - use them directly
    if start_date_str and end_date_str:
        start_date = pd.Timestamp(start_date_str, tz="UTC")
        end_date = pd.Timestamp(end_date_str, tz="UTC")

        # Make sure end_date isn't in the future beyond data availability
        end_date = min(end_date, today_lagged)

        if end_date < start_date:
            logger.warning(
                f"End date {end_date.strftime('%Y-%m-%d')} is before start date {start_date.strftime('%Y-%m-%d')}"
            )
            return None

        # Return dates and we'll append & sort later
        return start_date, end_date

    # Case 2: Only start_date provided - download from start_date to today_lagged
    if start_date_str and not end_date_str:
        start_date = pd.Timestamp(start_date_str, tz="UTC")
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
            last_date = pd.Timestamp(ds_check[time_coord].max().values, tz="UTC").normalize()
            logger.debug(f"Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Error opening Zarr store: {e}")
            return None  # Propagate error
        finally:
            if ds_check is not None:
                ds_check.close()
                logger.debug(f"Closed ds_check for {store_path} in determine_update_dates")
    else:
        logger.warning(f"Zarr store {store_path} does not exist, cannot determine last date")
        return None

    # Case 3: Only end_date provided - download from last_date+1 to end_date
    if end_date_str and not start_date_str:
        end_date = min(pd.Timestamp(end_date_str, tz="UTC"), today_lagged)
        start_date = last_date + pd.Timedelta(days=1)

        if end_date < start_date:
            logger.warning(
                f"No new dates to download between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
            )
            return None

        return start_date, end_date

    # Case 4: No dates provided - use standard update procedure (last_date+1 to today_lagged)
    start_date = last_date + pd.Timedelta(days=1)
    end_date = today_lagged

    # Ensure both start_date and end_date are tz-aware before comparison
    # start_date = start_date.tz_localize('UTC') if start_date.tzinfo is None else start_date
    # end_date = end_date.tz_localize('UTC') if end_date.tzinfo is None else end_date

    if end_date < start_date:
        logger.info(f"Dataset already up-to-date, last date is {last_date.strftime('%Y-%m-%d')}")
        return None

    return start_date, end_date


def data_for_update_already_exists(
    store_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp, weather_variable: str, force: bool
) -> bool:
    """
    Checks for existing data in the store_path for the given date range.
    If overlapping data exists and force is False, logs an error and returns True (skip).
    If force is True, removes overlapping dates from the Zarr store.
    Returns True if the update for this variable should be skipped, False otherwise.
    """
    if not check_zarr_exists(store_path):
        return False  # No existing data, proceed with update

    existing_ds = None
    try:
        existing_ds = xr.open_zarr(store_path)
        existing_times_utc = pd.DatetimeIndex(existing_ds.time.values).normalize().tz_localize("UTC")

        update_period_range_utc = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
        overlapping_dates_utc = existing_times_utc.intersection(update_period_range_utc)

        overlapping_dates = overlapping_dates_utc.tz_convert(None)

        if not overlapping_dates.empty:
            if force:
                logger.info(
                    f"Force update: Removing overlapping dates from existing dataset {store_path}: {overlapping_dates.strftime('%Y-%m-%d').tolist()}"
                )
                modified_ds = existing_ds.drop_sel(time=overlapping_dates)
                logger.info(
                    f"Loading data (excluding overlaps) into memory before overwriting Zarr store {store_path}..."
                )
                modified_ds = modified_ds.load()  # This can be memory intensive
                logger.info(f"Data loaded. Now saving to Zarr store {store_path}.")
                save_to_zarr(modified_ds, store_path, mode="w")
                del modified_ds
                logger.info(f"Successfully removed overlapping dates from {store_path}.")
                return False  # Overlapping dates have been erased, proceed with update for the period
            else:
                logger.error(
                    f"Update for {weather_variable} aborted. "
                    f"Dates {overlapping_dates.strftime('%Y-%m-%d').tolist()} already exist in {store_path}. "
                    f"Use --force to overwrite or ensure the update range is distinct."
                )
                return True  # Skip update for this variable
        return False  # No overlap, proceed with update
    except Exception as e:
        logger.error(
            f"Error during overlap check or data modification for {weather_variable} in {store_path}: {e}. "
            f"Skipping update for this variable to be safe.",
            exc_info=True,
        )
        return True  # Skip update for this variable
    finally:
        if existing_ds is not None:
            existing_ds.close()
            logger.debug(f"Closed existing_ds for {store_path} in handle_existing_data_for_update")


def download_era5_variable(
    full_var_name: str,  # e.g., 'temperature'
    start_date_str: str,  # 'YYYY-MM-DD'
    end_date_str: str,  # 'YYYY-MM-DD'
    store_path: Optional[str] = None,
    initial_download: bool = False,
    force: bool = False,
) -> None:
    """
    Downloads, processes, and stores ERA5 data for a single variable and specified date range.
    Orchestrates calls to helper functions for distinct steps.
    """
    std_var_name, src_var_name, src_var_short_name = get_config_variables(full_var_name)
    if not all([std_var_name, src_var_name, src_var_short_name]):
        return  # Error already logged

    logger.info(f"--- Processing ERA5 variable: {full_var_name} (Zarr: {store_path}) ---")

    # Define the actual date range for the data (UTC aware)
    # The 6-day buffer for rolling mean is handled internally by fetch_data
    start_date = pd.Timestamp(start_date_str, tz="UTC")
    end_date = pd.Timestamp(end_date_str, tz="UTC")

    today_lagged = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=LATENCY)

    processed_ds = None  # Initialize to ensure it's defined in finally block
    try:
        # Step 1: Fetch and process data for the determined period
        processed_ds = fetch_data(
            start_date,
            end_date,
            today_lagged,
            full_var_name,  # for logging
            src_var_name,
            std_var_name,
            src_var_short_name,
        )

        if processed_ds is None:
            logger.warning(
                f"No data fetched or processed for {full_var_name} in the period "
                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. Skipping save."
            )
            logger.info(f"--- Finished processing ERA5 variable: {full_var_name} ---")
            return

        # Step 2: Validate (NaN check, etc.) and store the processed data
        if not validate_and_store_processed_data(
            processed_ds, std_var_name, full_var_name, store_path, initial_download, force
        ):
            # Save was aborted (e.g., NaNs found or other save error), error already logged by helper
            logger.warning(
                f"Validation or storage failed for {full_var_name}. Check previous logs. Aborting for this variable."
            )
            # Ensure we still log the "Finished processing" message for this attempt.
        else:
            logger.info(f"Successfully processed and stored data for {full_var_name}.")

    except Exception as e:
        # This catches errors re-raised from _fetch_data_for_variable_period
        # or any other unexpected error in this main orchestration block.
        logger.error(f"Critical error during download/processing of {full_var_name}: {e}", exc_info=True)
    finally:
        if processed_ds is not None:
            try:
                processed_ds.close()
                logger.debug(f"Closed processed_ds for {full_var_name} in download_era5_variable.")
            except Exception as e_close:
                logger.warning(f"Error closing processed_ds for {full_var_name}: {e_close}", exc_info=True)

    logger.info(f"--- Finished processing ERA5 variable: {full_var_name} ---")


def get_config_variables(full_var_name: str) -> tuple[str | None, str | None, str | None]:
    """Retrieves ERA5 specific standard name, CDS variable name, and short code from config."""
    std_var_name = STD_VARIABLES.get(full_var_name)
    src_var_name = SRC_VARIABLES.get(full_var_name)
    src_var_short_name = SRC_VARIABLES_SHORTCODE.get(full_var_name)

    if not std_var_name:
        logger.error(f"Standard variable name for '{full_var_name}' not found in config.variables.")
    if not src_var_name:
        logger.error(f"CDS variable name for '{full_var_name}' not found in config.sources.era5.variables.")
    if not src_var_short_name:
        logger.error(f"Variable short code for '{full_var_name}' not found in config.sources.era5.variables_shortcode.")

    if not all([std_var_name, src_var_name, src_var_short_name]):
        logger.error(
            f"Full configuration for variable '{full_var_name}' is missing. Cannot proceed with download for this variable."
        )
        return None, None, None

    return std_var_name, src_var_name, src_var_short_name


def fetch_data(
    start_date_actual: pd.Timestamp,  # Actual start date for data (UTC aware)
    end_date_actual: pd.Timestamp,  # Actual end date for data (UTC aware)
    today_lagged: pd.Timestamp,  # Latest available date (UTC aware)
    full_var_name: str,  # Original variable name for logging
    src_var_name: str,
    std_var_name: str,
    src_var_short_name: str,
) -> Optional[xr.Dataset]:
    """
    Fetches and processes ERA5 data for a given variable and actual date range.
    A 6-day buffer is added to end_date_actual for rolling mean calculations.
    """
    # Add 6-day buffer to the end date for rolling mean calculation
    fetch_end_date_buffered = end_date_actual + pd.Timedelta(days=6)

    # Determine the period for ERA5 reanalysis data, ensuring not to request beyond today_lagged
    era5_period_start = start_date_actual
    era5_period_end = min(fetch_end_date_buffered, today_lagged)

    processed_ds = None
    if era5_period_start <= era5_period_end:
        logger.info(
            f"Requesting ERA5 reanalysis data for {full_var_name} "
            f"from {era5_period_start.strftime('%Y-%m-%d')} to {era5_period_end.strftime('%Y-%m-%d')}"
        )
        request = build_cds_request(src_var_name, era5_period_start, era5_period_end, product_type="reanalysis", full_var_name=full_var_name)
        try:
            processed_ds = fetch_and_process_cds_data(request, std_var_name, src_var_short_name, full_var_name)
        except Exception:
            logger.error(
                f"Failed to retrieve/process ERA5 data for {full_var_name} "
                f"from {era5_period_start.strftime('%Y-%m-%d')} to {era5_period_end.strftime('%Y-%m-%d')}.",
                exc_info=True,
            )
            raise  # Re-raise to be handled by the caller (download_era5_variable)
    else:
        logger.info(
            f"No ERA5 reanalysis data to download for {full_var_name}. "
            f"Calculated period {era5_period_start.strftime('%Y-%m-%d')} to {era5_period_end.strftime('%Y-%m-%d')} is invalid "
            f"(actual end: {end_date_actual.strftime('%Y-%m-%d')}, buffered end: {fetch_end_date_buffered.strftime('%Y-%m-%d')}, "
            f"today_lagged: {today_lagged.strftime('%Y-%m-%d')})."
        )

    return processed_ds


def build_cds_request(
    src_var_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
    product_type: str = "reanalysis", full_var_name: str = "",
) -> Dict[str, Any]:
    """Builds the dictionary for the CDS API request."""

    date_list = pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()

    # Precipitation uses hourly time steps; other variables use 6-hourly
    if full_var_name == "precipitation":
        time_steps = [f"{h:02d}:00" for h in range(0, 24, 1)]
    else:
        time_steps = [f"{h:02d}:00" for h in range(0, 24, 6)]  # 00:00, 06:00, 12:00, 18:00

    request = {
        "product_type": product_type,
        "variable": [src_var_name],
        "date": date_list,
        "time": time_steps,
        "format": "netcdf",
        "area": [90, 0, -90, 360],  # Global coverage N, W, S, E
        "grid": [1.5, 1.5],
    }
    logger.debug(f"Built CDS request for {src_var_name}: {request}")
    return request


def fetch_and_process_cds_data(
    request: Dict[str, Any], std_var_name: str, src_var_short_name: str, full_var_name: str
) -> Optional[xr.Dataset]:
    """
    Orchestrates fetching data via CDS API, processing it, and returning an xarray Dataset.
    Uses helper functions for each main step of processing.
    `full_var_name` is the original variable key used for logging purposes.
    """
    client = cdsapi.Client(wait_until_complete=True, delete=False)  # Keep downloaded file for inspection if needed

    # Use a temporary file for the download, with delete=False to keep it after closing
    tmpfile = None  # Initialize to ensure it's defined in finally block
    try:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        target_path = tmpfile.name
        tmpfile.close()
        logger.debug(f"Temporary file created for {full_var_name}: {target_path}")

        if not cds_retrieval_succeded(client, request, target_path, full_var_name):
            return None  # Retrieval failed

        ds, original_attributes = load_and_preprocess_raw_data(
            target_path, std_var_name, src_var_short_name, full_var_name
        )
        if ds is None:
            return None  # Loading/preprocessing failed

        expected_steps = 24 if full_var_name == "precipitation" else 4
        ds = validate_subdaily_completeness(ds, expected_steps=expected_steps, full_var_name=full_var_name)
        if full_var_name == "precipitation":
            ds = ensure_precipitation_ends_at_00h(ds, client, request, std_var_name, src_var_short_name, full_var_name)
        ds = apply_temporal_transformations(ds, std_var_name, full_var_name)
        ds = finalize_dataset_for_storage(ds, std_var_name, original_attributes, full_var_name)

        logger.info(f"Successfully processed data for {full_var_name}")
        return ds

    except Exception as e:
        logger.error(f"Unhandled error in fetch_and_process_cds_data for {full_var_name}: {e}", exc_info=True)
        return None
    finally:
        # Clean up the temporary file
        if tmpfile and os.path.exists(target_path):
            try:
                os.remove(target_path)
                logger.debug(f"Removed temporary file for {full_var_name}: {target_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {target_path} for {full_var_name}: {e}")


def cds_retrieval_succeded(
    client: cdsapi.Client, request: Dict[str, Any], target_path: str, full_var_name: str
) -> bool:
    """
    Performs the data retrieval using cdsapi.client.
    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Requesting CDS data for {full_var_name} to {target_path}")
        client.retrieve("reanalysis-era5-single-levels", request, target_path)
        logger.info(f"Successfully downloaded data for {full_var_name} to {target_path}")
        return True
    except Exception as e:
        logger.error(f"CDS API retrieval failed for {full_var_name}. Error: {e}", exc_info=True)
        return False


def load_and_preprocess_raw_data(
    target_path: str, std_var_name: str, src_var_short_name: str, full_var_name: str
) -> Tuple[Optional[xr.Dataset], Optional[Dict]]:
    """
    Loads the raw NetCDF data, renames variables/coordinates, drops unnecessary ones.
    Returns the processed dataset and original attributes of the main variable.
    """
    try:
        # ds = xr.open_dataset(target_path, engine='netcdf4', chunks={'time': 24}) # Chunk by day initially
        ds = xr.open_dataset(target_path, engine="netcdf4")
        logger.info(f"Opened {target_path} for {full_var_name}")

        logger.info(
            f"Renaming variables and coordinates for {full_var_name} to standard names ({std_var_name}; latitude, longitude, time)"
        )
        rename_dict = {src_var_short_name: std_var_name}
        if "valid_time" in ds.coords:
            rename_dict["valid_time"] = "time"
        ds = ds.rename(rename_dict)

        if "expver" in ds.coords:
            logger.debug(f"Dropping 'expver' coordinate for {full_var_name}")
            ds = ds.drop_vars("expver")
        if "number" in ds.coords:
            logger.debug(f"Dropping 'number' coordinate for {full_var_name}")
            ds = ds.drop_vars("number")

        original_attributes = ds[std_var_name].attrs if std_var_name in ds else {}
        return ds, original_attributes
    except Exception as e:
        logger.error(f"Error loading and preprocessing {target_path} for {full_var_name}: {e}", exc_info=True)
        return None, None


def validate_subdaily_completeness(
    ds: xr.Dataset, expected_steps: int = 4, full_var_name: str = ""
) -> xr.Dataset:
    """
    Validates that every calendar day in the dataset has all expected sub-daily
    time steps (00:00, 06:00, 12:00, 18:00).

    If incomplete days are found, the first such date and all subsequent dates
    are dropped. A warning is issued if usable data remains; a ValueError is
    raised if the entire dataset would be discarded.
    """
    times = pd.DatetimeIndex(ds.time.values)
    day_labels = times.normalize()
    steps_per_day = pd.Series(1, index=times).groupby(day_labels).count()

    incomplete_days = steps_per_day[steps_per_day < expected_steps]

    if incomplete_days.empty:
        logger.info(
            f"All {expected_steps} sub-daily time steps present for every day of {full_var_name}."
        )
        return ds

    # Identify the first incomplete date and drop it + everything after
    first_incomplete_date = incomplete_days.index.min()
    detail = ", ".join(
        f"{d.strftime('%Y-%m-%d')} ({count}/{expected_steps} steps)"
        for d, count in incomplete_days.items()
    )
    logger.info(
        f"Incomplete sub-daily data for {full_var_name}: {detail}. "
        f"Dropping data from {first_incomplete_date.strftime('%Y-%m-%d')} onwards."
    )

    ds = ds.sel(time=ds.time.values < first_incomplete_date)

    if ds.time.size == 0:
        raise ValueError(
            f"No complete days remain for {full_var_name} after dropping "
            f"incomplete dates starting at {first_incomplete_date.strftime('%Y-%m-%d')}."
        )

    logger.warning(
        f"Dropped incomplete dates for {full_var_name} starting at "
        f"{first_incomplete_date.strftime('%Y-%m-%d')}. "
        f"Continuing with {ds.time.size} remaining timesteps "
        f"({steps_per_day[steps_per_day >= expected_steps].size} complete days)."
    )
    return ds


def ensure_precipitation_ends_at_00h(
    ds: xr.Dataset,
    client: "cdsapi.Client",
    base_request: Dict[str, Any],
    std_var_name: str,
    src_var_short_name: str,
    full_var_name: str,
) -> xr.Dataset:
    """
    Ensures the precipitation dataset ends at a 00:00 timestamp.

    If the last calendar day is complete (24 hourly steps), attempts to
    download 00:00 of the following day.  Falls back to trimming the
    dataset to the latest available 00:00 timestamp.
    """
    times = pd.DatetimeIndex(ds.time.values)
    last_day = times.max().normalize()

    # Count hours on the last calendar day
    n_hours_last_day = int((times.normalize() == last_day).sum())

    if n_hours_last_day == 24:
        # Last day is complete – try to fetch 00h of next day
        next_day = last_day + pd.Timedelta(days=1)
        logger.info(
            f"Attempting to download 00h of {next_day.strftime('%Y-%m-%d')} for {full_var_name}"
        )
        extra_ds = _download_single_hour(
            client, base_request, next_day,
            std_var_name, src_var_short_name, full_var_name,
        )
        if extra_ds is not None:
            ds = xr.concat([ds, extra_ds], dim="time")
            extra_ds.close()
            logger.info(
                f"Successfully appended 00h of {next_day.strftime('%Y-%m-%d')} "
                f"for {full_var_name}"
            )
            return ds

        logger.warning(
            f"Could not download 00h of {next_day.strftime('%Y-%m-%d')} for "
            f"{full_var_name}. Trimming to latest available 00:00."
        )
    else:
        logger.warning(
            f"Last day {last_day.strftime('%Y-%m-%d')} has only "
            f"{n_hours_last_day} hourly steps for {full_var_name}. "
            f"Trimming to latest available 00:00."
        )

    # Fallback: trim to the latest 00:00 timestamp
    times_00h = times[times.hour == 0]
    if len(times_00h) == 0:
        raise ValueError(
            f"No 00:00 timestamps found in precipitation data for {full_var_name}"
        )
    latest_00h = times_00h.max()
    ds = ds.sel(time=slice(None, latest_00h))
    logger.info(f"Trimmed {full_var_name} dataset to end at {latest_00h}")
    return ds


def _download_single_hour(
    client: "cdsapi.Client",
    base_request: Dict[str, Any],
    date: pd.Timestamp,
    std_var_name: str,
    src_var_short_name: str,
    full_var_name: str,
) -> Optional[xr.Dataset]:
    """Downloads a single hour (00:00) for a single date. Returns None on failure."""
    request = base_request.copy()
    request["date"] = [date.strftime("%Y-%m-%d")]
    request["time"] = ["00:00"]

    tmpfile = None
    target_path = None
    try:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        target_path = tmpfile.name
        tmpfile.close()

        if not cds_retrieval_succeded(client, request, target_path, full_var_name):
            return None

        ds, _ = load_and_preprocess_raw_data(
            target_path, std_var_name, src_var_short_name, full_var_name
        )
        return ds
    except Exception as e:
        logger.warning(
            f"Failed to download 00h for {date.strftime('%Y-%m-%d')}: {e}"
        )
        return None
    finally:
        if target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass

def apply_temporal_transformations(ds: xr.Dataset, std_var_name: str, full_var_name: str) -> xr.Dataset:
    """Applies temporal aggregation and 7-day rolling aggregation with left-alignment.

    For precipitation (hourly data):
        Shift time by -1h so each "day" sums hours 01h–00h(next day),
        resample to daily sums, then apply a 7-day rolling sum (× 1000 for m→mm).
    For other variables (6-hourly data):
        Daily average → 7-day rolling mean.
    """
    if std_var_name == "pr":
        logger.info(f"Applying precipitation temporal transformations for {full_var_name}")

        # Shift by -1h: hour 01 → 00 (same day), hour 00 (next day) → 23 (current day)
        ds["time"] = ds["time"] - pd.Timedelta(hours=1)

        # Drop the stray timestamp that fell into the previous calendar day
        first_midnight = pd.Timestamp(ds.time.values[0]).ceil("D")
        ds = ds.sel(time=ds.time >= first_midnight)

        # Daily sum of shifted hourly data
        ds = ds.resample(time="1D").sum(keep_attrs=True)

        # 7-day rolling sum, convert m → mm
        logger.info(f"Applying 7-day rolling sum for {full_var_name}")
        ds = ds.rolling(time=7).sum() * 1000
    else:
        logger.info(f"Calculating daily average for {full_var_name}")
        ds = daily_average(ds)

        logger.info(f"Applying 7-day rolling mean for {full_var_name}")
        ds = ds.rolling(time=7).mean()

    # Make it left-aligned: time axis labels indicating the start of each seven-day period.
    ds["time"] = ds["time"] - pd.Timedelta(days=6)
    logger.debug(f"Applied left-alignment to time axis for {full_var_name}")
    return ds


def finalize_dataset_for_storage(
    ds: xr.Dataset, std_var_name: str, original_attributes: Dict, full_var_name: str
) -> xr.Dataset:
    """Selects final variable, sorts all coordinates, and sets metadata."""
    logger.info(f"Finalizing dataset for {full_var_name}: selecting variable, sorting, adding attributes.")
    # Select only the variable of interest and standard coordinates
    ds = ds[[std_var_name]].sortby(["time", "latitude", "longitude"])

    # Add attributes - copy from original if possible
    if original_attributes:
        ds[std_var_name].attrs = original_attributes
    ds.attrs["history"] = (
        f"Downloaded from CDS, daily averaged, 7-day aggregated, interpolated to {TARGET_RESOLUTION}deg."  # TARGET_RESOLUTION is global
    )

    # Ensure consistent chunking for Zarr compatibility (currently commented out in original)
    ds = ds.chunk(ERA5_CHUNK_CONFIG)
    return ds


def validate_and_store_processed_data(
    processed_ds: xr.Dataset,
    std_var_name: str,
    full_var_name: str,  # Original variable name for logging
    store_path: str,
    initial_download: bool,
    force: bool,
) -> bool:
    """
    Validates the processed dataset (drops initial 6 days, checks for NaNs) and saves it to Zarr.
    Returns True if saved successfully or if no data to save; False if NaNs were found and save was aborted.
    """
    logger.debug(
        f"Initial processed dataset for {full_var_name} has time range: "
        f"{processed_ds.time.min().values} to {processed_ds.time.max().values} "
        f"({processed_ds.time.size} points)"
    )

    # Drop the first 6 days which were used for the rolling aggregation window
    # These days will have NaN values if they are at the very start of the fetched data
    processed_ds_final = processed_ds.isel(time=slice(6, None))

    logger.info(f"Checking final dataset for NaNs for {full_var_name} (variable: {std_var_name})")
    # Ensure the variable actually exists in the dataset before checking for NaNs
    if std_var_name not in processed_ds_final:
        logger.error(
            f"Standard variable '{std_var_name}' not found in the final processed dataset for {full_var_name}. Cannot check NaNs or save."
        )
        return False

    daily_nan_check = processed_ds_final[std_var_name].isnull().any(dim=["latitude", "longitude"])
    dates_with_nans = processed_ds_final.time.where(daily_nan_check.compute(), drop=True)

    if not dates_with_nans.time.size == 0:
        dates_with_nans_str = [pd.Timestamp(t.item()).strftime("%Y-%m-%d") for t in dates_with_nans.values]
        logger.error(
            f"Aborting save for {full_var_name}. NaNs found in the final processed data on dates: {dates_with_nans_str}"
        )
        return False  # Abort due to NaNs

    mode = "w" if initial_download or not check_zarr_exists(store_path) else "a"
    append_dim = "time" if mode == "a" else None

    # Ensure the data being appended doesn't already exist in the existing dataset (e.g., for concurrency reasons)
    if mode == "a" and not force:
        existing_ds_check = None
        try:
            logger.info(f"Performing final duplicate check for {full_var_name} before appending to {store_path}.")
            existing_ds_check = xr.open_zarr(store_path)
            existing_times = pd.DatetimeIndex(existing_ds_check.time.values)
            new_times = pd.DatetimeIndex(processed_ds_final.time.values)

            # Find dates in the new data that are NOT in the existing data
            times_to_append = new_times.difference(existing_times)

            if len(times_to_append) < len(new_times):
                duplicates_found = len(new_times) - len(times_to_append)
                logger.warning(
                    f"Found and removed {duplicates_found} duplicate date(s) for {full_var_name} just before saving."
                )

            if times_to_append.empty:
                logger.info(f"No new data to append for {full_var_name} after final check. All dates already exist.")
                return True  # Nothing to save, but not an error.

            # Filter the dataset to only include the truly new dates
            processed_ds_final = processed_ds_final.sel(time=times_to_append)

        except Exception as e:
            logger.error(f"Error during final duplicate check for {full_var_name}: {e}. Aborting save.", exc_info=True)
            return False
        finally:
            if existing_ds_check is not None:
                existing_ds_check.close()

    logger.info(f"Saving validated data for {full_var_name} to {store_path} (mode: {mode}, append_dim: {append_dim})")
    try:
        save_to_zarr(processed_ds_final, store_path, mode=mode, append_dim=append_dim)
        logger.info(f"Successfully saved data for {full_var_name} to {store_path}.")
        return True  # Saved successfully
    except Exception as e:
        logger.error(f"Failed to save data for {full_var_name} to {store_path}. Error: {e}", exc_info=True)
        return False
