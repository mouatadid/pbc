import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import xarray as xr
from utils.file_io import set_recursive_permissions, make_directories, set_file_permissions

# Suppress specific Zarr consolidated metadata warning
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

STANDARD_TEMP = 'temperature'
STANDARD_PREC = 'precipitation'

STANDARD_VARIABLE_NAMES = {
    'tas': STANDARD_TEMP,
    'tmp2m': STANDARD_TEMP,
    'temp': STANDARD_TEMP,
    '2m_temperature': STANDARD_TEMP,
    't2m': STANDARD_TEMP,
    'tp': STANDARD_PREC,
    'total_precipitation': STANDARD_PREC,
}

DATA_DIR = "data"

logger = logging.getLogger(__name__)

def load_data(id: str, lsmask: bool = False) -> xr.Dataset:
    """
    Loads dataset corresponding to the data identifier (e.g., "tas", "tp", "mslp").
    Args:
        id: The identifier of the xarray Dataset to load.
        lsmask: Whether to apply land sea mask.
    """
    if id.startswith("ecmwf-") or id.startswith("debiased_ecmwf-"):
        # Load multiple ECMWF models as a single dataset
        prefix, suffix = id.split("-", maxsplit=1)
        models = ["cy41-47", "cy48", "cy49"]
        load_dir = os.path.join(DATA_DIR, "debiased_ecmwf") if prefix == "debiased_ecmwf" else DATA_DIR
        filenames =[os.path.join(load_dir, f"{prefix}_{model}-{suffix}.zarr") for model in models]
        try:
            ds = xr.open_mfdataset(filenames, decode_timedelta=True)
            if lsmask:
                filename_lsm = os.path.join(DATA_DIR, "lsmask.zarr")
                lsm = xr.open_dataarray(filename_lsm, decode_timedelta=True)
                ds = ds.where(lsm >= 0.5)
            return ds
        except Exception as e:
            logger.error(f"Could not read datasets in {filenames}: {e}")
            logger.info(f"Available files: {list_zarr_files('data')}")
            raise e
    # Otherwise, load a single dataset
    filename = os.path.join(DATA_DIR, f"{id}.zarr")
    try:
        ds = xr.open_zarr(filename, chunks={}, decode_timedelta=True)
        if lsmask: 
            filename_lsm = os.path.join(DATA_DIR, "lsmask.zarr")
            lsm = xr.open_dataarray(filename_lsm, decode_timedelta=True)
            ds = ds.where(lsm >= 0.5)         
        return ds
    except Exception as e:
        if lsmask:
            filename_lsm = os.path.join(DATA_DIR, "lsmask.zarr")
            logger.error(f"Could not read dataset in {filename} or array in {filename_lsm}: {e}")
        else:
            logger.error(f"Could not read dataset in {filename}: {e}")
        logger.info(f"Available files: {list_zarr_files('data')}")
        raise e
    
def save_data(
    ds: xr.Dataset,
    id: str,
    mode: str = 'w',
    append_dim: Optional[str] = None,
    consolidated: bool = True,
    encoding: Optional[dict] = None
) -> None:
    """
    Saves or appends an xarray Dataset to a Zarr store defined by the data identifier.

    Args:
        ds: The xarray Dataset to save.
        id: The identifier for the dataset (e.g., "tas", "tp").
        mode: 'w' for write (overwrite), 'a' for append.
        append_dim: The dimension along which to append (required if mode='a').
        consolidated: Whether to write consolidated metadata.
        encoding: Encoding dictionary for variables (e.g., for compression).
    """
    store_path = get_zarr_store_path(DATA_DIR, id)
    save_to_zarr(ds, store_path, mode=mode, append_dim=append_dim, consolidated=consolidated, encoding=encoding)

def list_zarr_files(directory):
    return [f for f in os.listdir(directory)
            if f.endswith('.zarr') and os.path.isdir(os.path.join(directory, f))]

def save_to_zarr(
    ds: xr.Dataset,
    store_path: Path,
    mode: str = 'w',
    append_dim: Optional[str] = None,
    consolidated: bool = True,
    encoding: Optional[dict] = None
) -> None:
    """
    Saves or appends an xarray Dataset to a Zarr store.

    Args:
        ds: The xarray Dataset to save.
        store_path: The path to the Zarr store.
        mode: 'w' for write (overwrite), 'a' for append.
        append_dim: The dimension along which to append (required if mode='a').
        consolidated: Whether to write consolidated metadata.
        encoding: Encoding dictionary for variables (e.g., for compression).
    """
    if ds is None or ds.nbytes == 0:
        logger.warning(f"Attempted to save an empty or None dataset to {store_path}. Skipping.")
        return

    if mode == 'a' and not append_dim:
        raise ValueError("append_dim must be specified when mode is 'a'")
    if mode == 'a' and not check_zarr_exists(store_path):
        logger.info(f"Zarr store {store_path} does not exist. Switching mode to 'w' for initial write.")
        mode = 'w' # Create if it doesn't exist for appending
        append_dim = None  # No append dimension needed for initial write

    try:
        logger.info(f"Saving dataset to {store_path} (mode: {mode}, append_dim: {append_dim})")
        # Ensure the parent directory exists
        make_directories(store_path.parent)

        # Construct the dask task
        task = ds.to_zarr(
            store_path,
            mode=mode,
            append_dim=append_dim,
            consolidated=consolidated,
            encoding=encoding,
            compute=False 
        )
        # Execute the save operation
        task.compute()
        # Ensure access permissions are set appropriately
        set_recursive_permissions(store_path)
        logger.info(f"Successfully saved dataset to {store_path}")

    except Exception as e:
        logger.error(f"Failed to save dataset to {store_path}: {e}", exc_info=True)
        # Consider cleanup if needed, e.g., removing a partially written store on error?
        # This can be complex, might be better to leave it for manual inspection.

def save_to_netcdf(
    ds: xr.Dataset,
    store_path: Path
) -> None:
    """
    Saves an xarray Dataset to a NetCDF file.

    Args:
        ds: The xarray Dataset to save.
        store_path: The path to the NetCDF file.
    """
    if ds is None or ds.nbytes == 0:
        logger.warning(f"Attempted to save an empty or None dataset to {store_path}. Skipping.")
        return

    try:
        logger.info(f"Saving dataset to {store_path}")
        # Ensure the parent directory exists
        make_directories(store_path.parent)

        # Save to path
        task = ds.to_netcdf(store_path)
        # Ensure access permissions are set appropriately
        set_file_permissions(store_path)
        logger.info(f"Successfully saved dataset to {store_path}")

    except Exception as e:
        logger.error(f"Failed to save dataset to {store_path}: {e}", exc_info=True)
        # Consider cleanup if needed, e.g., removing a partially written store on error?
        # This can be complex, might be better to leave it for manual inspection.

def delete_contents(path):
    # Walk the directory tree in bottom-up order
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
                
        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                os.rmdir(dir_path)
                print(f"Deleted directory: {dir_path}")
            except Exception as e:
                print(f"Failed to delete directory {dir_path}: {e}")
    

def get_standard_var_name(var_name: str) -> str:
    """
    Returns the standard variable name for a given variable name.
    If the variable name is not recognized, it returns the original name.
    """
    return STANDARD_VARIABLE_NAMES.get(var_name, var_name)

def get_zarr_store_path(base_dir: Union[str, Path], dataset_name: str) -> Path:
    """Constructs the path to the Zarr store for a given dataset."""
    return Path(base_dir).expanduser() / f"{dataset_name}.zarr"

def check_zarr_exists(store_path: Path) -> bool:
    """Checks if a Zarr store exists."""
    return store_path.exists() and store_path.is_dir()

def get_last_date_in_zarr(store_path: Path, time_coord: str = "time") -> Optional[pd.Timestamp]:
    """
    Reads the last date from the time coordinate of a Zarr store.
    Returns None if the store doesn't exist or the coordinate is empty/missing.
    """
    if not check_zarr_exists(store_path):
        return None
    try:
        with xr.open_zarr(store_path, consolidated=True, decode_timedelta=True) as ds:
            if time_coord not in ds.coords or ds[time_coord].size == 0:
                logger.warning(f"Time coordinate '{time_coord}' not found or empty in {store_path}")
                return None
            # Ensure time coordinate is loaded into memory before taking max
            times = ds[time_coord].compute()
            last_date = times.max().values
            # Convert numpy datetime64 to pandas Timestamp for consistency
            return pd.Timestamp(last_date)
    except Exception as e:
        logger.error(f"Error reading last date from {store_path}: {e}")
        return None

def find_date_range_for_update(
    store_path: Path,
    time_coord: str = "time",
    data_latency_days: int = 1,
    max_lookback_days: Optional[int] = None,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determines the date range needed for an update.

    Args:
        store_path: Path to the Zarr store.
        time_coord: Name of the time coordinate.
        data_latency_days: Estimated delay (in days) before data becomes available.
        max_lookback_days: Optional maximum number of days to fetch in one go.

    Returns:
        A tuple (start_date, end_date) for the update, or None if no update is needed
        or the store doesn't exist yet.
    """
    last_stored_date = get_last_date_in_zarr(store_path, time_coord)
    if last_stored_date is None:
        logger.info(f"Cannot determine last date for {store_path}. Assuming initial download is needed.")
        # This function is for updates; initial download logic should be separate.
        return None # Indicate initial download might be required

    # Start date is the day after the last stored date
    start_date = (last_stored_date + pd.Timedelta(days=1)).tz_convert("UTC")

    # End date is typically 'yesterday' or adjusted for data latency
    # Use UTC time for consistency
    end_date = pd.Timestamp.utcnow().normalize().tz_convert("UTC") - pd.Timedelta(days=data_latency_days)

    if start_date > end_date:
        logger.info(f"Data is up-to-date for {store_path}. Last date: {last_stored_date}, Target end date: {end_date}")
        return None # No update needed

    # Apply max lookback if specified
    if max_lookback_days is not None:
        max_end_date = start_date + pd.Timedelta(days=max_lookback_days - 1)
        end_date = min(end_date, max_end_date)
        logger.info(f"Applying max lookback of {max_lookback_days} days. Adjusted end date: {end_date}")


    logger.info(f"Update required for {store_path}. Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    return start_date, end_date
