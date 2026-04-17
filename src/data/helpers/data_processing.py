import xarray as xr
# import xesmf as xe
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def daily_average(ds: xr.Dataset, time_coord: str = "time") -> xr.Dataset:
    """Calculates the daily average for all data variables in the dataset."""
    logger.debug(f"Calculating daily average over '{time_coord}' dimension.")
    try:
        # Ensure time coordinate is compatible with resampling
        if np.issubdtype(ds[time_coord].dtype, np.datetime64):
             # Use floor('D') to ensure timestamps are at 00:00 UTC for the day
            return ds.resample({time_coord: "1D"}).mean(keep_attrs=True)
        else:
            logger.warning(f"Time coordinate '{time_coord}' is not datetime. Cannot perform daily resampling.")
            return ds
    except Exception as e:
        logger.error(f"Error during daily averaging: {e}")
        raise

def ensemble_mean(ds: xr.Dataset, ensemble_coord: str = "number") -> xr.Dataset:
    """Calculates the ensemble mean over the specified dimension."""
    if ensemble_coord in ds.dims:
        logger.debug(f"Calculating ensemble mean over '{ensemble_coord}' dimension.")
        try:
            return ds.mean(dim=ensemble_coord, keep_attrs=True)
        except Exception as e:
            logger.error(f"Error calculating ensemble mean: {e}")
            raise
    else:
        logger.warning(f"Ensemble coordinate '{ensemble_coord}' not found. Returning original dataset.")
        return ds

def create_target_grid(config: Dict[str, Any]) -> xr.Dataset:
    """Creates the target grid definition for xESMF based on config."""
    resolution = config['target_grid']['resolution_deg']
    # Create target grid latitudes and longitudes
    # Ensure longitude wraps correctly if needed (0 to 360-res)
    lon = np.arange(0, 360, resolution)
    # Ensure latitude covers pole to pole (-90 to 90)
    lat = np.arange(-90, 90 + resolution, resolution) # Include 90
    # Clip to ensure bounds are correct
    lat = np.clip(lat, -90, 90)

    target_grid = xr.Dataset(
        {
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),
        }
    )
    logger.debug(f"Created target grid: {resolution}x{resolution} degrees.")
    return target_grid

def regrid_data(
    ds: xr.Dataset,
    target_grid_def: xr.Dataset,
    method: str = 'bilinear',
    weights_path: Optional[str] = None # Optional path to save/load weights
) -> xr.Dataset:
    """
    Regrids the dataset to the target grid using xESMF.

    Args:
        ds: Input xarray Dataset with 'lat' and 'lon' coordinates.
        target_grid_def: xarray Dataset defining the target grid (from create_target_grid).
        method: Regridding method ('bilinear', 'conservative', 'nearest_s2d', etc.).
        weights_path: Path to save/load pre-calculated weights file.

    Returns:
        Regridded xarray Dataset.
    """
    logger.info(f"Regridding data using method: {method}")
    if not all(coord in ds.coords for coord in ['lat', 'lon']):
         # Try common alternatives
        rename_map = {}
        if 'latitude' in ds.coords and 'lat' not in ds.coords:
            rename_map['latitude'] = 'lat'
        if 'longitude' in ds.coords and 'lon' not in ds.coords:
            rename_map['longitude'] = 'lon'

        if rename_map:
            logger.debug(f"Renaming coordinates for regridding: {rename_map}")
            ds = ds.rename(rename_map)
        else:
            logger.error("Input dataset for regridding must contain 'lat' and 'lon' coordinates.")
            raise ValueError("Missing lat/lon coordinates for regridding")

    try:
        # Create the regridder object
        # periodic=True assumes global data wrapping longitude
        regridder = xe.Regridder(
            ds,
            target_grid_def,
            method=method,
            periodic=True,
            filename=weights_path, # Pass path for weight file handling
            reuse_weights=weights_path is not None and Path(weights_path).exists() # Reuse if path provided and file exists
        )

        # Perform the regridding - apply to the dataset
        # Important: xESMF works best if NaNs are handled. Fillna(0) might be okay for some vars,
        # but can distort means. Regridding should handle masks if present.
        # Let xesmf handle masking by default.
        ds_regridded = regridder(ds, keep_attrs=True)

        # Clean up - regridder can consume memory
        regridder.clean_weight_file() # Removes weights file if not reusing

        # Assign standard coordinate names if needed (xESMF might rename them)
        if 'latitude' not in ds_regridded.coords and 'lat' in ds_regridded.coords:
             ds_regridded = ds_regridded.rename({'lat': 'latitude'})
        if 'longitude' not in ds_regridded.coords and 'lon' in ds_regridded.coords:
             ds_regridded = ds_regridded.rename({'lon': 'longitude'})

        logger.info("Regridding successful.")
        return ds_regridded

    except Exception as e:
        logger.error(f"Error during regridding: {e}", exc_info=True)
        raise

def coarsen_data(ds: xr.Dataset, scale_factor: int, boundary: str = "trim") -> xr.Dataset:
    """
    Coarsens the dataset by averaging over blocks using xarray's coarsen method.
    Assumes 'lat' and 'lon' dimensions exist.

    Args:
        ds: Input xarray Dataset.
        scale_factor: The integer factor by which to coarsen (e.g., 6 for 0.25 -> 1.5 deg).
        boundary: How to handle boundaries ('trim', 'pad').

    Returns:
        Coarsened xarray Dataset.
    """
    logger.info(f"Coarsening data with factor {scale_factor} using mean.")
    if not all(coord in ds.dims for coord in ['lat', 'lon']):
        # # Try common alternatives
        # rename_map = {}
        # if 'latitude' in ds.dims and 'lat' not in ds.dims:
        #     rename_map['latitude'] = 'lat'
        # if 'longitude' in ds.dims and 'lon' not in ds.dims:
        #     rename_map['longitude'] = 'lon'
        #
        # if rename_map:
        #      logger.debug(f"Renaming dimensions for coarsening: {rename_map}")
        #      ds = ds.rename(rename_map)
        # else:
        #     logger.error("Input dataset for coarsening must contain 'lat' and 'lon' dimensions.")
        #     raise ValueError("Missing lat/lon dimensions for coarsening")
        raise ValueError("Missing lat/lon dimensions for coarsening")

    try:
        # Ensure coordinates are descending for latitude if necessary for coarsen
        if ds['lat'].values[0] < ds['lat'].values[-1]:
             logger.debug("Reversing latitude coordinate for coarsening.")
             ds = ds.reindex(lat=ds['lat'][::-1])

        ds_coarse = ds.coarsen(lat=scale_factor, lon=scale_factor, boundary=boundary).mean(keep_attrs=True)

        # # Rename back if necessary
        # if 'latitude' not in ds_coarse.coords and 'lat' in ds_coarse.coords:
        #      ds_coarse = ds_coarse.rename({'lat': 'latitude'})
        # if 'longitude' not in ds_coarse.coords and 'lon' in ds_coarse.coords:
        #      ds_coarse = ds_coarse.rename({'lon': 'longitude'})

        logger.info("Coarsening successful.")
        return ds_coarse
    except Exception as e:
        logger.error(f"Error during coarsening: {e}", exc_info=True)
        raise
