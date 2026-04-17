from ecmwfapi import ECMWFDataServer
from ecmwfapi.api import APIException
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import os
from typing import Dict, Any, Optional, Tuple, List

from utils.logging import setup_logging
from utils.file_io import make_directories, set_file_permissions

BASE_DIR = Path("data/ecmwf-backup/")
VARIABLES = {
    "temperature": {
        "short_name": "2t",
        "id_name": "t2m",
        "aiwq_name": "tas",
        "param": "167",
    },
    "precipitation": {
        "short_name": "tp",
        "id_name": "tp",
        "aiwq_name": "pr",
        "param": "228228",
    },
    "mean_sea_level_pressure": {
        "short_name": "msl",
        "id_name": "msl",
        "aiwq_name": "mslp",
        "param": "151",
    },
}
# For reforecasts, we allow end_date to be up to 20 days into the future since they are issued ahead of time
LATENCY = -20
HINDCAST_YEARS = 20
PERTURBED_MEMBERS = 10
REFERENCE_DATE = pd.Timestamp("1960-01-01")
HINDCAST_MONTH_OFFSET = 6

logger = logging.getLogger(__name__)


def initial_download(start_date_str: str, end_date_str: str, force: bool = False, weather_variables: List[str] = None) -> None:
    """
    Performs download for all configured ECMWF reforecast variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
        weather_variables: List of weather variable names to process. Defaults to all VARIABLES.
    """
    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    logger.info(f"===== Starting ECMWF Reforecast Download ({start_date_str} to {end_date_str}) =====")

    for weather_variable in weather_variables:
        try:
            date_range = pd.date_range(start=start_date_str, end=end_date_str, freq="D")
            for date in date_range:
                logger.info(f"Processing {weather_variable} for issuance date {date.strftime('%Y-%m-%d')}")
                try:
                    for forecast_type in ["control", "perturbed"]:
                        aiwq_name = VARIABLES[weather_variable]["aiwq_name"]
                        file_path = (
                            BASE_DIR
                            / f"ecmwf-reforecast-{aiwq_name}"
                            / f"{date.strftime('%Y%m%d')}-{forecast_type}.nc"
                        )

                        if force or not file_path.exists():
                            download_ecmwf_reforecast_date(
                                weather_variable=weather_variable,
                                date=date,
                                forecast_type=forecast_type,
                                file_path=file_path,
                            )
                        else:
                            logger.warning(f"Skipping existing file {file_path}. Use --force to overwrite.")
                except APIException as e:
                    logger.warning(
                        f"No reforecast issued for {date.strftime('%Y-%m-%d')} ({weather_variable}); skipping. Error: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Failed download for {weather_variable}. Error: {e}", exc_info=True)

    logger.info("===== Finished ECMWF Reforecast Download =====")


def update(start_date_str: str, end_date_str: str, force: bool = False, weather_variables: List[str] = None) -> None:
    """
    Downloads update for all configured ECMWF reforecast variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
        weather_variables: List of weather variable names to process. Defaults to all VARIABLES.
    """
    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    dates_to_download = determine_update_dates(
        time_coord="issuance_date",
        data_latency_days=LATENCY,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
        weather_variables=weather_variables,
    )

    if not dates_to_download:
        logger.warning("No new dates to download. Skipping update.")
        return

    start_date, end_date = dates_to_download
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    initial_download(start_date_str, end_date_str, force, weather_variables=weather_variables)


def download_ecmwf_reforecast_date(
    weather_variable: str, date: pd.Timestamp, forecast_type: str, file_path: Path
) -> None:
    tmp_filename = f"_{date.strftime('%Y-%m-%d')}-{forecast_type}-{weather_variable}.nc"
    hindcast_dates = hindcast_dates_for_issuance(date)

    request = build_request(weather_variable, date, forecast_type, tmp_filename, hindcast_dates)

    try:
        ds = fetch_data(request, tmp_filename)
        ds = process_data(ds, weather_variable, forecast_type, date, hindcast_dates)
        save_data(ds, file_path)
    finally:
        clean_tmp(tmp_filename)


def build_request(
    weather_variable: str,
    date: pd.Timestamp,
    forecast_type: str,
    target_filename: str,
    hindcast_dates: List[pd.Timestamp],
) -> Dict[str, Any]:
    param = VARIABLES[weather_variable]["param"]
    date_str = date.strftime("%Y-%m-%d")

    request = {
        "class": "s2",
        "dataset": "s2s",
        "date": f"{date_str}/to/{date_str}",
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": param,
        "stream": "enfh",
        "time": "00:00:00",
        "type": "cf" if forecast_type == "control" else "pf",
        "format": "netcdf",
        "target": target_filename,
    }

    if hindcast_dates:
        request["hdate"] = "/".join(hd.strftime("%Y-%m-%d") for hd in sorted(hindcast_dates))

    if weather_variable == "temperature":
        request["step"] = "/".join([f"{lead_day * 24}-{(lead_day + 1) * 24}" for lead_day in range(0, 26 + 6 + 1)])
    elif weather_variable == "precipitation":
        request["step"] = "/".join([f"{lead_day * 24}" for lead_day in range(0, 26 + 1 + 6 + 1)])
    elif weather_variable == "mean_sea_level_pressure":
        request["step"] = "/".join([f"{lead_day * 24}" for lead_day in range(0, 26 + 6 + 1)])

    if forecast_type == "perturbed":
        member_range = range(1, PERTURBED_MEMBERS + 1)
        request["number"] = "/".join(str(i) for i in member_range)

    return request


def hindcast_dates_for_issuance(issuance_date: pd.Timestamp) -> List[pd.Timestamp]:
    hindcast_dates = [
        (issuance_date - pd.DateOffset(years=years_back)).normalize() for years_back in range(1, HINDCAST_YEARS + 1)
    ]
    return sorted(hindcast_dates)


def _issuance_days_since_1960(issuance_date: pd.Timestamp) -> np.float32:
    return np.float32((issuance_date.normalize() - REFERENCE_DATE).days)


def _hindcast_months_since_1960_offset(
    issuance_date: pd.Timestamp, hindcast_dates: List[pd.Timestamp] | pd.DatetimeIndex
) -> np.ndarray:
    base_date = pd.Timestamp(f"1960-{issuance_date.month:02d}-{issuance_date.day:02d}")
    hindcast_index = pd.DatetimeIndex(hindcast_dates).normalize()
    month_offsets = (hindcast_index.year - base_date.year) * 12 + (hindcast_index.month - base_date.month)
    return (month_offsets + HINDCAST_MONTH_OFFSET).astype(np.float32)


def _coerce_hindcast_index(values: np.ndarray) -> Optional[pd.DatetimeIndex]:
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values).normalize()
    if values.dtype == object:
        try:
            return pd.to_datetime(values).normalize()
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def fetch_data(request: Dict[str, Any], tmp_filename: str) -> xr.Dataset:
    server = ECMWFDataServer()
    server.retrieve(request)
    logger.info(f"Downloaded data to temporary file: {tmp_filename}")
    ds = xr.open_dataset(tmp_filename, chunks={"number": 1})
    return ds


def process_data(
    ds: xr.Dataset,
    weather_variable: str,
    forecast_type: str,
    issuance_date: pd.Timestamp,
    hindcast_dates: List[pd.Timestamp],
) -> xr.Dataset:
    issuance_date = issuance_date.normalize()
    ds = ds.copy()
    issuance_days = _issuance_days_since_1960(issuance_date)
    ds.coords["issuance_date"] = xr.DataArray([issuance_days], dims=["issuance_date"], name="issuance_date")

    ds = ensure_hindcast_dimension(ds, issuance_date, hindcast_dates)

    if "time" not in ds.dims:
        if "step" in ds.dims:
            ds = ds.rename({"step": "time"})
        else:
            raise ValueError("Lead dimension ('time' or 'step') not found in dataset.")

    if weather_variable in ("temperature", "mean_sea_level_pressure"):
        ds = ds.rolling(time=7).mean()
        ds = ds.isel(time=slice(6, None))
    elif weather_variable == "precipitation":
        ds = ds.sortby("time")
        ds_shifted = ds.shift(time=-7)
        ds = ds_shifted - ds
        ds = ds.dropna(dim="time", how="all")

    lead_len = ds.sizes.get("time", 0)
    if lead_len == 0:
        raise ValueError("No lead times remaining after aggregation.")

    if weather_variable == "temperature":
        lead = np.arange(0.5, 0.5 + lead_len, 1, dtype=np.float32)
    else:
        lead = np.arange(0, lead_len, 1, dtype=np.float32)

    ds = ds.rename({"time": "lead"})
    ds = ds.assign_coords({"lead": pd.to_timedelta(lead, unit="D")})

    id_name = VARIABLES[weather_variable]["id_name"]
    short_name = VARIABLES[weather_variable]["short_name"]
    if id_name in ds.data_vars:
        ds = ds.rename_vars({id_name: short_name})

    if forecast_type == "perturbed":
        if "number" in ds.dims:
            ds = ds.rename({"number": "M"})
        member_count = ds.sizes.get("M")
        if member_count is None:
            ds = ds.expand_dims(M=np.arange(1, PERTURBED_MEMBERS + 1, dtype=np.float32))
        else:
            ds = ds.assign_coords(M=("M", np.arange(1, member_count + 1, dtype=np.float32)))
    else:
        for dim in ("number", "M"):
            if dim in ds.dims:
                ds = ds.squeeze(dim, drop=True)

    if "hindcast_date" in ds.coords:
        ds = ds.sortby("hindcast_date")
        ds.coords["hindcast_date"] = ds["hindcast_date"].astype("float32")

    ds = ds.sortby("latitude")
    ds = ds.astype("float32")
    ds[short_name] = ds[short_name].expand_dims(issuance_date=ds["issuance_date"])

    desired_dims = ["hindcast_date", "lead"]
    if "M" in ds.dims:
        desired_dims.append("M")
    desired_dims.append("issuance_date")
    desired_dims.extend(["latitude", "longitude"])
    ds = ds.transpose(*desired_dims)

    logger.info(f"Processed dataset for {weather_variable} (reforecast {forecast_type}) with shape {ds.sizes}")
    return ds


def ensure_hindcast_dimension(
    ds: xr.Dataset, issuance_date: pd.Timestamp, hindcast_dates: List[pd.Timestamp]
) -> xr.Dataset:
    for candidate in ("hdate", "hindcast_date"):
        if candidate in ds.dims or candidate in ds.coords:
            rename_map = {candidate: "hindcast_date"} if candidate != "hindcast_date" else {}
            ds = ds.rename(rename_map)
            hindcast_index = _coerce_hindcast_index(ds["hindcast_date"].values)
            if hindcast_index is None:
                if not hindcast_dates:
                    raise ValueError("Hindcast date coordinate not parseable and no hindcast dates provided.")
                hindcast_index = pd.DatetimeIndex(hindcast_dates).normalize()
            hindcast_months = _hindcast_months_since_1960_offset(issuance_date, hindcast_index)
            ds = ds.assign_coords({"hindcast_date": ("hindcast_date", hindcast_months)})
            ds = ds.sortby("hindcast_date")
            return ds

    if hindcast_dates:
        return reshape_time_for_hindcasts(ds, issuance_date, hindcast_dates)

    raise ValueError("Hindcast date coordinate not found in reforecast dataset.")


def reshape_time_for_hindcasts(
    ds: xr.Dataset, issuance_date: pd.Timestamp, hindcast_dates: List[pd.Timestamp]
) -> xr.Dataset:
    if "time" not in ds.dims:
        if "step" in ds.dims:
            ds = ds.rename({"step": "time"})
        else:
            raise ValueError("Cannot reshape hindcasts without a 'time' or 'step' dimension.")

    hindcast_dates_norm = [hd.normalize() for hd in hindcast_dates]
    hindcast_len = len(hindcast_dates_norm)
    time_len = ds.sizes["time"]
    if hindcast_len == 0:
        raise ValueError("No hindcast dates provided for reforecast processing.")
    if time_len % hindcast_len != 0:
        raise ValueError(
            f"Cannot reshape time dimension of length {time_len} into hindcast_date/lead with {hindcast_len} hindcast dates."
        )

    lead_len = time_len // hindcast_len
    hindcast_months = _hindcast_months_since_1960_offset(issuance_date, hindcast_dates_norm)
    hindcast_labels = np.repeat(hindcast_months, lead_len)
    lead_indices = np.tile(np.arange(lead_len, dtype=np.int32), hindcast_len)

    ds = ds.assign_coords(
        hindcast_date=("time", hindcast_labels),
        lead_index=("time", lead_indices),
    )
    ds = ds.set_index(time=["hindcast_date", "lead_index"])
    ds = ds.unstack("time")
    ds = ds.rename({"lead_index": "time"})
    ds = ds.sortby(["hindcast_date", "time"])
    return ds


def save_data(ds: xr.Dataset, file_path: Path) -> None:
    make_directories(file_path.parent)
    ds.to_netcdf(file_path)
    set_file_permissions(file_path)
    logger.info(f"Saved dataset to {file_path}")
    del ds


def clean_tmp(tmp_filename: str) -> None:
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)
    else:
        logger.warning(f"Temporary file {tmp_filename} does not exist, cannot clean up.")


def determine_update_dates(
    time_coord: str = "issuance_date",
    data_latency_days: Optional[int] = None,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    weather_variables: List[str] = None,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determines what date range to download based on existing data and provided arguments.

    Args:
        time_coord: Name of the time coordinate
        data_latency_days: Number of days to lag behind current date
        start_date_str: Optional user-specified start date
        end_date_str: Optional user-specified end date
        weather_variables: List of weather variable names to process. Defaults to all VARIABLES.

    Returns:
        Tuple of (start_date, end_date) timestamps to download, or None if no download needed
    """

    today_lagged = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=data_latency_days)

    if start_date_str and end_date_str:
        start_date = pd.Timestamp(start_date_str, tz="UTC")
        end_date = pd.Timestamp(end_date_str, tz="UTC")
        end_date = min(end_date, today_lagged)

        if end_date < start_date:
            logger.warning(
                f"End date {end_date.strftime('%Y-%m-%d')} is before start date {start_date.strftime('%Y-%m-%d')}"
            )
            return None

        return start_date, end_date

    if start_date_str and not end_date_str:
        start_date = pd.Timestamp(start_date_str, tz="UTC")
        end_date = today_lagged

        if end_date < start_date:
            logger.warning(f"Data not yet available for period starting on {start_date.strftime('%Y-%m-%d')}")
            return None

        return start_date, end_date

    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    last_dates = []
    for weather_variable in weather_variables:
        output_folder = BASE_DIR / f"ecmwf-reforecast-{VARIABLES[weather_variable]['aiwq_name']}"

        if os.path.isdir(output_folder):
            last_date_in_folder = get_last_date_from_filenames_in(output_folder)
            if last_date_in_folder:
                last_dates.append(last_date_in_folder)
            else:
                logger.warning(f"No control files found in {output_folder}; skipping for last date computation.")
        else:
            logger.error(f"The folder '{output_folder}' does not exist. Please rerun script with initial_download.")

    last_date = max(last_dates) if last_dates else None
    if last_date is None:
        logger.warning("No existing ECMWF reforecast data found.")
        return None
    last_date = pd.Timestamp(last_date, tz="UTC")

    if end_date_str and not start_date_str:
        end_date = min(pd.Timestamp(end_date_str, tz="UTC"), today_lagged)
        start_date = last_date + pd.Timedelta(days=1)

        if end_date < start_date:
            logger.warning(
                f"No new dates to download between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
            )
            return None

        return start_date, end_date

    start_date = last_date + pd.Timedelta(days=1)
    end_date = today_lagged

    if end_date < start_date:
        logger.info(f"Dataset already up-to-date, last date is {last_date.strftime('%Y-%m-%d')}")
        return None

    return start_date, end_date


def get_last_date_from_filenames_in(folder_path: Path) -> Optional[datetime]:
    files = [f for f in os.listdir(folder_path) if f.endswith("-control.nc")]

    dates = []
    for file in files:
        try:
            date_str = file.split("-")[0]
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            dates.append(date_obj)
        except ValueError:
            continue

    if dates:
        latest_date = max(dates)
        return latest_date
    else:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-date",
        "-sd",
        help="First date to download in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        "-ed",
        help="Last date to download in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--weather_variables",
        "-wv",
        nargs="+",
        choices=VARIABLES.keys(),
        default=list(VARIABLES.keys()),
        help="Weather variables to download. Defaults to all.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="If true, overwrite any existing date; otherwise, skip download.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    update(start_date_str=args.start_date, end_date_str=args.end_date, force=args.force, weather_variables=args.weather_variables)
