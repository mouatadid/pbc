from ecmwfapi import ECMWFDataServer
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import os
from typing import Dict, Any, Optional, Tuple, List

from utils.logging import setup_logging
from utils.file_io import set_file_permissions

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
LATENCY = 2  # Days to lag behind current date for data availability


logger = logging.getLogger(__name__)


def initial_download(start_date_str: str, end_date_str: str, force: bool = False, weather_variables: List[str] = None) -> None:
    """
    Performs download for all configured ECMWF forecast variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
        weather_variables: List of weather variable names to process. Defaults to all VARIABLES.
    """
    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    logger.info(f"===== Starting ECMWF Forecast Download ({start_date_str} to {end_date_str}) =====")

    for weather_variable in weather_variables:
        try:
            date_range = pd.date_range(start=start_date_str, end=end_date_str, freq="D")
            for date in date_range:
                logger.info(f"Processing {weather_variable} for date {date.strftime('%Y-%m-%d')}")
                for forecast_type in ["control", "perturbed"]:
                    file_path = (
                        BASE_DIR
                        / f"ecmwf_cy49-forecast-{VARIABLES[weather_variable]['aiwq_name']}"
                        / f"{date.strftime('%Y%m%d')}-{forecast_type}.nc"
                    )

                    if force or not file_path.exists():
                        download_ecmwf_forecast_date(
                            weather_variable=weather_variable,
                            date=date,
                            forecast_type=forecast_type,
                            file_path=file_path,
                        )
                    else:
                        logger.warning(f"Skipping existing file {file_path}. Use --force to overwrite.")

        except Exception as e:
            logger.error(f"Failed download for {weather_variable}. Error: {e}", exc_info=True)

    logger.info("===== Finished ECMWF Forecast Download =====")


def update(start_date_str: str, end_date_str: str, force: bool = False, weather_variables: List[str] = None) -> None:
    """
    Downloads update for all configured ECMWF forecast variables.

    Args:
        start_date_str: Start date in YYYY-MM-DD format
        end_date_str: End date in YYYY-MM-DD format
        force: If True, overwrite existing datasets. If False, error when dataset exists.
        weather_variables: List of weather variable names to process. Defaults to all VARIABLES.
    """
    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    # Determine what dates to download based on arguments
    dates_to_download = determine_update_dates(
        time_coord="time", 
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

    # Treat update as initial_download
    initial_download(start_date_str, end_date_str, force, weather_variables=weather_variables)


def download_ecmwf_forecast_date(
        weather_variable: str, date: pd.Timestamp, forecast_type: str, file_path: str
) -> None:
    tmp_filename = f"_{date.strftime('%Y-%m-%d')}-{forecast_type}-{weather_variable}.nc"

    request = build_request(weather_variable, date, forecast_type, tmp_filename)
    ds = fetch_data(request, tmp_filename)
    ds = process_data(ds, weather_variable, forecast_type)
    save_data(ds, file_path)
    clean_tmp(tmp_filename)


def build_request(
    weather_variable: str, date: pd.Timestamp, forecast_type: str, target_filename: str
) -> Dict[str, Any]:
    param = VARIABLES[weather_variable]["param"]
    date = date.strftime("%Y-%m-%d")

    request = {
        "class": "s2",
        "dataset": "s2s",
        "date": f"{date}/to/{date}",
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": param,
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf" if forecast_type == "control" else "pf",
        "format": "netcdf",
        "target": target_filename,
    }

    if weather_variable == "temperature":
        request["step"] = "/".join([f"{lead_day * 24}-{(lead_day + 1) * 24}" for lead_day in range(0, 26 + 6 + 1)])
    elif weather_variable == "precipitation":
        request["step"] = "/".join([f"{lead_day * 24}" for lead_day in range(0, 26 + 1 + 6 + 1)])
    elif weather_variable == "mean_sea_level_pressure":
        request["step"] = "/".join([f"{lead_day * 24}" for lead_day in range(0, 26 + 6 + 1)])

    if forecast_type == "perturbed":
        request["number"] = "/".join(str(i) for i in range(1, 100 + 1))  # Perturbed members 1-100.

    return request


def fetch_data(request: Dict[str, Any], tmp_filename: str) -> xr.Dataset:
    server = ECMWFDataServer()
    server.retrieve(request)
    logger.info(f"Downloaded data to temporary file: {tmp_filename}")
    ds = xr.open_dataset(tmp_filename, chunks={"number": 1})
    return ds


def process_data(ds: xr.Dataset, weather_variable: str, forecast_type: str) -> xr.Dataset:
    # Include issuance date as a coordinate
    ds.coords["issuance_date"] = xr.DataArray([ds["time"].values[0]], dims=["issuance_date"], name="issuance_date")

    if weather_variable == "temperature" or weather_variable == "mean_sea_level_pressure":
        # Average over 7 days
        ds = ds.rolling(time=7).mean()
        # Drop the first 6 time steps since aggregation is left-aligned
        ds = ds.isel(time=slice(6, None))

    elif weather_variable == "precipitation":
        # Weekly accumulated precipitation: day 8 - day 1, day 9 - day 2, etc.
        ds = ds.sortby("time")
        ds_shifted = ds.shift(time=-7)
        ds = ds_shifted - ds
        ds = ds.dropna(dim="time", how="all")

    # Add leads
    if weather_variable == "temperature":
        lead = np.arange(0.5, 27, 1, dtype=np.float32)
    else:
        lead = np.arange(0, 27, 1, dtype=np.float32)
    ds = ds.rename({"time": "lead"})
    ds = ds.assign_coords({"lead": lead})

    # Rename variables to match AIWQ conventions
    id_name = VARIABLES[weather_variable]["id_name"]
    short_name = VARIABLES[weather_variable]["short_name"]
    ds = ds.rename_vars({id_name: short_name})

    if forecast_type == "perturbed":
        ds = ds.rename({"number": "M"})
        ds.coords['M'] = ds.coords['M'].astype('float32')

    # Sort latitude in increasing order
    ds = ds.sortby("latitude")

    # Use float32 for data variables
    ds = ds.astype('float32')
    # Encode lead as timedelta
    lead_delta =  pd.to_timedelta(ds.coords['lead'].values, unit='days')
    ds = ds.assign_coords(lead=lead_delta)
    # Ensure data variable is indexed by issuance_date
    ds[short_name] = ds[short_name].expand_dims(issuance_date=ds["issuance_date"])

    logger.info(f"Processed dataset for {weather_variable} with shape {ds.sizes}")
    return ds


def save_data(ds: xr.Dataset, file_path: str) -> None:
    os.makedirs(file_path.parent, exist_ok=True)
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
    time_coord: str = "time",
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
    if weather_variables is None:
        weather_variables = list(VARIABLES.keys())

    last_dates = []
    for weather_variable in weather_variables:
        output_folder = BASE_DIR / f"ecmwf_cy49-forecast-{VARIABLES[weather_variable]['aiwq_name']}"

        if os.path.isdir(output_folder):
            last_dates.append(get_last_date_from_filenames_in(output_folder))
        else:
            logger.error(f"The folder '{output_folder}' does not exist. Please rerun script with initial_download.")

    last_date = max(last_dates) if last_dates else None
    last_date = pd.Timestamp(last_date, tz="UTC")

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


def get_last_date_from_filenames_in(folder_path: Path) -> Optional[datetime]:
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith("-control.nc")]

    # Extract date part from filenames and convert to datetime
    dates = []
    for file in files:
        try:
            date_str = file.split("-")[0]  # Get the date part (e.g., '20000101')
            date_obj = datetime.strptime(date_str, "%Y%m%d")  # Convert to datetime object
            dates.append(date_obj)
        except ValueError:
            # If the filename doesn't match the expected format, ignore it
            continue

    # Find the latest date
    if dates:
        latest_date = max(dates)
        return latest_date
    else:
        return None  # Return None if no valid dates are found


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

    # Setup Logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    update(start_date_str=args.start_date, end_date_str=args.end_date, force=args.force, weather_variables=args.weather_variables)
