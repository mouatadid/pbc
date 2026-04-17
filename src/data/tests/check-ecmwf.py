import argparse
import logging
import io
import pandas as pd
import xarray as xr
from pathlib import Path
from utils.logging import setup_logging

BASE_DIR = Path("data/ecmwf/")
BASE_DIR_BACKUP = Path("data/ecmwf-backup/")
VARIABLES = {
    "temperature": {
        "short_name": "2t",
        "aiwq_name": "tas",
        "tolerance": 1e-3,
    },
    "precipitation": {
        "short_name": "tp",
        "aiwq_name": "pr",
        "tolerance": 1e-1,
    },
    "mean_sea_level_pressure": {
        "short_name": "msl",
        "aiwq_name": "mslp",
        "tolerance": 5e-1,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start-date",
    "-sd",
    help="First date to check in YYYY-MM-DD format",
)
parser.add_argument(
    "--end-date",
    "-ed",
    help="Last date to check in YYYY-MM-DD format",
)
parser.add_argument(
    "-la",
    "--log_alert",
    action="store_true",
    help="Log alerts to file whenever there is a data mismatch (but not for missing files)",
)
parser.add_argument(
    "--weather_variables",
    "-wv",
    nargs="+",
    choices=VARIABLES.keys(),
    default=list(VARIABLES.keys()),
    help="Weather variables to check. Defaults to all.",
)
args = parser.parse_args()

date_range_str = args.start_date if args.start_date == args.end_date else f"{args.start_date}-{args.end_date}"
alert_stream = io.StringIO() if args.log_alert else None
setup_logging(level=logging.INFO, log_stream=alert_stream)
logger = logging.getLogger(__name__)

date_range = pd.date_range(start=args.start_date, end=args.end_date, freq="D")
missing_dates = set()
data_mismatch_found = False

for date in date_range:
    for weather_variable in args.weather_variables:
        for forecast_type in ["control", "perturbed"]:
            subfolder = f"ecmwf_cy49-forecast-{VARIABLES[weather_variable]['aiwq_name']}"
            filename = f"{date.strftime('%Y%m%d')}-{forecast_type}.nc"
            filepath = BASE_DIR / subfolder / filename
            filepath_backup = BASE_DIR_BACKUP / subfolder / filename

            try:
                ds = xr.open_dataset(filepath, decode_timedelta=True)
                ds_backup = xr.open_dataset(filepath_backup, decode_timedelta=True)
            except FileNotFoundError as e:
                missing_dates.add(date)
                logger.warning(
                    f"File not found: {e.filename}. Skipping date {date.strftime('%Y-%m-%d')} for variable {weather_variable}."
                )
                continue

            tolerance = VARIABLES[weather_variable]["tolerance"]
            var_short_name = VARIABLES[weather_variable]["short_name"]
            max_diff = (ds[var_short_name] - ds_backup[var_short_name]).max().item()
            if abs(max_diff) > tolerance:
                data_mismatch_found = True
                logger.error(
                    f"Data mismatch for {weather_variable} on {date.strftime('%Y-%m-%d')}: "
                    f"max difference {max_diff:.4f} (original: {filepath}, backup: {filepath_backup})"
                )
            else:
                logger.debug(
                    f"Data match for {weather_variable} on {date.strftime('%Y-%m-%d')}: "
                    f"max difference {max_diff:.4f} (original: {filepath}, backup: {filepath_backup})"
                )

if not data_mismatch_found:
    logger.info(f"No data mismatches found for the specified date range: {date_range_str}.")
if missing_dates:
    logger.warning(
        f"Missing files for the following dates: {', '.join(date.strftime('%Y-%m-%d') for date in missing_dates)}."
    )

if args.log_alert and data_mismatch_found:
    alert_file_path = Path(f"data/ALERT-check-ecmwf-{date_range_str}.err")
    with open(alert_file_path, "w") as f:
        f.write(alert_stream.getvalue())
    logger.info(f"Log output with errors saved to {alert_file_path}")

logging.shutdown()
