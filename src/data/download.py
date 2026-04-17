# Download or update datasets
#
# Example usages:
#   src/data/download.py update -d aiwq
#   src/data/download.py update -d era5
#   src/batch/batch_python.sh -c 12 -m 50 src/data/download.py update -d era5

import argparse
import inspect
import logging
from datetime import datetime, timedelta, timezone

from utils.logging import setup_logging
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Download and update subseasonal weather datasets.")
    parser.add_argument(
        "action",
        choices=["initial_download", "update"],
        help="Action to perform: 'initial_download' for first run, 'update' for daily checks."
    )
    parser.add_argument(
        "-d", "--datasets",
        nargs='+',
        default=["all"], # Default to process all configured datasets
        help="Specify which datasets to process (e.g., era5 (or aiwq), sst, s2s). Use 'all' for everything."
    )
    parser.add_argument(
        "-sd", "--start-date",
        help="Start date (YYYY-MM-DD) for download. If not provided, defaults to last date in dataset + 1"
    )
    parser.add_argument(
        "-ed", "--end-date",
        help="End date (YYYY-MM-DD) for download. If not provided, defaults to today-lag."
    )
    parser.add_argument(
        "-lf", "--log-file",
        default=None,
        help="Optional path to a log file."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase logging verbosity to DEBUG."
    )
    parser.add_argument(
        "-f", "--force", 
        action="store_true",
        help="Force overwrite of existing datasets during initial download."
    )
    parser.add_argument(
        "-wv", "--weather_variables",
        nargs='+',
        default=None,
        help="Weather variables to process (e.g., temperature precipitation mean_sea_level_pressure). "
             "Defaults to all variables defined in the dataset module."
    )

    args = parser.parse_args()

    # Setup Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    logger = logging.getLogger(__name__)

    logger.info(f"Starting data pipeline action: {args.action}")
    logger.info(f"Processing datasets: {', '.join(args.datasets)}")

    # --- Action Handling ---
    module_prefix = "data.datasets."
    if args.action == "initial_download":
        if not args.start_date:
            parser.error("--start-date is required for initial_download.")
        end_date = args.end_date if args.end_date else (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')

        logger.info(f"Initial download range: {args.start_date} to {end_date}")
        if args.force:
            logger.warning("Force flag set: existing datasets will be overwritten")

        datasets_to_process = args.datasets
        if "all" in datasets_to_process:
            # Define what 'all' means based on implemented downloaders
            datasets_to_process = ["era5", "sst"]

        for dataset in datasets_to_process:
            module_name = f"{module_prefix}{dataset}"
            try:
                module = import_module(module_name)
            except ImportError as e:
                logger.error(f"Dataset '{dataset}' not found: {e}")
                continue

            _call_module_fn(module.initial_download, args.start_date, end_date, args.force, args.weather_variables)

    elif args.action == "update":
        logger.info("Checking for updates...")

        # Process start and end dates if provided
        # When not provided, start date is last date in available data; end date is today-lag
        start_date_str = args.start_date if args.start_date else None
        end_date_str = args.end_date if args.end_date else None
        
        if start_date_str or end_date_str:
            logger.info(f"Update with date range: {start_date_str or 'auto'} to {end_date_str or 'auto'}")
        
        if args.force:
            logger.warning("Force flag set: existing dates will be overwritten")

        datasets_to_process = args.datasets
        if "all" in datasets_to_process:
            datasets_to_process = ["era5", "sst"]

        for dataset in datasets_to_process:
            module_name = f"{module_prefix}{dataset}"
            try:
                module = import_module(module_name)
            except ImportError as e:
                logger.error(f"Dataset '{dataset}' not found: {e}")
                continue

            _call_module_fn(module.update, args.start_date, args.end_date, args.force, args.weather_variables)

    logger.info("Data pipeline action finished.")


def _call_module_fn(fn, start_date, end_date, force, weather_variables):
    """Call a module's download/update function, passing weather_variables only if supported."""
    sig = inspect.signature(fn)
    if "weather_variables" in sig.parameters:
        fn(start_date, end_date, force, weather_variables=weather_variables)
    else:
        fn(start_date, end_date, force)


if __name__ == "__main__":
    main()
