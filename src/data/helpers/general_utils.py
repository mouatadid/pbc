import datetime
import os
import pathlib
import shutil as sh

# import subprocess
# import sys
import time

# import warnings
from typing import Dict, List, Union

import requests

DATETIME_FORMAT = '%Y%m%d'

def download_url(url: str, timeout: int = 600, retry: int = 30, cookies: Dict[str, str] = {}):
    """Download URL, waiting some time between retries."""
    r = None
    attempt = 0
    for i in range(retry):
        try:
            r = requests.get(url, timeout=timeout, cookies=cookies)
            return r
        except requests.exceptions.Timeout as e:
            # Wait until making another request
            if i == retry - 1:
                raise e
            attempt += 1
            print(f"[Attempt {attempt}] Request to url {url} has timed out after {timeout}s. Trying again...")
            time.sleep(3)
            timeout *= 2  # double the timeout for next try
    print(f"Failed to retrieve file after {retry} attempts. Stopping...")


def get_dates(date_str: str) -> List[datetime.datetime]:
    """Outputs the list of dates corresponding to input date string."""
    if "-" in date_str:
        # Input is of the form '20170101-20180130'
        first_date, last_date = date_str.split("-")
        first_date = string_to_dt(first_date)
        last_date = string_to_dt(last_date)
        dates = [first_date + datetime.timedelta(days=x) for x in range(0, (last_date - first_date).days + 1)]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif "," in date_str:
        # Input is of the form '20170101,20170102,20180309'
        dates = [datetime.datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 4:
        # Input '2017' is expanded to 20170101-20171231
        year = int(date_str)
        first_date = datetime.datetime(year=year, month=1, day=1)
        last_date = datetime.datetime(year=year, month=12, day=31)
        dates = [first_date + datetime.timedelta(days=x) for x in range(0, (last_date - first_date).days + 1)]
        return dates
    elif len(date_str) == 6:
        # Input '201701' is expanded to 20170101-20170131
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime.datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime.datetime(year=year + 1, month=1, day=1)
        else:
            last_date = datetime.datetime(year=year, month=month + 1, day=1)
        dates = [first_date + datetime.timedelta(days=x) for x in range(0, (last_date - first_date).days)]
        return dates
    elif len(date_str) == 8:
        # Input '20170101' is a date
        dates = [datetime.datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    else:
        raise NotImplementedError("Date string provided cannot be transformed " "into list of target dates.")


def print_fail(
    message: str = "FAIL",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in purple."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(f"{string_before}\x1b[1;30;45m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[35m{message}\x1b[0m{string_after}")


def print_error(
    message: str = "ERROR",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in red."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(f"{string_before}\x1b[1;30;41m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[31m{message}\x1b[0m{string_after}")


def print_warning(
    message: str = "WARNING",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in yellow."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(f"{string_before}\x1b[1;30;43m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[33m{message}\x1b[0m{string_after}")


def print_ok(
    message: str = "OK",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in green."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(f"{string_before}\x1b[1;30;42m[ {message} ]\x1b[0m{string_after}")
        else:
            print(f"{string_before}\x1b[32m{message}\x1b[0m{string_after}")


def print_info(
    message: str, verbose: bool = True, skip_line_before: bool = False, skip_line_after: bool = False
) -> None:
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        print(f"{string_before}{message}{string_after}")


def set_path_permission(
    file_path: pathlib.Path, mode: int = 0o777, recursive: bool = True, warning: bool = False
) -> None:
    if recursive:
        recursive_path = ""
        for dir in os.path.normpath(file_path).split(os.path.sep):
            recursive_path = os.path.join(recursive_path, dir)
            set_path_permission(recursive_path, mode=mode, recursive=False, warning=warning)
    else:
        try:
            os.chmod(file_path, mode)
            # sh.chown(file_path, group="sched_mit_hill")
        except PermissionError:
            if warning:
                print_warning(
                    f"Permission error: can't modify path ({file_path})", skip_line_before=False, skip_line_after=False
                )
            pass


def string_to_dt(string: str) -> datetime.datetime:
    """Transforms string to datetime."""
    return datetime.datetime.strptime(string, DATETIME_FORMAT)


def dt_to_string(dt: datetime.datetime) -> str:
    """Transforms datetime to string."""
    return datetime.datetime.strftime(dt, DATETIME_FORMAT)


def get_folder(folder_path: Union[str, pathlib.Path], verbose=True) -> pathlib.Path:
    """Creates folder, if it doesn't exist, and returns folder path.
    Args:
        folder_path (str): Folder path, either existing or to be created.
    Returns:
        str: folder path.
    """
    folder_path = pathlib.Path(folder_path)
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path
