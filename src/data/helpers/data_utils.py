import datetime
import pathlib
from typing import List, Optional, Tuple

import pandas as pd
import xarray as xr

from data.helpers.general_utils import string_to_dt

VALID_FORECAST_DATES = {
    "reforecast": {
        # Reforecasts for a given issuance date are released in advance of the issuance date
        "ecmwf_cy49": (string_to_dt("20241113"), datetime.datetime.today() + datetime.timedelta(days=6), "odd_no_leap"),
        "ecmwf_cy48": (string_to_dt("20230102"), string_to_dt("20241111"), "monday/thursday_no_leap"),
        "ecmwf_cy41-47": (string_to_dt("20150514"), string_to_dt("20221229"), "monday/thursday_no_leap"),
    },
    "forecast": {
        "ecmwf_cy49": (string_to_dt("20241112"), datetime.datetime.today(), "daily"),
        "ecmwf_cy48": (string_to_dt("20230628"), string_to_dt("20241111"), "daily"),
        "ecmwf_cy41-47": (string_to_dt("20150514"), string_to_dt("20230626"), "monday/thursday"),
    },
}

VALID_MEASUREMENT_DATES = {
    "tas": (string_to_dt("19790101"), datetime.datetime.today(), "daily"),
    "pr": (string_to_dt("19790101"), datetime.datetime.today(), "daily"),
    "mslp": (string_to_dt("19790101"), datetime.datetime.today(), "daily"),
}

IRI_GRID_NAMES = {
    "X": "longitude",
    "Y": "latitude",
    "S": "issuance_date",
    "M": "model_run",
    "LA": "lead",
    "L": "lead",
    "hdate": "hindcast_date",
    "T": "target_date",
}

REFORECAST_MODELS = ["ecmwf", "hmcr", "kma"]


def get_measurement_dataset(weather_variable: str, geographic_region: str, print_path: bool = False) -> xr.Dataset:
    """Getter function for aggregated measurement dataset, optionally printing the dataset path.

    Args:
        weather_variable (str): Name of weather variable.
        geographic_region (str): Identifier for geographic region.
        print_path (bool): If true, print path of dataset.

    Returns:
        xarray.Dataset: aggregated measurement dataset.
    """
    dataset_path = pathlib.Path(f"data/datasets/{weather_variable}-{geographic_region}.nc")
    if print_path:
        print(f"Dataset path: {dataset_path}")
    ds = xr.load_dataset(dataset_path)
    return ds


def get_forecast_dataset(
    model: str,
    weather_variable: str,
    lead_times: str,
    geographic_region: str,
    runs_mean: Optional[str] = None,
    control_forecast: Optional[bool] = None,
    reforecast: bool = False,
    print_path: bool = False,
) -> xr.Dataset:
    """Getter function for aggregated forecast dataset, optionally printing the dataset path.

    Args:
        model (str): Model name.
        weather_variable (str): Name of weather variable.
        lead_times (str): Identifier for lead times.
        geographic_region (str): Identifier for geographic region.
        control_forecast (bool): For s2s models, whether to download control or perturbed forecasts.
        reforecast (bool): For select s2s models, whether to download forecasts or reforecasts.
        print_path (bool): If true, print path of dataset.

    Returns:
        xarray.Dataset: aggregated forecast dataset.
    """
    is_s2s_model = model in ["ecmwf", "bom", "cma", "cnrm", "hmcr", "isac", "jma", "kma", "ncep"]
    if is_s2s_model:
        forecast_run = "cf" if control_forecast else "pf"
    else:
        forecast_run = ""

    dataset_path = pathlib.Path(
        f"data/datasets/{model}-{weather_variable}-{lead_times}-{geographic_region}"
        f"-{forecast_run}{'-reforecast' if reforecast else ''}{f'-{runs_mean}' if runs_mean else ''}.nc"
    )

    if print_path:
        print(f"Dataset path: {dataset_path}")
    ds = xr.load_dataset(dataset_path)
    return ds

def get_valid_forecast_dates(
    model: str, forecast_type: str = "forecast", 
    start_on: Optional[str] = None
) -> List[datetime.datetime]:
    """Returns a list of valid forecast dates for a given model and forecast type.

    Args:
        model (str): Model name.
        forecast_type (str): Type of forecast, either 'reforecast' or 'forecast'.
        start_on (str, optional): If provided, only return forecast dates on or after this date (inclusive). 
          Should be in "YYYYMMDD" format.

    Returns:
        List[datetime.datetime]: List of valid forecast dates.
    """
    try:
        start_date, end_date, date_frequency = VALID_FORECAST_DATES[forecast_type][model]
    except KeyError:
        raise ValueError(f"Invalid model '{model}' or forecast type '{forecast_type}'.")

    if start_on:
        start_on_date = string_to_dt(start_on)
        if start_on_date > end_date:
            raise ValueError(f"Provided 'start_on' date {start_on} is after the end date of valid forecasts.")
        start_date = max(start_date, start_on_date)

    return generate_dates_in_between(start_date, end_date, date_frequency) 

def generate_dates_in_between(
    first_date: datetime.datetime, last_date: datetime.datetime, date_frequency: str
) -> List[datetime.datetime]:

    if date_frequency == "monday/thursday":
        dates = [
            date
            for date in generate_dates_in_between(first_date, last_date, "daily")
            if date.strftime("%A") in ["Monday", "Thursday"]
        ]
        return dates
    elif date_frequency == "monday/thursday_no_leap":
        # Mondays and Thursdays except for Feb. 29
        dates = [
            date
            for date in generate_dates_in_between(first_date, last_date, "daily")
            if date.strftime("%A") in ["Monday", "Thursday"] and not (date.day == 29 and date.month == 2)
        ]
        return dates
    elif date_frequency == "odd_no_leap":
        # Odd days except for Feb. 29
        dates = [date for date in generate_dates_in_between(first_date, last_date, "daily") 
                 if (date.day % 2 == 1) and not (date.day == 29 and date.month == 2)]
        return dates
    elif date_frequency == "ends_in_1_6":
        dates = [date for date in generate_dates_in_between(first_date, last_date, "daily") if date.day % 10 in [1, 6]]
        return dates
    elif date_frequency == "in_1_15":
        dates = [date for date in generate_dates_in_between(first_date, last_date, "daily") if date.day in [1, 15]]
        return dates
    elif date_frequency == "weekly_hmcr":
        dates_wedn = []
        dates_thur = []
        if first_date < string_to_dt("20170531"):
            dates_wedn = [
                first_date + datetime.timedelta(days=x * 7)
                for x in range(0, int((string_to_dt("20170531") - first_date).days / 7) + 1)
            ]
        if last_date > string_to_dt("20170608"):
            dates_thur = [
                string_to_dt("20170608") + datetime.timedelta(days=x * 7)
                for x in range(0, int((last_date - string_to_dt("20170608")).days / 7) + 1)
            ]
        return dates_wedn + dates_thur
    else:
        frequency_to_int = {"daily": 1, "weekly": 7}
        dates = [
            first_date + datetime.timedelta(days=x * frequency_to_int[date_frequency])
            for x in range(
                0,
                int((last_date - first_date).days / (frequency_to_int[date_frequency])) + 1,
            )
        ]
        return dates


def is_valid_forecast_date(model: str, forecast_type: str, forecast_date: datetime.datetime) -> bool:
    assert isinstance(forecast_date, datetime.datetime)

    # # Treat s2s reforecast as a hindcast
    # if model in ["ecmwf", "bom", "cma", "cnrm", "hmcr", "isac", "jma", "kma", "ncep"] and forecast_type == "reforecast":
    #     forecast_type = "hindcast"
    try:
        return forecast_date in generate_dates_in_between(*VALID_FORECAST_DATES[forecast_type][model])
    except KeyError:
        return False


def is_valid_measurement_date(measurement: str, measurement_date: datetime.datetime) -> bool:
    assert isinstance(measurement_date, datetime.datetime)

    try:
        return measurement_date in generate_dates_in_between(*VALID_MEASUREMENT_DATES[measurement])
    except KeyError:
        return False


def get_grid(region_id: str) -> Tuple[List[str], List[str], str]:
    if region_id == "global1_5":
        longitudes = ["0", "358.5"]
        latitudes = ["-90.0", "90.0"]
        grid_size = "1.5"
    elif region_id == "global0_5":
        longitudes = ["0.25", "359.75"]
        latitudes = ["-89.75", "89.75"]
        grid_size = "0.5"
    elif region_id == "us1_0":
        # longitudes = ["-125.0", "-67.0"]
        longitudes = ["235.0", "293.0"]
        latitudes = ["25.0", "50.0"]
        grid_size = "1.0"
    elif region_id == "us1_5":
        # longitudes = ["-123", "-67.5"]
        longitudes = ["237", "292.5"]
        latitudes = ["25.5", "48"]
        grid_size = "1.5"
    else:
        raise NotImplementedError("Only grids global1_5, us1_0 and us1_5 have been implemented.")
    return longitudes, latitudes, grid_size


def get_forecast_type(model: str, forecast_date: datetime.datetime, reforecast: Optional[bool] = None) -> str:
    if reforecast:
        forecast_type = "reforecast"
    else:
        forecast_type = "forecast"
    return forecast_type


def df_contains_nas(file_path: pathlib.Path, column_name: str, how: str = "any") -> bool:
    try:
        df = xr.load_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = load_s2s_reforecast_dataset(file_path).to_dataframe().reset_index()
    nas_in_column = df.isna()[column_name]
    if how == "all":
        return nas_in_column.all()
    elif how == "any":
        return nas_in_column.any()
    else:
        raise NotImplementedError("Flag 'how' must receive 'any' or 'all'.")


def df_contains_multiple_dates(file_path: pathlib.Path, time_col: str = IRI_GRID_NAMES["S"]) -> bool:
    try:
        df = xr.load_dataset(file_path).to_dataframe().reset_index()
    except ValueError:
        df = load_s2s_reforecast_dataset(file_path).to_dataframe().reset_index()
    return len(df[time_col].unique()) > 1


def load_s2s_reforecast_dataset(file_path: pathlib.Path) -> xr.Dataset:
    """Loads an s2s reforecast nc file. It is not possible to use xr.load_dataset() because the encoding
    for the hindcast date (hdate) is given as months since 1960-01-01, which is not standard for xarray.
    Drops all hdates with no valid data and sorts latitudes in an increasing order.
    """

    # Note: if decode_timedelta = False, leads will be floats rather than timedelta64[ns]
    ds = xr.load_dataset(file_path, decode_times=False, decode_timedelta=True)
    # Drop hdates with no valid data
    ds = ds.dropna(dim=IRI_GRID_NAMES["hdate"], how="all")
    ds[IRI_GRID_NAMES["S"]] = pd.to_datetime(
        ds[IRI_GRID_NAMES["S"]].values, unit="D", origin=pd.Timestamp("1960-01-01")
    )

    model_issuance_day = ds[f'{IRI_GRID_NAMES["S"]}.day'].values[0]
    model_issuance_month = ds[f'{IRI_GRID_NAMES["S"]}.month'].values[0]
    model_issuance_date_in_1960 = pd.Timestamp(f"1960-{model_issuance_month}-{model_issuance_day}")
    # While hdates refer to years (for which the model issued in ds["S"] is initialized), its values are given as
    # months until the middle of the year, so 6 months are subtracted to yield the beginning of the year.
    ds[IRI_GRID_NAMES["hdate"]] = pd.to_datetime(
        [model_issuance_date_in_1960 + pd.DateOffset(months=x - 6) for x in ds[IRI_GRID_NAMES["hdate"]].values]
    )
    # Sort latitudes in increasing order
    ds = ds.sortby(ds[IRI_GRID_NAMES["Y"]])

    return ds