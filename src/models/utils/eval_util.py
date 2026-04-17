# Utility functions supporting evaluation
from datetime import datetime, timedelta
import os
import pandas as pd
from .models_util import get_selected_submodel_name
from .general_util import string_to_dt
from .experiments_util import pandas2hdf
from datetime import datetime
import calendar


def mean_rmse_to_score(mean_rmse):
    """Returns frii contest score associated with a mean RMSE value
    """
    return 100 / (0.1 * mean_rmse + 1)


def score_to_mean_rmse(score):
    """Returns mean RMSE value associated with an frii contest score
    """
    return ((100 / score) - 1) / 0.1


def first_day_of_week(year, day_of_week):
    """Returns the datetime of the first Monday, Tuesday, ..., or Sunday in a given
    year.

    Args:
      year: integer representing year
      day_of_week: string specifying target day of the week in
        {"Monday", ..., "Sunday"}
    """
    # Map day name to Pandas integer
    names_to_ints = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    dt = datetime(year=year, month=1, day=1)
    return dt + timedelta(days=(names_to_ints[day_of_week] - dt.weekday()) % 7)


def get_named_targets():
    """Return a list of named target date ranges"""
    return ["std_test", "std_future", "std_tune", "std_flood"]


def get_target_dates(date_str="std_test", horizon=None):
    """Return list of target date datetime objects for model evaluation.
        Note: returned object should always be a list, even for single target dates

    Args:
    ----------
    date_str : string
        Either a named set of target dates (
        "std_test" Mondays and Fridays from 2016-2024 inclusive,
        "std_tune" daily in 2013-2026 inclusive,
        "std_future" daily in 2025-2026 inclusive,
        "std_flood" daily in 2022-2025 inclusive,
        "std_fuxi" daily from 2017-2021 inclusive,
        "std_fuxi_forecast" bi-weekly (with varying days of week) from 2017-2021 inclusive,
        "std_fuxi_ecmwf" dates for which both fuxi and pbc_ecmwf forecasts are available,
        "std_msn" MSN forecast and reforecast dates in 2016-2024 inclusive,
        "std_msn_forecast" MSN forecast dates corresponding to issuance dates in 2024,
        "std_aifs_forecast" AIFS forecast target dates in 2025,
        "std_contest" Mondays from 20250814-20250817 inclusive as in the AIWQ contest,
        or a string with comma-separated dates in YYYYMMDD format
        (e.g., '20170101,20170102,20180309'))
    horizon: string, Either "19" or "26". Used for generated contest date periods.

    Returns:
    -------
    list
        List with datetime objects
    """
    if date_str == "std_test":
        first_year = 2016
        number_of_days = (365 * 6) + (366 * 3)
    elif date_str == "std_flood":
        first_year = 2022
        number_of_days = (365 * 3) + 366  # 2022-2025, 2024 is leap
    elif date_str == "std_future":
        first_year = 2025
        number_of_days = (365 * 2)
    elif date_str == "std_tune":
        ###
        first_year = 2013
        number_of_days = (365 * 11) + (366 * 3)
        ###
        # # First identify the ECMWF issuance dates
        # # Include Monday and Thursday dates between 1/1/2012 and 6/27/2023 inclusive
        # issuance_dates = pd.date_range(start='2012-01-01', end='2023-06-27', freq='W-MON')
        # issuance_dates = issuance_dates.union(
        #     pd.date_range(start='2012-01-01', end='2023-06-27', freq='W-THU'))
        # # Include every day between 6/28/2023 and 12/31/2026 inclusive
        # issuance_dates = issuance_dates.union(
        #     pd.date_range(start='2023-06-28', end='2026-12-31', freq='D')
        # ).to_pydatetime()
        # # Add horizon offset to get target dates
        # target_dates = [timedelta(days=int(horizon)-1) + date for date in issuance_dates]
        # # Filter out any target dates outside of 2013-2026 inclusive
        # return [d for d in target_dates if d.year in range(2013, 2026+1)]
    elif date_str == "std_fuxi":
        ###
        first_year = 2017
        number_of_days = (365 * 4) + (366 * 1)
    elif date_str == "std_fuxi_forecast":
        # Include Tuesday and Friday dates between 20170101 to 20171229 inclusive
        issuance_dates = pd.date_range(start='2017-01-01', end='2017-12-29', freq='W-TUE')
        issuance_dates = issuance_dates.union(pd.date_range(start='2017-01-01', end='2017-12-29', freq='W-FRI'))
        # Include Wednesday and Saturday dates between 20180103 to 20181229 inclusive
        issuance_dates = issuance_dates.union(pd.date_range(start='2018-01-03', end='2018-12-29', freq='W-WED'))
        issuance_dates = issuance_dates.union(pd.date_range(start='2018-01-03', end='2018-12-29', freq='W-SAT'))
        # Include Thursday and Sunday dates between 20190103 to 20191229 inclusive
        issuance_dates = issuance_dates.union(pd.date_range(start='2019-01-03', end='2019-12-29', freq='W-THU'))
        issuance_dates = issuance_dates.union(pd.date_range(start='2019-01-03', end='2019-12-29', freq='W-SUN'))
        # Include Monday and Friday dates between 20200103 to 20200224 inclusive
        issuance_dates = issuance_dates.union(pd.date_range(start='2020-01-03', end='2020-02-24', freq='W-MON'))
        issuance_dates = issuance_dates.union(pd.date_range(start='2020-01-03', end='2020-02-24', freq='W-FRI'))
        # Include Tuesday and Saturday dates between 20200229 to 20201229 inclusive
        issuance_dates = issuance_dates.union(pd.date_range(start='2020-02-29', end='2020-12-29', freq='W-TUE'))
        issuance_dates = issuance_dates.union(pd.date_range(start='2020-02-29', end='2020-12-29', freq='W-SAT'))
        # Include Wednesday and Sunday dates between 20210103 to 20211226 inclusive
        issuance_dates = issuance_dates.union(pd.date_range(start='2021-01-03', end='2021-12-31', freq='W-WED')) #26
        issuance_dates = issuance_dates.union(pd.date_range(start='2021-01-03', end='2021-12-31', freq='W-SUN'))  
        issuance_dates = issuance_dates.to_pydatetime()
        # Create target dates
        target_dates = [timedelta(days=int(horizon)-1) + date for date in issuance_dates]
        # target_dates = [date.to_pydatetime() for date in target_dates]
        return target_dates   
    elif date_str == "std_fuxi_ecmwf":
        target_date_strs = ['20190128', '20190204', '20190211', '20190218', '20190225', '20190304', '20190311', '20190318', '20190325', '20190401', '20190408', '20190415', '20190422', '20190429', '20190506', '20190513', '20190520', '20190527', '20190603', '20190610', '20190617', '20190624', '20190701', '20190708', '20190715', '20190722', '20190729', '20190805', '20190812', '20190819', '20190826', '20190902', '20190909', '20190916', '20190923', '20190930', '20191007', '20191014', '20191021', '20191028', '20191104', '20191111', '20191118', '20191125', '20191202', '20191209', '20191216', '20191223', '20191230', '20200106', '20200113', '20200131', '20200207', '20200214', '20200221', '20200228', '20200306', '20200313']
        if horizon == 19:
            target_date_strs += ['20190121', '20200124']
        elif horizon == 26:
            target_date_strs += ['20200120', '20200320']
        return [datetime.strptime(d, '%Y%m%d') for d in sorted(target_date_strs)]
    elif date_str == "std_msn":
        # First identify the MSN issuance dates
        # Include Monday and Thursday dates between 1/1/2024 and 11/10/2024 inclusive
        issuance_dates = pd.date_range(start='2024-01-01', end='2024-11-10', freq='W-MON')
        issuance_dates = issuance_dates.union(
            pd.date_range(start='2024-01-01', end='2024-11-10', freq='W-THU'))
        # Include odd dates between 11/11/2024 and 12/31/2024 inclusive
        issuance_dates = issuance_dates.union(
            pd.date_range(start='2024-11-11', end='2024-12-31', freq='2D')
        ).to_pydatetime()
        # Include all issuance dates in 2016-2024 inclusive with the same 
        # month/day combination as one of the 2024 issuance dates, replace
        # 2/29 with 2/28 in prior years, and add horizon offset to get target dates.
        target_dates = [timedelta(days=int(horizon)-1) + (
            date.replace(year=year, day=28) 
            if (year != 2024) and (date.day == 29) and (date.month == 2)
            else date.replace(year=year))
            for year in range(2015, 2025) for date in issuance_dates]
        # Filter out any target dates outside of 2016-2024 inclusive
        return [d for d in target_dates if d.year in range(2016, 2025)]
    elif date_str == "std_msn_forecast":
        # First identify the MSN issuance dates
        # Include Monday and Thursday dates between 1/1/2024 and 11/10/2024 inclusive
        issuance_dates = pd.date_range(start='2024-01-01', end='2024-11-10', freq='W-MON')
        issuance_dates = issuance_dates.union(
            pd.date_range(start='2024-01-01', end='2024-11-10', freq='W-THU'))
        # Include odd dates between 11/11/2024 and 12/31/2024 inclusive
        issuance_dates = issuance_dates.union(
            pd.date_range(start='2024-11-11', end='2024-12-31', freq='2D')
        ).to_pydatetime()
        # Add horizon offset to get target dates.
        target_dates = [timedelta(days=int(horizon)-1) + date for date in issuance_dates]
        return target_dates
    elif date_str == "std_aifs_forecast":
        # First identify the AIFS issuance dates
        # Include all odd dates in 2025 except for 31sts
        issuance_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        issuance_dates = issuance_dates[(issuance_dates.day % 2 != 0) & (issuance_dates.day != 31)].to_pydatetime()
        # Add horizon offset to get target dates
        target_dates = [timedelta(days=int(horizon)-1) + date for date in issuance_dates]
        return target_dates
    elif date_str == "std_contest":
        # First identify the AIWQ issuance dates
        # Include Thursday issuance dates between 20250814 and 20260806 inclusive
        issuance_dates = pd.date_range(start='2025-08-14', end='2026-08-06', freq='W-THU').to_pydatetime()
        # Add horizon offset to get target dates.
        target_dates = [timedelta(days=int(horizon)-1) + date for date in issuance_dates]
        return target_dates
    elif "," in date_str:
        # Input is a string of the form '20170101,20170102,20180309'
        dates = [datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 6:
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime(year=year+1, month=1, day=1)
        else:
            last_date = datetime(year=year, month=month+1, day=1)
        dates = [
            first_date + timedelta(days=x)
            for x in range(0, (last_date-first_date).days)
        ]
        return dates
    elif len(date_str) == 8:
        # Input is a string of the form '20170101', representing a single target date
        dates = [datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    elif len(date_str) == 4:
        # Input is a string of the form '2017', representing a single year
        first_year = int(date_str)
        number_of_days = 366
    else:
        raise NotImplementedError("Date string provided cannot be transformed "
                                  "into list of target dates.")

    # Return standard set of dates
    first_date = datetime(year=first_year, month=1, day=1)
    dates = [first_date + timedelta(days=x) for x in range(number_of_days)]
    if date_str in list(calendar.month_name):
        dates = [d for d in dates if d.month == list(calendar.month_name).index(date_str)]
    if date_str in ["std_test"]:
        dates = [d for d in dates if d.weekday() in [0, 4]] # Keep only Mondays and Fridays
    return dates


def get_task_metrics_dir(
    model="ecmwfpp", submodel=None, gt_id="era5-tas", horizon="19",
    target_dates=None):
    """Returns the directory in which evaluation metrics for a given submodel
    or model are stored

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns metrics
         directory associated with selected submodel or returns None if no
         submodel selected
       gt_id: e.g., "era5-tas", "era5-pr", "era5-f1_tas"
       horizon: 19 or 26
    """
    if submodel is None:
        submodel = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon,
                                              target_dates=target_dates)
        if submodel is None:
            return None
    return os.path.join('eval', 'metrics', model, 'submodel_forecasts', submodel, gt_id+'_'+horizon)


def get_contest_task_metrics_dir(
    model="spatiotemporal_mean", gt_id="contest_tmp2m", horizon="34w"
):
    """Returns the directory in which evaluation metrics for a given model contest
    prediction are stored

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: 34w or 56w
    """
    return os.path.join(
        "eval", "metrics", model, "contest_forecasts", f"{gt_id}_{horizon}"
    )


def get_metric_filename(
    model="spatiotemporal_mean",
    submodel=None,
    gt_id="contest_tmp2m",
    horizon="34w",
    target_dates="std_test",
    metric="rmse",
):
    """Returns the filename associated with metric data for a given (sub)model and task;
    returns None if submodel=None and model has no selected submodel

    Args:
      model - model name
      submodel - if None, identifies model's selected submodel
      gt_id - "contest_tmp2m" or "contest_precip"
      horizon - "34w" or "56w"
      target_dates - a valid input to get_target_dates
      metric - "rmse", "skill", or "score"
    """
    if submodel is None:
        # Identify the selected submodel name for this model
        submodel = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon,
                                              target_dates=target_dates)
    # If there is no selected submodel, return None
    if submodel is None:
        return None
    # Load and return metric data if it exists
    task = f"{gt_id}_{horizon}"
    return os.path.join(
        "eval",
        "metrics",
        model,
        "submodel_forecasts",
        submodel,
        task,
        f"{metric}-{task}-{target_dates}.h5",
    )


def save_metric(
    data,
    model="spatiotemporal_mean",
    submodel=None,
    gt_id="contest_tmp2m",
    horizon="34w",
    target_dates="std_test",
    metric="rmse",
):
    """Saves metric data to disk

    Args:
      data - metric data to save
      model - model name
      submodel - if None, saves metric data for model (i.e..,
        for the model's selected submodel); otherwise saves metric
        data for requested submodel
      gt_id - "contest_tmp2m" or "contest_precip"
      horizon - "34w" or "56w"
      target_dates - a valid input to get_target_dates
      metric - "rmse", "skill", or "score"
    """
    filename = get_metric_filename(
        model=model, submodel=submodel, gt_id=gt_id, horizon=horizon,
        target_dates=target_dates, metric=metric)
    # If there is no selected submodel, raise exception
    if submodel is None:
        raise ValueError(f"Could not find selected submodel for model {model}")
    # Otherwise save metric data
    pandas2hdf(data, filename)


def load_metric(
    model="spatiotemporal_mean",
    submodel=None,
    gt_id="contest_tmp2m",
    horizon="34w",
    target_dates="std_test",
    metric="rmse",
):
    """Loads metric data stored by src/eval/batch_metrics.py;
    returns None if requested data is unavailable.

    Args:
      model - model name
      submodel - if None, loads metric data stored for model (e.g.,
        for the model's selected submodel); otherwise loads metric
        data for requested submodel
      gt_id - "contest_tmp2m" or "contest_precip"
      horizon - "34w" or "56w"
      target_dates - a valid input to get_target_dates
      metric - "rmse", "skill", or "score"
    """
    filename = get_metric_filename(
        model=model, submodel=submodel, gt_id=gt_id, horizon=horizon,
        target_dates=target_dates, metric=metric)
    if os.path.exists(filename):
        return pd.read_hdf(filename)
    return None


def contest_year(target_date_obj, horizon, dow=1):
    """Returns the contest year associated with a target_date_obj, where
    a contest starts on the last Wednesday in October of a given year for 34w
    two weeks after that for 56w task.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    year = target_date_obj.year
    contest_start, contest_end = contest_start_end(horizon, year, dow)
    if target_date_obj >= contest_start:
        return year
    else:
        return year-1


def contest_start_end(horizon, year=2019, dow=1):
    """Returns the contest start and end date (inclusive) for a given year and horizon,
    where a contest starts on the last <dow> in Octobor of a given year for the 34w task
    and two weeks after that for the 56w task.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    # Find Wednesday in last 7 days of October each year
    for d in range(25, 32):
        date = datetime(year=year, month=10, day=d)
        # if date.weekday() == 1: # Tuesday
        if date.weekday() == dow:
            contest_start = date
            break

    contest_end = contest_start + timedelta(days=364)
    # Remove first or last two-weeks from contest period for 34w and 56w respectively
    if horizon == "12w":
        contest_end -= timedelta(weeks=4)
    elif horizon == "34w":
        contest_end -= timedelta(weeks=2)
    elif horizon == "56w":
        contest_start += timedelta(weeks=2)
    else:
        pass
        # printf(f"Unknown horizon {horizon}. Including full period.")
    return contest_start, contest_end


def contest_quarter_start_dates(horizon, year=2019, dow=1):
    """Returns the start day and month for each contest quarter associated with a
    given year as a list of datetime objects

    Args:
      horizon: "34w" or "56w" indicating contest forecast horizon
      year: year to associate with each quarter start day and month
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    # Get contest start/end
    cs, ce = contest_start_end(horizon, year, dow)

    # Set contest offsets
    q0 = 0  # offset in of q0 start from contest_start, in weeks
    q1 = 14 + q0  # offset of q1 from q0 start, in weeks
    q2 = 12 + q1  # offset of q2 from q1 start, in weeks
    q3 = 12 + q2  # offset of q3 from q2 start, in weeks

    # Get quarter offsets and return
    q_duration = [timedelta(weeks=q0), timedelta(weeks=q1),
                  timedelta(weeks=q2), timedelta(weeks=q3)]
    return [cs + q for q in q_duration]


def contest_quarter(target_date_obj, horizon, dow=1):
    """Returns the frii contest quarter (coded as 0,1,2,3) in which a given
    target datetime object lies.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    yy = contest_year(target_date_obj, horizon, dow)
    quarter_starts = contest_quarter_start_dates(horizon, yy, dow)
    for ii in range(0, 3):
        if (target_date_obj >= quarter_starts[ii]) and (target_date_obj < quarter_starts[ii + 1]):
            return ii
    # Otherwise, date lies in last quarter
    return 3


def year_quarter(target_date_obj):
    """Returns the yearly quarter (coded as 0,1,2,3) in which a given
    target datetime object lies.

    Args:
      target_date_obj: target date as a datetime object
    """
    m = target_date_obj.month
    if m >= 12 or m <= 2:  # December - February
        return 0
    elif m >= 3 and m <= 5:  # March - May
        return 1
    elif m >= 6 and m <= 8:  # June - August
        return 2
    elif m >= 9 and m <= 11:  # September - November
        return 3
    else:
        raise ValueError(f"Invalid month {m}")


def get_region_bounding_box(region=None, lon_type="0,360"):
    
    dic_regions = {'us': {'lon_min':-125, 'lon_max':-67, 'lat_min':26, 'lat_max':50},
                'europe': {'lon_min':-10, 'lon_max':50, 'lat_min':35, 'lat_max':70},
                'east_asia': {'lon_min':90, 'lon_max':145, 'lat_min':20, 'lat_max':50},
                'middle_east': {'lon_min':30, 'lon_max':60, 'lat_min':15, 'lat_max':40},
                'central_america': {'lon_min':-118, 'lon_max':-80, 'lat_min':8, 'lat_max':32},
                'south_america_nh': {'lon_min':-82, 'lon_max':-35, 'lat_min':0, 'lat_max':12},
                'south_america_sh': {'lon_min':-80, 'lon_max':-36, 'lat_min':-56, 'lat_max':0},
                'north_africa': {'lon_min':-18, 'lon_max':45, 'lat_min':0, 'lat_max':40},
                'southern_africa': {'lon_min':10, 'lon_max':40, 'lat_min':-35, 'lat_max':0},
                'australia': {'lon_min':110, 'lon_max':179, 'lat_min':-45, 'lat_max':-10},
                'maritime_continent': {'lon_min':90, 'lon_max':150, 'lat_min':-10, 'lat_max':20},
                'india': {'lon_min':65, 'lon_max':90, 'lat_min':5, 'lat_max':35},
               'northern_hemisphere': {'lon_min':-180, 'lon_max':180, 'lat_min':0, 'lat_max':90},
               'southern_hemisphere': {'lon_min':-180, 'lon_max':180, 'lat_min':-90, 'lat_max':0},
               'global': {'lon_min':-180, 'lon_max':180, 'lat_min':-90, 'lat_max':90}
              }

    # Handle default region
    if region is None:
        region = 'global'

    if region not in dic_regions:
        raise ValueError(f"Region '{region}' not found in dictionary.")

    # Copy to avoid mutating original dictionary
    bbox = dic_regions[region].copy()

    # Convert longitude if needed
    if lon_type == "0,360":
        bbox['lon_min'] = (bbox['lon_min'] + 360) % 360
        bbox['lon_max'] = (bbox['lon_max'] + 360) % 360
    elif lon_type == "-180,180":
        pass  # already in correct format
    else:
        raise ValueError("lon_type must be '0,360' or '-180,180'")

    return bbox

def subset_region(ds, bbox):
    """
    Subset an xarray Dataset/DataArray to a region defined by bbox.
    Handles longitude wrap-around (e.g., Europe in 0–360 coords).
    """

    lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
    lon_min, lon_max = bbox['lon_min'], bbox['lon_max']

    # Latitude slice (always safe)
    ds = ds.sel(latitude=slice(lat_min, lat_max))

    # Longitude handling
    if lon_min <= lon_max:
        ds = ds.sel(longitude=slice(lon_min, lon_max))
    else:
        # Wrap-around case when region crosses 0° (e.g., Europe: 350 → 50)
        ds_left = ds.sel(longitude=slice(lon_min, 360))
        ds_right = ds.sel(longitude=slice(0, lon_max))
        ds = xr.concat([ds_left, ds_right], dim="longitude")

    # Ensure consistent ordering
    ds = ds.sortby(['latitude', 'longitude'])

    return ds
