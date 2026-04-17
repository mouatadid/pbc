import numpy as np
import pandas as pd
import netCDF4
from models.utils.downloader_utils import get_local_file_path


def printf(str):
    """Print messages in real time.

    Calls print on given argument and then flushes
    stdout buffer to ensure printed message is displayed right away
    """
    print(str, flush=True)


def load_measurement(file_name, mask_df=None, shift=None):
    """Load measurement data from a given file name.

    Parameters
    ----------
    file_name: string
        Name of HDF5 file from which measurement data will be loaded.

    mask_df: pd.DataFrame, optional (default=None)
        Mask to use for filtering the data. Columns of dataframe should be lat, lon, and mask,
        where mask is a {0,1} variable indicating whether the grid point should be included (1) or excluded (0).

    shift: int, optional (default=None)
        Number of days by which ground truth measurements should be shifted forward.
        The date index will be extended upon shifting.

    Returns:
    -------
    measurement_df: pd.DataFrame
        Measurement data as a dataframe.
    """
    # Load ground-truth data
    df = pd.read_hdf(file_name, 'data')

    # Convert to dataframe if necessary
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    # Replace multiindex with start_date, lat, lon columns if necessary
    if isinstance(df.index, pd.MultiIndex):
        df.reset_index(inplace=True)
    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        df = subsetmask(df, mask_df)

    # Return dataframe with desired shift
    return shift_df(df, shift=shift, date_col='start_date', groupby_cols=['lat', 'lon'])


def print_missing_cols_func(df, target_date_obj, print_missing_cols):
    """Print missing columns for target_date_obj."""
    if print_missing_cols is True:
        missing_cols_in_target_date = df.loc[df["start_date"] == target_date_obj].isnull(
        ).any()
        if sum(missing_cols_in_target_date) > 0:
            printf("")
            printf("There is missing data for target_date. The following variables are missing: {}"
                   .format(df.columns[missing_cols_in_target_date].tolist()))
            printf("")


def subsetmask(df, mask_df):
    """Subsets df to rows with lat,lon pairs included in both df and mask_df.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with columns 'lat' and 'lon'.

    mask_df: pd.DataFrame
        Mask to use for filtering the data. Columns of dataframe should be lat, lon, and mask,
        where mask is a {0,1} variable indicating whether the grid point should be included (1) or excluded (0).
        Masks can be created using :func:`subseasonal_data.utils.subsetmask`.

    Returns:
    -------
    masked_df: pd.DataFrame
        Subsetted dataframe.
    """
    return pd.merge(df, mask_df, on=['lat', 'lon'], how='inner')


def df_merge(left, right, on=["lat", "lon", "start_date"], how="outer"):
    """Returns merger of pandas dataframes left and right on 'on' with merge type determined by 'how'.

    If left == None, simply returns right.
    """
    if left is None:
        return right
    else:
        return pd.merge(left, right, on=on, how=how)


def shift_df(df, shift=None, date_col='start_date', groupby_cols=['lat', 'lon'],
             rename_cols=True):
    """Shift dataframe features by a given amount.

    Return dataframe with all columns save for the date_col and groupby_cols
    shifted forward by a specified number of days within each group.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to shift forward.

    shift: int, optional (default=None)
        Number of days by which ground truth measurements
        should be shifted forward; date index will be extended upon shifting;
        if shift is None or shift == 0, original df is returned, unmodified.

    date_col: string, optional (default='start_date')
        Name of datetime column.

    groupby_cols: list of string, optional (default=['lat', 'lon'])
        If all groupby_cols exist, shifting performed separately on each group.
        Otherwise, shifting performed globally on the dataframe.

    rename_cols: bool, optional (default=True)
        Rename columns to reflect shift.

    Returns:
    -------
    shifted_df: pd.DataFrame
        Shifted data as a dataframe.
    """
    if shift is not None and shift != 0:
        # Get column names of all variables to be shifted
        # If any of groupby_cols+[date_col] do not exist, ignore error
        cols_to_shift = df.columns.drop(
            groupby_cols+[date_col], errors='ignore')
        # Function to shift data frame by shift and extend index

        def shift_grp_df(grp_df): return grp_df[cols_to_shift].set_index(
            grp_df[date_col]).shift(int(shift), freq="D")
        if set(groupby_cols).issubset(df.columns):
            # Shift ground truth measurements for each group
            df = df.groupby(groupby_cols).apply(shift_grp_df).reset_index()
        else:
            # Shift ground truth measurements
            df = shift_grp_df(df).reset_index()
        if rename_cols:
            # Rename variables to reflect shift
            df.rename(columns=dict(
                list(zip(cols_to_shift, [col+"_shift"+str(shift) for col in cols_to_shift]))),
                inplace=True)
    return df


def createmaskdf(mask_file):
    """Create mask dataframe from file.

    Load netCDF4 mask file and creates an equivalent dataframe with columns 'lat' and 'lon'
    and rows corresponding to (lat,lon) combinations with mask value == 1.

    Parameters
    ----------
    mask_file: string
        Name of netCDF4 mask file

    Returns:
    -------
    mask_df: pd.DataFrame
       Dataframe with one row for each (lat,lon) pair with mask value == 1.
    """
    fh = netCDF4.Dataset(mask_file, 'r')
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:] + 360
    mask = fh.variables['mask'][:]
    lon, lat = np.meshgrid(lon, lat)
    mask_df = pd.DataFrame({'lat': lat.flatten(),
                            'lon': lon.flatten(),
                            'mask': mask.data.flatten()})
    # Retain only those entries with a mask value of 1
    mask_df = mask_df.loc[mask_df['mask'] == 1]
    # Drop unnecessary 'mask' column
    return mask_df.drop('mask', axis=1)


def year_slice(df, first_year=None, date_col='start_date'):
    """Return slice of df containing all rows with df[date_col].dt.year >= first_year.

    Returns df if first_year is None.
    """
    if first_year is None:
        return df
    if first_year <= df[date_col].min().year:
        # No need to slice
        return df
    return df[df[date_col] >= f"{first_year}-01-01"]


def load_forecast_from_file(file_name, mask_df=None):
    """Load forecast data from file and returns as a dataframe.

    Parameters
    ----------
    file_name: string
        Path to HDF5 file containing forecast data.

    mask_df: pd.DataFrame
        Mask to use for filtering the data. Columns of dataframe should be lat, lon, and mask,
        where mask is a {0,1} variable indicating whether the grid point should be included (1) or excluded (0).
        Masks can be created using :func:`subseasonal_data.utils.subsetmask`.

    Returns:
    -------
    forecast_df: pd.DataFrame
        Dataframe with forecast data.
    """
    # Load forecast dataframe
    forecast = pd.read_hdf(file_name)

    # PY37
    if 'start_date' in forecast.columns:
        forecast.start_date = pd.to_datetime(forecast.start_date)
    if 'target_date' in forecast.columns:
        forecast.target_date = pd.to_datetime(forecast.target_date)

    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        forecast = subsetmask(forecast, mask_df)
    return forecast


def get_measurement_variable(gt_id, shift=None):
    """Return measurement variable name for the given ground truth id.

    Parameters
    ----------
    gt_id: ground truth data string accepted by :func:`~subseasonal_data.data_loaders.get_ground_truth`

    shift: int, optional (default=None)
        Number of days by which ground truth measurements should be shifted forward.
        The date index will be extended upon shifting.

    Returns:
    -------
    measurement_var: string
        Measurement variable corresponding to the gt_id.
    """
    suffix = "" if shift is None or shift == 0 else "_shift"+str(shift)
    if '-' in gt_id:
        var = gt_id.split('-', 1)[1]
        return var + suffix
    else:
        raise ValueError("Invalid gt_id " + gt_id)


def get_combined_data_filename(file_id, gt_id, target_horizon, sync=False, allow_write=False):
    """Return path to directory or file of interest.

    Parameters
    ----------
    file_id: string {"lat_lon_date_data", "lat_lon_data", "date_data", "all_data", "all_data_no_NA"}
        Identifier defining the combined data file of interest.

    gt_id: string {"contest_precip", "contest_tmp2m", "us_precip", "us_tmp2m"}
        Ground truth ID.

    target_horizon: string {"34w", "56w"}
        Target horizon for prediction, "34w" and "56w" corresponding
        to 3-4 weeks and 5-6 weeks, respectively.

    sync: bool, optional (default=True)
        Whether to download/sync the source files.

    allow_write: bool, (default=False)
        Whether to give write permissions to all users when syncing files.
        Recommended if working in shared directories. Users must be allowed
        to set permissions.

    Returns:
    -------
    """
    suffix = "feather"
    combined_data_path = get_local_file_path(
        data_subdir="combined_dataframes", fname="{}-{}_{}.{}".format(file_id, gt_id,
                                                                      target_horizon, suffix),
        sync=sync, allow_write=allow_write)
    return combined_data_path
