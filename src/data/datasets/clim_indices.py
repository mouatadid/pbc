import datetime as dt
from datetime import datetime, timedelta
import io
import logging
import os
import re
import tempfile
from pathlib import Path
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

from data.helpers.config import load_config # Assuming this path is correct relative to execution
from utils.data_io import get_zarr_store_path, save_to_zarr, check_zarr_exists # Assuming this path

logger = logging.getLogger(__name__)


try:
    CONFIG = load_config()
    BASE_DIR = Path(CONFIG['data_dir'])
    CI_CONFIG = CONFIG.get('sources', {}).get('clim_indices', {})
    DEFAULT_LATENCY_DAYS = CONFIG.get('latency', {}).get('clim_indices', 1)
    MIN_OUTPUT_START_DATE_CFG = pd.Timestamp(CI_CONFIG.get('min_start_date', "1980-01-01"))
except KeyError as e:
    logger.error(f"Configuration missing for clim_indices: {e}. Using defaults.")
    # Provide sensible defaults if config is missing
    BASE_DIR = Path("data")
    CI_CONFIG = {}
    DEFAULT_LATENCY_DAYS = 1
    MIN_OUTPUT_START_DATE_CFG = pd.Timestamp("1980-01-01")


ZARR_STORE_NAME = CI_CONFIG.get('zarr_store_name', "clim_indices")
ZARR_PATH = get_zarr_store_path(BASE_DIR, ZARR_STORE_NAME)


def initial_download(start_date_str: str, end_date_str: str, force: bool = False) -> None:
    """Performs an initial download for Climate Indices data."""
    logger.info(f"===== Starting Climate Indices Initial Download ({start_date_str} to {end_date_str}) =====")

    try:
        if check_zarr_exists(ZARR_PATH):
            if not force:
                logger.error(f"Dataset {ZARR_PATH} already exists. Use --force to overwrite.")
                return
            # else:
                # logger.warning(f"Force flag set. Dataset {ZARR_PATH} will be overwritten.")

        download_clim_indices_range(
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            initial_download=True,
        )
    except Exception as e:
        logger.error(f"Failed initial download for Climate Indices. Error: {e}", exc_info=True)

    logger.info("===== Finished Climate Indices Initial Download =====")


def update(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, force: bool = False) -> None:
    logger.warning("The update function for climate indices simply calls the initial_download function with the provided dates and force=True.")
    try:
        if check_zarr_exists(ZARR_PATH):
            if not force:
                logger.error(f"Dataset {ZARR_PATH} already exists. Use --force to overwrite.")
                return
            # else:
            #     logger.warning(f"Force flag set. Dataset {ZARR_PATH} will be overwritten.")

        download_clim_indices_range(
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            initial_download=True,
        )
    except Exception as e:
        logger.error(f"Failed update for Climate Indices. Error: {e}", exc_info=True)

    logger.info("===== Finished Climate Indices Update =====")


def download_clim_indices_range(
    start_date_str: str,
    end_date_str: str,
    initial_download: bool = False,
) -> None:
    """Downloads and processes climate indices data for a given date range and saves to Zarr."""

    parsed_start_date = pd.Timestamp(start_date_str)
    parsed_end_date = pd.Timestamp(end_date_str)

    # Determine the effective date range for the output Zarr store
    effective_zarr_start_date = max(parsed_start_date, MIN_OUTPUT_START_DATE_CFG)
    effective_zarr_end_date = parsed_end_date

    logger.info(f"--- Processing Climate Indices from {effective_zarr_start_date.strftime('%Y-%m-%d')} to {effective_zarr_end_date.strftime('%Y-%m-%d')} ---")
    logger.info(f"Target Zarr store: {ZARR_PATH}")

    # List of all fetch functions to be called
    fetch_functions = [
        fetch_enso, fetch_mjo_proj, fetch_mjo_romi, fetch_mjo_vpm, fetch_mjo_bom,
        fetch_nao, fetch_pna, fetch_qbo_30, fetch_qbo_50, fetch_iod, fetch_pdo,
        fetch_soi, fetch_mei, fetch_aao, fetch_ao, fetch_npgo, fetch_tpi, fetch_tsa,
        fetch_ea, fetch_sca, fetch_eawr, fetch_wp, fetch_epnp, fetch_wwv,
        fetch_polar_cap_height, fetch_ice_extent
    ]

    all_datasets = []
    for func in fetch_functions:
        try:
            logger.info(f"Fetching data from {func.__name__}...")
            ds = func()
            ds_subset = ds.sel(time=slice(effective_zarr_start_date, effective_zarr_end_date))

            if not ds_subset.time.size:
                logger.warning(f"No data for {func.__name__} in the specified date range. Skipping.")
                continue

            all_datasets.append(ds_subset)
            logger.info(f"Successfully fetched and processed {func.__name__}.")
        except Exception as e:
            logger.error(f"Failed to fetch or process data from {func.__name__}. Error: {e}", exc_info=True)

    if not all_datasets:
        logger.error("No datasets were successfully fetched. Aborting Zarr save.")
        return

    # Merge all collected datasets into a single one
    logger.info("Merging all datasets...")
    merged_ds = xr.merge(all_datasets, compat='override')

    # Save the final dataset to Zarr
    logger.info(f"Saving merged dataset to {ZARR_PATH}...")
    save_to_zarr(merged_ds, ZARR_PATH, mode='w') # Overwrite mode

    logger.info(f"--- Finished processing Climate Indices data. Saved to {ZARR_PATH} ---")


def fetch_enso():
    """
    Fetch the NOAA CPC Oceanic Niño Index (ONI) ASCII table and return an
    xarray.Dataset with:
      - time: daily DateTimeIndex spanning each 3-month season
      - enso_total: the 3-month mean SST for each day in that window
      - enso_anom: the 3-month SST anomaly for each day in that window
    """
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    resp = requests.get(url)
    resp.raise_for_status()
    
    # Split into lines and find header
    lines = resp.text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("SEAS")),
        None
    )
    if header_idx is None:
        raise ValueError("Could not find header row ('SEAS YR TOTAL ANOM') in ONI file.")
    
    # Read the table
    data = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(data),
        sep=r'\s+',  # Split on one-or-more whitespace
        engine='python'
    )
    
    # Define how each season maps to a 3-month window
    season_map = {
        'DJF': (-1, 12,  0,  2),
        'JFM': ( 0,  1,  0,  3),
        'FMA': ( 0,  2,  0,  4),
        'MAM': ( 0,  3,  0,  5),
        'AMJ': ( 0,  4,  0,  6),
        'MJJ': ( 0,  5,  0,  7),
        'JJA': ( 0,  6,  0,  8),
        'JAS': ( 0,  7,  0,  9),
        'ASO': ( 0,  8,  0, 10),
        'SON': ( 0,  9,  0, 11),
        'OND': ( 0, 10,  0, 12),
        'NDJ': ( 0, 11,  1,  1),
    }
    
    records = []
    for _, row in df.iterrows():
        season = row['SEAS']
        year = int(row['YR'])
        total = float(row['TOTAL'])
        anom  = float(row['ANOM'])
        
        month_offset_start, month_trimester_start, month_offset_end, month_trimester_end = season_map[season]
        start_year = year + month_offset_start
        end_year = year + month_offset_end
        
        start_date = dt.date(start_year, month_trimester_start, 1)
        if month_trimester_end != 12:
            # First day of month_trimester_end+1 minus one day
            next_date = dt.date(end_year, month_trimester_end + 1, 1)
            end_date = next_date - dt.timedelta(days=1)
        else:
            # Can't add a month to 12, so hardcode December 31st
            end_date = dt.date(end_year, 12, 31)
        
        daily_dates = pd.date_range(start_date, end_date, freq='D')
        for d in daily_dates:
            records.append((d, total, anom))
    
    daily = pd.DataFrame(records, columns=['time','enso_total','enso_anom'])
    daily = daily.drop_duplicates(subset='time', keep='first')
    
    ds = xr.Dataset(
        {
            'enso_total': ('time', daily['enso_total'].values),
            'enso_anom':  ('time', daily['enso_anom'].values),
        },
        coords={'time': daily['time'].values}
    )
    
    return ds


def fetch_mjo_proj():
    """
    Fetch CPC's projection-based MJO index from proj_norm_order.ascii and
    return an xarray.Dataset with:
      - time (daily; each pentad → 5 consecutive days)
      - mjo_index_1 ... mjo_index_10 (floats)
    """
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii"
    resp = requests.get(url)
    resp.raise_for_status()

    text  = resp.text
    lines = text.splitlines()

    # 1) figure out the order of the 10 INDEX_* columns
    #    (first line has exactly 10 entries: INDEX_9, INDEX_10, INDEX_1, … INDEX_8)
    idx_cols = lines[0].strip().split()

    # 2) read everything after the two header lines,
    #    bring in the "pentad" column + those 10 columns, treat "*****" as NaN
    df = pd.read_csv(
        io.StringIO(text),
        sep=r'\s+',            # whitespace delimiter
        engine='python',
        skiprows=2,            # drop both header lines
        header=None,
        names=['pentad'] + idx_cols,
        na_values=['*****']
    )

    # 3) drop any pentads with missing data
    df = df.dropna(subset=idx_cols)

    # 4) parse the pentad date (YYYYMMDD) into a Timestamp
    df['pentad'] = pd.to_datetime(df['pentad'], format='%Y%m%d')

    # 5) sort the INDEX cols so we output in numeric order INDEX_1…INDEX_10
    idx_cols_sorted = sorted(idx_cols, key=lambda s: int(s.split('_')[1]))

    # 6) expand each pentad (5-day block) into daily rows
    records = []
    for _, row in df.iterrows():
        start = row['pentad']
        for d in range(5):
            day = start + pd.Timedelta(days=d)
            rec = {'time': day}
            for col in idx_cols_sorted:
                n = col.split('_')[1]   # e.g. "1", "2", … "10"
                rec[f"mjo_proj_index_{n}"] = row[col]
            records.append(rec)

    daily = pd.DataFrame(records)

    # 7) build the xarray Dataset
    data_vars = {
        f"mjo_proj_index_{col.split('_')[1]}": ('time', daily[f"mjo_proj_index_{col.split('_')[1]}"].values)
        for col in idx_cols_sorted
    }
    ds = xr.Dataset(data_vars, coords={'time': daily['time'].values})

    return ds


def fetch_mjo_romi():
    """
    Fetch PSL's ROMI index (OLR‐only MJO) from romi.cpcolr.1x.txt.
    Note amplitude is sqrt(pc_1**2 + pc_2**2).
    """
    url = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=r'\s+',
        engine='python',
        header=None,
        names=['year','month','day','hour','pc1','pc2','amplitude'],
        comment='#'
    )
    df['time'] = pd.to_datetime(df[['year','month','day']])
    
    ds = xr.Dataset(
        {
            'mjo_romi_pc1':       ('time', df['pc1'].values),
            'mjo_romi_pc2':       ('time', df['pc2'].values),
            'mjo_romi_amplitude': ('time', df['amplitude'].values),
        },
        coords={'time': df['time'].values}
    )
    return ds


def fetch_mjo_vpm():
    """
    Fetch PSL's VPM index (velocity‐potential MJO) from vpm.1x.txt.
    Returns the same structure as fetch_mjo_proj().
    """
    url = "https://psl.noaa.gov/mjo/mjoindex/vpm.1x.txt"
    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=r'\s+',
        engine='python',
        header=None,
        names=['year','month','day','hour','pc1','pc2','amplitude'],
        comment='#'
    )
    df['time'] = pd.to_datetime(df[['year','month','day']])
    
    ds = xr.Dataset(
        {
            'mjo_vpm_pc1':       ('time', df['pc1'].values),
            'mjo_vpm_pc2':       ('time', df['pc2'].values),
            'mjo_vpm_amplitude': ('time', df['amplitude'].values),
        },
        coords={'time': df['time'].values}
    )
    return ds


def fetch_mjo_bom():
    """
    Fetch the BoM RMM index from IRI's .nc, load into xarray,
    clean, and assign a correct daily time axis from 1974-06-01.
    
    Returns an xarray.Dataset with:
      - time      : daily datetime64[ns], 1974-06-01 → 2025-07-10
      - rmm1      : RMM1 principal component
      - rmm2      : RMM2 principal component
      - phase     : MJO phase
      - amplitude : MJO amplitude
    """
    url = "https://iridl.ldeo.columbia.edu/SOURCES/.BoM/.MJO/.RMM/data.nc"

    # 1) Download to temp file
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        path = tmp.name
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)

    try:
        # 2) Open and let xarray mask the huge fill‐values for us
        ds = xr.open_dataset(path, decode_times=False)
        ds.load()
    finally:
        os.remove(path)

    # 3) Drop any extra vars and rename BoM names to our convention
    ds = ds.drop_vars("origin", errors="ignore")
    ds = ds.rename_vars({"RMM1": "rmm1", "RMM2": "rmm2"})

    # 4) Drop any days with NaNs in our core variables
    ds = ds.dropna(dim="T", how="any", subset=["rmm1", "rmm2", "phase", "amplitude"])

    # 5) Convert Julian days to datetime
    ds = ds.assign_coords({
        'T': pd.to_datetime(ds['T'].values, origin='julian', unit='D').normalize()
    })
    ds = ds.rename({'T': 'time', 'amplitude': 'mjo_bom_amplitude', 'phase': 'mjo_bom_phase', 'rmm1': 'mjo_bom_rmm1', 'rmm2': 'mjo_bom_rmm2'})
    
    return ds


def fetch_nao():
    """
    Fetch the CPC NAO index from the .table file, parse it, expand to daily,
    and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - nao  : float64, constant within each month
    """
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/" \
          "norm.nao.monthly.b5001.current.ascii.table"
    resp = requests.get(url)
    resp.raise_for_status()

    # Split into tokens
    tokens = resp.text.split()
    # First 12 tokens are the month headers
    months = tokens[:12]  # ["Jan","Feb",...,"Dec"]
    # Map header position → month number
    # e.g. months[0] == "Jan" → month_num = 1, etc.
    # We'll just use position+1 for month number.

    data = []
    i = 12
    n = len(tokens)
    while i < n:
        year = int(tokens[i])
        i += 1
        # Determine how many values follow (up to 12; last year may be partial)
        vals_left = n - i
        mcount   = min(12, vals_left)
        for m in range(mcount):
            val = float(tokens[i])
            data.append((year, m+1, val))
            i += 1

    # Build long-form DataFrame
    df = pd.DataFrame(data, columns=['year','month','nao'])

    # Drop any NaNs (shouldn't be any in this source)
    df = df.dropna(subset=['nao'])

    # First-of-month timestamp
    df['time'] = pd.to_datetime({
        'year':  df['year'].astype(int),
        'month': df['month'].astype(int),
        'day':   1
    })

    # Expand each month's value to every day in that month
    records = []
    for _, row in df.iterrows():
        start = row['time']
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq='D'):
            records.append((day, row['nao']))
    daily = pd.DataFrame(records, columns=['time','nao'])

    # Build xarray Dataset
    ds = xr.Dataset(
        {'nao': ('time', daily['nao'].values.astype('float64'))},
        coords={'time': daily['time'].values}
    )
    return ds


def fetch_pna():
    """
    Fetch the daily PNA index from CPC and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - pna  : PNA index as float64
    """
    url = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.cdas.z500.19500101_current.csv"
    resp = requests.get(url)
    resp.raise_for_status()

    # Read the CSV into a DataFrame
    # Expected columns: year, month, day, pna_index_cdas
    df = pd.read_csv(io.StringIO(resp.text))

    # Coerce the index column to float, drop any non‐numeric or missing entries
    df['pna_index_cdas'] = pd.to_numeric(df['pna_index_cdas'], errors='coerce')
    df = df.dropna(subset=['pna_index_cdas'])

    # Build a daily datetime index
    df['time'] = pd.to_datetime(df[['year','month','day']])

    # Assemble into an xarray Dataset
    ds = xr.Dataset(
        {
            'pna': ('time', df['pna_index_cdas'].values.astype('float64'))
        },
        coords={'time': df['time'].values}
    )

    return ds


def _fetch_qbo(level: str) -> xr.Dataset:
    """
    Generic helper to fetch CPC QBO data at the given pressure level (e.g. "30" or "50"),
    parse the three tables (original, anomaly, standardized), expand each month into daily
    values, and return an xarray.Dataset with:
      - time                        : daily datetime64[ns]
      - qbo_{level}_original        : float64 original monthly values
      - qbo_{level}_anomaly         : float64 monthly anomalies
      - qbo_{level}_standardized    : float64 monthly standardized values
    """
    url = f"https://www.cpc.ncep.noaa.gov/data/indices/qbo.u{level}.index"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/115.0.0.0 Safari/537.36'
        )
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    lines = resp.text.splitlines()

    # find the three header lines
    hdr_rx = re.compile(r'^\s*YEAR\s+JAN\s+FEB\s+MAR\s+APR\s+MAY\s+JUN\s+JUL\s+AUG\s+SEP\s+OCT\s+NOV\s+DEC\s*$')
    hdrs = [i for i, ln in enumerate(lines) if hdr_rx.match(ln)]
    if len(hdrs) < 3:
        raise RuntimeError(f"Expected 3 tables in QBO file for level {level}, found {len(hdrs)} headers")

    orig_block = lines[hdrs[0]+1 : hdrs[1]]
    anom_block = lines[hdrs[1]+1 : hdrs[2]]
    std_block  = lines[hdrs[2]+1 : ]

    def parse_block(block):
        d = {}
        for ln in block:
            # extract year + 12 floats (handles missing concatenation)
            nums = re.findall(r'-?\d+\.\d+|\d+', ln)
            if len(nums) < 13:
                continue
            year = int(nums[0])
            months = [float(x) for x in nums[1:13]]
            # skip entire year if all missing
            if all(v == -999.90 for v in months):
                continue
            for m, v in enumerate(months, start=1):
                if v == -999.90:
                    continue
                d[(year, m)] = v
        return d

    orig  = parse_block(orig_block)
    anom  = parse_block(anom_block)
    std   = parse_block(std_block)

    # only keep (year,month) present in all three tables
    keys = sorted(set(orig) & set(anom) & set(std))

    records = []
    for yr, mo in keys:
        vo = orig[(yr, mo)]
        va = anom[(yr, mo)]
        vs = std[(yr, mo)]
        start = pd.Timestamp(yr, mo, 1)
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq='D'):
            records.append((day, vo, va, vs))

    df = pd.DataFrame(records, columns=[
        'time',
        f'qbo_{level}_original',
        f'qbo_{level}_anomaly',
        f'qbo_{level}_standardized'
    ])

    return xr.Dataset(
        {
            f'qbo_{level}_original'    : ('time', df[f'qbo_{level}_original'].values),
            f'qbo_{level}_anomaly'     : ('time', df[f'qbo_{level}_anomaly'].values),
            f'qbo_{level}_standardized': ('time', df[f'qbo_{level}_standardized'].values),
        },
        coords={'time': df['time'].values}
    )

def fetch_qbo_30() -> xr.Dataset:
    """Daily QBO 30 hPa index (original, anomaly, standardized)."""
    return _fetch_qbo("30")

def fetch_qbo_50() -> xr.Dataset:
    """Daily QBO 50 hPa index (original, anomaly, standardized)."""
    return _fetch_qbo("50")


def fetch_iod():
    """
    Fetch the Indian Ocean Dipole index (DMI) from PSL CSV,
    skip the header, expand monthly → daily, and return an
    xarray.Dataset with:
      - time : daily datetime64[ns]
      - iod  : float64 DMI index
    """
    url = "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.csv"
    resp = requests.get(url)
    resp.raise_for_status()
    # Drop the first header line
    text = "\n".join(resp.text.splitlines()[1:])

    # Read two columns: date (YYYY-MM-DD) and iod, treating -9999 as NaN
    df = pd.read_csv(
        io.StringIO(text),
        sep=r'\s*,\s*',
        engine='python',
        names=['date','iod'],
        na_values=['-9999'],
        parse_dates=['date']
    )

    # Drop missing values
    df = df.dropna(subset=['iod'])

    # Expand each monthly value to every day in its month
    records = []
    for date, iod in zip(df['date'], df['iod']):
        # pd.tseries.offsets.MonthEnd rolls to month end
        end = date + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(date, end, freq="D"):
            records.append((day, iod))

    daily = pd.DataFrame(records, columns=['time','iod'])

    # Wrap into xarray Dataset
    ds = xr.Dataset(
        {'iod': ('time', daily['iod'].values.astype('float64'))},
        coords={'time': daily['time'].values}
    )
    return ds


def fetch_pdo():
    """
    Fetch the ERSSTv5 Pacific Decadal Oscillation (PDO) index from
    ersst.v5.pdo.dat, mask missing_value=99.99, expand monthly → daily,
    and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - pdo  : float64 PDO index
    """
    url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.splitlines()

    # Skip the first two lines (header + month names)
    data_lines = lines[2:]

    # Parse year + 12 monthly values
    records = []
    for ln in data_lines:
        parts = ln.strip().split()
        if len(parts) < 13:
            continue
        year = int(parts[0])
        for month, token in enumerate(parts[1:13], start=1):
            try:
                val = float(token)
            except ValueError:
                continue
            # 99.99 is the missing‐value flag
            if val == 99.99:
                continue
            records.append((year, month, val))

    # Build DataFrame of (year, month, pdo)
    df = pd.DataFrame(records, columns=['year','month','pdo'])

    # First‐of‐month timestamp
    df['time'] = pd.to_datetime({
        'year':  df['year'].astype(int),
        'month': df['month'].astype(int),
        'day':   1
    })

    # Expand each month → every day in that month
    daily_recs = []
    for _, row in df.iterrows():
        start = row['time']
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq='D'):
            daily_recs.append((day, row['pdo']))
    daily = pd.DataFrame(daily_recs, columns=['time','pdo'])

    # Wrap in xarray.Dataset
    ds = xr.Dataset(
        {'pdo': ('time', daily['pdo'].values.astype('float64'))},
        coords={'time': daily['time'].values}
    )

    return ds


def fetch_soi():
    """
    Fetch the Southern Oscillation Index (SOI) from PSL's soi.data file,
    parse the monthly table (skip first line), mask -99.99 as NaN, expand
    each monthly value to daily, and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - soi  : float64 SOI index
    """
    url = "https://psl.noaa.gov/data/correlation/soi.data"
    resp = requests.get(url)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    # 1) Drop the first line (start/end year header)
    data_lines = lines[1:]

    records = []
    missing = -99.99

    # 2) Parse each line until we hit the 'xmiss' line or non-year token
    for ln in data_lines:
        parts = ln.strip().split()
        # stop if this isn't a year row
        if not parts or not re.match(r'^\d{4}$', parts[0]):
            break
        year = int(parts[0])
        # next up to 12 tokens are Jan→Dec
        for month, tok in enumerate(parts[1:13], start=1):
            try:
                v = float(tok)
            except ValueError:
                continue
            if v == missing:
                continue
            records.append((year, month, v))

    # 3) Build a DataFrame of (year, month, soi)
    df = pd.DataFrame(records, columns=['year','month','soi'])

    # 4) Build first‐of‐month timestamps
    df['time'] = pd.to_datetime({
        'year':  df['year'].astype(int),
        'month': df['month'].astype(int),
        'day':   1
    })

    # 5) Expand each monthly value to all days in that month
    daily = []
    for _, row in df.iterrows():
        start = row['time']
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq='D'):
            daily.append((day, row['soi']))
    daily_df = pd.DataFrame(daily, columns=['time','soi'])

    # 6) Wrap into xarray.Dataset
    ds = xr.Dataset(
        {'soi': ('time', daily_df['soi'].values.astype('float64'))},
        coords={'time': daily_df['time'].values}
    )
    return ds


def fetch_mei():
    """
    Fetch the Multivariate ENSO Index version 2 (MEI.v2) from PSL,
    expand each 2-month season to daily resolution, and return an
    xarray.Dataset with:
      - time  : daily datetime64[ns]
      - mei_v2: float64 MEI.v2 index
    """
    url = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
    resp = requests.get(url)
    resp.raise_for_status()
    txt = resp.text

    # extract all numbers: start_year, end_year, then (year,12 values)…
    nums = re.findall(r'-?\d+\.\d+|-?\d+', txt)
    start_year, end_year = int(nums[0]), int(nums[1])
    data = nums[2:]
    n_years = end_year - start_year + 1
    block_size = 1 + 12  # one year + 12 seasons

    # define the 12 overlapping seasons and how they map to months
    seasons = ['DJ','JF','FM','MA','AM','MJ','JJ','JA','AS','SO','ON','ND']
    season_map = {
        'DJ': (-1,12, 0, 1),
        'JF': ( 0, 1, 0, 2),
        'FM': ( 0, 2, 0, 3),
        'MA': ( 0, 3, 0, 4),
        'AM': ( 0, 4, 0, 5),
        'MJ': ( 0, 5, 0, 6),
        'JJ': ( 0, 6, 0, 7),
        'JA': ( 0, 7, 0, 8),
        'AS': ( 0, 8, 0, 9),
        'SO': ( 0, 9, 0,10),
        'ON': ( 0,10, 0,11),
        'ND': ( 0,11, 0,12),
    }

    records = []
    for i in range(n_years):
        block = data[i*block_size : (i+1)*block_size]
        year = int(float(block[0]))
        # parse the 12 seasonal floats
        vals = [float(v) for v in block[1:13]]
        for seas, val in zip(seasons, vals):
            if val == -999.00:
                continue
            off_s, m_start, off_e, m_end = season_map[seas]
            sy = year + off_s
            ey = year + off_e
            start = dt.date(sy, m_start, 1)
            # compute end-of-window
            if m_end == 12:
                end = dt.date(ey, 12, 31)
            else:
                nxt = dt.date(ey, m_end+1, 1)
                end = nxt - dt.timedelta(days=1)
            # expand to daily
            for d in pd.date_range(start, end, freq='D'):
                records.append((d, val))

    df = pd.DataFrame(records, columns=['time','mei_v2'])
    df = df.drop_duplicates(subset='time', keep='first')
    ds = xr.Dataset(
        {'mei_v2': ('time', df['mei_v2'].values.astype('float64'))},
        coords={'time': df['time'].values}
    )
    return ds


def fetch_aao():
    """
    Fetch the daily AAO index (700 hPa) from CPC:
      https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.aao.cdas.z700.19790101_current.csv
    Returns an xarray.Dataset with:
      - time : daily datetime64[ns]
      - aao  : float64 AAO index (standardized anomaly)
    """
    url = "https://ftp.cpc.ncep.noaa.gov/cwlinks/" \
          "norm.daily.aao.cdas.z700.19790101_current.csv"
    resp = requests.get(url)
    resp.raise_for_status()

    # Read CSV into DataFrame
    df = pd.read_csv(io.StringIO(resp.text))

    # Identify the AAO index column (should contain 'aao' in its name)
    idx_col = next(
        col for col in df.columns
        if 'aao' in col.lower() and col.lower().endswith('cdas')
    )

    # Convert to float64, drop any rows where it fails (e.g. missing)
    df[idx_col] = pd.to_numeric(df[idx_col], errors='coerce')
    df = df.dropna(subset=[idx_col])

    # Build a daily time index
    df['time'] = pd.to_datetime(df[['year','month','day']])

    # Wrap in xarray Dataset
    ds = xr.Dataset(
        {
            'aao': ('time', df[idx_col].values.astype('float64'))
        },
        coords={'time': df['time'].values}
    )
    return ds


def fetch_ao():
    """
    Fetch the daily Arctic Oscillation (AO) index from CPC's legacy ASCII,
    parse the year, month, day, and value columns, drop missing (<= -90),
    and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - ao   : float64 AO index
    """
    url = "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    
    records = []
    for ln in resp.text.splitlines():
        parts = ln.strip().split()
        # look for exactly four columns: YYYY MM DD value
        if len(parts) != 4:
            continue
        year, month, day, val = parts
        try:
            y, m, d = int(year), int(month), int(day)
            v = float(val)
        except ValueError:
            continue
        # drop the missing-flag (–99.90 or anything ≤ –90)
        if v <= -90:
            continue
        records.append((pd.Timestamp(y, m, d), v))
    
    if not records:
        raise RuntimeError("No AO data could be parsed from the file.")
    
    df = pd.DataFrame(records, columns=['time','ao'])
    ds = xr.Dataset(
        {'ao': ('time', df['ao'].values.astype('float64'))},
        coords={'time': df['time'].values}
    )
    return ds


def fetch_npgo():
    """
    Fetch the monthly NPGO index from PSL’s correlation CSV,
    explicitly parse dates, expand each month to daily, and return
    an xarray.Dataset with:
      - time : daily datetime64[ns]
      - npgo : float64 NPGO index
    """
    url = "https://psl.noaa.gov/data/correlation/npgo.csv"
    resp = requests.get(url)
    resp.raise_for_status()

    # Drop commented lines, read two-column CSV: date, npgo
    text = "\n".join(line for line in resp.text.splitlines() if not line.startswith('#'))
    df = pd.read_csv(
        io.StringIO(text),
        sep=',',
        header=None,
        names=['date','npgo'],
        na_values=['-9999.000'],
    )

    # 1) Explicitly parse the date column (expecting YYYY-MM-DD or YYYY-MM-01)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    # 2) Drop any rows where date failed or npgo is NaN
    df = df.dropna(subset=['date','npgo'])

    # 3) Expand each month → daily
    records = []
    for date, val in zip(df['date'], df['npgo'].astype('float64')):
        # last day of the month
        end = date + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(date, end, freq='D'):
            records.append((day, val))

    daily = pd.DataFrame(records, columns=['time','npgo'])

    # 4) Wrap in xarray.Dataset
    ds = xr.Dataset(
        {'npgo': ('time', daily['npgo'].values)},
        coords={'time': daily['time'].values}
    )
    return ds


def fetch_tpi():
    """
    Fetch the filtered Tripole Index (TPI) from PSL,
    skip the first header line, parse year + 12 monthly values
    (stopping when the “TPI filtered” footer appears), mask -99.000 as NA,
    expand each month → daily, and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - tpi  : float64 TPI index
    """
    url = "https://psl.noaa.gov/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.filt.data"
    resp = requests.get(url)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    # 1) drop the first line
    data_lines = lines[1:]

    records = []
    for ln in data_lines:
        # stop at the footer
        if ln.startswith("TPI filtered"):
            break
        parts = ln.strip().split()
        # expect 1 year + 12 months
        if len(parts) < 13:
            continue
        # parse year
        try:
            year = int(parts[0])
        except ValueError:
            continue
        # parse 12 months
        for month, tok in enumerate(parts[1:13], start=1):
            try:
                v = float(tok)
            except ValueError:
                continue
            # drop missing‐value flag
            if v == -99.000:
                continue
            records.append((year, month, v))

    # build DataFrame
    df = pd.DataFrame(records, columns=["year", "month", "tpi"])
    # first‐of‐month timestamp
    df["time"] = pd.to_datetime({
        "year":  df["year"],
        "month": df["month"],
        "day":   1
    })

    # expand each month to daily
    daily = []
    for _, row in df.iterrows():
        start = row["time"]
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq="D"):
            daily.append((day, row["tpi"]))
    daily_df = pd.DataFrame(daily, columns=["time", "tpi"])

    # wrap in xarray
    ds = xr.Dataset(
        {"tpi": ("time", daily_df["tpi"].values.astype("float64"))},
        coords={"time": daily_df["time"].values}
    )
    return ds


# def fetch_tsa():
#     """
#     Fetch the Tropical Southern Atlantic (TSA) index from the NOAA State-of-the-Ocean
#     NetCDF, expand monthly values to daily, and return an xarray.Dataset with:
#       - time : daily datetime64[ns]
#       - tsa  : float64 TSA index
#     """
#     url = "https://stateoftheocean.osmc.noaa.gov/sur/data/tsa.nc"
#
#     # 1) Download to a temp file
#     resp = requests.get(url, stream=True, timeout=30)
#     resp.raise_for_status()
#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
#             tmp_path = tmp.name
#             for chunk in resp.iter_content(chunk_size=8192):
#                 tmp.write(chunk)
#
#         # 2) Open with xarray and load into memory
#         ds = xr.open_dataset(tmp_path, decode_times=True)
#         ds.load()
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)
#
#     # 3) Identify the TSA variable (assume it's the only data_var), rename it to 'tsa'
#     varname = list(ds.data_vars)[0]
#     da = ds[varname].rename("tsa")
#
#     # 4) Get its (single) dimension name and coordinate values
#     dim = da.dims[0]
#     times_monthly = pd.to_datetime(da[dim].values)
#
#     # 5) Build daily records by expanding each month to its days
#     records = []
#     for t, v in zip(times_monthly, da.values):
#         if pd.isna(v):
#             continue
#         # last day of that month
#         end = t + pd.tseries.offsets.MonthEnd(0)
#         for day in pd.date_range(t, end, freq="D"):
#             records.append((day, float(v)))
#
#     # 6) Assemble into a daily xarray.Dataset
#     daily = pd.DataFrame(records, columns=["time", "tsa"])
#     daily = daily.drop_duplicates(subset='time', keep='first')
#     return xr.Dataset(
#         {"tsa": ("time", daily["tsa"].values.astype("float64"))},
#         coords={"time": daily["time"].values}
#     )

def fetch_tsa():
    """
    Fetch the Tropical Southern Atlantic (TSA) index from the NOAA State-of-the-Ocean
    NetCDF, expand weekly values to daily, and return an xarray.Dataset with:
      - time : daily datetime64[ns]
      - tsa  : float64 TSA index
    """
    url = "https://stateoftheocean.osmc.noaa.gov/sur/data/tsa.nc"

    # 1) Download to a temp file
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)

        # 2) Open with xarray and load into memory
        ds = xr.open_dataset(tmp_path, decode_times=True)
        ds.load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3) Identify the TSA variable, rename it and its coordinate
    varname = list(ds.data_vars)[0]
    ds = ds.rename({varname: "tsa", "TIME": "time"})

    # 4) Drop missing values along the 'time' dimension
    ds = ds.dropna(dim="time")

    # 5) Resample the weekly data to daily frequency, using forward-fill
    daily_ds = ds.resample(time='1D').ffill()

    # 6) Assemble into a daily xarray.Dataset
    return daily_ds



def _fetch_cpc_tele(name: str, url: str, n_periods: int, missing: float = -99.900002) -> xr.Dataset:
    """
    Generic fetch for CPC NHTI indices via IRI OPeNDAP, without CF time decoding.
    - name     : output var name (e.g. "ea", "sca", …)
    - url      : the DODS endpoint
    - n_periods: number of monthly points (906 or 905)
    - missing  : the missing_value flag to drop
    Returns a daily‐expanded xarray.Dataset with coords time and var `name`.
    """
    # 1) open without decoding the non‐CF calendar
    ds = xr.open_dataset(url, decode_times=False)
    ds.load()
    var = list(ds.data_vars)[0]
    da  = ds[var].rename(name)
    dim = da.dims[0]

    # 2) build a proper monthly DatetimeIndex Jan 1950 → ...
    monthly = pd.date_range("1950-01-01", periods=n_periods, freq="MS")

    # 3) expand monthly values to daily, dropping missing
    records = []
    for t, v in zip(monthly, da.values.flatten()):
        if pd.isna(v) or v == missing:
            continue
        # last day of month
        end = t + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(t, end, freq="D"):
            records.append((day, float(v)))

    df = pd.DataFrame(records, columns=["time", name])

    return xr.Dataset(
        {name: ("time", df[name].values)},
        coords={"time": df["time"].values}
    )

def fetch_ea() -> xr.Dataset:
    """East Atlantic (EA) Pattern, Jan 1950–Jun 2025 (906 pts)."""
    url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Indices/.NHTI/.EA/dods"
    return _fetch_cpc_tele("ea", url, n_periods=906)

def fetch_sca() -> xr.Dataset:
    """Scandinavian (SCA) Pattern, Jan 1950–Jun 2025 (906 pts)."""
    url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Indices/.NHTI/.SCA/dods"
    return _fetch_cpc_tele("sca", url, n_periods=906)

def fetch_eawr() -> xr.Dataset:
    """East Atlantic–West Russia (EAWR), Jan 1950–Jun 2025 (906 pts)."""
    url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Indices/.NHTI/.EAWR/dods"
    return _fetch_cpc_tele("eawr", url, n_periods=906)

def fetch_wp() -> xr.Dataset:
    """West Pacific (WP) Pattern, Jan 1950–Jun 2025 (906 pts)."""
    url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Indices/.NHTI/.WP/dods"
    return _fetch_cpc_tele("wp", url, n_periods=906)

def fetch_epnp() -> xr.Dataset:
    """East Pacific/North Pacific (EPNP), Jan 1950–May 2025 (905 pts)."""
    url = "http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.CPC/.Indices/.NHTI/.EPNP/dods"
    return _fetch_cpc_tele("epnp", url, n_periods=905)


def fetch_wwv():
    """
    Fetch the TAO Warm Water Volume (WWV) data over HTTPS,
    parse the monthly Volume & Anomaly columns, and expand each
    to daily resolution. Returns an xarray.Dataset with:
      - time          : daily datetime64[ns]
      - wwv           : float64 monthly WWV (m^3), repeated per day
      - wwv_anomaly   : float64 monthly anomaly, repeated per day
    """
    # 1) Download and split into lines
    url = "https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()

    # 2) Find the header line and skip everything up to it
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if "Volume" in ln and "Anomaly" in ln),
        None
    )
    if header_idx is None:
        raise RuntimeError("WWV header not found")
    data_lines = lines[header_idx+1:]

    # 3) Parse YYYYMM, volume, anomaly
    records = []
    for ln in data_lines:
        parts = ln.strip().split()
        if len(parts) < 3:
            continue
        date_str, vol_str, anom_str = parts[:3]
        # parse year & month
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
        except ValueError:
            continue
        # parse numeric values
        try:
            vol  = float(vol_str)   # e.g. 0.2605404E+16
            anom = float(anom_str)  # e.g. 0.7657363E+14
        except ValueError:
            continue
        # expand this month into daily entries
        start = pd.Timestamp(year, month, 1)
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq="D"):
            records.append((day, vol, anom))

    # 4) Build DataFrame & xarray.Dataset
    if not records:
        return xr.Dataset(
            {"wwv":    ("time", []),
             "wwv_anomaly": ("time", [])},
            coords={"time": []}
        )

    df = pd.DataFrame(records, columns=["time","wwv","wwv_anomaly"])
    ds = xr.Dataset(
        {
            "wwv":         ("time", df["wwv"].values.astype("float64")),
            "wwv_anomaly": ("time", df["wwv_anomaly"].values.astype("float64")),
        },
        coords={"time": df["time"].values}
    )
    return ds


def fetch_polar_cap_height() -> xr.Dataset:
    """
    Fetch the 100 hPa polar‐cap geopotential height anomalies
    as a *small* remote‐subset via OPeNDAP, expand each monthly
    value to daily, and return:

      - time               : daily datetime64[ns]
      - polar_cap_height   : float64 [m]
    """
    # 1) Point at the THREDDS OPeNDAP URL — no full file download
    url = (
      "https://psl.noaa.gov/thredds/dodsC/"
      "Datasets/ncep.reanalysis2/Monthlies/pressure/hgt.mon.mean.nc"
    )
    ds = xr.open_dataset(url, decode_times=True)
    # return ds

    # 2) Subset on the server: 100 hPa & lat ≥ 65° N; average over lat/lon
    #    xarray/OPeNDAP will only fetch that tiny subset
    ts_monthly = (
        ds["hgt"]
          .sel(level=100, lat=slice(90, 65))
          .mean(dim=["lat","lon"])
          .rename("polar_cap_height")
    )

    # 3) Now pull the small series into memory (just ~900 monthly values)
    ts_monthly = ts_monthly.load()
    # return ts_monthly

    # 4) Expand each month → daily
    records = []
    for t, val in zip(ts_monthly["time"].values, ts_monthly.values):
        if pd.isna(val):
            continue
        start = pd.to_datetime(t).replace(day=1)
        end   = start + pd.tseries.offsets.MonthEnd(0)
        for day in pd.date_range(start, end, freq="D"):
            records.append((day, float(val)))

    df = pd.DataFrame(records, columns=["time","polar_cap_height"])

    return xr.Dataset(
        {"polar_cap_height": ("time", df["polar_cap_height"].values)},
        coords={"time": df["time"].values}
    )


def fetch_ice_extent() -> xr.Dataset:
    """
    Fetch daily Arctic sea‐ice extent from NSIDC over HTTPS,
    parse Year/Month/Day and Extent columns (skipping the descriptive row),
    and return as an xarray.Dataset with:
      - time              : daily datetime64[ns]
      - sea_ice_extent    : float64 (million km²)
    """
    # 1) Download the CSV via HTTPS
    url = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # 2) Read into pandas, skipping the second line which is just descriptive metadata,
    #    and only loading the four columns we need.
    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=[1],  # skip the first data-detail row
        skipinitialspace=True,
        usecols=["Year", "Month", "Day", "Extent"],
    )

    # 3) Drop any rows where Extent is missing or invalid
    #    (if NSIDC uses a sentinel like –999.9, mark those as NaN first)
    df["Extent"] = df["Extent"].replace(-999.9, pd.NA)
    df = df.dropna(subset=["Extent"])

    # 4) Build a proper datetime index from Year/Month/Day
    df["time"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    # 5) Wrap in xarray.Dataset
    ds = xr.Dataset(
        {
            "sea_ice_extent": ("time", df["Extent"].astype("float64").values)
        },
        coords={
            "time": df["time"].values
        }
    )

    return ds

