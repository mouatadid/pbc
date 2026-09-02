import os
import math
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from glob import glob
from itertools import product
from IPython.display import Markdown, display
from src.utils.data_io import *
from models.utils.general_util import printf
from models.utils.eval_util import get_target_dates
from models.utils.models_util import get_selected_submodel_name
from utils.file_io import make_directories
import cartopy.crs as ccrs
from string import Template
import copy
import pandas as pd
from pathlib import Path
import seaborn as sns
from arch.bootstrap import StationaryBootstrap, optimal_block_length
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import tempfile
import requests
from AI_WQ_package.plotting_forecast import create_colormap, convert_long
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER




OUT_DIR = os.path.join("viz", "pbc")
make_directories(OUT_DIR)
EXTREMES_OUT_DIR = os.path.join("viz", "pbc", "extremes")
make_directories(EXTREMES_OUT_DIR)
BIAS_OUT_DIR = os.path.join("viz", "pbc", "bias")
make_directories(BIAS_OUT_DIR)
SRC_DATA_DIR = os.path.join(OUT_DIR, "source_data")
make_directories(SRC_DATA_DIR)

#
# Dictionaries mapping all model names and tasks to their display names
#
tasks = {
      "era5-tas_19": "Temperature, week 3",
      "era5-tas_26": "Temperature, week 4",
      "era5-pr_19": "Precipitation, week 3",
      "era5-pr_26": "Precipitation, week 4",
      "era5-mslp_19": "Sea-level pressure, week 3",
      "era5-mslp_26": "Sea-level pressure, week 4",
      "era5-f1_tas_19": "tas-19 f1",
      "era5-f1_pr_19": "pr-19 f1",
      "era5-f1_mslp_19": "mslp-19 f1",
      "era5-f1_tas_26": "tas-26 f1",
      "era5-f1_pr_26": "pr-26 f1",
      "era5-f1_mslp_26": "mslp-26 f1",
      "era5-f2_tas_19": "tas-19 f2",
      "era5-f2_pr_19": "pr-19 f2",
      "era5-f2_mslp_19": "mslp-19 f2",
      "era5-f2_tas_26": "tas-26 f2",
      "era5-f2_pr_26": "pr-26 f2",
      "era5-f2_mslp_26": "mslp-26 f2",
      "era5-f3_tas_19": "tas-19 f3",
      "era5-f3_pr_19": "pr-19 f3",
      "era5-f3_mslp_19": "mslp-19 f3",
      "era5-f3_tas_26": "tas-26 f3",
      "era5-f3_pr_26": "pr-26 f3",
      "era5-f3_mslp_26": "mslp-26 f3",
      "era5-f4_tas_19": "tas-19 f4",
      "era5-f4_pr_19": "pr-19 f4",
      "era5-f4_mslp_19": "mslp-19 f4",
      "era5-f4_tas_26": "tas-26 f4",
      "era5-f4_pr_26": "pr-26 f4",
      "era5-f4_mslp_26": "mslp-26 f4"
}

all_model_names = {
    "climatology": "Climatology",
    
    # ECMWF
    "ecmwf": "Raw ECMWF",
    "perpp_ecmwf": "Persistence++-ECMWF",
    "proj_perpp_ecmwf": "Projected Persistence++-ECMWF",
    "tuned_ecmwfpp": "Debias++-ECMWF",
    "proj_tuned_ecmwfpp": "Projected Debias++-ECMWF",
    "pbc_ecmwf": "PBC-ECMWF",

    # Debiased ECMWF
    "debiased_ecmwf": "Debiased ECMWF",
    "perpp_debias": "Persistence++-ECMWF",
    "proj_perpp_debias": "Projected Persistence++-ECMWF",
    "tuned_debiaspp": "Debias++-Deb. ECMWF",
    "proj_tuned_debiaspp": "Projected Debias++-Deb. ECMWF",
    "pbc_debias": "PBC-Deb. ECMWF",

    # PBC Combo
    "pbc_ecmwf_combo": "PBC-ECMWF",

    # PoET
    "msn": "PoET",
    "proj_perpp_msn": "Projected Persistence++-PoET",
    "pbc_msn": "PBC-PoET",
    "msnpp": "Debias++-PoET",
    "perpp_msn": "Persistence++-PoET",
    "tuned_msnpp": "Debias++-PoET",
    "proj_tuned_msnpp": "Projected Debias++-PoET",

    # MicroDuet
    "duet": "MicroDuet",
    "duet_contest": "MicroDuet",
    
    # FuXi-S2S
    "fuxi": "FuXi-S2S",
    "debiased_fuxi": "FuXi-S2S",
    
    # AIFS
    "aifs": "AIFS-SUBS",
    "debiased_aifs": "AIFS-SUBS",
    "proj_perpp_aifs": "Persistence++-AIFS-SUBS",
    "proj_tuned_aifspp": "Debias++-AIFS-SUBS",
    "pbc_aifs": "PBC-AIFS-SUBS",
    "pbc_debias_aifs": "PBC-AIFS-SUBS"
}


global_tasks = {
    "era5-tas_19": "Temp., week 3",
    "era5-pr_19": "Precip., week 3",
    "era5-mslp_19": "MSLP, week 3",
    "era5-tas_26": "Temp., week 4",
    "era5-pr_26": "Precip., week 4",
    "era5-mslp_26": "MSLP, week 4",     
}

metric_names = {
    "lat_lon_rpss": "RPSS",
    "lat_lon_rps": "RPS",
    "rps": "RPS",
}

gt_id_names = {
    "era5-tas": "Temperature",
    "era5-pr": "Precipitation",
    "era5-mslp": "Sea Level Pressure",  
}

horizon_names = {
    19: "Week 3",
    26: "Week 4",
}
season_names = {
    'DJF': 'December-January-February', 
    'MAM': 'March-April-May',
    'JJA': 'June-July-August',
    'SON': 'September-October-November'
}

quintile_names = {
    'f1': 'Quintile 1',
    'f2': 'Quintiles 1-2',
    'f3': 'Quintiles 1-3',
    'f4': 'Quintiles 1-4',
}


dic_regions = {
    'us': {'lon_min':-125, 'lon_max':-67, 'lat_min':26, 'lat_max':50},
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
    # 'all': {'lon_min':-180, 'lon_max':180, 'lat_min':-90, 'lat_max':90}
}

region_names = {'all':'Global',
                'global': 'Global',
                'us': 'United States',
               'europe': 'Europe',
               'east_asia': 'East Asia',
               'middle_east': 'Middle East',
               'central_america': 'Central America',
               'south_america_nh': 'Northern South America',
               'south_america_sh': 'Southern South America',
               'north_africa': 'North Africa',
               'southern_africa': 'Southern Africa',
               'australia': 'Australia',
               'maritime_continent': 'Maritime continent',
               'india': 'India',
               'northern_hemisphere': 'Northern Hemisphere',
               'southern_hemisphere': 'Southern Hemisphere'}


model_colors = {
    "ecmwf": "purple",
    "debiased_ecmwf": "plum", 
    "pbc_ecmwf": "navy",
    "pbc_debias": "skyblue",
    "pbc_ecmwf_combo": "gold",

    "msn": "#80DEEA",                 # light cyan
    "proj_perpp_msn": "#009E73",      # bluish green
    "proj_tuned_msnpp": "#CC79A7",    # reddish purple
    "pbc_msn": "#00838F",             # dark teal

    "duet": 'gold', 
    "duet_contest": 'gold',
    "climatology": "gray",
    
    "fuxi": "#333333",
    "debiased_fuxi": "#333333",
    
    "aifs": "#87A96B", 
    "debiased_aifs": "#87A96B", 
    "pbc_debias_aifs": "orange", 

    "tuned_ecmwfpp": "#D55E00",      # brick
    "proj_tuned_ecmwfpp": "#F4A582", # light brick

    "perpp_ecmwf": "#0072B2",        # blue
    "proj_perpp_ecmwf": "#92C5DE",   # light blue

    "perpp_debias": "#0072B2",        # blue
    "proj_perpp_debias": "#92C5DE",   # light blue

    # "perpp_debias": "#1B9E77",       # blue-green
    # "proj_perpp_debias": "#A6DBD6",  # light blue-green

}




def get_metrics_ds(gt_id="era5-f1_tas",
                    horizon="19",
                    metric="wtd_mse",
                    target_dates="std_test",
                    model_names=["proj_perpp_ecmwf", "proj_tuned_ecmwfpp"],
                    regions=None,
                    combine_regions="concat",   # "mean" or "concat"
                    common_dates=True,
                    verbose=True):
    
    """
    Returns xarray Dataset containing metrics over input target dates.
    Args:
        gt_id (str): e.g. "era5-f1_tas", "era5-f2_pr", etc.
        horizon (str): "19" or "26"
        metric (str): "wtd_mse" or "rps"
        target_dates (str): target dates for metric calculations, e.g. "std_test"
        model_names (str): list of model names to be included.
        regions (list or None): e.g. ['us', 'europe']
        combine_regions (str):
            - "mean": average across regions (recommended)
            - "concat": keep region dimension
        common_dates (bool):
            - True: keep only dates where all models have data
            - False: keep dates where at least one model has data

    Returns:
        metrics_ds: xarray Dataset containing metrics over input target dates.
    """
    
    task = f"{gt_id}_{horizon}"
    if regions == 'all':
        regions = [r for r in dic_regions.keys()]
        
    base_template = os.path.join(
        "eval", "metrics", "$model", "submodel_forecasts", "$sn", task
    )

    metrics_ds = xr.Dataset(
        coords={"time": get_target_dates(target_dates, horizon=horizon)}
    )

    list_model_names = [x for x in os.listdir('models')]
    list_submodel_names = [
        os.path.basename(os.path.normpath(x))
        for x in glob(os.path.join("models", "*", "submodel_forecasts", "*"))
    ]

    def build_filename(model, sn, region=None):
        if region:
            fname = f"{metric}_{region}-{task}-{target_dates}.zarr"
        else:
            fname = f"{metric}-{task}-{target_dates}.zarr"
        return os.path.join(base_template, fname)\
                 .replace("$model", model)\
                 .replace("$sn", sn)

    def load_dataset(filename, model_name):
        if metric == "rps":
            ds = xr.open_zarr(filename).drop_vars("rpss").rename({metric: model_name}).load()
        else:
            ds = xr.open_zarr(filename).rename({metric: model_name}).load()
        return ds

    for model_name in model_names:

        # -------------------------
        # Resolve submodel name
        # -------------------------
        if model_name in list_model_names:
            if model_name.startswith("proj_") and 'clip' not in model_name:
                model_name_base = model_name.replace("proj_", "")
                sn = get_selected_submodel_name(
                    model=model_name_base,
                    gt_id=gt_id,
                    horizon=horizon,
                    target_dates=target_dates
                )
                sn = "proj_" + sn
            else:
                sn = get_selected_submodel_name(
                    model=model_name,
                    gt_id=gt_id,
                    horizon=horizon,
                    target_dates=target_dates
                )
            model_folder = model_name

        elif model_name in list_submodel_names:
            sn = model_name
            path = glob(os.path.join("models", "*", "submodel_forecasts", sn))[0]
            model_folder = os.path.basename(
                os.path.normpath(Path(path).resolve().parents[1])
            )
        else:
            continue

        # -------------------------
        # Load datasets
        # -------------------------
        if regions:
            regional_datasets = []
            loaded_regions = []
            missing_regions = []

            for region in regions:
                filename = build_filename(model_folder, sn, region)

                try:
                    ds_region = load_dataset(filename, model_name)

                    # Ensure region dimension exists
                    if "region" not in ds_region.dims:
                        ds_region = ds_region.expand_dims({"region": [region]})

                    regional_datasets.append(ds_region)
                    loaded_regions.append(region)

                except FileNotFoundError:
                    missing_regions.append(region)
                    if verbose:
                        print(f"\tMissing region '{region}' for model {model_name}")
                    continue

            # No regions found → skip model
            if len(regional_datasets) == 0:
                if verbose:
                    print(f"\tNo regional metrics found for model {model_name} (regions={regions})")
                continue

            # Warn if partial coverage
            if missing_regions and verbose:
                print(f"\tModel {model_name}: missing regions {missing_regions}, using {loaded_regions}")

            # Combine regions
            model_ds = xr.concat(regional_datasets, dim="region")

            if combine_regions == "mean":
                model_ds = model_ds.mean(dim="region")

        else:
            filename = build_filename(model_folder, sn)

            try:
                model_ds = load_dataset(filename, model_name)
            except FileNotFoundError:
                if verbose:
                    print(f"\tNo metrics for model {model_name}")
                continue

        # -------------------------
        # Align to target dates
        # -------------------------
        model_ds = model_ds.reindex(time=metrics_ds.time)

        # -------------------------
        # Detect missing dates
        # -------------------------
        data = model_ds[model_name]
        missing_mask = data.isnull()

        if "region" in data.dims:
            # consider missing only if ALL regions missing
            missing_mask = missing_mask.all(dim="region")

        missing_dates = model_ds.time.where(missing_mask, drop=True)

        if "msn" in model_name or "duet" in model_name or "best" in model_name:
            missing_dates = missing_dates.where(
                missing_dates.dt.year == 2024, drop=True
            )

        missing_dates = pd.to_datetime(missing_dates.values)\
                         .strftime('%Y-%m-%d')\
                         .tolist()

        if len(missing_dates) > 0 and verbose:
            print(f"Model {model_name} has NaN values on dates: {missing_dates}")

        # -------------------------
        # Drop NaNs (per-model)
        # -------------------------
        if common_dates:
            model_ds = model_ds.dropna(dim='time')
        else:
            model_ds = model_ds.dropna(dim='time', how='all')

        # -------------------------
        # Merge
        # -------------------------
        metrics_ds = xr.merge([metrics_ds, model_ds], join="left")

    # Final cleanup
    if common_dates:
        metrics_ds = metrics_ds.dropna(dim='time')
    else:
        metrics_ds = metrics_ds.dropna(dim='time', how='all')

    if verbose:
        # Display number of time points
        print(f"\tTask {task} for {target_dates} has n={len(metrics_ds.time)} time target dates.")

    if len(metrics_ds.data_vars) == 0:
        return None

    return metrics_ds




def get_all_metrics(model_names=['ecmwf', 'pbc_ecmwf', 'climatology'],
                    model_names_str="ECMWF-based models",
                    metrics=['wtd_mse'],
                    gt_ids=["era5-f1_tas", 
                           "era5-f2_tas", 
                           "era5-f3_tas", 
                           "era5-f4_tas", 
                           "era5-f1_pr", 
                           "era5-f2_pr", 
                           "era5-f3_pr", 
                           "era5-f4_pr", 
                           "era5-f1_mslp",
                           "era5-f2_mslp",
                           "era5-f3_mslp",
                           "era5-f4_mslp"],
                    horizons=["19", "26"],
                    target_dates_list=["std_test"],
                    common_dates=True,
                    verbose=True):

    """
    Generate a dictionary with metric values for all models and every combination of gt_id, 
    horizon, and target dates
    """
    all_metrics = {}
    
    # Get metrics for all experiments
    for metric, gt_id, horizon, target_dates in \
            [x for x in product(metrics, gt_ids, horizons, target_dates_list)]:  
        
            
        # Get task
        task = f"{gt_id}_{horizon}"
    
        if verbose:
            display(Markdown(f"### {model_names_str}: {metric}, {task}, {target_dates}"))
    
        # Get all metrics
        ds = get_metrics_ds(gt_id, horizon, metric, target_dates, model_names=model_names, common_dates=common_dates, verbose=verbose)
        # No models exist for this task    
        if ds is None: 
            continue
    
        all_metrics[(metric, task, target_dates)] = copy.copy(ds)
                
    return all_metrics

    
def get_all_rps(model_names=['ecmwf', 'pbc_ecmwf', 'climatology'],
                model_names_str="ECMWF-based models",
                gt_ids=["era5-tas", 
                        "era5-pr", 
                        "era5-mslp"],
                horizons=["19", "26"],
                target_dates_list=["std_test"],
                regions=None,  
                common_dates=True,
                verbose=True):

    """
    Generate a dictionary with rps values for all models and every combination 
    of gt_id, horizon, and target dates.

    Args:
        regions (list or None): e.g. ['us', 'europe']
    """

    all_rps = {}
    
    # Get metrics for all experiments
    for det_gt_id, horizon, target_dates in product(gt_ids, horizons, target_dates_list):  
            
        task = f"{det_gt_id}_{horizon}"
    
        if verbose:
            if regions:
                display(Markdown(f"### {model_names_str}: rps, {task}, {target_dates}, regions={regions}"))
            else:
                display(Markdown(f"### {model_names_str}: rps, {task}, {target_dates}"))
   
        # Use regional-aware function
        ds = get_metrics_ds(
            gt_id=det_gt_id, 
            horizon=horizon, 
            metric="rps", 
            target_dates=target_dates, 
            model_names=model_names,
            regions=regions,
            common_dates=common_dates,
            verbose=verbose
        )

        # No models exist for this task
        if ds is None:
            continue

        all_rps[(task, target_dates)] = copy.copy(ds)
                
    return all_rps


    
def get_seasonal_rpss(model_names=['ecmwf', 'pbc_ecmwf', 'climatology'],
                    model_names_str="ECMWF-based models",
                    gt_ids=["era5-tas", 
                            "era5-pr", 
                            "era5-mslp"],
                    horizons=["19", "26"],
                    target_dates_list=["std_test"],
                    common_dates=True,
                    verbose=True):

    all_rps = get_all_rps(model_names=list(set(model_names+['climatology'])),
                      model_names_str=model_names_str,
                      gt_ids=gt_ids,
                      horizons=horizons,
                      target_dates_list=target_dates_list,
                      common_dates=common_dates,
                      verbose=verbose)
    
    # Initialize empty dictionary of daily RPSS datasets
    all_seasonal_rpss = {}
    
    # Display daily RPSS for all models for each task 
    for (task, target_dates), ds in all_rps.items():
        if ds is None:
            continue
            
        if "climatology" in ds.data_vars:
            # Use resample to compute the mean over each season (DJF, MAM, JJA, SON)
            # with DJF assigned to the December year
            ds = ds.resample(time='QS-DEC').mean(dim="time")
            # Calculate RPSS for each model
            for model in ds.data_vars:
                if model != "climatology":
                    # Calculate daily RPSS: 1 - (model_rps / climatology_rps)
                    rpss = 1 - (ds[model] / ds["climatology"])
                    # Store in dictionary
                    if task not in all_seasonal_rpss:
                        all_seasonal_rpss[task] = rpss.to_dataset(name=model)
                    else:
                        all_seasonal_rpss[task][model] = rpss
            
        else:
            print(f"Warning: climatology RPS series not found for task {task}")
    
    return all_seasonal_rpss

       



def get_daily_rpss(model_names=['climatology', 'ecmwf', 'msn', 'pbc_msn', 'duet'],
                   model_names_str="ECMWF-based models",
                   gt_ids=["era5-tas", 
                           "era5-pr", 
                           "era5-mslp"],
                   horizons=["19", "26"],
                   target_dates_list=["std_test"],
                   regions=None,   
                   common_dates=True,
                   verbose=True):

    """
    Compute daily RPSS for all models.

    Args:
        regions (list or None): e.g. ['us', 'europe']
    """

    all_rps = get_all_rps(
        model_names=list(set(model_names + ['climatology'])),
        model_names_str=model_names_str,
        gt_ids=gt_ids,
        horizons=horizons,
        target_dates_list=target_dates_list,
        regions=regions,
        common_dates=common_dates,
        verbose=verbose
    )
    
    all_daily_rpss = {}
    
    for (task, target_dates), ds in all_rps.items():
        if ds is None:
            continue            
        if "climatology" not in ds.data_vars:
            print(f"Warning: climatology RPS series not found for task {task}")
            continue
        target_time = get_target_dates(target_dates, horizon=task.split('_')[1])
        ds = ds.reindex(time=target_time)
        clim = ds["climatology"]
        for model in ds.data_vars:
            if model == "climatology":
                continue
            model_rps = ds[model]
            rpss = 1 - (model_rps / clim)
            if task not in all_daily_rpss:
                all_daily_rpss[task] = rpss.to_dataset(name=model)
            else:
                all_daily_rpss[task][model] = rpss

    return all_daily_rpss

def lower_confidence_bound(input, clim=None, n_boot=5000, seed=42, 
                           confidence=0.95, verbose=False):
    """Return a lower confidence bound using the stationary block bootstrap 
    with automatic block length selection and bias-corrected and accelerated 
    (BCa) confidence intervals.

    If clim is None, provides a lower confidence bound for the expected input.
    Otherwise, provides a lower confidence bound for E[input] / E[clim].

    Args:
        input: 1-D array of input values
        clim: None or parallel array of climatology values
        n_boot: number of bootstrap replicates
        seed: random seed for reproducibility
        confidence: confidence level 
        verbose: print verbose messages?
    """
    # Select block length for stationary bootstrap using input values
    block_data = optimal_block_length(input)
    # Always select an even integer block length
    b_opt = int(block_data['stationary'].iloc[0])
    if b_opt % 2 == 1:
        b_opt += 1
    if verbose:
        printf(f"Selected block length: {b_opt}")

    # Initialize stationary bootstrap using input values or indices
    n = len(input)
    init = input if clim is None else np.arange(n)
    boot = StationaryBootstrap(b_opt, init, seed=seed)

    # Compute a lower BCa confidence bound
    def mean_ratio(inds):
        return input[inds].mean()/clim[inds].mean()
    lower_cb = boot.conf_int(
        func=np.mean if clim is None else mean_ratio, 
        reps=n_boot, 
        method='bca', 
        size=confidence,
        tail='lower'
    )[0][0]
    return lower_cb
        
def plot_rpss_barplot(
    all_daily_rps,
    model_names=None,
    variable_models=None,
    baseline_models=None,
    target_dates="std_test",
    show_fig=True,
    save_fig=True,
    by_season=False,
    y_bottom=None,
    y_top=None,
    week_gap=1.0,
    variable_gap=1.2,
    max_bar_width=0.18,
    legend_location="auto",      # "auto", "top", "right"
    legend_order=None,
    legend_ncols=None,
    n_boot=5000,
    seed=42,
    source_data=False,
    source_data_filename="fig_0-rpss_barplot.xlsx",
    source_data_sheet_prefix="Fig2a",
    verbose=False,
    suffix=""
):
    """
    Plots ranked probability skill score (RPSS) barplots. 

    Adds cross hatching if a non-baseline model improves significantly
    over all baseline models and single hatching if it improves
    significantly over at least one baseline model. Significance is determined 
    using lower_confidence_bound().

    Args:
        all_daily_rps: dictionary of daily ranked probability score (RPS) 
          datasets for each task
        model_names: list of model names to include (if variable_models 
          is None)
        variable_models: dictionary of variable names to lists of model names
        baseline_models: list of baseline model names (if None, all but the 
          last model in each variable_models list are baselines)
        target_dates: target dates for metric calculations, e.g. "std_test"
        show_fig: whether to display the figure
        save_fig: whether to save the figure
        by_season: whether to plot by season
        y_bottom: bottom limit for y-axis
        y_top: top limit for y-axis
        week_gap: gap between weeks in the plot
        variable_gap: gap between variables in the plot
        max_bar_width: maximum width of bars
        legend_location: location of the legend ("auto", "top", "right")
        legend_order: order of legend entries
        legend_ncols: number of columns in the legend
        n_boot: number of bootstrap samples
        seed: random seed for reproducibility
        verbose: whether to print verbose output
        suffix: suffix for figure filenames
    """
    variables = {
        "Temperature": "tas",
        "Precipitation": "pr",
        "Sea Level Pressure": "mslp",
    }

    horizons = {
        "Week 3": 19,
        "Week 4": 26,
    }

    # ----------------------------------------------------------
    # Normalize model specification
    # ----------------------------------------------------------
    fig_width = 16 
        
    if variable_models is None:
        if model_names is None:
            raise ValueError(
                "Either model_names or variable_models must be provided."
            )
        model_names = [m for m in model_names if m != "climatology"]
        variable_models = {
            var: list(model_names)
            for var in variables
        }
    else:
        variable_models = {
            var: [m for m in models if m != "climatology"]
            for var, models in variable_models.items()
        }

    if baseline_models is None:
        baseline_models_by_var = {
            var: models[:-1]
            for var, models in variable_models.items()
        }
        target_models_by_var = {
            var: models[-1:] if models else []
            for var, models in variable_models.items()
        }
    else:
        baseline_set = set(baseline_models)
        baseline_models_by_var = {
            var: [m for m in models if m in baseline_set]
            for var, models in variable_models.items()
        }
        target_models_by_var = {
            var: [m for m in models if m not in baseline_set]
            for var, models in variable_models.items()
        }

    # ----------------------------------------------------------
    # Compute mean RPSS + statistical significance
    # ----------------------------------------------------------   
    results = {}
    results_all_significant = {}
    results_some_significant = {}
    for var_name, var_code in variables.items():
        for week_name, horizon in horizons.items():
            ds = all_daily_rps[f"era5-{var_code}_{horizon}"]
            results[(var_name, week_name)] = {}
            for model in variable_models[var_name]:
                values = ds[model].values.flatten()
                results[(var_name, week_name)][model] =1-np.mean(values[~np.isnan(values)])/np.mean(ds["climatology"].values.flatten()[~np.isnan(values)])
            target_all_significant = {}
            target_some_significant = {}
            target_models = target_models_by_var[var_name]
            baseline_models_for_var = baseline_models_by_var[var_name]

            clim = ds["climatology"].values.flatten()
            for target_model in target_models:
                if target_model not in ds:
                    target_all_significant[target_model] = False
                    target_some_significant[target_model] = False
                    continue

                target = ds[target_model].values.flatten()
                all_significant = bool(baseline_models_for_var)
                some_significant = False

                for baseline_model in baseline_models_for_var:
                    if baseline_model not in ds:
                        all_significant = False
                        continue

                    baseline = ds[baseline_model].values.flatten()
                    mask = (
                        ~np.isnan(baseline)
                        & ~np.isnan(target)
                        & ~np.isnan(clim)
                    )

                    if not np.any(mask):
                        all_significant = False
                        continue

                    # Compute lower confidence bound for RPSS improvement
                    # of the target model over the baseline model.
                    lower_cb = lower_confidence_bound(
                        baseline[mask] - target[mask],
                        clim=clim[mask],
                        n_boot=n_boot,
                        seed=seed,
                    )
                    if verbose:
                        print(
                            f"  {var_name} | {week_name} | "
                            f"{target_model} - {baseline_model} RPSS confidence bound: {lower_cb}"
                        )
                    if lower_cb <= 0:
                        all_significant = False
                    else:
                        some_significant = True

                target_all_significant[target_model] = all_significant
                target_some_significant[target_model] = some_significant

            results_all_significant[(var_name, week_name)] = target_all_significant
            results_some_significant[(var_name, week_name)] = target_some_significant

    # ----------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------
    categories = [
        (v, w)
        for v in variables
        for w in horizons
    ]

    x = []
    current = 0
    for _ in variables:
        x.append(current)
        x.append(current + week_gap)
        current += week_gap + variable_gap

    x = np.asarray(x)
    max_models = max(
        len(v)
        for v in variable_models.values()
    )

    bar_width = min(0.8 / max_models, max_bar_width)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    legend_seen = set()
    
    # ----------------------------------------------------------
    # Draw bars
    # ----------------------------------------------------------
    legend_seen = set()
    for group_idx, (var_name, week_name) in enumerate(categories):
        models = variable_models[var_name]
        offsets = (
            np.arange(len(models))
            - (len(models) - 1) / 2
        ) * bar_width

        for offset, model in zip(offsets, models):
            label = (
                all_model_names[model]
                if model not in legend_seen
                else None
            )
            bars = ax.bar(
                x[group_idx] + offset,
                results[(var_name, week_name)][model],
                width=bar_width,
                color=model_colors.get(model, "gray"),
                label=label,
            )
            legend_seen.add(model)
            # Add hatch if this model is a target model that significantly
            # improved over every baseline model.
            if results_all_significant[(var_name, week_name)].get(model, False):
                for bar in bars:
                    bar.set_hatch("x")
            elif results_some_significant[(var_name, week_name)].get(model, False):
                for bar in bars:
                    bar.set_hatch("/")

    # ----------------------------------------------------------
    # X labels
    # ----------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        [week for _, week in categories],
        fontsize=20,
    )

    pair_centers = [
        np.mean(x[0:2]),
        np.mean(x[2:4]),
        np.mean(x[4:6]),
    ]

    for xc, var in zip(pair_centers, variables.keys()):

        ax.text(
            xc,
            -0.12,
            var,
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_ylabel(
        "Ranked probability skill score",
        fontsize=20,
        fontweight="bold",
    )

    ax.tick_params(axis="y", labelsize=15)

    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.xaxis.grid(False)

    if y_bottom is not None or y_top is not None:

        ax.set_ylim(
            bottom=y_bottom,
            top=y_top,
        )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()

    # Build plain (non-hatched) legend handles so significance hatching appears
    # only on bars, not in the legend.
    plain_handle_by_label = {
        label: plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=handle.patches[0].get_facecolor(),
        )
        for handle, label in zip(handles, labels)
    }
    handles = [plain_handle_by_label[label] for label in labels]

    if legend_order is not None:

        lookup = dict(zip(labels, handles))

        ordered_labels = [
            all_model_names[m]
            for m in legend_order
            if all_model_names[m] in lookup
        ]

        ordered_handles = [
            lookup[label]
            for label in ordered_labels
        ]

        handles = ordered_handles
        labels = ordered_labels

    if legend_location == "auto":

        same_models = all(
            variable_models[v] == next(iter(variable_models.values()))
            for v in variable_models
        )

        legend_location = "top" if same_models else "right"

    if legend_location == "top":

        if legend_ncols is None:

            if len(labels) <= 5:
                legend_ncols = len(labels)
            else:
                legend_ncols = math.ceil(len(labels) / 2)

        ax.legend(
            handles,
            labels,
            ncol=legend_ncols,
            fontsize=20,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.5,
            handletextpad=0.5,
        )

        plt.tight_layout()

    else:

        ax.legend(
            handles,
            labels,
            fontsize=15,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )

        plt.tight_layout(rect=[0, 0, 0.82, 1])


    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    def dict_to_dataframe(data, value_name):
        """
        Convert {(variable, horizon): {model: value}} to a tidy DataFrame.
        """
        rows = []

        for (variable, horizon), models in data.items():
            for model, value in models.items():
                rows.append({
                    "Variable": variable,
                    "Horizon": horizon,
                    "Model": model,
                    value_name: value,
                })

        return pd.DataFrame(rows)

    if source_data:

        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        results_df = dict_to_dataframe(
            results,
            "RPSS",
        )

        results_all_significant_df = dict_to_dataframe(
            results_all_significant,
            "Significant_All",
        )

        results_some_significant_df = dict_to_dataframe(
            results_some_significant,
            "Significant_Some",
        )

        # ------------------------------------------------------
        # Write to the existing workbook if it exists.
        #
        # This is important because Fig. 2a and Fig. 2b use
        # the same XLSX file. Only the sheets belonging to this
        # function are replaced; other sheets are preserved.
        # ------------------------------------------------------
        if os.path.exists(fig_filename):

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:

                results_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_results",
                    index=False,
                )

                results_all_significant_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_significant_all",
                    index=False,
                )

                results_some_significant_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_significant_some",
                    index=False,
                )

        else:

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="w",
            ) as writer:

                results_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_results",
                    index=False,
                )

                results_all_significant_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_significant_all",
                    index=False,
                )

                results_some_significant_df.to_excel(
                    writer,
                    sheet_name=f"{source_data_sheet_prefix}_significant_some",
                    index=False,
                )

        print(f"Source data saved: {fig_filename}")

                    
    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------

    if by_season:

        filename = os.path.join(
            OUT_DIR,
            f"barplot_seasonal_rpss{suffix}_{target_dates}.pdf",
        )

    else:

        filename = os.path.join(
            OUT_DIR,
            f"barplot_daily_rpss{suffix}_{target_dates}.pdf",
        )

    if save_fig:

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        plt.savefig(
            filename.replace(".pdf", ".jpeg"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        plt.savefig(
            filename.replace(".pdf", ".eps"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        print(f"Figure saved: {filename}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def confidence_interval(input, clim=None, n_boot=5000, seed=42, 
                        confidence=0.95, verbose=False):
    """Return a confidence interval using the stationary block bootstrap 
    with automatic block length selection and bias-corrected and accelerated 
    (BCa) confidence intervals.

    If clim is None, provides a confidence interval for the expected input.
    Otherwise, provides a confidence interval for E[input] / E[clim].

    Args:
        input: 1-D array of input values
        clim: None or parallel array of climatology values
        n_boot: number of bootstrap replicates
        seed: random seed for reproducibility
        confidence: confidence level 
        verbose: print verbose messages?
    """
    # Select block length for stationary bootstrap using input values
    block_data = optimal_block_length(input)
    # Always select an even integer block length
    b_opt = int(block_data['stationary'].iloc[0])
    if b_opt % 2 == 1:
        b_opt += 1
    if verbose:
        printf(f"Selected block length: {b_opt}")

    # Initialize stationary bootstrap using input values or indices
    n = len(input)
    init = input if clim is None else np.arange(n)
    boot = StationaryBootstrap(b_opt, init, seed=seed)

    # Compute a BCa confidence interval
    def mean_ratio(inds):
        return input[inds].mean()/clim[inds].mean()
    ci = boot.conf_int(
        func=np.mean if clim is None else mean_ratio, 
        reps=n_boot, 
        method='bca', 
        size=confidence,
        tail='two'
    )
    # Return the lower and upper limits
    return ci[0][0], ci[1][0]


def plot_rpss_ci_barplot(
    all_daily_rps,
    model_names=None,
    variable_models=None,
    target_dates="std_test",
    show_fig=True,
    save_fig=True,
    by_season=False,
    y_bottom=None,
    y_top=None,
    week_gap=1.0,
    variable_gap=1.2,
    max_bar_width=0.18,
    legend_location="auto",
    legend_order=None,
    legend_ncols=None,
    n_boot=5000,
    seed=42,
    source_data=False,
    source_data_filename="fig_0-rpss_ci_barplot.xlsx",
    verbose=False,
    suffix=""
):
    """
    Plots ranked probability skill score (RPSS) barplots with 95% bootstrap
    confidence intervals for each model bar.

    If source_data=True, saves the numerical data used to recreate the
    figure to an Excel workbook.

    Args:
        all_daily_rps: dictionary of daily ranked probability score (RPS)
          datasets for each task
        model_names: list of model names to include (if variable_models
          is None)
        variable_models: dictionary of variable names to lists of model names
        target_dates: target dates for metric calculations, e.g. "std_test"
        show_fig: whether to display the figure
        save_fig: whether to save the figure
        by_season: whether to plot by season
        y_bottom: bottom limit for y-axis
        y_top: top limit for y-axis
        week_gap: gap between weeks in the plot
        variable_gap: gap between variables in the plot
        max_bar_width: maximum width of bars
        legend_location: location of the legend ("auto", "top", "right")
        legend_order: order of legend entries
        legend_ncols: number of columns in the legend
        n_boot: number of bootstrap samples
        seed: random seed for reproducibility
        source_data: whether to save source data to Excel
        source_data_filename: filename for the Excel source-data file
        verbose: whether to print verbose output
        suffix: suffix for figure filenames
    """

    variables = {
        "Temperature": "tas",
        "Precipitation": "pr",
        "Sea Level Pressure": "mslp",
    }

    horizons = {
        "Week 3": 19,
        "Week 4": 26,
    }

    # ----------------------------------------------------------
    # Normalize model specification
    # ----------------------------------------------------------
    fig_width = 16 if variable_models is None else 22

    if variable_models is None:
        if model_names is None:
            raise ValueError(
                "Either model_names or variable_models must be provided."
            )

        model_names = [
            m for m in model_names
            if m != "climatology"
        ]

        variable_models = {
            var: list(model_names)
            for var in variables
        }

    else:
        variable_models = {
            var: [
                m for m in models
                if m != "climatology"
            ]
            for var, models in variable_models.items()
        }

    # ----------------------------------------------------------
    # Compute mean RPSS + confidence intervals
    # ----------------------------------------------------------
    results = {}
    ci_lower = {}
    ci_upper = {}

    for var_name, var_code in variables.items():

        for week_name, horizon in horizons.items():

            ds = all_daily_rps[
                f"era5-{var_code}_{horizon}"
            ]

            results[(var_name, week_name)] = {}
            ci_lower[(var_name, week_name)] = {}
            ci_upper[(var_name, week_name)] = {}

            clim = ds["climatology"].values.flatten()

            for model in variable_models[var_name]:

                values = ds[model].values.flatten()

                mask = (
                    (~np.isnan(values))
                    & (~np.isnan(clim))
                )

                if not np.any(mask):

                    results[(var_name, week_name)][model] = np.nan
                    ci_lower[(var_name, week_name)][model] = np.nan
                    ci_upper[(var_name, week_name)][model] = np.nan

                    continue

                values_masked = values[mask]
                clim_masked = clim[mask]

                # --------------------------------------------------
                # RPSS
                # --------------------------------------------------
                mean_ratio = (
                    np.mean(values_masked)
                    / np.mean(clim_masked)
                )

                rpss = 1 - mean_ratio

                results[
                    (var_name, week_name)
                ][model] = rpss

                # --------------------------------------------------
                # Bootstrap confidence interval
                # --------------------------------------------------
                ratio_ci_low, ratio_ci_high = confidence_interval(
                    values_masked,
                    clim=clim_masked,
                    n_boot=n_boot,
                    seed=seed,
                    verbose=verbose
                )

                # Ratio CI -> RPSS CI
                #
                # RPSS = 1 - ratio
                #
                # Therefore:
                # lower RPSS bound = 1 - upper ratio bound
                # upper RPSS bound = 1 - lower ratio bound
                rpss_ci_low = 1 - ratio_ci_high
                rpss_ci_high = 1 - ratio_ci_low

                ci_lower[
                    (var_name, week_name)
                ][model] = rpss_ci_low

                ci_upper[
                    (var_name, week_name)
                ][model] = rpss_ci_high

                if verbose:
                    print(
                        f"  {var_name} | {week_name} | {model} "
                        f"RPSS 95% CI: "
                        f"[{rpss_ci_low}, {rpss_ci_high}]"
                    )

    # ----------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------
    categories = [
        (v, w)
        for v in variables
        for w in horizons
    ]

    x = []
    current = 0

    for _ in variables:
        x.append(current)
        x.append(current + week_gap)
        current += week_gap + variable_gap

    x = np.asarray(x)

    max_models = max(
        len(v)
        for v in variable_models.values()
    )

    bar_width = min(
        0.8 / max_models,
        max_bar_width
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 6)
    )

    legend_seen = set()

    # ----------------------------------------------------------
    # Draw bars with CI error bars
    # ----------------------------------------------------------
    for group_idx, (var_name, week_name) in enumerate(categories):

        models = variable_models[var_name]

        offsets = (
            np.arange(len(models))
            - (len(models) - 1) / 2
        ) * bar_width

        for offset, model in zip(offsets, models):

            label = (
                all_model_names[model]
                if model not in legend_seen
                else None
            )

            y = results[
                (var_name, week_name)
            ][model]

            low = ci_lower[
                (var_name, week_name)
            ][model]

            high = ci_upper[
                (var_name, week_name)
            ][model]

            if (
                np.isnan(y)
                or np.isnan(low)
                or np.isnan(high)
            ):
                yerr = None
            else:
                yerr = np.array([
                    [max(0, y - low)],
                    [max(0, high - y)]
                ])

            ax.bar(
                x[group_idx] + offset,
                y,
                width=bar_width,
                color=model_colors.get(
                    model,
                    "gray"
                ),
                label=label,
                yerr=yerr,
                capsize=4,
                error_kw=dict(
                    ecolor="gray",
                    elinewidth=1.5,
                    capthick=1.5
                )
            )

            legend_seen.add(model)

    # ----------------------------------------------------------
    # X labels
    # ----------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        [week for _, week in categories],
        fontsize=20,
    )

    pair_centers = [
        np.mean(x[0:2]),
        np.mean(x[2:4]),
        np.mean(x[4:6]),
    ]

    for xc, var in zip(
        pair_centers,
        variables.keys()
    ):

        ax.text(
            xc,
            -0.12,
            var,
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_ylabel(
        "Ranked probability skill score",
        fontsize=20,
        fontweight="bold",
    )

    ax.tick_params(
        axis="y",
        labelsize=15
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    ax.xaxis.grid(False)

    if y_bottom is not None or y_top is not None:

        ax.set_ylim(
            bottom=y_bottom,
            top=y_top,
        )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    handles, labels = (
        ax.get_legend_handles_labels()
    )

    if legend_order is not None:

        lookup = dict(
            zip(labels, handles)
        )

        ordered_labels = [
            all_model_names[m]
            for m in legend_order
            if all_model_names[m] in lookup
        ]

        ordered_handles = [
            lookup[label]
            for label in ordered_labels
        ]

        handles = ordered_handles
        labels = ordered_labels

    if legend_location == "auto":

        same_models = all(
            variable_models[v]
            == next(
                iter(variable_models.values())
            )
            for v in variable_models
        )

        legend_location = (
            "top"
            if same_models
            else "right"
        )

    if legend_location == "top":

        if legend_ncols is None:

            if len(labels) <= 5:
                legend_ncols = len(labels)
            else:
                legend_ncols = math.ceil(
                    len(labels) / 2
                )

        ax.legend(
            handles,
            labels,
            ncol=legend_ncols,
            fontsize=20,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=1.5,
            handletextpad=0.5,
        )

        plt.tight_layout()

    else:

        ax.legend(
            handles,
            labels,
            fontsize=15,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )

        plt.tight_layout(
            rect=[0, 0, 0.82, 1]
        )

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        source_data_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        source_rows = []

        for (var_name, week_name) in categories:

            # Convert the human-readable week name back
            # to the numerical horizon.
            horizon = horizons[week_name]

            for model in variable_models[var_name]:

                rpss = results[
                    (var_name, week_name)
                ][model]

                lower = ci_lower[
                    (var_name, week_name)
                ][model]

                upper = ci_upper[
                    (var_name, week_name)
                ][model]

                if (
                    np.isnan(rpss)
                    or np.isnan(lower)
                    or np.isnan(upper)
                ):
                    error_lower = np.nan
                    error_upper = np.nan
                else:
                    error_lower = max(
                        0,
                        rpss - lower
                    )
                    error_upper = max(
                        0,
                        upper - rpss
                    )

                source_rows.append({
                    "Variable": var_name,
                    "Variable_code": variables[var_name],
                    "Horizon": horizon,
                    "Horizon_label": week_name,
                    "Model": model,
                    "Model_label": all_model_names.get(
                        model,
                        model
                    ),
                    "RPSS": rpss,
                    "CI_lower": lower,
                    "CI_upper": upper,
                    "CI_error_lower": error_lower,
                    "CI_error_upper": error_upper,
                })

        source_df = pd.DataFrame(
            source_rows
        )

        # Metadata is useful for documenting exactly how
        # the bootstrap CIs were generated.
        metadata_df = pd.DataFrame({
            "Parameter": [
                "target_dates",
                "n_boot",
                "seed",
                "by_season",
            ],
            "Value": [
                target_dates,
                n_boot,
                seed,
                by_season,
            ]
        })

        with pd.ExcelWriter(
            source_data_filename,
            engine="openpyxl",
            mode="w",
        ) as writer:

            source_df.to_excel(
                writer,
                sheet_name="results",
                index=False,
            )

            metadata_df.to_excel(
                writer,
                sheet_name="metadata",
                index=False,
            )

        print(
            f"Source data saved: "
            f"{source_data_filename}"
        )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if by_season:

        filename = os.path.join(
            OUT_DIR,
            f"barplot_seasonal_rpss_ci"
            f"{suffix}_{target_dates}.pdf",
        )

    else:

        filename = os.path.join(
            OUT_DIR,
            f"barplot_daily_rpss_ci"
            f"{suffix}_{target_dates}.pdf",
        )

    if save_fig:

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        plt.savefig(
            filename.replace(".pdf", ".jpeg"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        plt.savefig(
            filename.replace(".pdf", ".eps"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        print(
            f"Figure saved: {filename}"
        )

    if show_fig:
        plt.show()
    else:
        plt.close(fig)





def get_all_lat_lon_rpss(model_names = ['ecmwf', 'pbc_ecmwf', 'msn', 'pbc_msn', 'duet'],
                        gt_ids = ['era5-tas', 'era5-pr', 'era5-mslp'],
                        horizons = [19, 26],
                        metric = "lat_lon_rpss",
                        target_dates = "std_test"):

    metrics_dic = {}
    for gt_id, horizon in product(gt_ids, horizons):
        
        # Set task
        task = f"{gt_id}_{horizon}"
        
        # Create metric filename template
        in_file = Template(os.path.join("eval","metrics", "$model", "submodel_forecasts", 
                                            "$sn", task, f"{metric}-{task}-{target_dates}.zarr"))
        ds_list = []
        for model_name in model_names:
            sn = get_selected_submodel_name(model_name, gt_id, horizon)
            
           # Form the metric filename
            filename = in_file.substitute(model=model_name, sn=sn)
            if metric == 'lat_lon_rpss':
                filename = filename.replace('lat_lon_rpss','lat_lon_rps')
            
            # Open dataset and select metric
            da = xr.open_dataset(filename, engine="zarr")[metric]
            
            # Rename variable to model name
            da = da.rename(model_name)
            
            # Convert DataArray -> Dataset
            ds_model = da.to_dataset()
            
            ds_list.append(ds_model)
            
        # Merge all models into one dataset
        ds_merged = xr.merge(ds_list, compat="override")
        metrics_dic[task] = ds_merged
        
    return metrics_dic


def plot_rpss_bar_threshold(
    metrics_dic,
    model_names=['duet', 'pbc_msn', 'msn', 'ecmwf'],
    gt_id="era5-tas",
    horizon=19,
    target_dates="std_test",
    thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    show_fig=True,
    save_fig=True,
):

    task = f"{gt_id}_{horizon}"
    ds = metrics_dic[task]

    records = []

    # Prepare data
    for m in model_names:
        if m not in ds.data_vars:
            print(f"Missing model {m} in {task}")
            continue

        da = ds[m]

        total = da.notnull().sum().item()

        for thr in thresholds:
            count_above = (da >= thr).sum().item()
            frac = count_above / total if total > 0 else np.nan

            records.append({
                "model": m,
                "rpss_threshold": thr,
                "fraction_above": frac
            })

    df_barplot = pd.DataFrame(records)

    
    fig, ax = plt.subplots(figsize=(10, 8))

    n_models = len(model_names)
    n_thresholds = len(thresholds)
    width = 0.15  # width of each bar

    x = np.arange(n_thresholds)  # the x locations for the groups

    for i, m in enumerate(model_names):
        model_data = df_barplot[df_barplot["model"] == m]["fraction_above"].values
        ax.bar(
            x + (i - (n_models-1)/2) * width,  # center bars around x
            model_data,
            width=width,
            color=model_colors.get(m, "gray"),
            label=all_model_names.get(m, m)
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(thr) for thr in thresholds], fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("RPSS threshold", fontweight="bold", labelpad=15)
    ax.set_ylabel("Fraction of grid cells\nabove threshold", fontweight="bold", labelpad=15)
    ax.set_title(f"{gt_id_names[gt_id]}, {horizon_names[horizon]}", fontweight="bold", pad=20)
    ax.xaxis.grid(False)
    ax.legend(loc="upper right", fontsize=18)

    plt.tight_layout()

    filename = os.path.join(
        OUT_DIR,
        f"lat_lon_rpss_by_threshold_{task}_{target_dates}.pdf"
    )

    if save_fig:
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Figure saved: {filename}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_seasonal_rpss_all_tasks(all_seasonal_rpss,
                                 model_names=["ecmwf", "pbc_ecmwf", "msn", "duet"],
                                 gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
                                 horizons=[19, 26],
                                 metric='rpss',
                                 target_dates='std_test',   
                                 show_fig=True,
                                 save_fig=False):

    model_markers = {
        "ecmwf": "s",
        "debiased_ecmwf": "v",
        "pbc_ecmwf": "^",
        "msn": "D",
        "pbc_msn": "o",
        "duet": "X",
        "climatology": "P",
    }

    yaxis_limits = {
        "era5-tas": (-0.3, 0.3),
        "era5-pr": (-0.05, 0.1),
        "era5-mslp": (-0.2, 0.2),
    }

    num_horizons = len(horizons)
    num_gt_ids = len(gt_ids)

    fig, ax = plt.subplots(
        num_horizons,
        num_gt_ids,
        figsize=(18, 8),
        constrained_layout=True
    )

    # Ensure ax is always 2D
    if num_horizons == 1:
        ax = np.expand_dims(ax, axis=0)

    for i, horizon in enumerate(horizons):
        for j, gt_id in enumerate(gt_ids):

            task = f"{gt_id}_{horizon}"
            vmin, vmax = yaxis_limits[gt_id]

            if task not in all_seasonal_rpss:
                ax[i, j].set_visible(False)
                continue

            last_ds = None  # store for tick formatting

            for model in model_names:
                if model in all_seasonal_rpss[task].data_vars:

                    ds_seasonal = all_seasonal_rpss[task][model]

                    ds_seasonal.plot.line(
                        ax=ax[i, j],
                        label=all_model_names[model],
                        color=model_colors.get(model, "black"),
                        marker=model_markers.get(model, "o"),
                        markersize=7,
                        linewidth=1.8,
                        markerfacecolor=ax[i, j].get_facecolor(),
                        markeredgewidth=1
                    )

                    last_ds = ds_seasonal

            ax[i, j].set_title(tasks[task], size=16)
            ax[i, j].set_xlabel('')
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)

            if j == 0:
                ax[i, j].legend(fontsize=10, frameon=True, loc='lower left')
                ax[i, j].set_ylabel('Seasonal RPSS', size=16)
            else:
                if ax[i, j].legend_:
                    ax[i, j].legend_.remove()
                ax[i, j].set_ylabel('')

            if last_ds is not None:

                time_values = last_ds['time'].values
                ax[i, j].set_xticks(time_values)

                # Label as e.g. "2023 DJF"
                years = last_ds['time'].dt.year.values
                seasons = last_ds['time'].dt.season.values
                
                season_labels = [f"{y} {s}" for y, s in zip(years, seasons)]
                
                ax[i, j].set_xticklabels(season_labels, rotation=90, ha='right', size=14)

            if i == 0:
                ax[i, j].tick_params(axis='x',
                                     bottom=False,
                                     labelbottom=False)

            if vmin is not None and vmax is not None:
                ax[i, j].set_ylim(vmin, vmax)

            ax[i, j].axhline(y=0,
                             linestyle='--',
                             linewidth=1.2,
                             alpha=0.8,
                             color='gray')

            ax[i, j].grid(False)

    filename = os.path.join(OUT_DIR, f"seasonal_{metric}_{target_dates}.pdf")

    if save_fig:
        plt.savefig(filename, dpi=300, transparent=True)
        print(f"Figure saved: {filename}")

    if show_fig:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)




def plot_metric_diff_grid_6x4(
    model_names,
    gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
    horizons=[19, 26],
    metric="lat_lon_rps",
    target_dates="std_test",
    diff_cmap="seismic",
    skill_cmap="seismic",
    source_data=False,
    source_data_filename="fig_0-metric_diff_grid_6x4.xlsx",
    source_data_sheet_prefix="Fig2b",
    show_fig=True,
    save_fig=False,
):

    num_rows = len(gt_ids) * len(horizons)
    num_cols = len(model_names) + 2  # models + spacer + diff

    # Create a visible gap via a thin spacer column
    width_ratios = [1] * len(model_names) + [0.05, 1]

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(18, 18),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios}
    )

    # Load arid mask once if precipitation is being plotted
    if any(gt_id.endswith("pr") for gt_id in gt_ids):
        quintiles = xr.open_dataset(
            "data/era5-quintiles-pr.zarr",
            engine="zarr"
        ).load()

    # ----------------------------------------------------------
    # Storage for source data
    # ----------------------------------------------------------
    source_data_dict = {}
    improvement_summary = []

    curr_row = 0
    im_metric, im_diff = None, None

    for gt_id in gt_ids:

        for horizon in horizons:

            task = f"{gt_id}_{horizon}"

            arid_mask = None

            if gt_id.endswith("pr"):

                quintiles_sel = quintiles.sel(
                    time=get_target_dates(
                        target_dates,
                        horizon
                    )
                )

                arid_mask = (
                    quintiles_sel["pr"]
                    .isel(quantile=-1)
                    == 0
                ).any(dim="time")

            # --------------------------------------------------
            # Load spatial maps
            # --------------------------------------------------
            metrics = {}

            for model_name in model_names:

                sn = get_selected_submodel_name(
                    model_name,
                    gt_id,
                    horizon
                )

                metric_f = (
                    'lat_lon_rps'
                    if metric.startswith('lat_lon_rps')
                    else metric
                )

                filename = os.path.join(
                    'eval',
                    'metrics',
                    model_name,
                    'submodel_forecasts',
                    sn,
                    task,
                    f'{metric_f}-{task}-{target_dates}.zarr'
                )

                metrics[model_name] = xr.open_zarr(
                    filename
                ).load()

            # Difference between final two models
            metrics['diff'] = (
                metrics[model_names[-1]]
                - metrics[model_names[-2]]
            )

            row_axes = axes[curr_row]

            # --------------------------------------------------
            # Style all axes
            # --------------------------------------------------
            for ax in row_axes:

                ax.set_global()

                ax.coastlines(
                    color="black",
                    linewidth=0.6
                )

                ax.spines["geo"].set_linewidth(2.25)

            # Hide spacer column
            spacer_idx = len(model_names)
            row_axes[spacer_idx].set_visible(False)

            # Row label
            row_label = (
                f"{gt_id_names[gt_id]}\n"
                f"{horizon_names[horizon]}"
            )

            row_axes[0].text(
                -0.15,
                0.5,
                row_label,
                va="center",
                ha="center",
                rotation=90,
                transform=row_axes[0].transAxes,
                fontsize=20
            )

            # --------------------------------------------------
            # Plot model columns
            # --------------------------------------------------
            for i, model_name in enumerate(model_names):

                data = metrics[model_name][metric]

                if arid_mask is not None:
                    data = data.where(~arid_mask)

                im = data.plot(
                    ax=row_axes[i],
                    transform=ccrs.PlateCarree(),
                    cmap=skill_cmap,
                    vmin=-1,
                    vmax=1,
                    add_colorbar=False,
                    rasterized=True
                )

                if source_data:

                    source_data_dict[
                        f"{task}_{model_name}"
                    ] = data

            # --------------------------------------------------
            # Diff column
            # --------------------------------------------------
            diff_ax = row_axes[-1]

            diff_data = metrics["diff"][metric]

            if arid_mask is not None:
                diff_data = diff_data.where(~arid_mask)

            im2 = diff_data.plot(
                ax=diff_ax,
                transform=ccrs.PlateCarree(),
                cmap=diff_cmap,
                vmin=-0.2,
                vmax=0.2,
                add_colorbar=False,
                rasterized=True
            )

            if source_data:

                source_data_dict[
                    f"{task}_diff"
                ] = diff_data

            # --------------------------------------------------
            # Improvement statistics
            # --------------------------------------------------
            nonnull = diff_data.notnull()

            num_nonnull = nonnull.sum().values

            num_improved = (
                nonnull & (diff_data > 0)
            ).sum().values

            fraction_improved = float(
                (
                    (nonnull & (diff_data > 0)).sum()
                    / num_nonnull
                ).values
            )

            print(
                f"{task}: % grid cells improved: "
                f"{fraction_improved} "
                f"of {num_nonnull}"
            )

            if source_data:

                improvement_summary.append(
                    {
                        "Task": task,
                        "Variable": gt_id_names[gt_id],
                        "Horizon": horizon_names[horizon],
                        "Number of grid cells": int(num_nonnull),
                        "Number improved": int(num_improved),
                        "Fraction improved": fraction_improved,
                        "Percent improved": (
                            100 * fraction_improved
                            if not np.isnan(fraction_improved)
                            else np.nan
                        ),
                    }
                )

            # --------------------------------------------------
            # Titles
            # --------------------------------------------------
            if curr_row == 0:

                for i, model_name in enumerate(model_names):

                    row_axes[i].set_title(
                        all_model_names[model_name],
                        fontsize=20,
                        pad=30
                    )

                diff_ax.set_title(
                    f"{all_model_names[model_names[-1]]} - "
                    f"{all_model_names[model_names[-2]]}",
                    fontsize=20,
                    pad=30
                )

            else:

                for ax in row_axes:
                    ax.set_title("")

            im_metric, im_diff = im, im2

            curr_row += 1

    # ----------------------------------------------------------
    # Colorbars
    # ----------------------------------------------------------
    cax1 = fig.add_axes(
        [0.11, -0.02, 0.6, 0.02]
    )

    cax2 = fig.add_axes(
        [0.79, -0.02, 0.19, 0.02]
    )

    cbar1 = fig.colorbar(
        im_metric,
        cax=cax1,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40
    )

    cbar1.set_label(
        metric_names[metric].replace(
            'RPSS',
            'Ranked probability skill score (RPSS)'
        ),
        fontsize=20,
        labelpad=10
    )

    cbar1.ax.tick_params(
        labelsize=20
    )

    cbar2 = fig.colorbar(
        im_diff,
        cax=cax2,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40
    )

    cbar2.set_label(
        f"{metric_names[metric]} difference",
        fontsize=20,
        labelpad=10
    )

    cbar2.ax.tick_params(
        labelsize=20
    )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if save_fig:

        outfile = os.path.join(
            'viz',
            'pbc',
            f"{metric}_{target_dates}.pdf"
        )

        plt.savefig(
            outfile,
            dpi=100,
            bbox_inches='tight'
        )

        print(f"Saved: {outfile}")

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        # Both Fig. 2a and Fig. 2b must resolve to exactly the
        # same workbook.
        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename
        )

        # Make sure the source-data directory exists.
        os.makedirs(
            SRC_DATA_DIR,
            exist_ok=True
        )

        # ------------------------------------------------------
        # Write to existing workbook if present.
        #
        # mode="a" preserves the Fig2a sheets written by the
        # first function.
        #
        # if_sheet_exists="replace" means that rerunning this
        # function updates only its own sheets.
        # ------------------------------------------------------
        if os.path.exists(fig_filename):

            writer_mode = "a"

        else:

            writer_mode = "w"

        with pd.ExcelWriter(
            fig_filename,
            engine="openpyxl",
            mode=writer_mode,
            if_sheet_exists=(
                "replace"
                if writer_mode == "a"
                else None
            ),
        ) as writer:

            # ----------------------------------------------
            # Improvement summary
            # ----------------------------------------------
            improvement_df = pd.DataFrame(
                improvement_summary
            )

            improvement_df.to_excel(
                writer,
                sheet_name=f"{source_data_sheet_prefix}_summary",
                index=False,
            )

            # ----------------------------------------------
            # Save each spatial map
            # ----------------------------------------------
            for sheet_name, data in source_data_dict.items():
                sheet_name = sheet_name.replace('era5-','')

                # Convert DataArray to DataFrame.
                #
                # reset_index() puts latitude/longitude and
                # other coordinates into ordinary columns.
                data_df = data.to_dataframe(
                    name=metric
                ).reset_index()

                # Excel sheet names have a maximum length of
                # 31 characters.
                safe_sheet_name = (
                    f"{source_data_sheet_prefix}_{sheet_name}"
                )[:31]

                data_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name,
                    index=False,
                    na_rep="NaN",
                )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Display / close
    # ----------------------------------------------------------
    if show_fig:
        plt.show()
    else:
        plt.close(fig)



def plot_metric_diff_grid_6x3(
    model_names,
    gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
    horizons=[19, 26],
    metric="lat_lon_rps",
    target_dates="std_test",
    diff_cmap="seismic",
    skill_cmap="seismic",
    source_data=False,
    source_data_filename="fig_2-rpss_ci_barplot_ecmwf.xlsx",
    show_fig=True,
    save_fig=False
):

    num_rows = len(gt_ids) * len(horizons)
    num_cols = 3

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(14, 18),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True
    )

    # Make axes 2D if there is only one row
    if num_rows == 1:
        axes = axes[np.newaxis, :]

    # ----------------------------------------------------------
    # Load arid mask once if precipitation is being plotted
    # ----------------------------------------------------------
    if any(gt_id.endswith("pr") for gt_id in gt_ids):
        quintiles = xr.open_dataset(
            "data/era5-quintiles-pr.zarr",
            engine="zarr"
        ).load()

    # ----------------------------------------------------------
    # Storage for source data
    # ----------------------------------------------------------
    source_data_dict = {}
    improvement_summary = []

    curr_row = 0
    im_metric, im_diff = None, None

    # ----------------------------------------------------------
    # Loop over variables and horizons
    # ----------------------------------------------------------
    for gt_id in gt_ids:
        for horizon in horizons:

            task = f"{gt_id}_{horizon}"

            # --------------------------------------------------
            # Arid precipitation mask
            # --------------------------------------------------
            arid_mask = None

            if gt_id.endswith("pr"):
                quintiles_sel = quintiles.sel(
                    time=get_target_dates(
                        target_dates,
                        horizon
                    )
                )

                arid_mask = (
                    quintiles_sel["pr"]
                    .isel(quantile=-1)
                    == 0
                ).any(dim="time")

            # --------------------------------------------------
            # Load spatial metric maps
            # --------------------------------------------------
            metrics = {}

            for model_name in model_names:

                sn = get_selected_submodel_name(
                    model_name,
                    gt_id,
                    horizon
                )

                metric_f = (
                    'lat_lon_rps'
                    if metric.startswith('lat_lon_rps')
                    else metric
                )

                filename = os.path.join(
                    'eval',
                    'metrics',
                    model_name,
                    'submodel_forecasts',
                    sn,
                    task,
                    f'{metric_f}-{task}-{target_dates}.zarr'
                )

                metrics[model_name] = xr.open_zarr(
                    filename
                )

            # Difference: Model B - Model A
            metrics['diff'] = (
                metrics[model_names[1]]
                - metrics[model_names[0]]
            )

            # --------------------------------------------------
            # Plot row
            # --------------------------------------------------
            row_axes = axes[curr_row]

            for ax in row_axes:
                ax.set_global()
                ax.coastlines(
                    color="black",
                    linewidth=0.6
                )
                ax.spines["geo"].set_linewidth(1.5)

            # Row label
            row_label = (
                f"{gt_id_names[gt_id]}\n"
                f"{horizon_names[horizon]}"
            )

            row_axes[0].text(
                -0.15,
                0.5,
                row_label,
                va="center",
                ha="center",
                rotation=90,
                transform=row_axes[0].transAxes,
                fontsize=20
            )

            # --------------------------------------------------
            # Column 0: Model A
            # --------------------------------------------------
            data_model_a = metrics[
                model_names[0]
            ][metric]

            if arid_mask is not None:
                data_model_a = data_model_a.where(
                    ~arid_mask
                )

            im0 = data_model_a.plot(
                ax=row_axes[0],
                transform=ccrs.PlateCarree(),
                cmap=skill_cmap,
                vmin=-1,
                vmax=1,
                add_colorbar=False,
                rasterized=True
            )

            # --------------------------------------------------
            # Column 1: Model B
            # --------------------------------------------------
            data_model_b = metrics[
                model_names[1]
            ][metric]

            if arid_mask is not None:
                data_model_b = data_model_b.where(
                    ~arid_mask
                )

            im1 = data_model_b.plot(
                ax=row_axes[1],
                transform=ccrs.PlateCarree(),
                cmap=skill_cmap,
                vmin=-1,
                vmax=1,
                add_colorbar=False,
                rasterized=True
            )

            # --------------------------------------------------
            # Column 2: Difference
            # --------------------------------------------------
            diff_data = metrics["diff"][metric]

            if arid_mask is not None:
                diff_data = diff_data.where(
                    ~arid_mask
                )

            im2 = diff_data.plot(
                ax=row_axes[2],
                transform=ccrs.PlateCarree(),
                cmap=diff_cmap,
                vmin=-0.5,
                vmax=0.5,
                add_colorbar=False,
                rasterized=True
            )

            # --------------------------------------------------
            # Improvement statistics
            # --------------------------------------------------
            nonnull = diff_data.notnull()

            num_nonnull = int(
                nonnull.sum().values
            )

            num_improved = int(
                (
                    nonnull
                    & (diff_data > 0)
                ).sum().values
            )

            if num_nonnull > 0:
                fraction_improved = (
                    num_improved
                    / num_nonnull
                )
            else:
                fraction_improved = np.nan

            print(
                f"{task}: % grid cells improved: "
                f"{fraction_improved} "
                f"of {num_nonnull}"
            )

            # Save summary for source data
            if source_data:

                improvement_summary.append({
                    "Task": task,
                    "Variable": gt_id_names[gt_id],
                    "GT_ID": gt_id,
                    "Horizon": horizon_names[horizon],
                    "Horizon_value": horizon,
                    "Model_A": model_names[0],
                    "Model_B": model_names[1],
                    "Difference": (
                        f"{model_names[1]} - "
                        f"{model_names[0]}"
                    ),
                    "Number of grid cells": num_nonnull,
                    "Number improved": num_improved,
                    "Fraction improved": fraction_improved,
                    "Percent improved": (
                        100 * fraction_improved
                        if not np.isnan(fraction_improved)
                        else np.nan
                    ),
                })

                # --------------------------------------------------
                # Store the actual plotted data
                # --------------------------------------------------
                source_data_dict[
                    f"{task}_{model_names[0]}"
                ] = data_model_a

                source_data_dict[
                    f"{task}_{model_names[1]}"
                ] = data_model_b

                source_data_dict[
                    f"{task}_diff"
                ] = diff_data

            # --------------------------------------------------
            # Column titles only once
            # --------------------------------------------------
            if curr_row == 0:

                row_axes[0].set_title(
                    all_model_names[model_names[0]],
                    fontsize=20,
                    pad=30
                )

                row_axes[1].set_title(
                    all_model_names[model_names[1]],
                    fontsize=20,
                    pad=30
                )

                row_axes[2].set_title(
                    f"{all_model_names[model_names[1]]} - "
                    f"{all_model_names[model_names[0]]}",
                    fontsize=22,
                    pad=30
                )

            else:

                for ax in row_axes:
                    ax.set_title("")

            im_metric, im_diff = im0, im2

            curr_row += 1

    # ----------------------------------------------------------
    # Colorbars
    # ----------------------------------------------------------
    cbar1 = fig.colorbar(
        im_metric,
        ax=axes[:, :2],
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40
    )

    cbar1.set_label(
        metric_names[metric].replace(
            'RPSS',
            'Ranked probability skill score (RPSS)'
        ),
        fontsize=20,
        labelpad=10
    )

    cbar1.ax.tick_params(
        labelsize=20
    )

    cbar2 = fig.colorbar(
        im_diff,
        ax=axes[:, 2],
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=20
    )

    cbar2.set_label(
        f"{metric_names[metric]} difference",
        fontsize=20,
        labelpad=10
    )

    cbar2.ax.tick_params(
        labelsize=20
    )

    # ----------------------------------------------------------
    # Save source data -- APPEND to existing workbook
    # ----------------------------------------------------------
    if source_data:

        source_data_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        # ------------------------------------------------------
        # Determine whether workbook already exists
        # ------------------------------------------------------
        workbook_exists = os.path.exists(
            source_data_filename
        )

        writer_kwargs = {
            "engine": "openpyxl",
            "mode": "a" if workbook_exists else "w",
        }

        # When appending, replace sheets if this function has
        # previously been run with the same filename.
        if workbook_exists:
            writer_kwargs["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(
            source_data_filename,
            **writer_kwargs
        ) as writer:

            # --------------------------------------------------
            # Improvement summary
            # --------------------------------------------------
            improvement_df = pd.DataFrame(
                improvement_summary
            )

            improvement_df.to_excel(
                writer,
                sheet_name="metric_diff_summary",
                index=False
            )

            # --------------------------------------------------
            # Spatial maps
            # --------------------------------------------------
            for sheet_name, data in source_data_dict.items():

                # Convert DataArray to a self-contained table
                data_df = data.to_dataframe(
                    name=metric
                ).reset_index()

                # Excel sheet names max out at 31 characters
                safe_sheet_name = sheet_name[:31]

                data_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name,
                    index=False,
                    na_rep="NaN"
                )

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------
            metadata_df = pd.DataFrame({
                "Parameter": [
                    "metric",
                    "target_dates",
                    "Model_A",
                    "Model_B",
                    "skill_cmap",
                    "diff_cmap",
                    "skill_vmin",
                    "skill_vmax",
                    "diff_vmin",
                    "diff_vmax",
                ],
                "Value": [
                    metric,
                    target_dates,
                    model_names[0],
                    model_names[1],
                    skill_cmap,
                    diff_cmap,
                    -1,
                    1,
                    -0.5,
                    0.5,
                ]
            })

            metadata_df.to_excel(
                writer,
                sheet_name="metric_diff_metadata",
                index=False
            )

        print(
            f"Source data appended to: "
            f"{source_data_filename}"
        )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if save_fig:
        outfile = os.path.join('viz', 'pbc', f"{metric}_{target_dates}.pdf")
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.savefig(outfile.replace(".pdf", ".eps"), dpi=100, bbox_inches='tight')

    if show_fig:
        plt.show()
    else:
        plt.close(fig)






def plot_seasonal_rpss_grouped_bar(
    all_daily_rps,
    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
    gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
    horizons=[19, 26],
    target_dates='std_test',
    n_boot=5000,
    seed=42,
    show_fig=True,
    save_fig=False,
    source_data=False,
    source_data_filename="fig_s1-seasonal_rpss.xlsx",
    source_data_sheet_prefix="Fig_s1",
    verbose=False,
):

    season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }

    seasons = ['DJF', 'MAM', 'JJA', 'SON']

    width = 0.25
    n_models = len(model_names)
    n_vars = len(gt_ids)
    n_horizons = len(horizons)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        constrained_layout=True
    )

    axes = axes.flatten()

    # ----------------------------------------------------------
    # Aggregate RPSS and store raw daily values for bootstrap
    # ----------------------------------------------------------
    agg_rpss = {
        season: {
            var: {
                h: {
                    m: np.nan
                    for m in model_names
                }
                for h in horizons
            }
            for var in gt_ids
        }
        for season in seasons
    }

    agg_rpss_raw = {
        season: {
            var: {
                h: {
                    m: []
                    for m in model_names
                }
                for h in horizons
            }
            for var in gt_ids
        }
        for season in seasons
    }

    # Store climatology separately because it is used for the
    # RPSS calculation and significance testing.
    agg_rpss_raw_climatology = {
        season: {
            var: {
                h: []
                for h in horizons
            }
            for var in gt_ids
        }
        for season in seasons
    }

    for task_key, ds in all_daily_rps.items():

        var, horizon_str = task_key.split('_')
        horizon = int(horizon_str)

        if var not in gt_ids or horizon not in horizons:
            continue

        months = pd.DatetimeIndex(
            ds['time'].values
        ).month

        ds_season = [
            season_map[m]
            for m in months
        ]

        for season in seasons:

            mask = np.array(
                ds_season
            ) == season

            clim = ds['climatology'].values[mask]

            # Store climatology for possible source-data use
            agg_rpss_raw_climatology[
                season
            ][var][horizon] = clim

            for model in model_names:

                if model not in ds.data_vars:
                    continue

                data = ds[model].values
                vals = data[mask]

                vals = vals[
                    ~np.isnan(vals)
                ]

                if len(vals) > 0:

                    agg_rpss[
                        season
                    ][var][horizon][model] = (
                        1
                        - np.mean(vals)
                        / np.mean(clim)
                    )

                    agg_rpss_raw[
                        season
                    ][var][horizon][model] = vals

    # ----------------------------------------------------------
    # Determine whether target model significantly improves
    # over all baselines
    # ----------------------------------------------------------
    target_model = model_names[-1]

    target_all_significant = {
        season: {
            var: {
                h: False
                for h in horizons
            }
            for var in gt_ids
        }
        for season in seasons
    }

    for season in seasons:

        for var in gt_ids:

            for horizon in horizons:

                target_vals = np.asarray(
                    agg_rpss_raw[
                        season
                    ][var][horizon].get(
                        target_model,
                        []
                    )
                )

                if target_vals.size == 0:
                    continue

                clim_vals = np.asarray(
                    agg_rpss_raw_climatology[
                        season
                    ][var][horizon]
                )

                all_significant = True

                for baseline_model in model_names:

                    if baseline_model == target_model:
                        continue

                    baseline_vals = np.asarray(
                        agg_rpss_raw[
                            season
                        ][var][horizon].get(
                            baseline_model,
                            []
                        )
                    )

                    if baseline_vals.size == 0:

                        all_significant = False
                        break

                    n = min(
                        target_vals.size,
                        baseline_vals.size
                    )

                    if n == 0:

                        all_significant = False
                        break

                    target_aligned = target_vals[:n]
                    baseline_aligned = baseline_vals[:n]
                    clim_aligned = clim_vals[:n]

                    mask = (
                        ~np.isnan(target_aligned)
                        & ~np.isnan(baseline_aligned)
                        & ~np.isnan(clim_aligned)
                    )

                    if not np.any(mask):

                        all_significant = False
                        break

                    lower_cb = lower_confidence_bound(
                        baseline_aligned[mask]
                        - target_aligned[mask],
                        clim=clim_aligned[mask],
                        n_boot=n_boot,
                        seed=seed,
                    )

                    if verbose:

                        print(
                            f"Season {season} | "
                            f"{var}_{horizon} | "
                            f"{target_model} - "
                            f"{baseline_model} RPSS "
                            f"confidence bound: {lower_cb}"
                        )

                    if lower_cb <= 0:

                        all_significant = False
                        break

                target_all_significant[
                    season
                ][var][horizon] = all_significant

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------
    for i, season in enumerate(seasons):

        ax = axes[i]

        group_gap = 0.75
        week_gap = 0.3

        group_width = (
            n_horizons
            * n_models
            * width
            + group_gap
        )

        x_base = (
            np.arange(n_vars)
            * group_width
        )

        # ------------------------------------------------------
        # Bars
        # ------------------------------------------------------
        for vi, var in enumerate(gt_ids):

            for hi, horizon in enumerate(horizons):

                offset = (
                    hi
                    * (
                        n_models * width
                        + week_gap
                    )
                )

                for mi, model in enumerate(model_names):

                    if verbose and mi == 0:

                        print(
                            f"Task {var}_{horizon} "
                            f"for {target_dates} "
                            f"season {season} has "
                            f"{len(agg_rpss_raw[season][var][horizon][model])} "
                            f"target dates"
                        )

                    xpos = (
                        x_base[vi]
                        + offset
                        + mi * width
                    )

                    val = agg_rpss[
                        season
                    ][var][horizon][model]

                    bars = ax.bar(
                        xpos,
                        val,
                        width,
                        color=model_colors.get(
                            model,
                            'gray'
                        ),
                        label=(
                            model.capitalize().replace(
                                '_',
                                ' '
                            )
                            if (
                                vi == 0
                                and hi == 0
                            )
                            else None
                        ),
                    )

                    if (
                        model == target_model
                        and target_all_significant[
                            season
                        ][var][horizon]
                    ):

                        for bar in bars:
                            bar.set_hatch("x")

        # ------------------------------------------------------
        # Week-level x-axis labels
        # ------------------------------------------------------
        week_centers = []
        week_labels = []

        for vi in range(n_vars):

            for hi, horizon in enumerate(horizons):

                center = (
                    x_base[vi]
                    + hi
                    * (
                        n_models * width
                        + week_gap
                    )
                    + (
                        n_models * width
                    ) / 2
                    - width / 2
                )

                week_centers.append(center)
                week_labels.append(
                    horizon_names[horizon]
                )

        ax.set_xticks(week_centers)

        ax.set_xticklabels(
            week_labels,
            fontsize=20
        )

        # ------------------------------------------------------
        # Variable labels
        # ------------------------------------------------------
        pair_centers = [
            x_base[vi]
            + group_width / 2
            - width / 2
            for vi in range(n_vars)
        ]

        var_labels = [
            gt_id_names[g]
            for g in gt_ids
        ]

        for xc, var in zip(
            pair_centers,
            var_labels
        ):

            var_label = (
                f"{var}\n"
                if i in [0, 1]
                else var
            )

            ax.text(
                xc - 0.2,
                -0.12,
                var_label,
                ha='center',
                va='top',
                fontsize=20,
                fontweight='bold',
                transform=ax.get_xaxis_transform()
            )

        ax.set_title(
            season_names[season],
            fontsize=24,
            fontweight='bold'
        )

        ax.axhline(
            0,
            color='gray',
            linestyle='--',
            linewidth=1.2,
            alpha=0.7
        )

        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.4
        )

        ax.set_ylim(
            -0.3,
            0.3
        )

        ax.tick_params(
            axis='y',
            labelsize=18
        )

    # ----------------------------------------------------------
    # Shared Y-axis label
    # ----------------------------------------------------------
    fig.text(
        -0.01,
        0.5,
        'Ranked probability skill score (RPSS)',
        va='center',
        ha='center',
        rotation='vertical',
        fontsize=26,
        fontweight='bold'
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    handles = []
    labels = []

    for model in model_names:

        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=model_colors.get(
                    model,
                    'gray'
                )
            )
        )

        labels.append(
            all_model_names[model]
        )

    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(model_names),
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.5, 1.1)
    )

    plt.subplots_adjust(
        hspace=0.4,
        left=0.08
    )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if save_fig:
        outfile = os.path.join(OUT_DIR, f"rpss_by_season_{target_dates}.pdf")
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Figure saved: {outfile}")
        plt.savefig(outfile.replace('.pdf', '.eps'), dpi=100, bbox_inches='tight')

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename
        )

        os.makedirs(
            SRC_DATA_DIR,
            exist_ok=True
        )

        # ------------------------------------------------------
        # Build tidy source-data table containing every plotted
        # bar.
        # ------------------------------------------------------
        source_rows = []

        for season in seasons:

            for var in gt_ids:

                for horizon in horizons:

                    for model in model_names:

                        raw_values = np.asarray(
                            agg_rpss_raw[
                                season
                            ][var][horizon].get(
                                model,
                                []
                            )
                        )

                        source_rows.append(
                            {
                                "Season": season,
                                "Variable": gt_id_names[var],
                                "Variable_ID": var,
                                "Horizon": horizon,
                                "Horizon_Label": horizon_names[horizon],
                                "Model": model,
                                "Model_Label": all_model_names.get(
                                    model,
                                    model
                                ),
                                "RPSS": agg_rpss[
                                    season
                                ][var][horizon].get(
                                    model,
                                    np.nan
                                ),
                                "Significant_All_Baselines": (
                                    target_all_significant[
                                        season
                                    ][var][horizon]
                                    if model == target_model
                                    else False
                                ),
                                "N_Raw_Values": len(raw_values),
                            }
                        )

        source_df = pd.DataFrame(
            source_rows
        )

        # ------------------------------------------------------
        # Significance summary
        # ------------------------------------------------------
        significance_rows = []

        for season in seasons:

            for var in gt_ids:

                for horizon in horizons:

                    significance_rows.append(
                        {
                            "Season": season,
                            "Variable": gt_id_names[var],
                            "Variable_ID": var,
                            "Horizon": horizon,
                            "Horizon_Label": horizon_names[horizon],
                            "Target_Model": target_model,
                            "Significant_All_Baselines": (
                                target_all_significant[
                                    season
                                ][var][horizon]
                            ),
                            "N_Bootstrap": n_boot,
                            "Random_Seed": seed,
                        }
                    )

        significance_df = pd.DataFrame(
            significance_rows
        )

        # ------------------------------------------------------
        # Optional summary of the raw sample sizes.
        # This makes it easy to understand how many daily values
        # contributed to each plotted RPSS value.
        # ------------------------------------------------------
        sample_rows = []

        for season in seasons:

            for var in gt_ids:

                for horizon in horizons:

                    clim_values = np.asarray(
                        agg_rpss_raw_climatology[
                            season
                        ][var][horizon]
                    )

                    for model in model_names:

                        values = np.asarray(
                            agg_rpss_raw[
                                season
                            ][var][horizon].get(
                                model,
                                []
                            )
                        )

                        sample_rows.append(
                            {
                                "Season": season,
                                "Variable": gt_id_names[var],
                                "Variable_ID": var,
                                "Horizon": horizon,
                                "Horizon_Label": horizon_names[horizon],
                                "Model": model,
                                "Model_Label": all_model_names.get(
                                    model,
                                    model
                                ),
                                "N_Model_Values": len(values),
                                "N_Climatology_Values": len(clim_values),
                            }
                        )

        sample_df = pd.DataFrame(
            sample_rows
        )

        # ------------------------------------------------------
        # Append to existing workbook if it exists.
        #
        # This preserves source data written by other subfigures.
        # ------------------------------------------------------
        if os.path.exists(fig_filename):

            writer_mode = "a"

        else:

            writer_mode = "w"

        with pd.ExcelWriter(
            fig_filename,
            engine="openpyxl",
            mode=writer_mode,
            if_sheet_exists=(
                "replace"
                if writer_mode == "a"
                else None
            ),
        ) as writer:

            source_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_values"
                )[:31],
                index=False
            )

            significance_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_significance"
                )[:31],
                index=False
            )

            sample_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_samples"
                )[:31],
                index=False
            )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Display / close
    # ----------------------------------------------------------
    if show_fig:

        plt.show()

    else:

        plt.close(fig)






def get_regional_rpss(model_names=['ecmwf', 'pbc_ecmwf', 'msn', 'pbc_msn', 'duet'],
                         gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
                         horizons=[19, 26],
                         target_dates="std_test",
                         regions=['europe', 'north_africa']):

    metrics_dic = {}
    if regions == 'all':
        regions = [r for r in dic_regions.keys()] 

    for gt_id, horizon in product(gt_ids, horizons):

        task = f"{gt_id}_{horizon}"
        metrics_dic[task] = {}

        for region in regions:

            ds_list = []

            for model_name in model_names:

                sn = get_selected_submodel_name(model_name, gt_id, horizon)

                filename = os.path.join(
                    "eval", "metrics", model_name, "submodel_forecasts",
                    sn, task,
                    f"rps_{region}-{task}-{target_dates}.zarr"
                )

                ds = xr.open_dataset(filename, engine="zarr")

                # keep time series
                da = ds["rpss"]

                da = da.rename(model_name)
                ds_model = da.to_dataset()

                ds_list.append(ds_model)

            ds_merged = xr.merge(ds_list, compat="override")
            metrics_dic[task][region] = ds_merged

    return metrics_dic


def plot_bias_maps_4x3(model_names,
                      gt_id='era5-mslp',
                      horizon=19,
                      fs=[1, 2, 3, 4],
                      target_dates="std_test",
                      cmap="RdBu_r",
                      vmin=-5,
                      vmax=5,
                      show_fig=True,
                      save_fig=False):



    measurement = gt_id.replace('era5-', '')
    num_rows = len(fs)
    num_cols = len(model_names)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(4*num_cols, 2.25*num_rows),
                             subplot_kw={"projection": ccrs.Robinson()},
                             constrained_layout=True)

    if num_rows == 1:
        axes = axes[np.newaxis, :]

    im = None

    for i, f in enumerate(fs):

        # -------------------------
        # Load GT (truth)
        # -------------------------
        gt_path = os.path.join(
            "eval", "metrics", "gt", "submodel_forecasts", "gt",
            f"era5-f{f}_{measurement}_{horizon}",
            f"lat_lon_pred-era5-f{f}_{measurement}_{horizon}-{target_dates}.zarr"
        )

        gt_ds = xr.open_zarr(gt_path)
        truth = gt_ds["lat_lon_pred"]

        for j, model_name in enumerate(model_names):

            ax = axes[i, j]
            ax.set_global()
            ax.coastlines(color="grey", linewidth=0.6)
            ax.spines["geo"].set_linewidth(1.2)

            # -------------------------
            # Load MODEL prediction
            # -------------------------
            sn = get_selected_submodel_name(model_name, gt_id, horizon)

            model_path = os.path.join(
                "eval", "metrics", model_name, "submodel_forecasts", sn,
                f"era5-f{f}_{measurement}_{horizon}",
                f"lat_lon_pred-era5-f{f}_{measurement}_{horizon}-{target_dates}.zarr"
            )

            if not os.path.exists(model_path):
                print(f"Missing: {model_name}, f{f}")
                continue

            model_ds = xr.open_zarr(model_path)
            pred = model_ds["lat_lon_pred"]

            # -------------------------
            # Align 
            # -------------------------
            pred, truth_aligned = xr.align(pred, truth)

            # -------------------------
            # Compute bias
            # -------------------------
            bias = pred - truth_aligned

            # -------------------------
            # Plot
            # -------------------------
            im = bias.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                rasterized=True
            )

            # Row labels
            if j == 0:
                ax.text(-0.1, 0.48, quintile_names[f"f{f}"],
                        transform=ax.transAxes,
                        rotation=90,
                        va="center", ha="center",
                        fontsize=16)

            # Column titles
            if i == 0:
                ax.set_title(all_model_names[model_name],
                             fontsize=16, pad=15)
            else:
                ax.set_title("")

    # -------------------------
    # Title
    # -------------------------
    title = f"{gt_id_names[gt_id]} ({horizon_names[horizon]})"
    fig.suptitle(title, fontsize=18, y=1.02)

    # -------------------------
    # Colorbar
    # -------------------------
    cbar = fig.colorbar(im,
                        ax=axes,
                        orientation="horizontal",
                        fraction=0.03,
                        pad=0.05,
                        aspect=40)

    cbar.set_label("Bias (prediction - truth)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # -------------------------
    # Save / Show
    # -------------------------
    if save_fig:
        outfile = os.path.join(BIAS_OUT_DIR,
                               f"bias_maps_{gt_id}_{horizon}_{target_dates}.pdf")

        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile}")

        plt.savefig(outfile.replace('.pdf', '.png'),
                    dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile.replace('.pdf','.png')}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)



def plot_rpss_by_region_all(
    metrics_dict,
    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
    gt_ids=['era5-tas', 'era5-pr', 'era5-mslp'],
    horizons=[19, 26],
    target_dates="std_test",
    regions='all',
    show_fig=True,
    save_fig=True,
    n_boot=5000,
    seed=42,
    source_data=False,
    source_data_filename="fig_s2-regional_rpss.xlsx",
    source_data_sheet_prefix="Fig_s2",
    verbose=False
):

    """
    Regional RPSS bar plotting using precomputed regional time series.

    metrics_dict: output of get_daily_rpss
                  {task: xr.Dataset(region, time)}

    Source data:
        <prefix>_values
            One row per plotted bar.

        <prefix>_significance
            Significance/hatching information for the target model.

        <prefix>_regions
            Region names and bounding boxes used in the figure.
    """

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    from matplotlib.ticker import MultipleLocator

    alpha = 0.05

    # ----------------------------------------------------------
    # Regions
    # ----------------------------------------------------------
    if regions == 'all':
        regions = list(dic_regions.keys())

    row_ylims = {
        'era5-tas': (-1, 0.4),
        'era5-pr': (-0.2, 0.3),
        'era5-mslp': (-0.8, 0.3),
    }

    region_labels = [
        region_names[r]
        for r in regions
    ]

    x = np.arange(
        len(region_labels)
    )

    width = 0.26

    fig, axes = plt.subplots(
        nrows=len(gt_ids),
        ncols=len(horizons),
        figsize=(18, 18),
        sharex=True,
        sharey=False
    )

    # Make axes consistently 2-D even if only one row/column
    axes = np.asarray(axes)

    if axes.ndim == 1:

        if len(gt_ids) == 1:
            axes = axes.reshape(1, -1)

        elif len(horizons) == 1:
            axes = axes.reshape(-1, 1)

    # ----------------------------------------------------------
    # Source-data storage
    # ----------------------------------------------------------
    source_value_rows = []
    significance_rows = []

    # ----------------------------------------------------------
    # Main plotting loop
    # ----------------------------------------------------------
    for row, gt_id in enumerate(gt_ids):

        if gt_id in row_ylims:

            ymin, ymax = row_ylims[gt_id]

            tick_spacing = 0.1

            major_locator = MultipleLocator(
                tick_spacing
            )

            # Set consistent ylim, ticks, and grid
            # for all axes in this row.
            for c in range(len(horizons)):

                axes[row, c].set_ylim(
                    ymin,
                    ymax
                )

                axes[row, c].yaxis.set_major_locator(
                    major_locator
                )

                axes[row, c].grid(
                    axis='y',
                    linestyle='--',
                    alpha=0.4
                )

        for col, horizon in enumerate(horizons):

            ax = axes[row, col]

            task = f"{gt_id}_{horizon}"

            if task not in metrics_dict:

                if verbose:
                    print(
                        f"Warning: task {task} "
                        f"not found in metrics_dict"
                    )

                continue

            ds = metrics_dict[task]

            if "region" not in ds.dims:

                raise ValueError(
                    f"Dataset for task {task} "
                    f"has no 'region' dimension"
                )

            results = {
                m: []
                for m in model_names
            }

            target_model = model_names[-1]

            target_all_significant = []
            target_some_significant = []

            # --------------------------------------------------
            # Calculate regional RPSS
            # --------------------------------------------------
            for region in regions:

                if region not in ds.region.values:

                    if verbose:
                        print(
                            f"Warning: region '{region}' "
                            f"not found in task {task}"
                        )

                    for m in model_names:
                        results[m].append(
                            np.nan
                        )

                    target_all_significant.append(
                        False
                    )

                    target_some_significant.append(
                        False
                    )

                    # Preserve one source-data row per model
                    for model in model_names:

                        source_value_rows.append(
                            {
                                "Task": task,
                                "Variable_ID": gt_id,
                                "Variable": gt_id_names.get(
                                    gt_id,
                                    gt_id
                                ),
                                "Horizon": horizon,
                                "Horizon_Label": horizon_names.get(
                                    horizon,
                                    str(horizon)
                                ),
                                "Region_ID": region,
                                "Region": region_names.get(
                                    region,
                                    region
                                ),
                                "Model": model,
                                "Model_Label": all_model_names.get(
                                    model,
                                    model
                                ),
                                "RPSS": np.nan,
                                "Region_Found": False,
                            }
                        )

                    significance_rows.append(
                        {
                            "Task": task,
                            "Variable_ID": gt_id,
                            "Variable": gt_id_names.get(
                                gt_id,
                                gt_id
                            ),
                            "Horizon": horizon,
                            "Horizon_Label": horizon_names.get(
                                horizon,
                                str(horizon)
                            ),
                            "Region_ID": region,
                            "Region": region_names.get(
                                region,
                                region
                            ),
                            "Target_Model": target_model,
                            "Significant_All_Baselines": False,
                            "Significant_Some_Baseline": False,
                        }
                    )

                    continue

                ds_region = ds.sel(
                    region=region
                )

                clim = ds_region[
                    "climatology"
                ].values

                # --------------------------------------------------
                # Calculate RPSS for each model
                # --------------------------------------------------
                for model in model_names:

                    if model not in ds_region:

                        results[model].append(
                            np.nan
                        )

                        rpss_value = np.nan

                    else:

                        values = ds_region[
                            model
                        ].values

                        mask = (
                            ~np.isnan(values)
                            & ~np.isnan(clim)
                        )

                        values_masked = values[
                            mask
                        ]

                        masked_clim = clim[
                            mask
                        ]

                        if len(values_masked) == 0:

                            rpss_value = np.nan

                        else:

                            rpss_value = (
                                1
                                - float(
                                    np.mean(
                                        values_masked
                                    )
                                    / np.mean(
                                        masked_clim
                                    )
                                )
                            )

                        results[model].append(
                            rpss_value
                        )

                    # ----------------------------------------------
                    # Source data for the plotted bar
                    # ----------------------------------------------
                    source_value_rows.append(
                        {
                            "Task": task,
                            "Variable_ID": gt_id,
                            "Variable": gt_id_names.get(
                                gt_id,
                                gt_id
                            ),
                            "Horizon": horizon,
                            "Horizon_Label": horizon_names.get(
                                horizon,
                                str(horizon)
                            ),
                            "Region_ID": region,
                            "Region": region_names.get(
                                region,
                                region
                            ),
                            "Model": model,
                            "Model_Label": all_model_names.get(
                                model,
                                model
                            ),
                            "RPSS": rpss_value,
                            "Region_Found": True,
                        }
                    )

                # --------------------------------------------------
                # Test significance of target model against
                # all baseline models
                # --------------------------------------------------
                if target_model not in ds_region:

                    target_all_significant.append(
                        False
                    )

                    target_some_significant.append(
                        False
                    )

                    significance_rows.append(
                        {
                            "Task": task,
                            "Variable_ID": gt_id,
                            "Variable": gt_id_names.get(
                                gt_id,
                                gt_id
                            ),
                            "Horizon": horizon,
                            "Horizon_Label": horizon_names.get(
                                horizon,
                                str(horizon)
                            ),
                            "Region_ID": region,
                            "Region": region_names.get(
                                region,
                                region
                            ),
                            "Target_Model": target_model,
                            "Significant_All_Baselines": False,
                            "Significant_Some_Baseline": False,
                        }
                    )

                    continue

                target_vals = ds_region[
                    target_model
                ].values

                all_significant = True
                some_significant = False

                for baseline_model in model_names:

                    if baseline_model == target_model:
                        continue

                    if baseline_model not in ds_region:

                        all_significant = False
                        some_significant = False
                        break

                    baseline_vals = ds_region[
                        baseline_model
                    ].values

                    mask = (
                        ~np.isnan(target_vals)
                        & ~np.isnan(baseline_vals)
                        & ~np.isnan(clim)
                    )

                    if not np.any(mask):

                        all_significant = False
                        some_significant = False
                        break

                    lower_cb = lower_confidence_bound(
                        baseline_vals[mask]
                        - target_vals[mask],
                        clim=clim[mask],
                        n_boot=n_boot,
                        seed=seed,
                        verbose=verbose
                    )

                    if verbose:

                        print(
                            f"  {task} | {region} | "
                            f"{target_model} - "
                            f"{baseline_model} "
                            f"RPSS confidence bound: "
                            f"{lower_cb}"
                        )

                    if lower_cb <= 0:

                        all_significant = False

                    else:

                        some_significant = True

                target_all_significant.append(
                    all_significant
                )

                target_some_significant.append(
                    some_significant
                )

                # ----------------------------------------------
                # Source significance data
                # ----------------------------------------------
                significance_rows.append(
                    {
                        "Task": task,
                        "Variable_ID": gt_id,
                        "Variable": gt_id_names.get(
                            gt_id,
                            gt_id
                        ),
                        "Horizon": horizon,
                        "Horizon_Label": horizon_names.get(
                            horizon,
                            str(horizon)
                        ),
                        "Region_ID": region,
                        "Region": region_names.get(
                            region,
                            region
                        ),
                        "Target_Model": target_model,
                        "Significant_All_Baselines": all_significant,
                        "Significant_Some_Baseline": some_significant,
                    }
                )

            # --------------------------------------------------
            # Plot bars
            # --------------------------------------------------
            for i, model in enumerate(model_names):

                bars = ax.bar(
                    x + (i - 1.5) * width,
                    results[model],
                    width,
                    label=all_model_names.get(
                        model,
                        model
                    ),
                    color=model_colors.get(
                        model,
                        'gray'
                    ),
                )

                if model == target_model:

                    for j, bar in enumerate(bars):

                        if target_all_significant[j]:

                            bar.set_hatch("xx")

                        elif target_some_significant[j]:

                            bar.set_hatch("//")

            # --------------------------------------------------
            # Titles
            # --------------------------------------------------
            if row == 0:

                ax.set_title(
                    horizon_names[horizon].replace(
                        'week',
                        'Week'
                    ),
                    fontsize=22,
                    fontweight='bold'
                )

            if col == 0:

                ax.tick_params(
                    axis='y',
                    labelsize=14
                )

                plt.setp(
                    ax.get_yticklabels(),
                    fontweight='bold'
                )

            else:

                ax.tick_params(
                    axis='y',
                    labelleft=False
                )

            if col == 1:

                ax.set_ylabel(
                    gt_id_names[gt_id],
                    fontsize=22,
                    fontweight='bold',
                    rotation=270,
                    labelpad=30
                )

                ax.yaxis.set_label_position(
                    "right"
                )

    # ----------------------------------------------------------
    # X labels
    # ----------------------------------------------------------
    for ax in axes[-1, :]:

        ax.set_xticks(x)

        ax.set_xticklabels(
            region_labels,
            rotation=90,
            ha="right",
            fontsize=16,
            fontweight='bold'
        )

    fig.supylabel(
        "Ranked probability skill score (RPSS)",
        fontsize=22,
        fontweight='bold',
        x=0.07
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=model_colors.get(
                model,
                'gray'
            )
        )
        for model in model_names
    ]

    labels = [
        all_model_names.get(
            model,
            model
        )
        for model in model_names
    ]

    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(model_names),
        fontsize=22,
        frameon=False
    )

    # ----------------------------------------------------------
    # Region table
    # ----------------------------------------------------------
    def format_bbox(
        lat_min,
        lat_max,
        lon_min,
        lon_max
    ):

        def lat_str(x):

            return (
                f"{abs(x)}°"
                f"{'N' if x >= 0 else 'S'}"
            )

        def lon_str(x):

            return (
                f"{abs(x)}°"
                f"{'E' if x >= 0 else 'W'}"
            )

        return (
            f"{lat_str(lat_min)}–"
            f"{lat_str(lat_max)}, "
            f"{lon_str(lon_min)}–"
            f"{lon_str(lon_max)}"
        )

    all_regions = list(regions)

    n_rows, n_cols = 7, 4

    table_data = [
        ["" for _ in range(n_cols)]
        for _ in range(n_rows)
    ]

    for i, region in enumerate(all_regions):

        row = i % n_rows
        col = (i // n_rows) * 2

        if col < n_cols:

            table_data[row][col] = (
                region_names[region]
            )

            table_data[row][col + 1] = (
                format_bbox(
                    **dic_regions[region]
                )
            )

    ax_table = fig.add_axes(
        [0.145, -0.15, 0.86, 0.15]
    )

    ax_table.axis('off')

    ax_table.add_patch(
        patches.Rectangle(
            (0, 0),
            0.985,
            1,
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            transform=ax_table.transAxes,
            zorder=10
        )
    )

    tbl = ax_table.table(
        cellText=table_data,
        cellLoc='left',
        colLoc='left',
        loc='center'
    )

    for (row_idx, col_idx), cell in tbl.get_celld().items():

        cell.set_linewidth(0)

        cell.set_height(
            cell.get_height() * 2.0
        )

        if col_idx in [0, 2]:

            cell.get_text().set_fontweight(
                'bold'
            )

        cell.get_text().set_fontsize(
            24
        )

    plt.tight_layout(
        rect=[0.06, 0, 1, 0.96]
    )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    filename = os.path.join(
        OUT_DIR,
        f"regional_rpss_{target_dates}.pdf"
    )

    if save_fig:

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches='tight'
        )

        plt.savefig(
            filename.replace(
                '.pdf',
                '.png'
            ),
            dpi=300,
            transparent=True,
            bbox_inches='tight'
        )

        plt.savefig(
            filename.replace(
                '.pdf',
                '.eps'
            ),
            dpi=300,
            bbox_inches='tight'
        )

        print(
            f"Figure saved: {filename}"
        )

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename
        )

        os.makedirs(
            SRC_DATA_DIR,
            exist_ok=True
        )

        # ------------------------------------------------------
        # Values table
        # ------------------------------------------------------
        values_df = pd.DataFrame(
            source_value_rows
        )

        # ------------------------------------------------------
        # Significance table
        # ------------------------------------------------------
        significance_df = pd.DataFrame(
            significance_rows
        )

        # ------------------------------------------------------
        # Region definitions
        # ------------------------------------------------------
        region_rows = []

        for region in regions:

            bbox = dic_regions[region]

            region_rows.append(
                {
                    "Region_ID": region,
                    "Region": region_names.get(
                        region,
                        region
                    ),
                    "Latitude_Min": bbox["lat_min"],
                    "Latitude_Max": bbox["lat_max"],
                    "Longitude_Min": bbox["lon_min"],
                    "Longitude_Max": bbox["lon_max"],
                    "Bounding_Box": format_bbox(
                        **bbox
                    ),
                }
            )

        regions_df = pd.DataFrame(
            region_rows
        )

        # ------------------------------------------------------
        # Metadata table
        # ------------------------------------------------------
        metadata_df = pd.DataFrame(
            [
                {
                    "Parameter": "Target dates",
                    "Value": target_dates
                },
                {
                    "Parameter": "Target model",
                    "Value": target_model
                },
                {
                    "Parameter": "Bootstrap samples",
                    "Value": n_boot
                },
                {
                    "Parameter": "Random seed",
                    "Value": seed
                },
                {
                    "Parameter": "Models",
                    "Value": ", ".join(model_names)
                },
                {
                    "Parameter": "Variables",
                    "Value": ", ".join(gt_ids)
                },
                {
                    "Parameter": "Horizons",
                    "Value": ", ".join(
                        map(str, horizons)
                    )
                },
            ]
        )

        # ------------------------------------------------------
        # Append to existing workbook if present.
        #
        # This preserves source data from other subfigures.
        # ------------------------------------------------------
        if os.path.exists(fig_filename):

            writer_mode = "a"

        else:

            writer_mode = "w"

        with pd.ExcelWriter(
            fig_filename,
            engine="openpyxl",
            mode=writer_mode,
            if_sheet_exists=(
                "replace"
                if writer_mode == "a"
                else None
            ),
        ) as writer:

            values_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_values"
                )[:31],
                index=False
            )

            significance_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_significance"
                )[:31],
                index=False
            )

            regions_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_regions"
                )[:31],
                index=False
            )

            metadata_df.to_excel(
                writer,
                sheet_name=(
                    f"{source_data_sheet_prefix}_metadata"
                )[:31],
                index=False
            )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Display / close
    # ----------------------------------------------------------
    if show_fig:

        plt.show()

    else:

        plt.close(fig)





def print_improvements(scores, model_name='pbc_ecmwf_combo', baseline_models=['ecmwf', 'debiased_ecmwf']):
    """
    Print percentage improvements of a model over baseline models.

    Args:
        scores (dict): dict of {task: xr.Dataset or DataArray-like} scores (i.e., errors)
        model_name (str): target model to evaluate (e.g., 'pbc_ecmwf_combo')
        baseline_models (list): list of baseline model names to compare against
    """
    from pprint import pprint

    # Compute skill score for each task and model
    ss = {}
    for task in scores.keys():
        task_scores = scores[task]
        # Divide each task score by the climatology mean over the corresponding set 
        # of nonnull times
        ss[task] = task_scores.mean()
        for model in task_scores.data_vars:
            ss[task][model] = 1 - ss[task][model] / task_scores["climatology"][task_scores[model].notnull()].mean()

    # Compute improvements for each baseline
    improvements = {}

    for baseline in baseline_models:
        improvements[baseline] = {
            task: {
                model_name: float(ss[task][model_name].values),
                baseline: float(ss[task][baseline].values),
                '% improvement': (
                    (float(ss[task][model_name]) - float(ss[task][baseline]))
                    / float(ss[task][baseline])
                ) * 100
            }
            for task in ss.keys()
        }

    # Print results
    for baseline in baseline_models:
        print(f"Improvements over {all_model_names[baseline]}:")
        pprint(improvements[baseline])


def plot_bias_maps_3x4(
    results_dict,
    model_names=["ecmwf", "debiased_ecmwf", "pbc_ecmwf_combo"],
    gt_id='era5-mslp',
    horizon=19,
    fs=[1, 2, 3, 4],
    target_dates="std_test",
    cmap="RdBu_r",
    vmin=-5,
    vmax=5,
    show_cbar=True,
    show_fig=True,
    save_fig=False,
    source_data=False,
    source_data_filename="fig_s3-bias_maps_precip.xlsx",
    source_data_sheet_prefix="Fig_s3"
):
    """
    Plots bias maps by extracting the relevant dataset from results_dict
    using (gt_id, horizon) as the key.

    Source data
    -----------
    If source_data=True, an Excel workbook is created/updated containing:

        <prefix>_maps
            Spatial bias values for every model and frequency/quintile.

        <prefix>_metadata
            Figure and plotting parameters.

    Each row in <prefix>_maps corresponds to one grid cell of one
    model/frequency combination and contains latitude, longitude,
    model, frequency, and mean bias.
    """

    # ----------------------------------------------------------
    # 1. Extract the specific dataset from the results dictionary
    # ----------------------------------------------------------
    if (gt_id, horizon) not in results_dict:

        print(
            f"Error: Key {(gt_id, horizon)} "
            f"not found in results_dict."
        )

        return

    ds = results_dict[
        (gt_id, horizon)
    ]

    # ----------------------------------------------------------
    # 2. Setup metadata and dimensions
    # ----------------------------------------------------------
    measurement = gt_id.replace(
        'era5-',
        ''
    )

    # Filter models to those actually present
    # in the dataset.
    model_names = [
        m
        for m in model_names
        if m in ds.model.values
        and m != 'gt'
    ]

    # ----------------------------------------------------------
    # Load arid mask once if precipitation is plotted
    # ----------------------------------------------------------
    arid_mask = None

    if gt_id.endswith("pr"):

        quintiles = xr.open_dataset(
            "data/era5-quintiles-pr.zarr",
            engine="zarr"
        ).load()

        quintiles_sel = quintiles.sel(
            time=get_target_dates(
                target_dates,
                horizon
            )
        )

        arid_mask = (
            quintiles_sel["pr"]
            .isel(quantile=-1)
            == 0
        ).any(
            dim="time"
        )

    num_rows = len(model_names)
    num_cols = len(fs)

    # ----------------------------------------------------------
    # Figure scaling
    # ----------------------------------------------------------
    fig_x = 3
    fig_y = (
        2.15
        if show_cbar
        else 1.95
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(
            fig_x * num_cols,
            fig_y * num_rows
        ),
        subplot_kw={
            "projection": ccrs.Robinson()
        },
        constrained_layout=True
    )

    # Ensure axes is always 2D
    if num_rows == 1 and num_cols == 1:

        axes = np.asarray(
            [[axes]]
        )

    elif num_rows == 1:

        axes = axes[
            np.newaxis,
            :
        ]

    elif num_cols == 1:

        axes = axes[
            :,
            np.newaxis
        ]

    im = None

    # ----------------------------------------------------------
    # Source-data storage
    # ----------------------------------------------------------
    source_data_rows = []

    # ----------------------------------------------------------
    # Loop over models + quintiles
    # ----------------------------------------------------------
    for i, model_name in enumerate(
        model_names
    ):

        for j, f in enumerate(fs):

            ax = axes[i, j]

            ax.set_global()

            ax.coastlines(
                color="grey",
                linewidth=0.6
            )

            ax.spines[
                "geo"
            ].set_linewidth(1.2)

            var_name = (
                f"f{f}_{measurement}"
            )

            if var_name not in ds:

                print(
                    f"Missing variable {var_name} "
                    f"in dataset for {gt_id}"
                )

                continue

            # --------------------------------------------------
            # Extract prediction
            # --------------------------------------------------
            pred = ds[
                var_name
            ].sel(
                model=model_name
            )

            # --------------------------------------------------
            # Ground truth
            # --------------------------------------------------
            gt = ds[
                var_name
            ].sel(
                model='gt'
            )

            # --------------------------------------------------
            # Calculate mean bias
            # --------------------------------------------------
            bias = (
                pred - gt
            ).mean(
                dim="time"
            )

            # Apply precipitation arid mask
            if arid_mask is not None:

                bias = bias.where(
                    ~arid_mask
                )

            # --------------------------------------------------
            # Plot
            # --------------------------------------------------
            im = bias.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=False,
                rasterized=True
            )

            # --------------------------------------------------
            # Save source data
            #
            # Convert the DataArray to a tidy table containing
            # latitude/longitude and the exact value plotted.
            # --------------------------------------------------
            if source_data:

                bias_df = bias.to_dataframe(
                    name="Mean_Bias"
                ).reset_index()

                # Add metadata identifying this panel.
                bias_df["Model"] = model_name
                bias_df["Model_Label"] = (
                    all_model_names.get(
                        model_name,
                        model_name
                    )
                    if 'all_model_names'
                    in globals()
                    else model_name
                )

                bias_df["Frequency"] = f
                bias_df["Frequency_ID"] = (
                    f"f{f}"
                )

                bias_df["Variable_ID"] = gt_id

                bias_df["Variable"] = (
                    gt_id_names.get(
                        gt_id,
                        gt_id
                    )
                    if 'gt_id_names'
                    in globals()
                    else gt_id
                )

                bias_df["Horizon"] = horizon

                bias_df["Horizon_Label"] = (
                    horizon_names.get(
                        horizon,
                        f"Horizon {horizon}"
                    )
                    if 'horizon_names'
                    in globals()
                    else f"Horizon {horizon}"
                )

                bias_df["Target_Dates"] = (
                    target_dates
                )

                # Put identifying columns first.
                preferred_columns = [
                    "Variable_ID",
                    "Variable",
                    "Horizon",
                    "Horizon_Label",
                    "Target_Dates",
                    "Model",
                    "Model_Label",
                    "Frequency",
                    "Frequency_ID",
                ]

                remaining_columns = [
                    c
                    for c in bias_df.columns
                    if c not in preferred_columns
                    and c != "Mean_Bias"
                ]

                bias_df = bias_df[
                    preferred_columns
                    + remaining_columns
                    + ["Mean_Bias"]
                ]

                source_data_rows.append(
                    bias_df
                )

            # --------------------------------------------------
            # Row labels
            # --------------------------------------------------
            if j == 0:

                label = (
                    all_model_names.get(
                        model_name,
                        model_name
                    )
                    if 'all_model_names'
                    in globals()
                    else model_name
                )

                ax.text(
                    -0.1,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight='bold'
                )

            # --------------------------------------------------
            # Column titles
            # --------------------------------------------------
            if i == 0:

                q_label = (
                    quintile_names.get(
                        f"f{f}",
                        f"f{f}"
                    )
                    if 'quintile_names'
                    in globals()
                    else f"f{f}"
                )

                ax.set_title(
                    q_label,
                    fontsize=16,
                    pad=15
                )

            else:

                ax.set_title("")

    # ----------------------------------------------------------
    # Main title
    # ----------------------------------------------------------
    pretty_gt = (
        gt_id_names.get(
            gt_id,
            gt_id
        )
        if 'gt_id_names'
        in globals()
        else gt_id
    )

    pretty_horizon = (
        horizon_names.get(
            horizon,
            f"H{horizon}"
        )
        if 'horizon_names'
        in globals()
        else f"Horizon {horizon}"
    )

    fig.suptitle(
        f"{pretty_gt} ({pretty_horizon})",
        fontsize=18,
        y=1.07
    )

    # ----------------------------------------------------------
    # Colorbar
    # ----------------------------------------------------------
    if show_cbar and im is not None:

        cbar = fig.colorbar(
            im,
            ax=axes,
            orientation="horizontal",
            fraction=0.05,
            pad=0.05,
            aspect=60
        )

        cbar.set_label(
            "Mean Bias (Prediction - Truth)",
            fontsize=14
        )

        cbar.ax.tick_params(
            labelsize=12
        )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if save_fig:
        out_path = os.path.join(OUT_DIR, f"bias_maps_{gt_id}_{horizon}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {out_path}")
        plt.savefig(out_path.replace('.pdf', '.eps'), dpi=300, bbox_inches='tight')

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        # If you want to share one workbook between multiple
        # subfigures, save it in SRC_DATA_DIR.
        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename
        )

        os.makedirs(
            SRC_DATA_DIR,
            exist_ok=True
        )

        # ----------------------------------------------
        # Combine all spatial maps
        # ----------------------------------------------
        if source_data_rows:

            source_maps_df = pd.concat(
                source_data_rows,
                ignore_index=True
            )

        else:

            source_maps_df = pd.DataFrame()

        # ----------------------------------------------
        # Metadata
        # ----------------------------------------------
        metadata_df = pd.DataFrame(
            [
                {
                    "Parameter": "Variable ID",
                    "Value": gt_id
                },
                {
                    "Parameter": "Variable",
                    "Value": pretty_gt
                },
                {
                    "Parameter": "Horizon",
                    "Value": horizon
                },
                {
                    "Parameter": "Horizon label",
                    "Value": pretty_horizon
                },
                {
                    "Parameter": "Target dates",
                    "Value": target_dates
                },
                {
                    "Parameter": "Models",
                    "Value": ", ".join(
                        model_names
                    )
                },
                {
                    "Parameter": "Frequencies",
                    "Value": ", ".join(
                        map(str, fs)
                    )
                },
                {
                    "Parameter": "Colormap",
                    "Value": cmap
                },
                {
                    "Parameter": "Colorbar minimum",
                    "Value": vmin
                },
                {
                    "Parameter": "Colorbar maximum",
                    "Value": vmax
                },
                {
                    "Parameter": "Arid mask applied",
                    "Value": (
                        gt_id.endswith("pr")
                    )
                },
                {
                    "Parameter": "Quantity",
                    "Value": (
                        "Mean Bias "
                        "(Prediction - Truth)"
                    )
                },
            ]
        )

        # ----------------------------------------------
        # Save to workbook
        #
        # Append if workbook already exists so this
        # function can share a workbook with other
        # subfigures.
        # ----------------------------------------------
        if os.path.exists(
            fig_filename
        ):

            writer_mode = "a"

        else:

            writer_mode = "w"

        with pd.ExcelWriter(
            fig_filename,
            engine="openpyxl",
            mode=writer_mode,
            if_sheet_exists=(
                "replace"
                if writer_mode == "a"
                else None
            )
        ) as writer:

            maps_sheet = (
                f"{source_data_sheet_prefix}_maps"
            )[:31]

            metadata_sheet = (
                f"{source_data_sheet_prefix}_metadata"
            )[:31]

            source_maps_df.to_excel(
                writer,
                sheet_name=maps_sheet,
                index=False,
                na_rep="NaN"
            )

            metadata_df.to_excel(
                writer,
                sheet_name=metadata_sheet,
                index=False
            )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Show / close
    # ----------------------------------------------------------
    if show_fig:

        plt.show()

    else:

        plt.close(fig)


def get_all_preds(model_names,
                     gt_ids=['era5-tas'],
                     horizons=[19],
                     fs=[1, 2, 3, 4],
                     target_dates="std_test",
                     verbose=True):
    """
    Loads Zarr prediction files for multiple gt_ids and horizons,
    reporting missing dates for each model relative to requested target dates.
    """
    # Get the full set of expected dates for reference
    expected_dates_list = get_target_dates(target_dates)
    expected_dates_set = set(pd.to_datetime(expected_dates_list))
    
    results_dict = {}

    for gt_id in gt_ids:
        measurement = gt_id.replace('era5-', '')
        
        for horizon in horizons:
            model_datasets = {}
            model_to_available_dates = {}

            for model_name in model_names:
                sn = get_selected_submodel_name(model_name, gt_id, horizon)
                quintile_ds_list = []
                common_dates_in_model = None

                for f in fs:
                    folder_prefix = f"era5-f{f}_{measurement}_{horizon}"
                    zarr_path = os.path.join(
                        "eval", "metrics", model_name, "submodel_forecasts", sn,
                        folder_prefix, f"lat_lon_pred_time-{folder_prefix}-{target_dates}.zarr"
                    )

                    if os.path.exists(zarr_path):
                        ds_f = xr.open_zarr(zarr_path)
                        var_name = f"f{f}_{measurement}"
                        ds_f = ds_f.rename({"lat_lon_pred_time": var_name})
                        quintile_ds_list.append(ds_f)
                        
                        # Intersection of dates across all fs (quintiles) for THIS model
                        current_dates = set(pd.to_datetime(ds_f.time.values))
                        if common_dates_in_model is None:
                            common_dates_in_model = current_dates
                        else:
                            common_dates_in_model = common_dates_in_model.intersection(current_dates)
                    else:
                        if verbose:
                            print(f"Warning: File missing -> {zarr_path}")
                        # If one quintile file is totally missing, the model has 0 common dates
                        common_dates_in_model = set()

                if quintile_ds_list and common_dates_in_model:
                    ds_merged = xr.merge(quintile_ds_list, join="inner")
                    model_datasets[model_name] = ds_merged
                    model_to_available_dates[model_name] = common_dates_in_model
                else:
                    model_to_available_dates[model_name] = set()

            if not model_datasets:
                continue

            # Find dates that exist in EVERY model
            all_model_common_dates = set.intersection(*model_to_available_dates.values())
            all_model_common_dates_sorted = sorted(list(all_model_common_dates))

            # Reporting Section
            if verbose:
                print(f"\n--- Report for {gt_id} | Horizon: {horizon} ---")
                for model_name, available_dates in model_to_available_dates.items():
                    # Missing relative to the original 'std_test' request
                    missing_from_target = sorted(list(expected_dates_set - available_dates))
                    
                    count = len(available_dates)
                    total = len(expected_dates_set)
                    print(f"Model {model_name}: {count}/{total} dates found.")
                    
                    if missing_from_target:
                        missing_strs = [d.strftime('%Y-%m-%d') for d in missing_from_target]
                        # Only print the first few if the list is huge
                        if len(missing_strs) > 10:
                            print(f"   ==> Missing: {', '.join(missing_strs[:10])}... ({len(missing_strs)} total)")
                        else:
                            print(f"   ==> Missing: {', '.join(missing_strs)}")

                print(f"Intersection: {len(all_model_common_dates_sorted)} dates shared by all models.")

            # Align and Combine
            final_model_list = []
            for model_name in sorted(model_datasets.keys()):
                ds = model_datasets[model_name]
                ds_aligned = ds.sel(time=all_model_common_dates_sorted)
                ds_aligned = ds_aligned.assign_coords(model=model_name).expand_dims("model")
                final_model_list.append(ds_aligned)

            if final_model_list:
                full_ds = xr.concat(final_model_list, dim="model")
                full_ds = full_ds.transpose("model", "latitude", "longitude", "time")
                results_dict[(gt_id, horizon)] = full_ds

    return results_dict


def print_model_bias(
    results_dict,
    model_names,
    gt_id='era5-mslp',
    horizon=19,
    fs=[1, 2, 3, 4],
    source_data=False,
    source_data_filename="fig_s3-bias_maps_precip.xlsx",
    source_data_sheet_prefix="fig_s3b"
):
    """
    Computes bias metrics using pre-loaded results_dict.

    Uses latitude-weighting for global average accuracy.

    If source_data=True, appends the summary statistics to the
    same Excel workbook used by plot_bias_maps_3x4.

    Source data sheet:
        <prefix>_stats

    Columns:
        Variable_ID
        Variable
        Horizon
        Horizon_Label
        Model
        Model_Label
        Frequency
        Frequency_ID
        Mean_Bias
        Mean_Abs_Bias
        RMS_Bias
        Overall
    """

    import os
    import numpy as np
    import pandas as pd
    import xarray as xr

    measurement = gt_id.replace(
        'era5-',
        ''
    )

    results = []

    # ----------------------------------------------------------
    # Get data from dictionary
    # ----------------------------------------------------------
    if (gt_id, horizon) not in results_dict:

        print(
            f"Key {(gt_id, horizon)} "
            f"not found in results_dict."
        )

        return []

    ds = results_dict[
        (gt_id, horizon)
    ]

    # ----------------------------------------------------------
    # Pretty names
    # ----------------------------------------------------------
    pretty_name = (
        gt_id_names.get(
            gt_id,
            gt_id
        )
        if 'gt_id_names' in globals()
        else gt_id
    )

    pretty_hor = (
        horizon_names.get(
            horizon,
            horizon
        )
        if 'horizon_names' in globals()
        else horizon
    )

    # ----------------------------------------------------------
    # Header printing
    # ----------------------------------------------------------
    print(
        f"\n--- Bias Analysis: "
        f"{pretty_name} ({pretty_hor}) ---"
    )

    print(
        f"{'Model':<20} | "
        f"{'Quintile':<8} | "
        f"{'Mean Bias':<12} | "
        f"{'Mean Abs Bias':<12} | "
        f"{'RMS Bias':<12}"
    )

    print("-" * 80)

    # ----------------------------------------------------------
    # Source-data rows
    # ----------------------------------------------------------
    source_rows = []

    # ----------------------------------------------------------
    # Loop over models
    # ----------------------------------------------------------
    for model_name in [
        m for m in model_names
        if m != 'gt'
    ]:

        model_mbs = []
        model_mabs = []
        model_rms = []

        # ------------------------------------------------------
        # Loop over frequencies/quintiles
        # ------------------------------------------------------
        for f in fs:

            var_name = (
                f"f{f}_{measurement}"
            )

            if var_name not in ds:

                continue

            # --------------------------------------------------
            # Select prediction and truth
            # --------------------------------------------------
            pred = ds[
                var_name
            ].sel(
                model=model_name
            )

            truth = ds[
                var_name
            ].sel(
                model='gt'
            )

            # Align times
            pred, truth_aligned = xr.align(
                pred,
                truth
            )

            bias = (
                pred
                - truth_aligned
            )

            # --------------------------------------------------
            # Latitude weighting
            # --------------------------------------------------
            weights = np.cos(
                np.deg2rad(
                    bias['latitude']
                )
            )

            # --------------------------------------------------
            # Weighted statistics
            # --------------------------------------------------
            mb = (
                bias
                .weighted(weights)
                .mean()
                .compute()
                .item()
            )

            mab = (
                np.abs(bias)
                .weighted(weights)
                .mean()
                .compute()
                .item()
            )

            rms = np.sqrt(
                (
                    bias ** 2
                )
                .weighted(weights)
                .mean()
                .compute()
                .item()
            )

            # --------------------------------------------------
            # Print
            # --------------------------------------------------
            print(
                f"{model_name:<20} | "
                f"f{f:<7} | "
                f"{mb:>12.4f} | "
                f"{mab:>12.4f} | "
                f"{rms:>12.4f}"
            )

            model_mbs.append(mb)
            model_mabs.append(mab)
            model_rms.append(rms)

            # --------------------------------------------------
            # Return-value data
            # --------------------------------------------------
            results.append(
                {
                    'model': model_name,
                    'quintile': f,
                    'mean_bias': mb,
                    'mean_abs_bias': mab,
                    'rms_bias': rms
                }
            )

            # --------------------------------------------------
            # Source-data row
            # --------------------------------------------------
            model_label = (
                all_model_names.get(
                    model_name,
                    model_name
                )
                if 'all_model_names'
                in globals()
                else model_name
            )

            source_rows.append(
                {
                    "Variable_ID": gt_id,
                    "Variable": pretty_name,
                    "Horizon": horizon,
                    "Horizon_Label": pretty_hor,
                    "Model": model_name,
                    "Model_Label": model_label,
                    "Frequency": f,
                    "Frequency_ID": f"f{f}",
                    "Mean_Bias": mb,
                    "Mean_Abs_Bias": mab,
                    "RMS_Bias": rms,
                    "Overall": False,
                }
            )

        # ------------------------------------------------------
        # Overall statistics for model
        # ------------------------------------------------------
        if model_mbs:

            avg_mb = np.mean(
                model_mbs
            )

            avg_mab = np.mean(
                model_mabs
            )

            avg_rms = np.sqrt(
                np.mean(
                    np.square(
                        model_rms
                    )
                )
            )

            print(
                f"{' '*20} | "
                f"{'OVERALL':<8} | "
                f"{avg_mb:>12.4f} | "
                f"{avg_mab:>12.4f} | "
                f"{avg_rms:>12.4f}"
            )

            model_label = (
                all_model_names.get(
                    model_name,
                    model_name
                )
                if 'all_model_names'
                in globals()
                else model_name
            )

            # ----------------------------------------------
            # Save overall row
            # ----------------------------------------------
            source_rows.append(
                {
                    "Variable_ID": gt_id,
                    "Variable": pretty_name,
                    "Horizon": horizon,
                    "Horizon_Label": pretty_hor,
                    "Model": model_name,
                    "Model_Label": model_label,
                    "Frequency": "OVERALL",
                    "Frequency_ID": "overall",
                    "Mean_Bias": avg_mb,
                    "Mean_Abs_Bias": avg_mab,
                    "RMS_Bias": avg_rms,
                    "Overall": True,
                }
            )

        print("-" * 80)

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        # Same directory convention as plot_bias_maps_3x4
        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename
        )

        os.makedirs(
            SRC_DATA_DIR,
            exist_ok=True
        )

        stats_df = pd.DataFrame(
            source_rows
        )

        # ------------------------------------------------------
        # Metadata
        # ------------------------------------------------------
        metadata_df = pd.DataFrame(
            [
                {
                    "Parameter": "Variable ID",
                    "Value": gt_id
                },
                {
                    "Parameter": "Variable",
                    "Value": pretty_name
                },
                {
                    "Parameter": "Horizon",
                    "Value": horizon
                },
                {
                    "Parameter": "Horizon label",
                    "Value": pretty_hor
                },
                {
                    "Parameter": "Models",
                    "Value": ", ".join(
                        [
                            m
                            for m in model_names
                            if m != "gt"
                        ]
                    )
                },
                {
                    "Parameter": "Frequencies",
                    "Value": ", ".join(
                        map(str, fs)
                    )
                },
                {
                    "Parameter": "Weighting",
                    "Value": "Latitude weighting: cos(latitude)"
                },
                {
                    "Parameter": "Mean Bias",
                    "Value": "Weighted mean of Prediction - Truth"
                },
                {
                    "Parameter": "Mean Absolute Bias",
                    "Value": "Weighted mean of abs(Prediction - Truth)"
                },
                {
                    "Parameter": "RMS Bias",
                    "Value": "sqrt(weighted mean of (Prediction - Truth)^2)"
                },
            ]
        )

        # ------------------------------------------------------
        # Append to existing workbook
        # ------------------------------------------------------
        if os.path.exists(
            fig_filename
        ):

            writer_mode = "a"

        else:

            writer_mode = "w"

        with pd.ExcelWriter(
            fig_filename,
            engine="openpyxl",
            mode=writer_mode,
            if_sheet_exists=(
                "replace"
                if writer_mode == "a"
                else None
            )
        ) as writer:

            stats_sheet = (
                f"{source_data_sheet_prefix}_stats"
            )[:31]

            metadata_sheet = (
                f"{source_data_sheet_prefix}_stats_metadata"
            )[:31]

            stats_df.to_excel(
                writer,
                sheet_name=stats_sheet,
                index=False,
                na_rep="NaN"
            )

            metadata_df.to_excel(
                writer,
                sheet_name=metadata_sheet,
                index=False
            )

        print(
            f"Source data appended: {fig_filename}"
        )

    return results




def plot_single_extreme_bss_barplot(all_wtd_mse,
                    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
                    target_dates='std_test',
                    show_fig=True,
                    save_fig=True,
                    n_boot=5000,
                    seed=42, 
                    prefix="F95_",
                    verbose=False,
                    y_bottom=None):
    """
    Plot a barplot of the Brier Skill Score (BSS) for multiple models.

    Args:
        all_wtd_mse (dict): Dictionary containing weighted MSE for each model and variable
        model_names (list): List of model names to include in the plot
        target_dates (str): Target dates for the evaluation
        show_fig (bool): Whether to display the figure
        save_fig (bool): Whether to save the figure to disk
        n_boot (int): Number of bootstrap samples for confidence intervals
        seed (int): Random seed for reproducibility
        prefix (str): Prefix for the variable names in the dataset
        verbose (bool): Whether to print verbose output
        y_bottom (float): Bottom limit for the y-axis
    """

    model_names = [m for m in model_names if m != 'climatology']

    variables = {
        "Extreme Temperature": "tas",
        "Extreme Precipitation": "pr",
        "Extreme Pressure": "mslp",
    }
    
    horizons = {
        "Week 3": 19,
        "Week 4": 26,
    }

    # Compute BSS and test for significant improvements of  
    # target model over baselines
    results_bss = {}
    results_significant = {}
    target_model = model_names[-1]
    for var_name, var_code in variables.items():
        for week_name, horizon in horizons.items():
            key = f"era5-{prefix}{var_code}_{horizon}"
            ds = all_wtd_mse[key].dropna("time", how="any")

            n = ds.time.size

            bss = {}

            pbc = ds[target_model].values
            clim = ds["climatology"].values
            # Test if target model improves significantly over all baselines
            all_significant = True
            for m in model_names:
                baseline = ds[m].values
                bss[m] = 1 - (np.mean(baseline) / np.mean(clim))
                if m != target_model:
                    # Compute lower confidence bound for BSS improvement
                    # of target model over the baseline model
                    lower_cb = lower_confidence_bound(
                        baseline-pbc,
                        clim=clim,
                        n_boot=n_boot,
                        seed=seed,
                    )
                    if verbose:
                        printf(f"  {var_name} | {week_name} | {target_model} - {m} BSS confidence bound: {lower_cb}")
                    if lower_cb <= 0:
                        all_significant = False
            results_bss[(var_name, week_name)] = bss
            results_significant[(var_name, week_name)] = all_significant
            

    categories = [(v, w) for v in variables for w in horizons]

    values = np.array([
        [results_bss[c][model_names[i]] for c in categories]
        for i in range(len(model_names))
    ])

    x = np.arange(len(horizons))
    width = 0.22

    # =========================================================
    # Single plot with shared y axis
    # =========================================================

    fig, ax = plt.subplots(figsize=(14, 6))

    n_vars = len(variables)
    n_weeks = len(horizons)

    group_gap = .0   # space between variable blocks
    bar_w = width

    # Build grouped x positions
    x_positions = []
    for v in range(n_vars):
        base = v * (n_weeks + group_gap)
        for w in range(n_weeks):
            x_positions.append(base + w)

    x_positions = np.array(x_positions)

    # Plot bars
    for i, model in enumerate(model_names):
        bars = ax.bar(
            x_positions + (i - (len(model_names) - 1) / 2) * bar_w,
            values[i],
            bar_w,
            capsize=4,
            label=all_model_names[model],
            color=model_colors.get(model, "gray"),
        )
        # Add hatch if target model improved significantly over all baselines
        if model == target_model and results_significant[(var_name, week_name)]:
            for bar in bars:
                bar.set_hatch("x")

    # ---- X tick labels (weeks repeated per variable) ----
    week_labels = list(horizons.keys())
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels * n_vars, fontsize=20)

    # ---- Variable labels centered under each group ----
    var_names = list(variables.keys())
    for v in range(n_vars):
        base = v * (n_weeks + group_gap)
        center = base + (n_weeks - 1) / 2

        ax.text(
            center,
            -0.12,
            var_names[v],
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
            transform=ax.get_xaxis_transform()
        )

    # ---- y-axis label ----
    percentile = int(prefix.rstrip("_").lstrip("F"))
    label_suffix = f"Top {100-percentile}%" if percentile > 50 else f"Bottom {percentile}%"

    ax.set_ylabel(
        f"Brier skill score ({label_suffix})",
        fontsize=20,
        fontweight="bold"
    )

    ax.tick_params(axis="y", labelsize=16)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=model_colors.get(model, "gray"))
        for model in model_names
    ]
    legend_labels = [all_model_names[model] for model in model_names]

    ax.legend(
        legend_handles,
        legend_labels,
        ncol=len(model_names),
        fontsize=18,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=y_bottom)
    plt.tight_layout()
    fig_to_save = fig

    
        
    # =========================================================
    # save and show
    # =========================================================
    filename = os.path.join(
        EXTREMES_OUT_DIR,
        f"barplot_{prefix}bss_{target_dates}.pdf"
    )

    if save_fig:
        fig_to_save.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
        fig_to_save.savefig(filename.replace('.pdf', '.jpeg'),
                            dpi=300, transparent=True, bbox_inches='tight')
        print(f"Figure saved: {filename}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig_to_save)




def plot_single_horizon_bss_barplot(
    all_wtd_mse,
    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
    horizon=19,
    prefixes=['F95_', 'F5_'],
    target_dates='std_test',
    show_fig=True,
    save_fig=True,
    n_boot=5000,
    seed=42,
    verbose=False,
    y_bottom=None,
    source_data=False,
    source_data_filename="fig_s9-bss_plots.xlsx",
    source_data_sheet_prefix="FigS9b",
):
    """
    Plot a barplot of Brier Skill Score (BSS) for one horizon and two extremes.

    For each variable, the two bar groups correspond to the two entries in
    prefixes (for example, ['F95_', 'F5_'] for extreme highs and lows).

    If source_data=True, the data underlying the plotted bars and the
    significance information used for hatching are saved to an Excel
    workbook.

    Args:
        all_wtd_mse: Dictionary of weighted MSE datasets for each task.
        model_names: Models to include in the plot.
        horizon: Forecast horizon.
        prefixes: Exactly two prefixes corresponding to the two extremes.
        target_dates: Target-date identifier.
        show_fig: Whether to display the figure.
        save_fig: Whether to save the figure.
        n_boot: Number of bootstrap samples for significance testing.
        seed: Random seed for bootstrap significance testing.
        verbose: Whether to print verbose output.
        y_bottom: Lower y-axis limit.
        source_data: Whether to save source data.
        source_data_filename: XLSX filename for source data.
        source_data_sheet_prefix: Prefix for source-data worksheet names.
    """

    if len(prefixes) != 2:
        raise ValueError(
            "prefixes must contain exactly two entries, "
            "e.g. ['F95_', 'F5_']"
        )

    model_names = [m for m in model_names if m != 'climatology']

    if len(model_names) == 0:
        raise ValueError("model_names must contain at least one model.")

    variables = {
        "Temperature": "tas",
        "Precipitation": "pr",
        "Sea Level Pressure": "mslp",
    }

    def get_extreme_label(prefix):
        percentile = int(prefix.rstrip("_").lstrip("F"))
        if percentile > 50:
            return "Very High"
        return "Very Low"

    extreme_labels = [
        get_extreme_label(prefix)
        for prefix in prefixes
    ]

    # ----------------------------------------------------------
    # Compute BSS and test for significant improvements
    # ----------------------------------------------------------
    results_bss = {}
    results_significant = {}

    target_model = model_names[-1]

    for var_name, var_code in variables.items():

        for prefix in prefixes:

            key = f"era5-{prefix}{var_code}_{horizon}"

            ds = all_wtd_mse[key].dropna("time", how="any")

            bss = {}

            target = ds[target_model].values
            clim = ds["climatology"].values

            all_significant = True

            for model_name in model_names:

                baseline = ds[model_name].values

                bss[model_name] = (
                    1
                    - (
                        np.mean(baseline)
                        / np.mean(clim)
                    )
                )

                if model_name != target_model:

                    lower_cb = lower_confidence_bound(
                        baseline - target,
                        clim=clim,
                        n_boot=n_boot,
                        seed=seed,
                    )

                    if verbose:
                        printf(
                            f"  {var_name} | H{horizon} | {prefix} | "
                            f"{target_model} - {model_name} "
                            f"BSS confidence bound: {lower_cb}"
                        )

                    if lower_cb <= 0:
                        all_significant = False

            results_bss[(var_name, prefix)] = bss
            results_significant[(var_name, prefix)] = all_significant

    # ----------------------------------------------------------
    # Organize categories and values
    # ----------------------------------------------------------
    categories = [
        (var_name, prefix)
        for var_name in variables
        for prefix in prefixes
    ]

    values = np.array([
        [
            results_bss[category][model_names[i]]
            for category in categories
        ]
        for i in range(len(model_names))
    ])

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        # ------------------------------------------------------
        # BSS values used directly for the plotted bars
        # ------------------------------------------------------
        results_rows = []

        for (variable, prefix), models in results_bss.items():

            extreme_label = get_extreme_label(prefix)

            for model, bss in models.items():

                results_rows.append({
                    "Variable": variable,
                    "Prefix": prefix,
                    "Extreme": extreme_label,
                    "Horizon": horizon,
                    "Target dates": target_dates,
                    "Model": model,
                    "BSS": bss,
                })

        results_df = pd.DataFrame(results_rows)

        # ------------------------------------------------------
        # Significance information used to hatch target bars
        # ------------------------------------------------------
        significance_rows = []

        for (variable, prefix), significant in results_significant.items():

            significance_rows.append({
                "Variable": variable,
                "Prefix": prefix,
                "Extreme": get_extreme_label(prefix),
                "Horizon": horizon,
                "Target dates": target_dates,
                "Target model": target_model,
                "Significant_All": significant,
            })

        significance_df = pd.DataFrame(significance_rows)

        # ------------------------------------------------------
        # Write to the existing workbook if it exists.
        #
        # This preserves source-data sheets written by other
        # subfigures/functions using the same XLSX file.
        # ------------------------------------------------------
        results_sheet_name = (
            f"{source_data_sheet_prefix}_results"
        )

        significance_sheet_name = (
            f"{source_data_sheet_prefix}_significance"
        )

        # Excel worksheet names are limited to 31 characters.
        if len(results_sheet_name) > 31:
            raise ValueError(
                f"Source-data sheet name '{results_sheet_name}' "
                "is longer than Excel's 31-character limit."
            )

        if len(significance_sheet_name) > 31:
            raise ValueError(
                f"Source-data sheet name '{significance_sheet_name}' "
                "is longer than Excel's 31-character limit."
            )

        if os.path.exists(fig_filename):

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:

                results_df.to_excel(
                    writer,
                    sheet_name=results_sheet_name,
                    index=False,
                )

                significance_df.to_excel(
                    writer,
                    sheet_name=significance_sheet_name,
                    index=False,
                )

        else:

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="w",
            ) as writer:

                results_df.to_excel(
                    writer,
                    sheet_name=results_sheet_name,
                    index=False,
                )

                significance_df.to_excel(
                    writer,
                    sheet_name=significance_sheet_name,
                    index=False,
                )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    n_vars = len(variables)
    n_extremes = len(prefixes)

    width = 0.22
    group_gap = 0.0

    x_positions = []

    for var_idx in range(n_vars):

        base = var_idx * (
            n_extremes + group_gap
        )

        for extreme_idx in range(n_extremes):

            x_positions.append(
                base + extreme_idx
            )

    x_positions = np.array(x_positions)

    # ----------------------------------------------------------
    # Draw bars
    # ----------------------------------------------------------
    for i, model_name in enumerate(model_names):

        bars = ax.bar(
            x_positions
            + (
                i
                - (len(model_names) - 1) / 2
            ) * width,
            values[i],
            width,
            capsize=4,
            label=all_model_names[model_name],
            color=model_colors.get(
                model_name,
                "gray",
            ),
        )

        if model_name == target_model:

            for bar_idx, bar in enumerate(bars):

                if results_significant[
                    categories[bar_idx]
                ]:

                    bar.set_hatch("x")

    # ----------------------------------------------------------
    # X labels
    # ----------------------------------------------------------
    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        extreme_labels * n_vars,
        fontsize=20,
    )

    # Variable labels centered below each pair of extremes.
    variable_names = list(variables.keys())

    for var_idx in range(n_vars):

        base = var_idx * (
            n_extremes + group_gap
        )

        center = (
            base
            + (n_extremes - 1) / 2
        )

        ax.text(
            center,
            -0.12,
            variable_names[var_idx],
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_ylabel(
        "Brier skill score",
        fontsize=20,
        fontweight="bold",
    )

    ax.tick_params(
        axis="y",
        labelsize=16,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    ax.set_ylim(
        bottom=y_bottom
    )

    # ----------------------------------------------------------
    # Legend
    # ----------------------------------------------------------
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=model_colors.get(
                model_name,
                "gray",
            ),
        )
        for model_name in model_names
    ]

    legend_labels = [
        all_model_names[model_name]
        for model_name in model_names
    ]

    ax.legend(
        legend_handles,
        legend_labels,
        ncol=len(model_names),
        fontsize=18,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    plt.tight_layout()

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    prefix_name = "_".join(
        prefix.rstrip("_")
        for prefix in prefixes
    )

    filename = os.path.join(
        EXTREMES_OUT_DIR,
        f"barplot_{prefix_name}_bss_{horizon}_{target_dates}.pdf",
    )

    if save_fig:

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        plt.savefig(
            filename.replace(".pdf", ".jpeg"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        print(
            f"Figure saved: {filename}"
        )

        plt.savefig(filename.replace('.pdf', '.eps'), dpi=300, transparent=True, bbox_inches='tight')
        print(f"Figure saved: {filename.replace('.pdf', '.eps')}") 

    if show_fig:
        plt.show()
    else:
        plt.close(fig)




def plot_single_extreme_bss_diff_grid_6x4(
    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
    prefix="F95_",
    target_dates="std_test",
    horizons=[19, 26],
    diff_cmap="bwr",
    skill_cmap="RdBu_r",
    show_fig=True,
    save_fig=False):

    gt_id_names={f'era5-{prefix}tas' : "Temperature", 
                 f'era5-{prefix}pr' : "Precipitation", 
                 f'era5-{prefix}mslp' : "Sea Level Pressure"}
    gt_ids = list(gt_id_names.keys())

    num_rows = len(gt_ids) * len(horizons)
    num_cols = len(model_names) + 2  # models + spacer + diff

    # Create a visible gap via a thin spacer column
    width_ratios = [1]*len(model_names) + [0.05, 1]

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(18, 18),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios}
    )

    # Load arid mask once if precipitation is being plotted
    if any(gt_id.endswith("pr") for gt_id in gt_ids):
        # print("Computing zero quintiles mask")
        quintiles = xr.open_dataset("data/era5-quintiles-pr.zarr", engine="zarr").load()

    curr_row = 0
    im_metric, im_diff = None, None

    metric = 'lat_lon_mse'
    for gt_id in gt_ids:
        for horizon in horizons:

            arid_mask = None
            if gt_id.endswith("pr"):
                quintiles_sel = quintiles.sel(time=get_target_dates(target_dates, horizon))
                arid_mask = (quintiles_sel["pr"].isel(quantile=-1) == 0).any(dim="time")    

            task = f"{gt_id}_{horizon}"

            # Load climatology lat_lon_mse
            sn = get_selected_submodel_name("climatology", gt_id, horizon)
            filename = os.path.join('eval', 'metrics', "climatology",
                                    'submodel_forecasts', sn,
                                    task,
                                    f'{metric}-{task}-{target_dates}.zarr')
            clim = xr.open_zarr(filename).load()
            
            # Compute spatial BSS for each model
            metrics = {}
            for model_name in model_names:
                sn = get_selected_submodel_name(model_name, gt_id, horizon)

                filename = os.path.join('eval', 'metrics', model_name,
                                        'submodel_forecasts', sn,
                                        task,
                                        f'{metric}-{task}-{target_dates}.zarr')

                # Spatial BSS = 1 - (model_lat_lon_mse / climatology_lat_lon_mse)
                metrics[model_name] = 1 - xr.open_zarr(filename).load()/clim

            metrics['diff'] = metrics[model_names[-1]] - metrics[model_names[-2]]

            row_axes = axes[curr_row]

            # Style all axes
            for ax in row_axes:
                ax.set_global()
                ax.coastlines(color="black", linewidth=0.6)
                ax.spines["geo"].set_linewidth(2.25)

            # Hide spacer column
            spacer_idx = len(model_names)
            row_axes[spacer_idx].set_visible(False)

            # Row label
            row_label = f"{gt_id_names[gt_id]}\n{horizon_names[horizon]}"
            row_axes[0].text(-0.15, 0.5, row_label,
                             va="center", ha="center",
                             rotation=90,
                             transform=row_axes[0].transAxes,
                             fontsize=20)

            # Plot model columns
            for i, model_name in enumerate(model_names):
                data = metrics[model_name][metric]
                if arid_mask is not None:
                    data = data.where(~arid_mask)
                im = data.plot(
                    ax=row_axes[i],
                    transform=ccrs.PlateCarree(),
                    cmap=skill_cmap,
                    vmin=-1, vmax=1,
                    add_colorbar=False,
                    rasterized=True
                )

            # Diff column (last column)
            diff_ax = row_axes[-1]

            diff_data = metrics['diff'][metric]
            if arid_mask is not None:
                diff_data = diff_data.where(~arid_mask)
            im2 = diff_data.plot(
                ax=diff_ax,
                transform=ccrs.PlateCarree(),
                cmap=diff_cmap,
                vmin=-0.2, vmax=0.2,
                add_colorbar=False,
                rasterized=True
            )

            nonnull = diff_data.notnull()
            num_nonnull = nonnull.sum().values
            print(f"{task}: % grid cells improved: "
                  f"{float(((nonnull & (diff_data > 0)).sum() / num_nonnull).values)} "
                  f"of {num_nonnull}")


            # Titles (top row only)
            if curr_row == 0:
                for i, model_name in enumerate(model_names):
                    row_axes[i].set_title(
                        all_model_names[model_name],
                        fontsize=20,
                        pad=30
                    )

                diff_ax.set_title(
                    f"{all_model_names[model_names[-1]]} - {all_model_names[model_names[-2]]}",
                    fontsize=20,
                    pad=30
                )
            else:
                for ax in row_axes:
                    ax.set_title("")

            im_metric, im_diff = im, im2
            curr_row += 1


    # Colorbars
    cax1 = fig.add_axes([0.11, -0.02, 0.6, 0.02])
    cax2 = fig.add_axes([0.79, -0.02, 0.19, 0.02])
    
    cbar1 = fig.colorbar(im_metric, cax=cax1,
                         orientation="horizontal",
                         fraction=0.02, pad=0.02, aspect=40)

    # Extract percentile from prefix for y-axis label
    percentile = int(prefix.rstrip("_").lstrip("F"))
    label_suffix = f"Extreme highs (Top {100-percentile}%)" if percentile > 50 else f"Extreme lows (Bottom {percentile}%)"
    cbar1.set_label(f'Brier skill score (BSS): {label_suffix}', fontsize=20, labelpad=10)
    cbar1.ax.tick_params(labelsize=20)

    cbar2 = fig.colorbar(im_diff, cax=cax2,
                         orientation="horizontal",
                         fraction=0.02, pad=0.02, aspect=40)
    cbar2.set_label(f"BSS difference",
                    fontsize=20, labelpad=10)
    cbar2.ax.tick_params(labelsize=20)

    if save_fig:
        outfile = os.path.join(
            EXTREMES_OUT_DIR,
            f"lat_lon_{prefix}bss_{target_dates}.pdf"
        )
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Saved: {outfile}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)



def plot_single_horizon_bss_diff_grid_6x4(
    model_names=['ecmwf', 'debiased_ecmwf', 'pbc_ecmwf_combo'],
    prefixes=['F95_', 'F5_'],
    horizon=19,
    target_dates="std_test",
    diff_cmap="bwr",
    skill_cmap="RdBu_r",
    show_fig=True,
    save_fig=False,
    source_data=False,
    source_data_filename="fig_s9-bss_plots.xlsx",
    source_data_sheet_prefix="FigS9b_diff",
):

    if len(prefixes) != 2:
        raise ValueError(
            "prefixes must contain exactly two entries, "
            "e.g. ['F95_', 'F5_']"
        )

    variables = {
        "tas": "Temperature",
        "pr": "Precipitation",
        "mslp": "Sea Level Pressure",
    }

    # Load arid mask once if precipitation is being plotted
    if any(var_code == "pr" for var_code in variables):

        quintiles = xr.open_dataset(
            "data/era5-quintiles-pr.zarr",
            engine="zarr",
        ).load()

    def get_extreme_label(prefix):
        percentile = int(
            prefix.rstrip("_").lstrip("F")
        )

        if percentile > 50:
            return "Very High"

        return "Very Low"

    num_rows = len(variables) * len(prefixes)
    num_cols = len(model_names) + 2

    # Create a visible gap via a thin spacer column
    width_ratios = (
        [1] * len(model_names)
        + [0.05, 1]
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(18, 18),
        subplot_kw={
            "projection": ccrs.Robinson()
        },
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": width_ratios
        },
    )

    # ----------------------------------------------------------
    # Source-data storage
    # ----------------------------------------------------------
    #
    # Store the actual spatial BSS fields after application of
    # the precipitation arid mask, as well as the BSS difference
    # field used in the final column.
    #
    # Each entry is an xarray DataArray corresponding to one
    # plotted map.
    # ----------------------------------------------------------
    source_bss_data = []
    source_diff_data = []

    curr_row = 0
    im_metric, im_diff = None, None

    metric = 'lat_lon_mse'

    for var_code, var_name in variables.items():

        for prefix in prefixes:

            gt_id = f"era5-{prefix}{var_code}"
            task = f"{gt_id}_{horizon}"

            # --------------------------------------------------
            # Precipitation arid mask
            # --------------------------------------------------
            arid_mask = None

            if gt_id.endswith("pr"):

                quintiles_sel = quintiles.sel(
                    time=get_target_dates(
                        target_dates,
                        horizon,
                    )
                )

                arid_mask = (
                    quintiles_sel["pr"]
                    .isel(quantile=-1)
                    == 0
                ).any(dim="time")

            # --------------------------------------------------
            # Load climatology lat_lon_mse
            # --------------------------------------------------
            sn = get_selected_submodel_name(
                "climatology",
                gt_id,
                horizon,
            )

            filename = os.path.join(
                'eval',
                'metrics',
                "climatology",
                'submodel_forecasts',
                sn,
                task,
                f'{metric}-{task}-{target_dates}.zarr'
            )

            clim = xr.open_zarr(filename).load()

            # --------------------------------------------------
            # Compute spatial BSS for each model
            # --------------------------------------------------
            metrics = {}

            for model_name in model_names:

                sn = get_selected_submodel_name(
                    model_name,
                    gt_id,
                    horizon,
                )

                filename = os.path.join(
                    'eval',
                    'metrics',
                    model_name,
                    'submodel_forecasts',
                    sn,
                    task,
                    f'{metric}-{task}-{target_dates}.zarr'
                )

                # Spatial BSS =
                # 1 - (model lat_lon_mse / climatology lat_lon_mse)
                metrics[model_name] = (
                    1
                    - xr.open_zarr(filename).load()
                    / clim
                )

            # Difference between target and baseline model.
            metrics['diff'] = (
                metrics[model_names[-1]]
                - metrics[model_names[-2]]
            )

            row_axes = axes[curr_row]

            # --------------------------------------------------
            # Style all axes
            # --------------------------------------------------
            for ax in row_axes:

                ax.set_global()

                ax.coastlines(
                    color="black",
                    linewidth=0.6,
                )

                ax.spines["geo"].set_linewidth(2.25)

            # Hide spacer column
            spacer_idx = len(model_names)
            row_axes[spacer_idx].set_visible(False)

            # --------------------------------------------------
            # Row label
            # --------------------------------------------------
            row_label = (
                f"{var_name}\n"
                f"{get_extreme_label(prefix)}"
            )

            row_axes[0].text(
                -0.15,
                0.5,
                row_label,
                va="center",
                ha="center",
                rotation=90,
                transform=row_axes[0].transAxes,
                fontsize=20,
            )

            # --------------------------------------------------
            # Plot model BSS columns
            # --------------------------------------------------
            for i, model_name in enumerate(model_names):

                data = metrics[model_name][metric]

                if arid_mask is not None:
                    data = data.where(~arid_mask)

                # Store exactly the data being plotted.
                if source_data:

                    source_bss_data.append({
                        "Variable": var_name,
                        "Variable code": var_code,
                        "Prefix": prefix,
                        "Extreme": get_extreme_label(prefix),
                        "Horizon": horizon,
                        "Target dates": target_dates,
                        "Model": model_name,
                        "Data": data,
                    })

                im = data.plot(
                    ax=row_axes[i],
                    transform=ccrs.PlateCarree(),
                    cmap=skill_cmap,
                    vmin=-1,
                    vmax=1,
                    add_colorbar=False,
                    rasterized=True,
                )

            # --------------------------------------------------
            # Plot difference column
            # --------------------------------------------------
            diff_ax = row_axes[-1]

            diff_data = metrics['diff'][metric]

            if arid_mask is not None:
                diff_data = diff_data.where(~arid_mask)

            # Store exactly the difference field being plotted.
            if source_data:

                source_diff_data.append({
                    "Variable": var_name,
                    "Variable code": var_code,
                    "Prefix": prefix,
                    "Extreme": get_extreme_label(prefix),
                    "Horizon": horizon,
                    "Target dates": target_dates,
                    "Target model": model_names[-1],
                    "Baseline model": model_names[-2],
                    "Data": diff_data,
                })

            im2 = diff_data.plot(
                ax=diff_ax,
                transform=ccrs.PlateCarree(),
                cmap=diff_cmap,
                vmin=-0.2,
                vmax=0.2,
                add_colorbar=False,
                rasterized=True,
            )

            # --------------------------------------------------
            # Print fraction of grid cells improved
            # --------------------------------------------------
            nonnull = diff_data.notnull()
            num_nonnull = nonnull.sum().values

            print(
                f"{task}: % grid cells improved: "
                f"{float((
                    (nonnull & (diff_data > 0)).sum()
                    / num_nonnull
                ).values)} "
                f"of {num_nonnull}"
            )

            # --------------------------------------------------
            # Titles (top row only)
            # --------------------------------------------------
            if curr_row == 0:

                for i, model_name in enumerate(model_names):

                    row_axes[i].set_title(
                        all_model_names[model_name],
                        fontsize=20,
                        pad=30,
                    )

                diff_ax.set_title(
                    f"{all_model_names[model_names[-1]]} - "
                    f"{all_model_names[model_names[-2]]}",
                    fontsize=20,
                    pad=30,
                )

            else:

                for ax in row_axes:
                    ax.set_title("")

            im_metric = im
            im_diff = im2

            curr_row += 1

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------
    if source_data:

        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        # ------------------------------------------------------
        # Convert spatial BSS fields to tidy DataFrame
        # ------------------------------------------------------
        bss_rows = []

        for entry in source_bss_data:

            data = entry["Data"]

            # Remove singleton dimensions, while preserving the
            # spatial latitude/longitude dimensions.
            data = data.squeeze()

            df = data.to_dataframe(
                name="BSS"
            ).reset_index()

            # Identify latitude/longitude column names. In case
            # the dataset uses lat/lon or latitude/longitude,
            # retain the original names.
            for _, row in df.iterrows():

                row_data = {
                    "Variable": entry["Variable"],
                    "Variable code": entry["Variable code"],
                    "Prefix": entry["Prefix"],
                    "Extreme": entry["Extreme"],
                    "Horizon": entry["Horizon"],
                    "Target dates": entry["Target dates"],
                    "Model": entry["Model"],
                    "BSS": row["BSS"],
                }

                # Add all coordinate columns.
                for coord_name in df.columns:

                    if coord_name == "BSS":
                        continue

                    row_data[coord_name] = row[coord_name]

                bss_rows.append(row_data)

        bss_df = pd.DataFrame(bss_rows)

        # ------------------------------------------------------
        # Convert difference fields to tidy DataFrame
        # ------------------------------------------------------
        diff_rows = []

        for entry in source_diff_data:

            data = entry["Data"]
            data = data.squeeze()

            df = data.to_dataframe(
                name="BSS_difference"
            ).reset_index()

            for _, row in df.iterrows():

                row_data = {
                    "Variable": entry["Variable"],
                    "Variable code": entry["Variable code"],
                    "Prefix": entry["Prefix"],
                    "Extreme": entry["Extreme"],
                    "Horizon": entry["Horizon"],
                    "Target dates": entry["Target dates"],
                    "Target model": entry["Target model"],
                    "Baseline model": entry["Baseline model"],
                    "BSS_difference": row["BSS_difference"],
                }

                # Add all coordinate columns.
                for coord_name in df.columns:

                    if coord_name == "BSS_difference":
                        continue

                    row_data[coord_name] = row[coord_name]

                diff_rows.append(row_data)

        diff_df = pd.DataFrame(diff_rows)

        # ------------------------------------------------------
        # Write to shared workbook
        #
        # Other source-data sheets are preserved. Only the two
        # sheets belonging to this function are replaced.
        # ------------------------------------------------------
        bss_sheet_name = (
            f"{source_data_sheet_prefix}_BSS"
        )

        diff_sheet_name = (
            f"{source_data_sheet_prefix}_diff"
        )

        if len(bss_sheet_name) > 31:
            raise ValueError(
                f"Source-data sheet name '{bss_sheet_name}' "
                "is longer than Excel's 31-character limit."
            )

        if len(diff_sheet_name) > 31:
            raise ValueError(
                f"Source-data sheet name '{diff_sheet_name}' "
                "is longer than Excel's 31-character limit."
            )

        if os.path.exists(fig_filename):

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:

                bss_df.to_excel(
                    writer,
                    sheet_name=bss_sheet_name,
                    index=False,
                )

                diff_df.to_excel(
                    writer,
                    sheet_name=diff_sheet_name,
                    index=False,
                )

        else:

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="w",
            ) as writer:

                bss_df.to_excel(
                    writer,
                    sheet_name=bss_sheet_name,
                    index=False,
                )

                diff_df.to_excel(
                    writer,
                    sheet_name=diff_sheet_name,
                    index=False,
                )

        print(
            f"Source data saved: {fig_filename}"
        )

    # ----------------------------------------------------------
    # Colorbars
    # ----------------------------------------------------------
    cax1 = fig.add_axes(
        [0.11, -0.02, 0.6, 0.02]
    )

    cax2 = fig.add_axes(
        [0.79, -0.02, 0.19, 0.02]
    )

    cbar1 = fig.colorbar(
        im_metric,
        cax=cax1,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40,
    )

    cbar1.set_label(
        'Brier skill score (BSS)',
        fontsize=20,
        labelpad=10,
    )

    cbar1.ax.tick_params(
        labelsize=20
    )

    cbar2 = fig.colorbar(
        im_diff,
        cax=cax2,
        orientation="horizontal",
        fraction=0.02,
        pad=0.02,
        aspect=40,
    )

    cbar2.set_label(
        "BSS difference",
        fontsize=20,
        labelpad=10,
    )

    cbar2.ax.tick_params(
        labelsize=20
    )

    # ----------------------------------------------------------
    # Save figure
    # ----------------------------------------------------------
    if save_fig:

        prefix_str = "_".join(
            p.rstrip("_")
            for p in prefixes
        )

        outfile = os.path.join(
            EXTREMES_OUT_DIR,
            f"lat_lon_{prefix_str}_bss_{horizon}_{target_dates}.pdf"
        )

        plt.savefig(
            outfile,
            dpi=100,
            bbox_inches='tight',
        )

        print(
            f"Saved: {outfile}"
        )

        plt.savefig(outfile.replace(".pdf", ".eps"), dpi=100, bbox_inches='tight')
        print(f"Saved: {outfile.replace('.pdf', '.eps')}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def subset_latlon(da, bbox):
    """
    Subset a DataArray using (west, east, south, north).

    All longitudes are converted to the [-180, 180) convention before
    subsetting, ensuring monotonic coordinates suitable for interpolation.
    """

    west, east, south, north = bbox

    # Convert dataset longitudes from [0,360) -> [-180,180)
    da = da.assign_coords(
        longitude=((da.longitude + 180) % 360) - 180
    )

    # Ensure monotonic longitude
    da = da.sortby("longitude")

    # Longitude selection
    da = da.sel(longitude=slice(west, east))

    # Latitude selection 
    if da.latitude[0] > da.latitude[-1]:
        da = da.sel(latitude=slice(north, south))
    else:
        da = da.sel(latitude=slice(south, north))

    return da


def interpolate(field, resolution=0.1):

    lon = np.arange(
        field.longitude.min().item(),
        field.longitude.max().item() + 0.5 * resolution,
        resolution,
    )

    if field.latitude[0] > field.latitude[-1]:
        lat = np.arange(
            field.latitude.max().item(),
            field.latitude.min().item() - 0.5 * resolution,
            -resolution,
        )
    else:
        lat = np.arange(
            field.latitude.min().item(),
            field.latitude.max().item() + 0.5 * resolution,
            resolution,
        )

    return field.interp(
        longitude=lon,
        latitude=lat,
        method="linear",
    )


def smooth_field(field, sigma=1.2):
    """
    Gaussian smoothing while preserving NaNs.
    """

    values = field.values

    mask = np.isfinite(values)

    values0 = np.where(mask, values, 0)

    smooth = ndimage.gaussian_filter(values0, sigma=sigma)

    weights = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)

    smooth /= np.maximum(weights, 1e-6)

    smooth[weights < 0.05] = np.nan

    return xr.DataArray(
        smooth,
        coords=field.coords,
        dims=field.dims,
    )

def open_remote_dataset(url):

    r = requests.get(url)
    r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.write(r.content)
    tmp.close()

    return xr.open_dataset(tmp.name)
    


def plot_probability_maps(
    gt_id='era5-pr',
    horizon=19,
    target_date=None,
    issuance_date='20260101',
    team_name='Dynamical_S2SDatabase',
    model_name='ECMWF',
    bbox=None,
    bbox_name=None,
    quintile=0.8,
    y_suptitle=0.89,
    show_fig=True,
    save_fig=True,
    source_data=False,
    source_data_filename="source_data.xlsx",
):
    """
    Plot verifying ERA5 (ground truth) anomalies alongside MicroDuet and
    ECMWF/AIFS-Gaia accumulated precipitation quintile probabilities.

    If source_data=True, the plotted data are appended to
    `source_data_filename` as Excel sheets.

    Saved source data:
        - ERA5 observed anomaly
        - MicroDuet probability
        - selected model probability
        - metadata describing the figure/data
    """

    # ----------------------------------------------------------
    # Process parameters
    # ----------------------------------------------------------

    gt_var = gt_id.split('-')[-1]

    horizon_2_fc_period = {
        '19': '1',
        '26': '2'
    }

    fc_period = horizon_2_fc_period[str(horizon)]

    if target_date is None:
        target_date = (
            datetime.strptime(issuance_date, "%Y%m%d")
            + timedelta(days=int(horizon))
            - timedelta(days=1)
        ).strftime("%Y%m%d")

    if issuance_date is None:
        issuance_date = (
            datetime.strptime(target_date, "%Y%m%d")
            - timedelta(days=int(horizon))
            + timedelta(days=1)
        ).strftime("%Y%m%d")

    cbar_titles = {
        'pr': 'Precipitation',
        'tas': "Temperature",
        'mslp': 'Mean sea level pressure'
    }

    cbar_units = {
        'pr': '(mm)',
        'tas': "(°C)",
        'mslp': '(hPa)'
    }

    var = "__xarray_dataarray_variable__"

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    era5 = load_data(f"era5-{gt_var}")

    gt_ds = load_data(f"aiwq-{gt_var}")
    gt_ds = gt_ds.combine_first(era5)[gt_var]

    BASE_URL = "https://data.ecmwf.int/ai-weatherquest/by_fc_date"
    LOCAL_DIR = os.path.join('data', 'aiwq_forecasts')

    other_url = (
        f"{BASE_URL}/"
        f"{issuance_date}/"
        f"{team_name}/"
        f"{model_name}/"
        f"{gt_var}_{issuance_date}_p{fc_period}_{team_name}_{model_name}.nc"
    )

    microduet_url = (
        f"{BASE_URL}/"
        f"{issuance_date}/"
        f"MicroEnsemble/"
        f"MicroDuet/"
        f"{gt_var}_{issuance_date}_p{fc_period}_MicroEnsemble_MicroDuet.nc"
    )

    other_filename = os.path.join(
        LOCAL_DIR,
        other_url.split("/")[-1]
    )

    microduet_filename = os.path.join(
        LOCAL_DIR,
        microduet_url.split("/")[-1]
    )

    if os.path.isfile(other_filename):
        other_ds = xr.open_dataset(other_filename)
    else:
        other_ds = open_remote_dataset(other_url)

    if os.path.isfile(microduet_filename):
        microduet_ds = xr.open_dataset(microduet_filename)
    else:
        microduet_ds = open_remote_dataset(microduet_url)

    # ----------------------------------------------------------
    # Domain
    # ----------------------------------------------------------

    bbox_dict = {
        "us": (-130, -65, 25, 49),
        "eastern_us": (-102, -74, 28, 44),
        "europe": (-10, 45, 30, 60),
    }

    if bbox is None and bbox_name is None:
        bbox_name = "us"

    if bbox_name is not None:
        if bbox_name not in bbox_dict:
            raise ValueError(
                f"Unknown bbox_name '{bbox_name}'. "
                f"Choose from {list(bbox_dict.keys())}."
            )

        bbox = bbox_dict[bbox_name]

    elif bbox in bbox_dict.values():
        bbox_name = next(
            name
            for name, value in bbox_dict.items()
            if value == bbox
        )

    else:
        bbox_name = "custom"

    # ----------------------------------------------------------
    # Projection
    # ----------------------------------------------------------

    if bbox_name == "europe":
        proj = ccrs.LambertConformal(
            central_longitude=15,
            central_latitude=45,
            standard_parallels=(35, 65),
        )
    else:
        proj = ccrs.LambertConformal(
            central_longitude=-97,
            central_latitude=38,
            standard_parallels=(30, 60),
        )

    # ----------------------------------------------------------
    # Retrieve data
    # ----------------------------------------------------------

    valid_time = pd.Timestamp(
        other_ds.forecast_period_start.values
    )

    # Compute climatology for valid_time
    clim_years = 20

    clim = gt_ds.where(
        (gt_ds.time.dt.month == valid_time.month)
        & (gt_ds.time.dt.day == valid_time.day)
        & (gt_ds.time.dt.year < valid_time.year)
        & (gt_ds.time.dt.year >= valid_time.year - clim_years),
        drop=True
    ).mean(dim="time").load()

    # Ground-truth anomaly
    gt = subset_latlon(
        gt_ds.sel(time=valid_time) - clim,
        bbox,
    )

    # Land mask for precipitation / temperature
    if gt_var in ['tas', 'pr']:
        lsm = xr.open_dataarray(
            'data/lsmask.zarr',
            decode_timedelta=True
        )

        lsm = lsm.reindex_like(microduet_ds)

        microduet_ds = microduet_ds.where(
            lsm >= 0.5,
            None
        )

    # Forecast probabilities
    md = subset_latlon(
        microduet_ds[var].sel(quintile=quintile),
        bbox,
    ) * 100

    other = subset_latlon(
        other_ds[var].sel(quintile=quintile),
        bbox,
    ) * 100

    # Align forecast fields
    md, other = xr.align(md, other)

    land = md.notnull()

    other = other.where(land)

    # Align forecasts with observations
    md, gt = xr.align(md, gt)

    gt = gt.where(land)

    # ----------------------------------------------------------
    # Probability colour map
    # ----------------------------------------------------------

    bounds = [
        0, 3, 6, 9, 12, 15,
        25, 30, 35, 40, 45,
        50, 55, 60, 65, 70,
        75, 80, 85, 90, 95, 100,
    ]

    colors = [
        "#2b2b2b",
        "#46515e",
        "#68737e",
        "#8f98a2",
        "#bdbdbd",
        "#dfdfdf",
        "#63d1b0",
        "#48c8b7",
        "#35c0bf",
        "#78b4ab",
        "#7f9ea9",
        "#5e9fda",
        "#3a87ec",
        "#2d5fe4",
        "#211de3",
        "#4314b8",
        "#6940a7",
        "#7e42c8",
        "#9737d9",
        "#b018d5",
        "#4c003f",
    ]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # ----------------------------------------------------------
    # Ground truth colour map
    # ----------------------------------------------------------

    vmin = float(gt.quantile(0.02))
    vmax = float(gt.quantile(0.98))

    gt_levels = np.linspace(vmin, vmax, 13)
    gt_cmap = plt.get_cmap("RdBu_r")
    gt_norm = mcolors.BoundaryNorm(
        gt_levels,
        gt_cmap.N
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15.5, 6),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    # ----------------------------------------------------------
    # Figure title information
    # ----------------------------------------------------------

    start = mdates.num2date(
        mdates.date2num(
            other_ds.forecast_period_start.values
        )
    )

    end = start + timedelta(days=6)

    if start.month == end.month:
        date_str = (
            f"{start:%b} {start:%d}–{end:%d, %Y}"
        )
    else:
        date_str = (
            f"{start:%d %b %Y} – {end:%d %b %Y}"
        )

    # ----------------------------------------------------------
    # Common map styling
    # ----------------------------------------------------------

    def decorate(ax):

        ax.set_extent(
            bbox,
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(
            cfeature.LAND,
            facecolor="white",
            edgecolor="black",
        )

        ax.add_feature(
            cfeature.OCEAN,
            facecolor="white",
            zorder=0,
        )

        ax.add_feature(
            cfeature.COASTLINE,
            edgecolor="black",
        )

        ax.add_feature(
            cfeature.BORDERS,
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.add_feature(
            cfeature.LAKES,
            edgecolor="black",
            linewidth=0.5,
        )

    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    decorate(axs[0])

    gt_im = axs[0].pcolormesh(
        gt.longitude,
        gt.latitude,
        gt,
        cmap=gt_cmap,
        norm=gt_norm,
        shading="nearest",
        transform=ccrs.PlateCarree(),
    )

    axs[0].set_title(
        "Observed (ERA5)",
        fontsize=18,
        weight="bold",
        pad=16,
    )

    # ----------------------------------------------------------
    # Forecast probabilities
    # ----------------------------------------------------------

    for ax, field, title in zip(
        axs[1:],
        [md, other],
        ["MicroDuet", model_name],
    ):

        decorate(ax)

        im = ax.pcolormesh(
            field.longitude,
            field.latitude,
            field,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(
            f"Debiased {title}"
            if title == "ECMWF"
            else title,
            fontsize=18,
            weight="bold",
            pad=16,
        )

    # ----------------------------------------------------------
    # Colourbars
    # ----------------------------------------------------------

    cbar_height = 0.018
    cbar_y = 0.25

    cbar_gt_ax = fig.add_axes(
        [0.04, cbar_y, 0.25, cbar_height]
    )

    cbar_gt = fig.colorbar(
        gt_im,
        cax=cbar_gt_ax,
        orientation="horizontal",
    )

    cbar_gt.set_label(
        f"{cbar_titles[gt_var]} anomaly "
        f"{cbar_units[gt_var]}: {date_str}",
        fontsize=14,
    )

    cbar_gt.ax.tick_params(
        labelsize=12
    )

    cbar_prob_ax = fig.add_axes(
        [0.39, cbar_y, 0.55, cbar_height]
    )

    cbar_prob = fig.colorbar(
        im,
        cax=cbar_prob_ax,
        orientation="horizontal",
        ticks=bounds,
    )

    quintile_str = (
        'highest'
        if quintile == 1
        else 'lowest'
    )

    cbar_prob.set_label(
        f"Predicted probability of {quintile_str} "
        f"{cbar_titles[gt_var].lower()} quintile: "
        f"{date_str}",
        fontsize=14,
    )

    cbar_prob.ax.tick_params(
        labelsize=12
    )

    # ----------------------------------------------------------
    # Save source data
    # ----------------------------------------------------------

    if source_data:

        # ------------------------------------------------------
        # This is a SEPARATE workbook for probability maps.
        # It is not the workbook used by plot_bias_maps_3x4.
        # ------------------------------------------------------

        source_data_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        # Make sure the directory exists
        os.makedirs(
            os.path.dirname(source_data_filename),
            exist_ok=True
        )

        # Use append mode if this probability-map workbook
        # already exists; otherwise create a new workbook.
        file_exists = os.path.exists(source_data_filename)

        if file_exists:
            writer_kwargs = {
                "path": source_data_filename,
                "engine": "openpyxl",
                "mode": "a",
                "if_sheet_exists": "new",
            }
        else:
            writer_kwargs = {
                "path": source_data_filename,
                "engine": "openpyxl",
                "mode": "w",
            }

        # ------------------------------------------------------
        # Base name identifies this particular forecast.
        # ------------------------------------------------------

        base_name = (
            f"{gt_var}_{issuance_date}_"
            f"p{fc_period}_{bbox_name}"
        )

        def make_sheet_name(base, suffix):
            """
            Generate a valid Excel sheet name.

            Excel limits sheet names to 31 characters.
            """

            name = f"{base}_{suffix}"

            # Remove characters that Excel does not allow
            invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
            for char in invalid_chars:
                name = name.replace(char, '_')

            # Excel sheet names max out at 31 characters.
            return name[:31]

        # ------------------------------------------------------
        # Data to save
        #
        # These are exactly the fields displayed in the figure.
        # ------------------------------------------------------

        source_fields = {
            "observed": gt,
            "microduet": md,
            model_name.lower(): other,
        }

        # ------------------------------------------------------
        # Metadata
        # ------------------------------------------------------

        metadata = {
            "gt_id": gt_id,
            "gt_variable": gt_var,
            "horizon": horizon,
            "forecast_period": fc_period,
            "issuance_date": issuance_date,
            "target_date": target_date,
            "valid_time": str(valid_time),
            "team_name": team_name,
            "model_name": model_name,
            "quintile": quintile,
            "bbox_name": bbox_name,
            "bbox_west": bbox[0],
            "bbox_east": bbox[1],
            "bbox_south": bbox[2],
            "bbox_north": bbox[3],
            "climatology_years": clim_years,
            "source_other_url": other_url,
            "source_microduet_url": microduet_url,
        }

        metadata_df = pd.DataFrame(
            list(metadata.items()),
            columns=["Parameter", "Value"]
        )

        # ------------------------------------------------------
        # Write to the separate probability-map workbook
        # ------------------------------------------------------

        with pd.ExcelWriter(**writer_kwargs) as writer:

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            metadata_sheet = make_sheet_name(
                base_name,
                "metadata"
            )

            metadata_df.to_excel(
                writer,
                sheet_name=metadata_sheet,
                index=False,
            )

            # --------------------------------------------------
            # Spatial fields
            # --------------------------------------------------

            for field_name, data in source_fields.items():

                # Convert DataArray to a self-contained table
                # containing latitude/longitude and values.
                data_df = data.to_dataframe(
                    name="value"
                ).reset_index()

                sheet_name = make_sheet_name(
                    base_name,
                    field_name
                )

                data_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    na_rep="NaN",
                )

        print(
            f"Source data saved to: "
            f"{source_data_filename}"
        )

    # ----------------------------------------------------------
    # Save / Show
    # ----------------------------------------------------------

    filename_bbox = (
        bbox_name
        if bbox_name
        else '_'.join(map(str, bbox))
    )

    filename = os.path.join(
        OUT_DIR,
        f"extreme_weather_{gt_var}_"
        f"{issuance_date}_p{fc_period}_"
        f"{filename_bbox}.pdf"
    )

    if save_fig:

        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches='tight'
        )

        print(
            f"Figure saved: {filename}"
        )

        plt.savefig(filename.replace(".pdf", ".eps"), dpi=300, transparent=True, bbox_inches='tight')

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_probability_maps_aiwq(
    gt_id="era5-pr",
    horizon=19,
    target_date=None,
    issuance_date="20260101",
    team_name="AIFS",
    model_name="AIFShera",
    bbox=None,
    bbox_name=None,
    quintile=0.8,
    y_suptitle=0.89,
    show_fig=True,
    save_fig=True,
):
    """
    Plot verifying ERA5 anomalies alongside MicroDuet and AIFS
    quintile probability forecasts using the AI Weather Quest style.
    """

    #
    # Process parameters
    #

    gt_var = gt_id.split("-")[-1]

    horizon_2_fc_period = {
        "19": "1",
        "26": "2",
    }

    fc_period = horizon_2_fc_period[str(horizon)]

    if target_date is None:
        target_date = (
            datetime.strptime(issuance_date, "%Y%m%d")
            + timedelta(days=int(horizon))
            - timedelta(days=1)
        ).strftime("%Y%m%d")

    if issuance_date is None:
        issuance_date = (
            datetime.strptime(target_date, "%Y%m%d")
            - timedelta(days=int(horizon))
            + timedelta(days=1)
        ).strftime("%Y%m%d")

    cbar_titles = {
        "pr": "Precipitation (mm)",
        "tas": "Temperature (°C)",
        "mslp": "Mean sea level pressure",
    }

    var = "__xarray_dataarray_variable__"

    #
    # Load data
    #

    era5 = load_data(f"era5-{gt_var}")
    gt_ds = load_data(f"aiwq-{gt_var}")
    gt_ds = gt_ds.combine_first(era5)[gt_var]

    BASE_URL = "https://data.ecmwf.int/ai-weatherquest/by_fc_date"
    LOCAL_DIR = os.path.join("data", "aiwq_forecasts")

    aifs_url = (
        f"{BASE_URL}/"
        f"{issuance_date}/"
        f"{team_name}/"
        f"{model_name}/"
        f"{gt_var}_{issuance_date}_p{fc_period}_{team_name}_{model_name}.nc"
    )

    microduet_url = (
        f"{BASE_URL}/"
        f"{issuance_date}/"
        f"MicroEnsemble/"
        f"MicroDuet/"
        f"{gt_var}_{issuance_date}_p{fc_period}_MicroEnsemble_MicroDuet.nc"
    )

    aifs_filename = os.path.join(
        LOCAL_DIR,
        aifs_url.split("/")[-1],
    )

    microduet_filename = os.path.join(
        LOCAL_DIR,
        microduet_url.split("/")[-1],
    )

    if os.path.isfile(aifs_filename):
        aifs_ds = xr.open_dataset(aifs_filename)
    else:
        aifs_ds = open_remote_dataset(aifs_url)

    if os.path.isfile(microduet_filename):
        microduet_ds = xr.open_dataset(microduet_filename)
    else:
        microduet_ds = open_remote_dataset(microduet_url)

    #
    # Domain
    #

    bbox_dict = {
        "us": (-130, -65, 25, 49),
        "eastern_us": (-102, -74, 28, 44),
        "europe": (-10, 45, 30, 60),
    }

    if bbox is None and bbox_name is None:
        bbox_name = "us"

    if bbox_name is not None:
        if bbox_name not in bbox_dict:
            raise ValueError(
                f"Unknown bbox_name '{bbox_name}'. "
                f"Choose from {list(bbox_dict.keys())}."
            )
        bbox = bbox_dict[bbox_name]

    elif bbox in bbox_dict.values():
        bbox_name = next(
            name for name, value in bbox_dict.items()
            if value == bbox
        )
    else:
        bbox_name = "custom"

    #
    # Projection
    #

    if bbox_name == "europe":
        proj = ccrs.LambertConformal(
            central_longitude=15,
            central_latitude=45,
            standard_parallels=(35, 65),
        )
    else:
        proj = ccrs.LambertConformal(
            central_longitude=-97,
            central_latitude=38,
            standard_parallels=(30, 60),
        )

    #
    # Retrieve data
    #

    valid_time = pd.Timestamp(aifs_ds.forecast_period_start.values)

    clim_years = 20

    clim = gt_ds.where(
        (gt_ds.time.dt.month == valid_time.month)
        & (gt_ds.time.dt.day == valid_time.day)
        & (gt_ds.time.dt.year < valid_time.year)
        & (gt_ds.time.dt.year >= valid_time.year - clim_years),
        drop=True,
    ).mean(dim="time").load()

    gt = subset_latlon(
        gt_ds.sel(time=valid_time) - clim,
        bbox,
    )

    if gt_var in ["tas", "pr"]:
        lsm = xr.open_dataarray(
            "data/lsmask.zarr",
            decode_timedelta=True,
        )
        lsm = lsm.reindex_like(microduet_ds)
        microduet_ds = microduet_ds.where(lsm >= 0.5)

    md = (
        subset_latlon(
            microduet_ds[var].sel(quintile=quintile),
            bbox,
        )
        * 100
    )

    aifs = (
        subset_latlon(
            aifs_ds[var].sel(quintile=quintile),
            bbox,
        )
        * 100
    )

    # Match plot_forecast(): convert longitudes to [-180,180]
    # and sort before plotting.
    def prepare_forecast_field(field):
        field = field.assign_coords(
            longitude=convert_long(field.longitude)
        )
        return field.sortby("longitude")

    md = prepare_forecast_field(md)
    aifs = prepare_forecast_field(aifs)

    md, aifs = xr.align(md, aifs)

    land = md.notnull()
    aifs = aifs.where(land)

    md, gt = xr.align(md, gt)
    gt = gt.where(land)

    #
    # AI Weather Quest probability colour map
    #

    ai_wq_cmap, bounds = create_colormap()

    #
    # Ground-truth colour map
    #

    vmin = float(gt.quantile(0.02))
    vmax = float(gt.quantile(0.98))

    gt_levels = np.linspace(vmin, vmax, 13)
    gt_cmap = plt.get_cmap("RdBu_r")
    gt_norm = mcolors.BoundaryNorm(gt_levels, gt_cmap.N)

    #
    # Figure
    #

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15.5, 6),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    #
    # Dates
    #

    start = pd.Timestamp(aifs_ds.forecast_period_start.values)
    end = start + timedelta(days=6)

    date_str = f"{start:%d %b %Y} - {end:%d %b %Y}"

    #
    # AI Weather Quest map styling
    #

    def decorate(ax):

        ax.set_extent(
            bbox,
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(
            cfeature.BORDERS,
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.add_feature(
            cfeature.LAND,
            facecolor="grey",
            edgecolor="black",
        )

        ax.add_feature(
            cfeature.COASTLINE,
            edgecolor="black",
        )

        ax.add_feature(
            cfeature.LAKES,
            edgecolor="black",
            linewidth=0.5,
        )

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.4,
            color="0.7",
            alpha=0.6,
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl.xlabel_style = {
            "size": 9,
        }

        gl.ylabel_style = {
            "size": 9,
        }

    #
    # Ground truth
    #

    decorate(axs[0])

    gt_im = axs[0].pcolormesh(
        gt.longitude,
        gt.latitude,
        gt,
        cmap=gt_cmap,
        norm=gt_norm,
        shading="nearest",
        transform=ccrs.PlateCarree(),
    )

    axs[0].set_title(
        "Observed (ERA5)",
        fontsize=11,
    )

    #
    # Forecast probabilities
    #

    for ax, field, title in zip(
        axs[1:],
        [md, aifs],
        ["MicroDuet", model_name],
    ):

        decorate(ax)

        im = ax.contourf(
            field.longitude,
            field.latitude,
            field,
            cmap=ai_wq_cmap,
            levels=bounds,
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(
            title,
            fontsize=11,
        )

    #
    # Colourbars
    #

    cbar_gt = fig.colorbar(
        gt_im,
        ax=axs[0],
        orientation="horizontal",
        shrink=0.9,
        pad=0.06,
        aspect=50,
    )

    cbar_gt.set_label(
        f"{cbar_titles[gt_var]}, {date_str}",
        fontsize=10,
    )

    cbar_gt.ax.tick_params(
        labelsize=8,
    )

    cbar_prob = fig.colorbar(
        im,
        ax=axs[1:],
        orientation="horizontal",
        pad=0.06,
        ticks=bounds,
        aspect=70,
    )

    quintile_str = (
        "top"
        if quintile == 1
        else "bottom"
    )

    cbar_prob.set_label(
        f"{cbar_titles[gt_var]} {quintile_str} quintile probabilities, {date_str}",
        fontsize=10,
    )

    cbar_prob.ax.tick_params(
        labelsize=8,
    )

    #
    # Save / show
    #

    filename_bbox = (
        bbox_name
        if bbox_name
        else "_".join(map(str, bbox))
    )

    filename = os.path.join(
        OUT_DIR,
        f"extreme_weather_{gt_var}_{issuance_date}_p{fc_period}_{filename_bbox}_aiwq_format.pdf",
    )

    if save_fig:
        plt.savefig(
            filename,
            dpi=300,
            transparent=True,
            bbox_inches="tight",
        )

        print(
            f"Figure saved: {filename}"
        )

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


