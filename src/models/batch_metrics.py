# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: aiwqd
#     language: python
#     name: python3
# ---

# %%
''' 
Evaluate error metrics for model or submodel for a specified set of test dates. 
Output is stored in eval/metrics/MODEL_NAME/submodel_forecasts/SUBMODEL_NAME. 
If no submodel is provided, the selected submodel for a model is evaluated.

Example usage: 
python src/models/batch_metrics.py era5-f1_tas 19 -mn proj_tuned_ecmwfpp -sn proj_tuned_ecmwfpp_on_years3_marginNone -t std_future -m wtd_mse 
python src/models/batch_metrics.py era5-f3_tas 26 -mn ecmwf -sn ecmwfpp-debiasFalse_years20_margin0_days1_leads26-26_lossmse -t std_tune -m wtd_mse
python src/models/batch_metrics.py era5-f1_tas 26 -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t std_test -m lat_lon_mse
src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/batch_metrics.py era5-f1_tas 26 -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t std_test -m wtd_mse 

for dates in std_test 2024; do
for var in tas mslp pr; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
    gt_id="era5-${f}_${var}"
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn ecmwf -sn ecmwfpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_ecmwfpp -sn proj_tuned_ecmwfpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_perpp_ecmwf -sn proj_perpp_ecmwf-yearsall_marginNone_clim20 -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t $dates -m wtd_mse lat_lon_mse

    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn msn -sn msnpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_msnpp -sn proj_tuned_msnpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_perpp_msn -sn proj_perpp_msn-yearsall_marginNone_clim20 -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn pbc_msn -sn pbc_msn-yearsall_marginNone_equal -t $dates -m wtd_mse lat_lon_mse

    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn duet -sn duet -t $dates -m wtd_mse lat_lon_mse
    done
  done
done
done

for dates in std_aifs_forecast; do
for var in tas pr mslp; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
    gt_id="era5-${f}_${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn climatology -sn climatology -t $dates -m wtd_mse lat_lon_mse

    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn ecmwf -sn ecmwfpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_ecmwfpp -sn proj_tuned_ecmwfpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_perpp_ecmwf -sn proj_perpp_ecmwf-yearsall_marginNone_clim20 -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t $dates -m wtd_mse lat_lon_mse

    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn msn -sn msnpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_msnpp -sn proj_tuned_msnpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_perpp_msn -sn proj_perpp_msn-yearsall_marginNone_clim20 -t $dates -m wtd_mse lat_lon_mse
    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn pbc_msn -sn pbc_msn-yearsall_marginNone_equal -t $dates -m wtd_mse lat_lon_mse

    # src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn duet -sn duet -t $dates -m wtd_mse lat_lon_mse
    done
  done
done
done

for dates in std_msn; do
for var in tas pr mslp; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
    gt_id="era5-${f}_${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_ecmwfpp -sn proj_tuned_ecmwfpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn proj_tuned_msnpp -sn proj_tuned_msnpp_on_years3_marginNone -t $dates -m wtd_mse lat_lon_mse
    done
  done
done
done

for var in tas pr mslp; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
      gt_id="era5-${f}_${var}"
      src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn tuned_ecmwfpp -sn tuned_ecmwfpp_on_years3_marginNone -t std_test -m wtd_mse
    done
  done
done
for var in tas pr mslp; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
      gt_id="era5-${f}_${var}"
      src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn climatology -sn climatology -t std_future -m wtd_mse
    done
  done
done
for var in tas pr mslp; do
  for f in "f1" "f2" "f3" "f4"; do
    for horizon in "19" "26"; do
      gt_id="era5-${f}_${var}"
      src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/batch_metrics.py "$gt_id" "$horizon" -mn ecmwfpp -t std_test -m wtd_mse
    done
  done
done

Positional args: 
gt_id: e.g. era5-tas, era5-pr, era5-f1_tas 
horizon: 19 or 26

Named args: 
--target_dates (-t): target dates for batch prediction (e.g., 'std_test', 'std_future') (default: 'std_tune') 
--metrics (-m): Space-separated list of error metrics to compute (e.g., -m wtd_mse lat_lon_mse)
--model_name (-mn): name of model, e.g, pbc_ecmwf (default: None) 
--submodel_name (-sn): name of submodel, e.g., spatiotemporal_mean-1981_2010 (default: None) 
--region (-r): name of region, e.g., 'us', 'central_america' (default: None)
'''

# %%
import os
from utils.notebook import isnotebook
if isnotebook():
    # Change to aiwq working directory
    home_dir = os.path.expanduser("~")
    os.chdir(os.path.join(home_dir, "aiwq"))
    # Autoreload packages that are modified
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
else:
    from argparse import ArgumentParser

# %%
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from filelock import FileLock
from argparse import ArgumentParser
import xarray as xr
from pathlib import Path
from models.utils.data_utils import get_measurement_variable
from models.utils.eval_util import get_target_dates, mean_rmse_to_score, get_task_metrics_dir, get_region_bounding_box, subset_region
from models.utils.models_util import get_task_forecast_dir, get_selected_submodel_name
from utils.data_io import load_data, save_to_zarr
from utils.file_io import make_directories
from utils.logging import printf
from utils.timing import tic, toc

#
# Error and skill measures
#
def get_rmse(pred, gt):
    return np.sqrt(np.square(pred-gt).mean())

def get_mse(pred, gt):
    return np.square(pred-gt).mean()

def get_wtd_mse(pred, gt, weights):
    # Returns weighted average of squared errors over non-na values: 
    # sum_i weights_i (pred_i - gt_i)^2/sum_i weights_i
    vals = np.square(pred-gt)
    not_nas = ~np.isnan(vals)
    return np.average(vals[not_nas], weights = weights[not_nas])

def get_skill(pred, gt, clim):
    return 1 - cosine(pred-clim, gt-clim)

def get_anom(pred, clim):
    return (pred-clim).mean()

def get_error(pred, gt):
    return (pred-gt).mean()


# %%
#
# Specify model parameters
#
if __name__ == "__main__":
    if not isnotebook():
        # If notebook run as a script, parse command-line arguments
        parser = ArgumentParser()
        parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
        parser.add_argument('--target_dates', '-t', default='std_tune')
        # For metrics, 1 or more values expected => creates a list
        parser.add_argument("--metrics", '-m', nargs="+", type=str, default=['wtd_mse'], 
                            help="Space-separated list of error metrics to compute (e.g., 'wtd_mse', 'lat_lon_mse')")    
        parser.add_argument('--region', '-r', default=None,
                    help="Region name (choices: 'us', 'europe', 'east_asia', 'middle_east', 'central_america', 'south_america_nh', 'south_america_sh', 'north_africa', 'southern_africa', 'australia', 'maritime_continent', 'india', 'northern_hemisphere', 'southern_hemisphere')")
        parser.add_argument('--model_name', '-mn', default=None)
        parser.add_argument('--submodel_name', '-sn', default=None)
        args, opt = parser.parse_known_args()
    
        # Assign variables
        gt_id = args.pos_vars[0]  # e.g. era5-tas, era5-pr, era5-f1_tas
        horizon = args.pos_vars[1]  # "19" or "26"
        target_dates = args.target_dates
        metrics = args.metrics
        region = args.region
        model_name = args.model_name
        submodel_name = args.submodel_name  
    else:
        # Otherwise, specify arguments interactively
        gt_id = "era5-f2_pr"
        horizon = "19"
        target_dates = "std_test" 
        metrics = ['lat_lon_mse']
        region = None
        model_name = "pbc_ecmwf" 
        submodel_name = "pbc_ecmwf-yearsall_marginNone_equal"  

    """ 
    Process model parameters
    """
    measurement_variable = get_measurement_variable(gt_id)
    agg_period = 7 
    dataset = gt_id.split('-')[0]  # e.g. era5
    
    # Set input folder (with pred files) and output folder (for metrics)  
    if submodel_name is None:
        submodel_name = get_selected_submodel_name(model_name, gt_id, horizon)
    
    # Set input folder (with pred files) and output folder (for metrics)
    preds_folder = get_task_forecast_dir(
        model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
        target_dates=target_dates)
    output_folder = get_task_metrics_dir(
        model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon,
        target_dates=target_dates)
    
    # Get preds filenames
    printf('Getting prediction file paths and target dates')
    tic()
    # Use set of test dates to determine which preds dfs to load
    target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
    file_names = [f"{gt_id}_{horizon}-{datetime.strftime(target_date,'%Y%m%d')}.nc" for target_date in target_date_objs]
    
    # Get list of sorted preds file paths and target dates
    file_paths = sorted([f"{preds_folder}/{file_name}" for file_name in file_names])
    
    # Extract date from file name as the penultimate list element 
    # after splitting on periods and dashes
    target_date_strs = [file_name.replace('-', '.').split('.')[-2] for file_name in file_names]
    target_date_objs = [datetime.strptime(date_str, '%Y%m%d') for date_str in target_date_strs]
    toc()


    """
    Load and merge ground truth and dropping extraneous columns
    """
    printf("Loading ground truth data")
    # Load ground truth data
    tic()
    if gt_id.endswith("tas") or gt_id.endswith("pr"):
        gt_ds = load_data(gt_id, lsmask=True)
    elif gt_id.endswith("mslp"): 
        gt_ds = load_data(gt_id, lsmask=False)
    else:
        raise ValueError(f"Unknown ground truth id {gt_id}")
    # Load data into memory (in place)
    gt_ds.load()
    toc()
    # Set region's bounding box
    if region is not None:
        bbox = get_region_bounding_box(region=region, lon_type="0,360")
    
    # Restrict to start_dates in target dates and
    # convert xarray object to an in-memory Pandas dataframe
    available_dates = set(pd.to_datetime(gt_ds['time'].values))
    target_date_objs = [d for d in target_date_objs if d in available_dates]

    # If target variable is precipitation, load in quintiles
    if gt_id.endswith("pr"):
        printf("Computing zero quintiles mask")
        tic()
        quintiles = xr.open_dataset("data/era5-quintiles-pr.zarr", engine="zarr").load()
        zero_quintile_masks = (quintiles['pr'] == 0).all(dim="quantile")
        toc()
    
    # Create error dfs; populate start_date column with target_date_strs
    metric_dfs = {}
    for metric in metrics:
        if metric == 'lat_lon_rmse':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates = 0
            continue
        if metric == 'lat_lon_mse':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates_llm = 0
            continue
        # Index by target dates
        metric_dfs[metric] = pd.Series(name=metric, index=target_date_objs, dtype=np.float64)
        metric_dfs[metric].index.name = 'time'
        if metric == 'wtd_mse':
            base_weights = None
        if 'skill' in metric or 'anom' in metric:
            # Load climatology
            printf('Loading climatology and replacing start date with month-day')
            tic()
            clim = get_climatology(gt_id, sync=False)
            clim = clim.set_index(
                [clim.time.dt.month,clim.time.dt.day,'lat','lon']
            ).drop(columns='time').squeeze().sort_index()
            toc()
        if metric == 'lat_lon_skill':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates_lls = 0
        if metric == 'lat_lon_anom':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates_lla = 0
        if metric == 'lat_lon_pred':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates_llp = 0
        if metric == 'lat_lon_error':
            # Keep track of number of dates contributing to error calculation
            # Initialize dataframe later
            num_dates_lle = 0
    
    # Fill the error dfs for given target date    
    for file_path, target_date_obj in zip(file_paths, target_date_objs):
        printf(f'Getting metrics for {target_date_obj}')
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        tic()
        if np.datetime64(target_date_obj) not in gt_ds.time.values:
            printf(f"Warning: {target_date_obj} has no ground truth; skipping")
            continue
        if model_name == 'gt':
            preds = gt_ds.sel(time=np.datetime64(target_date_obj))[measurement_variable]
        else:
            # Check if metrics file is missing
            if os.path.exists(file_path) is False:
                printf(f"Warning: {file_path} does not exist; skipping")
                continue
            # Obtain a lock on the file to deal with multiple process file access
            with FileLock(file_path+"lock"):
                preds = xr.open_dataset(file_path)[measurement_variable].squeeze()
            # Suppress error messages when removing the lock file
            subprocess.call(f"rm {file_path}lock", shell=True, stderr=subprocess.DEVNULL)
     
        if len(preds) == 0:
            printf(f"There are no predictions in {file_path}; skipping")
            continue

        if np.all(np.isnan(preds.values)):
            printf(f"All predictions are NaN in {file_path}; skipping")
            continue
            
        preds = preds.sortby(['latitude', 'longitude']).astype('float64')
        # assert len(preds) == len(gt.loc[target_date_obj]), f"Differing lengths for prediction ({len(preds)}) and ground truth ({len(gt.loc[target_date_obj])})"
    
        printf('-Calculating metrics')
        gt_sel = gt_ds.sel(time=np.datetime64(target_date_obj))[measurement_variable]

        if region is not None:
            gt_sel = subset_region(gt_sel, bbox)
            preds = subset_region(preds, bbox)
            if gt_sel.size == 0:
                raise ValueError(f"Empty region after subsetting: {region}")   
            if preds.shape != gt_sel.shape:
                raise ValueError(f"Shape mismatch: preds {preds.shape}, gt {gt_sel.shape}")

        # Transpose preds to match ground truth dimension order, if necessary
        if preds.dims != gt_sel.dims:
            preds = preds.transpose('latitude', 'longitude')

        if 'wtd_mse' in metrics:
            if base_weights is None: 
                lats = gt_sel.latitude.values
                base_weights = np.cos(np.deg2rad(np.abs(lats)))
                base_weights = np.broadcast_to(base_weights[:, None], gt_sel.shape)
            if gt_id.endswith("pr"): 
                # For precipitation, weights change for every target date
                zero_quintile_mask = zero_quintile_masks.sel(time=np.datetime64(target_date_obj))
                if region is not None:
                    zero_quintile_mask = subset_region(zero_quintile_mask, bbox)
                zero_quintile_mask = zero_quintile_mask.values
                weights = base_weights * (~zero_quintile_mask)
            else: 
                # For other variables, weights stay constant
                weights = base_weights
            metric_dfs['wtd_mse'].loc[target_date_obj] = get_wtd_mse(preds.values, gt_sel.values, weights)
        if 'mse' in metrics:
            metric_dfs['mse'].loc[target_date_obj] = get_mse(preds.values, gt_sel.values)
        if 'rmse' in metrics or 'score' in metrics:
            rmse = get_rmse(preds.values, gt_sel.values)
            if 'rmse' in metrics:
                metric_dfs['rmse'].loc[target_date_obj] = rmse
            if 'score' in metrics:
                metric_dfs['score'].loc[target_date_obj] = mean_rmse_to_score(rmse)
        if 'anom' in metrics:
            month_day = (target_date_obj.month, target_date_obj.day)
            if month_day == (2,29):
                printf('--Using Feb. 28 climatology for Feb. 29')
                month_day = (2,28)
            anom = get_anom(preds.values, clim.loc[month_day].values)
            metric_dfs['anom'].loc[target_date_obj] = anom
        if 'error' in metrics:
            error = get_error(preds.values, gt_sel.values)
            metric_dfs['error'].loc[target_date_obj] = error
        if 'skill' in metrics:
            month_day = (target_date_obj.month, target_date_obj.day)
            if month_day == (2,29):
                printf('--Using Feb. 28 climatology for Feb. 29')
                month_day = (2,28)
            metric_dfs['skill'].loc[target_date_obj] = get_skill(
                preds.values, gt_sel.values, clim.loc[month_day].values)
        if 'lat_lon_rmse' in metrics:
            sqd_error = np.square(preds - gt_sel)
            if num_dates == 0:
                metric_dfs['lat_lon_rmse'] = sqd_error
                metric_dfs['lat_lon_rmse'].name = 'lat_lon_rmse'
            else:
                metric_dfs['lat_lon_rmse'] += sqd_error
            num_dates += 1
        if 'lat_lon_mse' in metrics:
            sqd_error = np.square(preds - gt_sel)
            if num_dates_llm == 0:
                metric_dfs['lat_lon_mse'] = sqd_error
                metric_dfs['lat_lon_mse'].name = 'lat_lon_mse'
            else:
                metric_dfs['lat_lon_mse'] += sqd_error
            num_dates_llm += 1
        if 'lat_lon_error' in metrics:
            error = preds - gt_sel
            if num_dates_lle == 0:
                metric_dfs['lat_lon_error'] = error
                metric_dfs['lat_lon_error'].name = 'lat_lon_error'
            else:
                metric_dfs['lat_lon_error'] += error
            num_dates_lle += 1
        if 'lat_lon_skill' in metrics:
            month_day = (target_date_obj.month, target_date_obj.day)
            if month_day == (2,29):
                printf('--Using Feb. 28 climatology for Feb. 29')
                month_day = (2,28)
            if num_dates_lls ==0:
                metric_dfs['lat_lon_skill'] = pd.DataFrame(index=preds.index, columns=['lat_lon_skill'])
                lat_lon_skill_u, lat_lon_skill_v = pd.DataFrame(index=preds.index), pd.DataFrame(index=preds.index) 
            lat_lon_skill_u[target_date_str] = (preds - clim.loc[month_day]).values.flatten()
            lat_lon_skill_v[target_date_str] = (gt_sel - clim.loc[month_day]).values.flatten()
            num_dates_lls += 1    
        if 'lat_lon_anom' in metrics:
            month_day = (target_date_obj.month, target_date_obj.day)
            if month_day == (2,29):
                printf('--Using Feb. 28 climatology for Feb. 29')
                month_day = (2,28)
            anom = preds - clim.loc[month_day]
            if num_dates_lla == 0:
                metric_dfs['lat_lon_anom'] = anom
                metric_dfs['lat_lon_anom'].name = 'lat_lon_anom'
            else:
                metric_dfs['lat_lon_anom'] += anom
            num_dates_lla += 1    
        if 'lat_lon_pred' in metrics:
            if num_dates_llp == 0:
                metric_dfs['lat_lon_pred'] = preds
                metric_dfs['lat_lon_pred'].name = 'lat_lon_pred'
            else:
                metric_dfs['lat_lon_pred'] += preds
            num_dates_llp += 1    
        toc()
    
    if 'lat_lon_mse' in metric_dfs:
        # Replace error sum with MSE
        metric_dfs['lat_lon_mse'] /= num_dates_llm

    if 'lat_lon_pred' in metric_dfs:
        # Replace preds sum with mean preds
        metric_dfs['lat_lon_pred'] /= num_dates_llp
        metric_dfs['lat_lon_pred'] = metric_dfs['lat_lon_pred']
    
    if 'lat_lon_anom' in metric_dfs:
        # Replace preds sum with mean preds
        metric_dfs['lat_lon_anom'] /= num_dates_lla
        metric_dfs['lat_lon_anom'] = metric_dfs['lat_lon_anom']   
    
    if 'lat_lon_skill' in metric_dfs:
        # If no predictions were made, keep NAs; otherwise calculate lat_lon_skill
        if num_dates_lls > 0:
            # Calculate skill between u and v vectors
            l = [(1 - cosine(lat_lon_skill_u.loc[i], lat_lon_skill_v.loc[i])) for i in lat_lon_skill_u.index]  
            metric_dfs['lat_lon_skill']['lat_lon_skill'] = l
            metric_dfs['lat_lon_skill'] = metric_dfs['lat_lon_skill'].squeeze()    
    
    if 'lat_lon_rmse' in metric_dfs:
        # Replace error sum with RMSE
        metric_dfs['lat_lon_rmse'] /= num_dates
        metric_dfs['lat_lon_rmse'] = np.sqrt(metric_dfs['lat_lon_rmse'])
    
    if 'lat_lon_error' in metric_dfs:
        # Replace error sum with RMSE
        metric_dfs['lat_lon_error'] /= num_dates_lle  
    
    # Set error columns to float and print diagnostics
    for metric, df in metric_dfs.items():
        if metric.startswith('lat_lon_'):
            continue
        printf(f'\n\n{metric}')
        printf(', '.join([f'{statistic}:{np.round(value, 3)}'
                          for statistic, value in df.describe()[1:].items()]))
        if metric == 'rmse':
            printf(f'- bonus score: {mean_rmse_to_score(df.mean())}')
    printf('')
    
    # Create output directory if it doesn't exist
    make_directories(output_folder)
    
    # Save error dfs
    for metric, df in metric_dfs.items():
        region_suffix = f"_{region}" if region is not None else ""
        metric_file_path = f'{output_folder}/{metric}{region_suffix}-{gt_id}_{horizon}-{target_dates}.zarr'
        if metric.startswith('lat_lon_'):
            if df.isnull().all():
                printf(f'{metric} dataset is empty; not saving')
                continue
            printf(f"Saving to {metric_file_path}")
            tic()
            save_to_zarr(df, Path(metric_file_path))
            toc()
        else:
            if df.isna().all():
                printf(f'{metric} dataframe is empty; not saving')
                continue
            printf(f"Saving to {metric_file_path}")
            tic()
            ds = df.to_frame().to_xarray()
            save_to_zarr(ds, Path(metric_file_path))
            toc()
