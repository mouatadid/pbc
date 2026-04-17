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
Evaluate RPSS for model for a specified set of test dates.

Example usage: 
python src/models/rpss.py era5-tas 19 -mn proj_perpp_ecmwf -t std_future
src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/rpss.py era5-tas 19 -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t std_test

for dates in std_future ; do
for var in tas mslp pr; do
  for horizon in "19" "26"; do
    gt_id="era5-${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn ecmwf -sn ecmwfpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn proj_tuned_ecmwfpp -sn proj_tuned_ecmwfpp_on_years3_marginNone -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn proj_perpp_ecmwf -sn proj_perpp_ecmwf-yearsall_marginNone_clim20 -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t $dates

    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn msn -sn msnpp-debiasFalse_years20_margin0_days1_leads${horizon}-${horizon}_lossmse -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn proj_tuned_msnpp -sn proj_tuned_msnpp_on_years3_marginNone -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn proj_perpp_msn -sn proj_perpp_msn-yearsall_marginNone_clim20 -t $dates
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn pbc_msn -sn pbc_msn-yearsall_marginNone_equal -t $dates
    
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn duet -sn duet -t $dates
  done
done
done

for var in mslp pr tas; do
  for horizon in "19" "26"; do
    gt_id="era5-${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn tuned_ecmwfpp -sn tuned_ecmwfpp_on_years3_marginNone -t std_test
  done
done
for var in mslp pr tas; do
  for horizon in "19" "26"; do
    gt_id="era5-${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn climatology -sn climatology -t std_future
  done
done
for var in mslp pr tas; do
  for horizon in "19" "26"; do
    gt_id="era5-${var}"
    src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/rpss.py "$gt_id" "$horizon" -mn ecmwfpp -t std_test
  done
done

Positional args: 
deterministic gt_id: e.g. era5-tas, era5-pr, era5-mslp
horizon: 19 or 26

Named args: 
--target_dates (-t): target dates for batch prediction, e.g., std_tune, std_test (default: 'std_test') 
--model_name (-mn): name of model, e.g., pbc_ecmwf (default: None) 
--submodel_name (-sn): name of submodel, e.g., pbc_ecmwf-yearsall_marginNone_equal (default: None) 
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

import numpy as np
import pandas as pd
from datetime import datetime
import xarray as xr
from pathlib import Path
from utils.data_io import save_to_zarr
from models.utils.models_util import get_selected_submodel_name

if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # deterministic gt_id, horizon
    parser.add_argument('--target_dates', '-t', default='std_test')
    parser.add_argument('--model_name', '-mn', default=None)
    parser.add_argument('--submodel_name', '-sn', default=None)
    parser.add_argument('--region', '-r', default=None,
                    help="Region name (choices: 'us', 'europe', 'east_asia', 'middle_east', 'central_america', 'south_america_nh', 'south_america_sh', 'north_africa', 'southern_africa', 'australia', 'maritime_continent', 'india', 'northern_hemisphere', 'southern_hemisphere')")
    args, opt = parser.parse_known_args()

    # Assign variables
    gt_id = args.pos_vars[0]  # e.g. era5-tas, era5-pr, era5-mslp
    horizon = args.pos_vars[1]  # e.g. "19" or "26"
    target_dates = args.target_dates
    model_name = args.model_name
    submodel_name = args.submodel_name
    region = args.region
else:
    # Otherwise, specify arguments interactively
    gt_id = "era5-tas"
    horizon = "19"
    target_dates = "std_test" 
    model_name = 'proj_perpp_ecmwf'
    submodel_name = 'proj_perpp_ecmwf-yearsall_marginNone_clim20'
    region = None

if submodel_name is None:
    submodel_name = get_selected_submodel_name(model_name, gt_id, horizon)
task = f'{gt_id}_{horizon}'
region_suffix = f"_{region}" if region is not None else ""


# %%
def get_prob_ds(gt_id):
    '''Get the names of the quintile datasets based on the deterministic gt_id.'''
    dataset = gt_id.split('-')[0]
    target_variable = gt_id.split('-')[1]
    return [f"{dataset}-f{q}_{target_variable}_{horizon}" for q in range(1, 5)]

def get_rps(model_name, submodel_name):
    '''Calculate the RPS over time and quintile for the specific model and submodel.'''
    all_mse = [] # List of DataArrays containing weighted MSEs for each quintile 
    prob_ds = get_prob_ds(gt_id)

    # Get all metric files for each quintile
    for q in range(1, 5):
        mse_file = f"eval/metrics/{model_name}/submodel_forecasts/{submodel_name}/{prob_ds[q-1]}/wtd_mse{region_suffix}-{prob_ds[q-1]}-{target_dates}.zarr"
        if os.path.exists(mse_file):
            mse_ds = xr.open_zarr(mse_file)
            all_mse.append(mse_ds['wtd_mse'])
        else:
            raise FileNotFoundError(f"Metrics file {mse_file} does not exist.")

    # Combine along a new 'quintile' dimension and compute RPS per timestep
    mse_stack = xr.concat(all_mse, dim='quintile')
    rps_series = mse_stack.sum(dim='quintile', skipna=False)
    rps_series = rps_series.dropna(dim='time')

    # Return full time series of RPS values
    return rps_series 


# %%
# Get model RPS
model_rps_series = get_rps(model_name, submodel_name).load()
# Compute model RPSS (relative to climatology)
if model_name == 'climatology':
    # By definition, RPSS for climatology is 0.0
    rpss_value = 0.0
else:
    # Get Climatology RPS
    clim_model_name = 'climatology'
    clim_submodel_name = 'climatology'
    clim_rps_path = f"eval/metrics/{clim_model_name}/submodel_forecasts/{clim_submodel_name}/{task}/rps{region_suffix}-{task}-{target_dates}.zarr"

    if os.path.exists(clim_rps_path):
        print(f"Loading climatology RPS series from {clim_rps_path}")
        clim_rps_series = xr.open_zarr(clim_rps_path)['rps'].load()
    else:
        print(f"Climatology RPS file not found; calculating RPS for climatology.")
        clim_rps_series = get_rps(clim_model_name, clim_submodel_name)
        clim_rps_ds = xr.Dataset({
            'rps': clim_rps_series,
            'rpss': xr.DataArray(0.0, dims=())  # By definition, RPSS for climatology is 0.0
        })
        save_to_zarr(clim_rps_ds, Path(clim_rps_path))
        print(f"Saved climatology RPS series to {clim_rps_path}")

    # Exclude dates not present in both models
    model_targets = set(model_rps_series.time.values)
    clim_targets = set(clim_rps_series.time.values)
    common_targets = list(model_targets.intersection(clim_targets))

    model_rps_mean = model_rps_series.sel(time=common_targets).mean(dim='time').compute().item()
    clim_rps_mean = clim_rps_series.sel(time=common_targets).mean(dim='time').compute().item()

    # Calculate RPSS
    rpss_value = 1 - model_rps_mean / clim_rps_mean

# Save model RPS and RPSS
output_path = f"eval/metrics/{model_name}/submodel_forecasts/{submodel_name}/{task}/rps{region_suffix}-{task}-{target_dates}.zarr"


output_ds = xr.Dataset({
    'rps': model_rps_series,
    'rpss': xr.DataArray(rpss_value, dims=())
})
save_to_zarr(output_ds, Path(output_path))
print(f"Saved {model_name} RPS{region_suffix} and RPSS{region_suffix} to {output_path}")
print(f"RPSS{region_suffix} for {model_name}: {rpss_value:.4f}")
