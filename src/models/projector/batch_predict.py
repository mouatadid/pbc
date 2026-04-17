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
Projector: For a given model and submodel, project the predicted f1-4 probabilities onto the valid CDF space.

Example usages: 
  python src/models/projector/batch_predict.py era5-tas 19 -mn perpp_msn -t std_future
  python src/models/projector/batch_predict.py era5-tas 19 -mn tuned_msnpp -t std_future
  python src/models/projector/batch_predict.py era5-tas 19 -mn perpp_ecmwf -sn perpp_ecmwf-yearsall_marginNone_clim20 -t std_future
  python src/models/projector/batch_predict.py era5-mslp 19 -mn perpp_debias -sn perpp_debias-yearsall_marginNone_clim20 -t std_future
  src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py era5-tas 19 -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t std_test
  for var in mslp; do
    for horizon in 19 26; do
      echo "Running batch prediction for variable: $var, horizon: $horizon"
      src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn pbc_ecmwf -sn pbc_ecmwf-yearsall_marginNone_equal -t std_test
    done
  done 
  # (include -o to overwrite existing predictions)
  for dates in std_test; do
    for model in tuned_ecmwfpp; do
      for var in tas pr mslp; do
        for horizon in 19 26; do
          echo "Running batch prediction for variable: $var, horizon: $horizon, model: $model, dates: $dates"
          src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn $model -sn ${model}_on_years3_marginNone -t $dates -o
        done
      done
    done
  done
  
  for var in pr tas mslp; do
    for horizon in 19 26; do
      echo "Running batch prediction for variable: $var, horizon: $horizon"
      src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn perpp_ecmwf -sn perpp_ecmwf-yearsall_marginNone_clim20 -t std_test
    done
  done 

  for var in pr tas mslp; do
    for horizon in 19 26; do
      echo "Running batch prediction for variable: $var, horizon: $horizon"
      src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn perpp_debias -sn perpp_debias-yearsall_marginNone_clim20 -t std_test
    done
  done 

  for var in tas pr mslp; do
    for horizon in 19 26; do
      echo "Running batch prediction for variable: $var, horizon: $horizon"
      src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn tuned_ecmwfpp -sn tuned_ecmwfpp_on_years3_marginNone -t std_future -o
    done
  done 
  for var in pr tas mslp; do
    for horizon in 19 26; do
      echo "Running batch prediction for variable: $var, horizon: $horizon"
      src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/projector/batch_predict.py "era5-$var" "$horizon" -mn perpp_ecmwf -sn perpp_ecmwf-yearsall_marginNone_clim20 -t std_future -o
    done
  done 

Positional args:
  deterministic gt_id: e.g. era5-tas, era5-pr, era5-mslp
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction
  --model_name (-mn): name of model to tune
  --submodel_name (-sn): (optional) name of submodel to project; if None, name model's selected submodel name will be used (default: None)
'''

# %%
# Ensure notebook is being run from base repository directory
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
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
from sklearn.isotonic import isotonic_regression
from models.utils.eval_util import get_target_dates
from utils.file_io import make_directories
from utils.data_io import save_to_netcdf
from utils.logging import printf
from utils.timing import tic, toc
from models.utils.models_util import get_selected_submodel_name

# %%
#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # deterministic gt_id and horizon 
    parser.add_argument('--model_name', '-mn', default="perpp_ecmwf")
    parser.add_argument('--submodel_name', '-sn', default=None)                                                                                 
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                        help="overwrite existing prediction files")
    args = parser.parse_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0]                                                                
    horizon = args.pos_vars[1]
    model_name = args.model_name
    submodel_name = args.submodel_name
    target_dates = args.target_dates
    overwrite = args.overwrite
else:
    # Otherwise, specify arguments interactively
    gt_id = "era5-pr" 
    horizon = "19"
    model_name = "perpp_ecmwf" 
    submodel_name = "perpp_ecmwf-yearsall_marginNone_clim20"
    target_dates = "std_test" ###"20160101"
    overwrite = False

if submodel_name is None:
    submodel_name = get_selected_submodel_name(model_name, gt_id, horizon)
    printf(f"Selected submodel name: {submodel_name}")

# Name of the projected model and submodel
output_model_name = f'proj_{model_name}'
output_submodel_name = f'proj_{submodel_name}'

# Create a projected model src folder with attributes.py
# if one does not already exist
dst_dir = os.path.join('src', 'models', output_model_name)
proj_dir = os.path.join('src', 'models', 'projector')
if not os.path.exists(dst_dir):
    tic()
    printf(f'\nCreating {dst_dir}')
    make_directories(dst_dir)
    # Copy attributes file from projector folder
    shutil.copy(os.path.join(proj_dir, "attributes.py"), os.path.join(dst_dir, "attributes.py"))
    # Add base model_name to the attributes file
    filename = os.path.join(dst_dir, "attributes.py")
    with open(filename, "r") as f:
        newText=f.read().replace('BASE_MODEL_NAME = ""', f'BASE_MODEL_NAME = "{model_name}"')
    with open(filename, "w") as f:
        f.write(newText)
    toc()

# %%
# Get prediction and projected prediction folders
preds_folder = os.path.join('models', model_name, 'submodel_forecasts', submodel_name)
output_folder = os.path.join('models', output_model_name, 'submodel_forecasts', output_submodel_name)

# Get target dates
target_date_objs = get_target_dates(target_dates, horizon)
target_date_strs = sorted([date_obj.strftime('%Y%m%d') for date_obj in target_date_objs])

# %%
#
# Project the original probabilities onto the valid CDF space
#
dataset = gt_id.split('-')[0]
base_target_variable = f'{gt_id.split("-")[1]}'
for target_date in target_date_strs: 
    f_preds = []
    pred_datasets = []

    printf(f"Loading predicted probabilities per quintile for target date {target_date}")
    tic()
    for q in range(1, 5): 
        # Process parameters
        target_variable = f'f{q}_{base_target_variable}'
        prob_id = f'{dataset}-{target_variable}_{horizon}'

        # Check if projected probabilities already exist for this target date
        output_path = os.path.join(output_folder, prob_id, f'{prob_id}-{target_date}.nc')
        if not overwrite and os.path.exists(output_path):
            printf(f"Projected file {output_path} already exists. Skipping...")
            continue
        
        # Load the predicted probabilities for each quintile
        preds_file = os.path.join(preds_folder, prob_id, f'{prob_id}-{target_date}.nc')
        if not os.path.exists(preds_file):
            printf(f"Warning: Predictions file {preds_file} does not exist. Skipping...")
            continue
        pred_ds = xr.open_dataset(preds_file)
        if np.all(np.isnan(pred_ds[target_variable]).values):
            printf(f"Warning: All predictions are NaN in {preds_file}. Skipping...")
            continue
        pred_ds = pred_ds.transpose('time', 'latitude', 'longitude')
        f_preds.append(pred_ds[target_variable].squeeze(dim='time').values)
        pred_datasets.append(pred_ds)
    toc()

    # If not all quintile predictions were found, skip to the next date
    if len(f_preds) < 4:
        printf(f"Only {len(f_preds)}/4 quintile files found for target date {target_date}. Skipping to next date.")
        continue

    printf(f"Projecting into [0, 1]^4")
    tic()
    # Stack the predictions for each quintile; shape: (4, latitude, longitude);
    # will overwrite those predictions with their projections
    f_projs = np.clip(np.stack(f_preds, axis=0), 0, 1)
    toc()

    # Loop through each grid point and apply isotonic regression to get projected probabilities
    printf(f"Calculating projected probabilities for target date {target_date}")
    tic()
    for i in range(f_projs.shape[1]):
        for j in range(f_projs.shape[2]):
            # Get the predicted probabilities for the current grid point
            preds = f_projs[:, i, j]
            # Preserve any null predictions
            if np.any(np.isnan(preds)): 
                continue
            if np.any(preds[:-1] > preds[1:]): 
                # Project onto space of valid CDFs if predictions are not monotonically increasing
                f_projs[:, i, j] = isotonic_regression(preds, y_min=0, y_max=1, increasing=True)
    toc()

    # Save projected probabilities for each quintile separately
    printf(f"Saving projected probabilities for target date {target_date}")
    tic()
    for q_idx, q in enumerate(range(1, 5)):
        # Process parameters for this quintile
        target_variable = f'f{q}_{base_target_variable}'
        prob_id = f'{dataset}-{target_variable}_{horizon}'
        
        # Create DataArray for this quintile's projections
        proj_da = xr.DataArray(
            f_projs[q_idx], 
            dims=('latitude', 'longitude'), 
            coords={
                'latitude': pred_datasets[q_idx]['latitude'].values,
                'longitude': pred_datasets[q_idx]['longitude'].values
            }, 
            name=target_variable
        )
        
        # Add date to the DataArray
        date_val = pred_datasets[q_idx]['time'].values[0]
        proj_ds = proj_da.expand_dims(time=[date_val]).to_dataset()
        
        # Save as netCDF
        output_path = os.path.join(output_folder, prob_id, f'{prob_id}-{target_date}.nc')
        printf(f"Saving {target_variable} projections to {output_path}")
        save_to_netcdf(proj_ds, Path(output_path))
    toc()
