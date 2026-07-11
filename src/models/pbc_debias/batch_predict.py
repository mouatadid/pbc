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
""" 
Probabilistic Bias Correction (PBC) for Debiased ECMWF dynamical model

For each gridpoint and target date, regress onto the specified ensemble members' forecasts 

Example usages:
  python src/models/pbc_debias/batch_predict.py era5-f1_pr 19 -t std_test -y all -m None -e True
  for dates in std_future; do
  for var in mslp; do
    for f in {1..4}; do
      for horizon in 19 26; do
        python src/models/pbc_debias/batch_predict.py era5-f${f}_${var} ${horizon} -t ${dates} -y all -m None -e True
      done
    done
  done
  done

Positional args:
  gt_id: era5-f1_tas, era5-f2_pr, era5-f3_mslp, era5-F10_tas, era5-F5_pr, era5-F95_mslp, etc.
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction 
  --train_years (-y): number of years to use in training ("all" or integer)
  --margin_in_days (-m): number of month-day combinations on either side of the target combination to include when training
    Set to 0 to include only target month-day combo
    Set to "None" to include entire year
"""

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

# Imports 
import numpy as np
import pandas as pd
import xarray as xr
import sys
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from models.utils.general_util import printf
from models.utils.eval_util import get_target_dates
from models.utils.data_utils import get_measurement_variable
from models.utils.experiments_util import get_start_delta
from models.utils.models_util import get_selected_submodel_name
from utils.timing import tic, toc
from utils.data_io import load_data, save_to_netcdf
from models.pbc_debias.attributes import get_submodel_name

#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--train_years', '-y', default="all",
                        help='number of years to use in debiasing ("all" or integer)')
    parser.add_argument('--margin_in_days', '-m', default="None",
                        help="number of month-day combinations on either side of the target combination "
                             "to include when training; set to 0 include only target month-day combo; "
                             "set to None to include entire year")
    parser.add_argument('--equal', '-e', default="True",
                        choices=['True', 'False'],
                        help="Use equal weights?")
    parser.add_argument('--simplex', '-s', default="False",
                        choices=['True', 'False'],
                        help="Constrain weights to simplex?")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                        help="overwrite existing prediction files")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0]                                                                          
    horizon = args.pos_vars[1]                                                                                     
    target_dates = args.target_dates
    train_years = args.train_years
    if train_years != "all":
        train_years = int(train_years)
    if args.margin_in_days == "None":
        margin_in_days = None
    else:
        margin_in_days = int(args.margin_in_days)
    equal = args.equal
    if equal == "False":
        equal = False
    elif equal == "True":
        equal = True
    else:
        raise ValueError(f"unrecognized value {equal} for equal") 
    simplex = args.simplex
    if simplex == "False":
        simplex = False
    elif simplex == "True":
        simplex = True
    else:
        raise ValueError(f"unrecognized value {simplex} for simplex")  
    overwrite = args.overwrite
else:
    # Otherwise, specify arguments interactively 
    gt_id = "era5-f1_tas"
    horizon = "19"
    target_dates = "20170102, 20170106, 20170109"
    train_years = "all"
    margin_in_days = None
    equal = True
    simplex = False
    overwrite = False

if margin_in_days is not None:
    raise ValueError("margin_in_days is not currently supported in this model. Set to None.")
if train_years != "all":
    raise ValueError("train_years is not currently supported in this model. Set to 'all'.")

# Get forecasting task
forecast = "debias"
task = f"{gt_id}_{horizon}"
measurement_variable = get_measurement_variable(gt_id)
model_name = f'pbc_{forecast}'

# Get list of ensemble member model names
if gt_id.startswith("era5-f"):
    pp_name = f'proj_tuned_ecmwfpp'
    perpp_name = 'proj_perpp_' + forecast
elif gt_id.startswith("era5-F"):
    pp_name = f'tuned_ecmwfpp'
    perpp_name = 'perpp_' + forecast
else:
    raise ValueError(f"Unsupported gt_id {gt_id}")
ensemble_members = [pp_name, perpp_name]

# Get selected submodel names for each ensemble member
submodel_names = []
printf("Ensemble members:")
for model in ensemble_members:
    sn = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon, 
                                   target_dates=target_dates)
    printf(f"\t{model}: {sn}")
    submodel_names.append(sn)

# Specify regression parameters
gt_col = measurement_variable
if equal:
    x_cols = ensemble_members
elif simplex:
    pp_minus_perpp = f'{pp_name}_minus_{perpp_name}'
    x_cols = [pp_minus_perpp]
else:
    x_cols = ensemble_members + ['ones']
group_by_cols = ['latitude', 'longitude']

# Get list of target date objects and corresponding strings
target_date_objs = get_target_dates(date_str=target_dates,horizon=horizon)

# For a given target date, the last observable training date is target date - gt_delta
# as gt_delta is the gap between the start of the target date and the start of the
# last ground truth period that's fully observable at the time of forecast issuance
gt_delta = timedelta(days=get_start_delta(horizon, gt_id))

# Record model and submodel name
submodel_name = get_submodel_name(train_years=train_years, margin_in_days=margin_in_days,
                                  equal=equal, simplex=simplex)
printf(f"Submodel name: {submodel_name}")

# %%
if equal:
    # Average predictions of each ensemble member

    # Predictions directory
    preds_dir = os.path.join('models', model_name, 'submodel_forecasts',
                             submodel_name, task)
    for target_date_obj in target_date_objs:
        target_date_str = datetime.strftime(target_date_obj, "%Y%m%d")
        preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
        if not overwrite and os.path.exists(preds_f):
            printf(f"\nprior forecast exists for target {target_date_str}")
            continue
        else: 
            printf(f"\nProcessing target date {target_date_str}...")
            file_template = os.path.join("models", "{}", "submodel_forecasts", "{}", task, f"{task}-{target_date_str}.nc")
            members_exist = True
            for i, model in enumerate(ensemble_members):
                model_filename = file_template.format(model, submodel_names[i])
                if not os.path.exists(model_filename):
                    printf(f'{model} is missing forecasts for {target_date_str}; skipping')
                    members_exist = False
                    break

                # Sum predictions from each model
                if i == 0:
                    preds = xr.open_dataset(model_filename)
                else:
                    preds += xr.open_dataset(model_filename)
            if not members_exist:
                continue

            # Normalize by number of models
            preds /= len(ensemble_members)
            
            # Save prediction 
            tic()
            preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
            printf(f"Saving to {preds_f}")
            save_to_netcdf(preds, Path(preds_f))
            toc()

    # Exit with success status
    sys.exit(0)

# %%
#
# Load ground truth data
#
printf("Loading ground truth data")
tic()
if gt_id.endswith("tas") or gt_id.endswith("pr"):
    gt_ds = load_data(gt_id, lsmask=True)
elif gt_id.endswith("mslp"): 
    gt_ds = load_data(gt_id, lsmask=False)
else:
    raise ValueError(f"Unknown ground truth id {gt_id}")
# Explicitly load gt_ds into memory
gt_ds.load()
toc()

# Store all lat/lon coordinates for reindexing later
all_lats = gt_ds.latitude.values.copy()
all_lons = gt_ds.longitude.values.copy()

# Remove NA latitudes
gt_ds = gt_ds.dropna(dim="latitude", how="all")

# %%
#
# Find dates with forecasts from all models
#
print("Finding dates with forecasts from all models")
tic()
available_dates = set(gt_ds.time.dt.strftime('%Y%m%d').values)
for i, model in enumerate(ensemble_members): 
    sn = submodel_names[i]
    model_dir = os.path.join('models', model, 'submodel_forecasts', sn, task)
    model_dates = set()
    if os.path.exists(model_dir): 
        files = os.listdir(model_dir)
        for file in files: 
            date_str = file.split('-')[2].split('.')[0]  # Extract date from filename
            model_dates.add(date_str)

        available_dates = available_dates.intersection(model_dates)

available_dates = sorted(list(available_dates))
toc()

# %%
#
# Load in model forecasts
#
lld_data = gt_ds
for i, model in enumerate(ensemble_members): 
    print(f"Loading forecasts for {model}")
    tic()
    sn = submodel_names[i]
    filenames = [os.path.join('models', model, 'submodel_forecasts', sn, task, f'{task}-{target_date_str}.nc') for target_date_str in available_dates]
    forecast_ds = xr.open_mfdataset(filenames)
    forecast_ds.load() # Load forecasts into memory
    forecast_ds = forecast_ds.rename({measurement_variable: model})
    forecast_ds = forecast_ds.dropna(dim="latitude", how="all")
    toc()
    print(f"Merging {model} forecast into lld_data")
    tic()
    lld_data = xr.merge([lld_data, forecast_ds], join="right")
    toc()

# %%
#
# Add additional columns
#
if 'ones' in x_cols:
    # Add intercept column
    printf("Adding ones")
    tic()
    lld_data.update({'ones': xr.ones_like(lld_data[measurement_variable])})
    toc()
if simplex:
    # Add features for simplex regression
    printf(f"Adding {pp_minus_perpp} feature for simplex regression")
    tic()
    lld_data.update({pp_minus_perpp: lld_data[pp_name] - lld_data[perpp_name]})
    toc()

# %%
# Convert to DataFrame
printf("Converting to DataFrame")
tic()
lld_data = lld_data.to_dataframe()
toc()

# Drop NA predictions
printf("Dropping rows with missing values")
tic()
lld_data = lld_data.dropna(subset=x_cols)
toc()

# Sort index
printf("Sorting index")
tic()
lld_data = lld_data.sort_index()
toc()


# %%
def apply_parallel(df_grouped, func, num_cores=os.process_cpu_count(), **kwargs):
    """Apply func to each group dataframe in df_grouped in parallel

    Args:
        df_grouped: output of grouby applied to pandas DataFrame
        func: function to apply to each group dataframe in df_grouped
        num_cores: number of CPU cores to use
        kwargs: additional keyword args to pass to func
    """
    # Associate only one OpenMP thread with each core
    os.environ['OMP_NUM_THREADS'] = str(1)
    pool = Pool(num_cores)
    # Pass additional keyword arguments to func using partial
    results = pool.map(partial(func, **kwargs), [group for name, group in df_grouped])
    pool.close()
    # Unset environment variable
    del os.environ['OMP_NUM_THREADS']

    # Separate predictions and coefficients
    ret_list = [result[0] for result in results]
    coef_list = [result[1] for result in results]

    return pd.concat(ret_list, ignore_index=False), pd.concat(coef_list, ignore_index=False)

def fit_and_predict(df, gt_col=None, x_cols=None, base_col=None, target_dates=None, 
                    gt_delta=None, simplex=False):
    """Fits model using rolling linear regression framework for a single gridpoint.

    Args:
        df: Dataframe with 'time' in index and gt_col, x_cols columns
        gt_col: Name of ground truth column in df
        x_cols: Names of columns used as input features
        base_col: Name of base column; if not None, will be subtracted away from gt_col
          to form regression target and added when making predictions
        target_dates: Nonempty list of target dates in ascending order for prediction
        gt_delta: Timedelta for computing last training date
        simplex: If True, clips coefficients to [0, 1]

    Returns DataFrame mapping target_date to prediction dataframe
    """
    # Initialize sufficient statistics with training data from first target date
    last_stored_date = target_dates[0] - gt_delta
    t = df.index.get_level_values('time')
    date_block = (t <= last_stored_date)
    train_data = df[date_block]
    n_train = len(train_data)

    if n_train > 0:
        X_train = train_data[x_cols].values
        if base_col:
            # Subtract base column prediction from ground truth
            y_val = train_data[gt_col].values - train_data[base_col].values
        else:
            y_val = train_data[gt_col].values
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_val
    else: 
        # If no training data, initialize sufficient statistics to zero
        n_features = len(x_cols)
        XtX = np.zeros((n_features, n_features))
        Xty = np.zeros(n_features)

    # Store predictions for all target dates
    all_predictions = []
    all_coefficients = []
    for target_date in target_dates:
        # Find the last observable training date for this target
        last_train_date = target_date - gt_delta        
        date_block = ((t <= last_train_date) & (t > last_stored_date))
        new_data = df[date_block]

        last_stored_date = last_train_date
        n_train += len(new_data)

        if len(new_data) > 0:
            X_train = new_data[x_cols].values
            y_val = new_data[gt_col].values

            # Update sufficient statistics
            XtX += X_train.T @ X_train
            Xty += X_train.T @ y_val

        # If there's at least one training date
        if n_train > 0:
            try:
                # Solve linear system when XtX full rank
                coef = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Otherwise, find minimum norm solution
                coef = np.linalg.lstsq(XtX, Xty)[0]
            
            if simplex:
                # Clip coefficients to [0, 1] for simplex regression
                coef = np.clip(coef, 0, 1)

            # Store prediction in dataframe
            X_test = df.loc[t == target_date, x_cols]
            if base_col:
                # Incorporate base column into prediction
                pred_df = pd.DataFrame(
                    {gt_col: [np.dot(X_test.values[0], coef) + 
                              df.loc[t == target_date, base_col].values[0]]},
                    index=X_test.index
                )
            else:
                pred_df = pd.DataFrame(
                    {gt_col: [np.dot(X_test.values[0], coef)]},
                    index=X_test.index
                )
            all_predictions.append(pred_df)

            ### TODO: store coefficients in a separate visualization script to avoid slowdown?
            if False: 
                # Store coefficients with feature names
                coef_data = {f'coef_{x_cols[i]}': coef[i] for i in range(len(x_cols))}
                coef_df = pd.DataFrame([coef_data], index=X_test.index)
                all_coefficients.append(coef_df)
    
    # return single dataframe with all predictions
    return (pd.concat(all_predictions, ignore_index=False), 
            pd.concat(all_coefficients, ignore_index=False) if all_coefficients else pd.DataFrame())


# %%
# Predictions directory
preds_dir = os.path.join('models', model_name, 'submodel_forecasts',
                         submodel_name, task)

# Sort target dates in ascending order
sorted_targets = sorted(target_date_objs)

# Check if target dates already have predictions or if some features
# are unavailable for prediction
printf("Identifying viable target dates for prediction")
tic()
dates_with_features = lld_data.index.unique(level='time')
targets_to_process = []
for target_date_obj in sorted_targets:
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
    if os.path.exists(preds_f):
        printf(f"prior forecast exists for target {target_date_str}")
        continue
    if target_date_obj not in dates_with_features:
        printf(f"warning: some features unavailable for target={target_date_obj}; skipping")
        continue
    targets_to_process.append(target_date_obj)
toc()
        
if not targets_to_process:
    printf("No target dates available for prediction; exiting")
else:
    printf(f"Processing {len(targets_to_process)} target dates with rolling regression")
    # Apply rolling regression to each grid point in parallel
    if simplex:
        # Use perpp as a base prediction
        prediction_func = partial(fit_and_predict, 
                                  gt_col=gt_col, 
                                  x_cols=x_cols,
                                  base_col=perpp_name,
                                  target_dates=targets_to_process,
                                  gt_delta=gt_delta,
                                  simplex=True)
    else:
         prediction_func = partial(fit_and_predict, 
                                   gt_col=gt_col, 
                                   x_cols=x_cols, 
                                   target_dates=targets_to_process,
                                   gt_delta=gt_delta)
    num_cores = os.process_cpu_count()
    tic()
    all_preds, all_coefs = apply_parallel(
        lld_data.groupby(group_by_cols),
        prediction_func,
        num_cores=num_cores
    )
    toc()
    printf(f"Clipping probabilistic forecasts to [0, 1]")
    tic()
    all_preds = all_preds.clip(0,1)
    toc()

    # Save predictions for each target date to disk
    t = all_preds.index.get_level_values('time')
    for target_date_obj in targets_to_process:
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        printf(f'Saving predictions for target {target_date_str}')
        
        tic()
        # Get predictions for this date
        date_preds = all_preds[t == target_date_obj]
        # Convert to xarray and reindex to full lat-lon grid
        preds_i = date_preds.to_xarray()
        preds_i = preds_i.reindex(
            time=[target_date_obj], 
            latitude=all_lats, 
            longitude=all_lons, 
            fill_value=np.nan
        )
        toc()

        # Save prediction 
        tic()
        preds_f = os.path.join(preds_dir, f'{task}-{target_date_str}.nc')
        printf(f"Saving to {preds_f}")
        save_to_netcdf(preds_i, Path(preds_f))
        toc()
