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
Tuner: For a given model and each target date, use the forecast of the submodel
with the most accurate predictions over all dates in the past num_years years
in a window of margin_in_days around the target month-day combination
python src/models/tuner/batch_predict.py era5-f1_tas 26 -mn ecmwfpp -t std_test
Example usage:
  python src/models/tuner/batch_predict.py era5-f1_mslp 19 -mn ecmwfpp -y 3 -t std_test
  for dates in std_test; do 
    for var in tas pr mslp; do
      for f in f1 f2 f3 f4; do
        for horizon in 19 26; do
        for model in ecmwfpp; do
            src/batch/batch_python.sh -m 10 --cores 1 --hours 1 src/models/tuner/batch_predict.py era5-${f}_$var ${horizon} -mn $model -y 3 -t $dates
        done
        done
      done
    done
  done
Positional args:
  gt_id: e.g. era5-tas, era5-pr, era5-mslp
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction
  --model_name (-mn): name of model to tune; (default: "ecmwfpp")
  --num_years (-y): number of years to use in tuning ("all" for all years
    or positive integer); (default: "3")
  --margin_in_days (-m): number of month-day combinations on either side of 
    the target combination to include; set to 0 to include only target 
    month-day combo; set to None to include entire year; (default: None)
'''

# %%
# Ensure notebook is being run from base repository directory
import os, sys
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
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
from models.utils.data_utils import get_measurement_variable
from models.utils.general_util import printf, make_directories, tic, toc
from models.utils.models_util import (start_logger, log_params)
from models.utils.eval_util import get_target_dates
from models.tuner.util import *
from utils.data_io import load_data
from utils.file_io import symlink

# %%
if __name__ == "__main__":
    #
    # Specify model parameters
    #
    if not isnotebook():
        # If notebook run as a script, parse command-line arguments
        parser = ArgumentParser()
        parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon 
        parser.add_argument('--model_name', '-mn', default="ecmwfpp")                                                                                 
        parser.add_argument('--target_dates', '-t', default="std_test")
        parser.add_argument('--num_years', '-y', default="3",
                           help="Number of years to use in training (all or integer)")
        parser.add_argument('--margin_in_days', '-m', default="None", 
                           help="Number of month-day combinations on either side of the target combination to include; "
                                "set to 0 to include only target month-day combo; set to None to include entire year; "
                                "None by default")
        parser.add_argument('--skip_existing', '-se', default=False, action='store_true',  
                        help="if False, overwrite existing forecasts")
        args = parser.parse_args()
        
        # Assign variables                                                                                                                                     
        gt_id = args.pos_vars[0]                                                                
        horizon = args.pos_vars[1]
        model_name = args.model_name
        target_dates = args.target_dates
        num_years = args.num_years
        if num_years != "all":
            num_years = int(num_years)
        margin_in_days = args.margin_in_days
        if margin_in_days == "None":
            margin_in_days = None
        else:
            margin_in_days = int(args.margin_in_days)
        skip_existing = args.skip_existing
    else:
        # Otherwise, specify arguments interactively
        gt_id = "era5-f1_tas" 
        horizon = "19"
        model_name = "msnpp" 
        target_dates = "std_msn"
        num_years = 3
        margin_in_days = None
        skip_existing = False
    
    """ 
    Process model parameters
    """
    task = f'{gt_id}_{horizon}'
    agg_period = 7 
    measurement_variable = get_measurement_variable(gt_id)

    # Record output model name and submodel name
    output_model_name = f"tuned_{model_name}"
    submodel_name = get_tuner_submodel_name(
        output_model_name=output_model_name, num_years=num_years, 
        margin_in_days=margin_in_days)

    # Prepare a directory to store tuned model attributes and configuration files
    src_dir = os.path.join('src', 'models', 'tuner')
    dst_dir = os.path.join('src', 'models', output_model_name)
    if not os.path.exists(dst_dir):
        tic()
        printf(f'\nCreating {dst_dir}')
        make_directories(dst_dir)
        # copy attributes and selected submodel files to output model folder
        shutil.copy(os.path.join(src_dir, "attributes.py"), os.path.join(dst_dir, "attributes.py"))
        shutil.copy(os.path.join(src_dir, "selected_submodel.json"), os.path.join(dst_dir, "selected_submodel.json"))
        # update MODEL_NAME in the attribute file
        filename = os.path.join(dst_dir, "attributes.py")
        with open(filename, "r") as f:
            newText=f.read().replace("tuner", output_model_name)
        with open(filename, "w") as f:
            f.write(newText)
        toc()

    # Create directory for storing forecasts if one does not already exist
    out_dir = os.path.join("models", output_model_name, "submodel_forecasts", 
                        submodel_name, f"{gt_id}_{horizon}")
    if not os.path.exists(out_dir):
        make_directories(out_dir)

    if not isnotebook():
        # Save output to log file
        logger = start_logger(model=output_model_name,submodel=submodel_name,gt_id=gt_id,
                              horizon=horizon,target_dates=target_dates)
        # Store parameter values in log
        params_names = ['gt_id', 'horizon', 'model_name', 
                        'target_dates', 'num_years', 'margin_in_days']
        params_values = [eval(param) for param in params_names]
        log_params(params_names, params_values)

    # Select target dates and restrict to dates with available ground truth data
    target_date_objs = get_target_dates(target_dates, horizon=horizon)

    printf(f'\nLoading metrics of {model_name} submodels')
    tic()
    metric_df = load_metric_df(gt_id=gt_id, target_horizon=horizon, model_name=model_name, metric='wtd_mse')
    toc()

    """
    Load and merge ground truth 
    """
    printf("Loading ground truth data")
    # Do not load anomalies (load raw ground truth)
    load_anomalies = False
    # Define climatology range (these years have no impact on model since
    # load_anomalies is False but are still required by ground-truth loader)
    clim_first_year = 1985
    clim_last_year = 2014
    start_time = datetime.strftime(min(target_date_objs), "%Y-%m-%d")
    end_time = datetime.strftime(max(target_date_objs), "%Y-%m-%d")
    # Load ground truth data
    # tic()
    if gt_id.endswith("tas") or gt_id.endswith("pr"):
        gt_ds = load_data(gt_id, lsmask=True)
    elif gt_id.endswith("mslp"): 
        gt_ds = load_data(gt_id, lsmask=False)
    else:
        raise ValueError(f"Unknown ground truth id {gt_id}")
    # toc()
    # Load data into memory (in place)
    tic()
    gt_ds.load()
    toc()


    #
    # Generate predictions
    #
    # Template for selected submodel forecast file
    
    file_template = os.path.join("models", "{}", "submodel_forecasts", "{}", 
                                     f"{gt_id}_{horizon}", f"{gt_id}_{horizon}"+"-{}.nc")
    
    # Auxiliary dataframe used to identify tuning dates
    X = pd.DataFrame(index=metric_df.index, columns = ["delta", "dividend", "remainder"], 
                     dtype=np.float64)
    # rmses = pd.Series(index=target_date_objs, dtype=np.float64)
    for target_date_obj in target_date_objs:
        tic()
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        # Determine which dates will be used to assess submodels
        printf(f"\n\nObtaining tuning dates for target date {target_date_str}")
        tic()
        tuning_dates = get_tuning_dates(gt_id, horizon, target_date_obj, 
                                        num_years, margin_in_days, X)
        
        #print(tuning_dates)
        if not tuning_dates.any():
            printf(f"Warning: No tuning dates have latitude-weighted MSEs for target date {target_date_str}; skipping")
            continue
        toc()
        printf(f"Selecting most accurate submodel")
        tic()
        # Select most accurate submodel across these dates
        model_selected_submodel = metric_df[tuning_dates].mean().idxmin()
        # Form forecast by softlinking to selected submodel forecast
        src_file = file_template.format(model_name, model_selected_submodel, target_date_str)
        dst_file = file_template.format(output_model_name, submodel_name, target_date_str)
        src_file_exists = os.path.exists(src_file)
        dst_file_exists = os.path.islink(dst_file)

        if skip_existing and dst_file_exists:
            printf(f"Forecasts already exist for target date {target_date_str}; skipping")
            continue
        toc()
        
        printf(f"Selected predictions -- {model_selected_submodel} for target_date {target_date_str}")
        
        if src_file_exists:
            tic()
            #if target_date_obj > last_train_date:
            printf(f"Saving predictions")
            symlink(src_file, dst_file, use_abs_path=True)
            toc()
            # if target_date_obj in gt.index:
            #     printf(f"Calculating latitude-weighted rmse")
            #     tic()
            #     preds = xr.open_dataset(src_file, engine='nc')
            #     preds = preds.to_dataframe().reset_index()
            #     rmse = np.sqrt(np.square(preds.set_index(['start_date'])["pred"] - gt.loc[target_date_obj,:][measurement_variable]).mean())
            #     rmses.loc[target_date_obj] = wtd_rmse
            #     toc()
        else:
            printf(f"Warning: Missing file:\n{src_file}")
        printf(f"Total processing time")
        toc()

