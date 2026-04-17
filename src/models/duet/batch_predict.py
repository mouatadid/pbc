""" 
Duet: For each gridpoint and target date, output an equal average of pbc-ecmwf and pbc-msn.

Example usage:
  python src/models/duet/batch_predict.py era5-f1_pr 19 -t std_test
  # (include -o to overwrite existing predictions)
  for dates in std_test std_future; do
  for var in pr tas mslp; do
    for f in {1..4}; do
      for horizon in 19 26; do
        python src/models/duet/batch_predict.py "era5-f${f}_$var" "$horizon" -t ${dates} -o
      done
    done
  done
  done

  for dates in std_test std_future; do
  for var in pr tas mslp; do
    for f in {1..4}; do
      for horizon in 19 26; do
        src/batch/batch_python.sh -m 1 --cores 1 --hours 1 src/models/duet/batch_predict.py "era5-f${f}_$var" "$horizon" -t ${dates} -o
      done
    done
  done
  done

Positional args:
  gt_id: era5-tas, era5-pr, era5-mslp, etc.
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction 
  --overwrite (-o): overwrite existing prediction files
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
from datetime import datetime
from pathlib import Path
from models.utils.general_util import printf
from models.utils.eval_util import get_target_dates
from models.utils.data_utils import get_measurement_variable
from models.utils.models_util import get_submodel_name
from utils.timing import tic, toc
from utils.data_io import save_to_netcdf

#
# Specify model parameters
#
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon                                                                                  
    parser.add_argument('--target_dates', '-t', default="std_test")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                        help="overwrite existing prediction files")
    args, opt = parser.parse_known_args()
    
    # Assign variables                                                                                                                                     
    gt_id = args.pos_vars[0]                                                                          
    horizon = args.pos_vars[1]                                                                                     
    target_dates = args.target_dates  
    overwrite = args.overwrite
else:
    # Otherwise, specify arguments interactively 
    gt_id = "era5-f1_tas"
    horizon = "19"
    target_dates = "std_test"
    overwrite = False

""" 
Process model parameters
"""
model_name = "duet"
task = f'{gt_id}_{horizon}'
measurement_variable = get_measurement_variable(gt_id) 
target_date_objs = pd.Series(get_target_dates(date_str=target_dates, horizon=horizon))

# Get ecmwf and msn model names
ecmwf_name = "pbc_ecmwf"
msn_name = "pbc_msn"
ensemble_members = [ecmwf_name, msn_name]

# Record model and submodel name
submodel_name = get_submodel_name(model_name)
printf(f"Submodel name: {submodel_name}")

#
# Average predictions of each ensemble member
#
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
            if model.startswith('pbc'): 
                sn = f'{model}-yearsall_marginNone_equal'
            else: 
                raise ValueError(f"Unknown model: {model}")
            model_filename = file_template.format(model, sn)
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