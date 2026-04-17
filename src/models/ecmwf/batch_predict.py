"""
Predicts outcomes using raw (undebiased) ecmwf ensemble forecast

Example usage:
  python src/models/ecmwf/batch_predict.py era5-f1_tas 26 -t std_future -fl 26 -ll 26

Positional args:
  gt_id: e.g., era5-f1_tas
  horizon: 19 or 26

Named args:
   --target_dates (-t): target dates for batch prediction (default: std_test)
"""
import os
import subprocess
import shlex
import sys
from argparse import ArgumentParser
from models.utils.models_util import get_submodel_name, get_task_forecast_dir
from models.utils.general_util import make_parent_directories
from utils.logging import printf
from utils.file_io import symlink

model_name = "ecmwf"
base_model_name = "ecmwfpp"

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--first_lead', '-fl', default=1, 
                    help="first ecmwf lead to average into forecast (0-26)")
parser.add_argument('--last_lead', '-ll', default=1, 
                    help="last ecmwf lead to average into forecast (0-26)")
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0] # e.g., "era5-f1_tas"
horizon = args.pos_vars[1] # e.g., "19" or "26"

# Create dictionary of default values for additional arguments
# used by the base model
default_args = {'fit_intercept': False,
                'years': 20, 
                'margin_in_days': 0,
                'days': 1,
                'loss': "mse"}
default_arg_str = " ".join(f"--{key} {value}" for key, value in default_args.items())

# Reconstruct command-line arguments and run base model
cmd_args = " ".join(map(shlex.quote, sys.argv[1:]))
cmd_args = f"{cmd_args} {default_arg_str}"
predict_script = os.path.join("src","models",base_model_name,"batch_predict.py")
cmd = f"python {predict_script} {cmd_args}"

printf(f"Running {cmd}")
subprocess.call(cmd, shell=True)

# Remove target dates and positional arguments from args dictionary
# so that remaining arguments are model-specific
del args.pos_vars
del args.target_dates

# Soft link prediction directory for this model to relevant base model 
# prediction directory
base_submodel_name = get_submodel_name(
    model=base_model_name, **default_args, **vars(args))
base_submodel_dir = get_task_forecast_dir(
    model=base_model_name, submodel=base_submodel_name, 
    gt_id=gt_id, horizon=horizon)
# Same submodel name used for model and base model
submodel_dir = get_task_forecast_dir(
    model=model_name, submodel=base_submodel_name, 
    gt_id=gt_id, horizon=horizon)

# Ensure all parent directories of submodel dir exist with correct permissions
make_parent_directories(submodel_dir)

printf(f"Soft-linking\n-src: {base_submodel_dir}\n-dest: {submodel_dir}")
symlink(base_submodel_dir, submodel_dir, use_abs_path=True)