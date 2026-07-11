"""Generates predictions for each of the ECMWF parameter configurations

Example usages:
  python src/models/ecmwf/bulk_batch_predict.py era5-f1_tas 19 -t std_test -c "src/batch/batch_python.sh --memory 10 --cores 1 --hours 1"
  python src/models/ecmwf/bulk_batch_predict.py era5-F10_tas 19 -t std_test -c "src/batch/batch_python.sh --memory 10 --cores 1 --hours 1"
  for dates in std_test std_future; do
  for f in {1..4}; do
    for var in pr tas mslp; do
      for horizon in 19 26; do
        python src/models/ecmwf/bulk_batch_predict.py era5-f${f}_${var} ${horizon} -t ${dates} -c "src/batch/batch_python.sh --memory 15 --cores 1 --hours 1"
      done
    done
  done
  done
  for dates in std_test std_future; do
  for f in "F5" "F10" "F90" "F95"; do
    for var in pr tas mslp; do
      for horizon in 19 26; do
        python src/models/ecmwf/bulk_batch_predict.py era5-f${f}_${var} ${horizon} -t ${dates} -c "src/batch/batch_python.sh --memory 15 --cores 1 --hours 1"
      done
    done
  done
  done
      
Positional args:
  gt_id: e.g., era5-f1_tas, era5-F10_tas
  horizon: e.g., 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction
  --cmd_prefix (-c): prefix of command used to execute batch_predict.py
    (default: "python"); e.g., "python" to run locally,
    "src/batch/batch_python.sh --memory 1 --cores 1 --hours 1" to
    submit to batch queue
"""
import os
import subprocess
from argparse import ArgumentParser
from models.utils.general_util import printf

forecast = "ecmwf"
model_name = forecast

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--cmd_prefix', '-c', default="python")

args = parser.parse_args()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates = args.target_dates
cmd_prefix = args.cmd_prefix

# Specify list of parameter settings to run
if horizon == "19":
    first_lead, last_lead = 19, 19
elif horizon == "26":
    first_lead, last_lead = 26, 26
else:
    raise ValueError(f"unsupported horizon {horizon}")

module_str = os.path.join("src","models",model_name,"batch_predict.py")

task_str = f"{gt_id} {horizon} -t {target_dates}"
# Run batch predict for this configuration
param_str=f"--first_lead {first_lead} --last_lead {last_lead}"
cmd=f"{cmd_prefix} {module_str} {task_str} {param_str}"
printf(f"Running {cmd}")
subprocess.call(cmd, shell=True)