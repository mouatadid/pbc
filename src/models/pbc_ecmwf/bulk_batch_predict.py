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
Generates predictions for each of the PBC-ECMWF parameter configurations

Example usage: (include -p to preview commands without running batch_predict.py)
  python src/models/pbc_ecmwf/bulk_batch_predict.py era5-f1_tas 19 -t std_test -c "src/batch/batch_python.sh -m 20 -c 8" -p

Example usage: (exclude -p to run batch_predict.py)
  python src/models/pbc_ecmwf/bulk_batch_predict.py era5-f1_tas 19 -t std_test -c "src/batch/batch_python.sh -m 20 -c 8" 
  python src/models/pbc_ecmwf/bulk_batch_predict.py era5-f1_tas 19 -t std_test -c "src/batch/batch_python.sh -m 1 -c 1" 
  # (include -o to overwrite existing predictions)
  for dates in std_future; do
  for var in tas pr mslp; do
    for f in {1..4}; do
      for horizon in 19 26; do
        python src/models/pbc_ecmwf/bulk_batch_predict.py era5-f${f}_${var} ${horizon} -t ${dates} -o -c "src/batch/batch_python.sh -m 1 -c 1"
      done
    done
  done
  done

  for dates in std_test; do
  for var in mslp; do
    for F in 90; do
      for horizon in 19 26; do
        python src/models/pbc_ecmwf/bulk_batch_predict.py era5-F${F}_${var} ${horizon} -t ${dates} -o -c "src/batch/batch_python.sh -m 1 -c 1 -h 12"
      done
    done
  done
  done

Positional args:
  gt_id: e.g., era5-f1_tas, era5-f1_pr, era5-f1_mslp, era5-F10_tas, era5-F5_pr, era5-F95_mslp
  horizon: e.g., 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction
  --cmd_prefix (-c): prefix of command used to execute batch_predict.py
    (default: "python"); e.g., "python" to run locally
  --preview (-p): preview batch commands to be run without executing them;
    (default: False)
  --num_seeds (-n): number of date_order_seeds to use for parallelizing batch predict;
    (default: 0)
'''

# %%
import os
import time
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
import subprocess
from pkg_resources import resource_filename
from models.utils.general_util import printf

# %%
forecast = "ecmwf"
model_name = f"pbc_{forecast}"

# %%
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
    parser.add_argument('--target_dates', '-t', default="std_contest")
    parser.add_argument('--cmd_prefix', '-c', default="python")
    parser.add_argument('--preview', '-p', default=False, action='store_true',
                       help="preview batch commands to be run without executing them")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                        help="overwrite existing prediction files")
    # Assign variables
    args = parser.parse_args()
    gt_id = args.pos_vars[0]  
    horizon = args.pos_vars[1] 
    target_dates = args.target_dates
    cmd_prefix = args.cmd_prefix.strip()
    preview = args.preview
    overwrite = args.overwrite
else:
    # Otherwise, specify arguments interactively
    gt_id = "era5-f1_tas"
    horizon = "19"
    target_dates = "20180105" 
    cmd_prefix = "python" 
    preview = True
    overwrite = False

# %%
# Specify list of parameter settings to run
train_years = "all"
margin = "None"

# %%
predict_script = resource_filename(__name__, os.path.join('batch_predict.py'))
task_str = f"{gt_id} {horizon} -t {target_dates}"
if overwrite:
    task_str += " -o"
if cmd_prefix is None:
    cmd_prefix = "python"

simplex = "False"
equal = "True"
param_strs=[f"-y {train_years} -m {margin} -s {simplex} -e {equal}"]

for i, param_str in enumerate(param_strs):
    if 'python' in cmd_prefix:
        cmd = f"{cmd_prefix} {predict_script} {task_str} {param_str}"
    else:
        cmd = f"{cmd_prefix} \"python {predict_script} {task_str} {param_str}\" &"
    printf(f"Running:\n{cmd}")
    if not preview:
        subprocess.call(cmd, shell=True)
        sleep_time = .0
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
