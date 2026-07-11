# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
'''
Generates predictions for each of the ECMWF parameter configurations
used by the tuner

Example usage: (include -p to preview commands without running batch_predict.py)
  python src/models/ecmwfpp/bulk_batch_predict.py era5-f4_mslp 19 -t std_tune -c "src/batch/batch_python.sh -m 15 -c 1"
  python src/models/ecmwfpp/bulk_batch_predict.py era5-F10_mslp 19 -t std_tune -c "src/batch/batch_python.sh -m 15 -c 1"

Example usages: (include -o to overwrite existing predictions, -wm to generate metrics after predictions)
  for var in tas pr; do
    for horizon in 19 26; do
        for f in f4 f3 f2 f1; do
            python src/models/ecmwfpp/bulk_batch_predict.py era5-${f}_${var} $horizon -t std_tune -c "src/batch/batch_python.sh -m 15 -c 1" -wm -o
        done
    done
  done
  for var in mslp; do
    for horizon in 19 26; do
        for f in f4 f3 f2 f1; do
            python src/models/ecmwfpp/bulk_batch_predict.py era5-${f}_${var} $horizon -t std_tune -c "src/batch/batch_python.sh -m 20 -c 1" -wm -o 
        done
    done
  done
  for var in tas pr; do
    for horizon in 19 26; do
        for F in F5 F10 F90 F95; do
            python src/models/ecmwfpp/bulk_batch_predict.py era5-${F}_${var} $horizon -t std_tune -c "src/batch/batch_python.sh -m 15 -c 1" -wm -o
        done
    done
  done
  for var in mslp; do
    for horizon in 19 26; do
        for F in F5 F10 F90 F95; do
            python src/models/ecmwfpp/bulk_batch_predict.py era5-${F}_${var} $horizon -t std_tune -c "src/batch/batch_python.sh -m 20 -c 1" -wm -o
        done
    done
  done
  # Use -mo to generate metrics only
  for var in pr tas mslp; do
    for horizon in 19 26; do
        for f in f4 f3 f2 f1; do
            python src/models/ecmwfpp/bulk_batch_predict.py era5-${f}_${var} $horizon -t std_tune -mo -c "src/batch/batch_python.sh -m 10 -c 1"
        done
    done
  done

Positional args:
  gt_id: era5-f1_tas, era5-f1_pr, era5-f1_mslp, era5-F5_tas, era5-F10_tas, era5-F90_tas etc.
  horizon: 19 or 26

Named args:
  --target_dates (-t): target dates for batch prediction
  --cmd_prefix (-c): prefix of command used to execute batch_predict.py
    (default: "python")
  --preview (-p): preview batch commands to be run without executing them;
    (default: False)
  --metrics_only (-mo): only generate metrics not predictions (default: False)
  --with_metrics (-wm): generate both predictions and corresponding metrics (default: False)
  --overwrite (-o): overwrite existing prediction files (default: False)
  --skip_existing (-se): skip submodels that have existing subfolders (default: False)
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
from subprocess import DEVNULL
import shlex
from pkg_resources import resource_filename
from models.utils.data_utils import get_measurement_variable
from models.utils.general_util import printf, tic, toc
from models.ecmwfpp.attributes import get_submodel_name
from models.utils.eval_util import get_named_targets


# %%
model_name = "ecmwfpp"

# %%
if not isnotebook():
    # If notebook run as a script, parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("pos_vars", nargs="*")  # gt_id and target_horizon
    parser.add_argument('--target_dates', '-t', default="std_tune")
    parser.add_argument('--cmd_prefix', '-c', default="python")
    parser.add_argument('--metrics_only', '-mo', default=False, action='store_true',
                        help="only generate metrics not predictions")
    parser.add_argument('--with_metrics', '-wm', default=False, action='store_true',
                        help="generate both predictions and corresponding metrics")
    parser.add_argument('--preview', '-p', default=False, action='store_true',
                       help="preview batch commands to be run without executing them")
    parser.add_argument('--overwrite', '-o', default=False, action='store_true',
                       help="overwrite existing prediction files")
    parser.add_argument('--skip_existing', '-se', default=False, action='store_true',
                        help="skip submodels that have existing subfolders")

    # Assign variables
    args = parser.parse_args()
    gt_id = args.pos_vars[0]
    horizon = args.pos_vars[1]
    target_dates = args.target_dates
    cmd_prefix = args.cmd_prefix
    metrics_only = args.metrics_only
    with_metrics = args.with_metrics
    preview = args.preview
    overwrite = args.overwrite
    skip_existing = args.skip_existing
else:
    # Otherwise, specify arguments interactively
    gt_id = "era5-F10_pr"
    horizon = "19"
    target_dates = "std_tune" 
    cmd_prefix = 'batch_python.sh' #'python'
    metrics_only = False
    with_metrics = True
    preview = True
    overwrite = False
    skip_existing = False

if with_metrics and 'batch_python.sh' not in cmd_prefix: 
    raise ValueError("with_metrics option only works with batch commands (e.g., batch_python.sh)")

# %%
measurement_variable = get_measurement_variable(gt_id)
task = f"{gt_id}_{horizon}"
# Specify list of parameter settings to run
years = 20
fit_intercept = True
days = [1] 
margins = [14, 28, 35] 
loss = 'mse'
# Specify parallel arrays of first and last leads
if horizon == "19":
    first_leads = [19]
    last_leads = [19]
elif horizon == "26":
    first_leads = [26]
    last_leads = [26]
else:
    raise ValueError(f"invalid horizon {horizon}")


# %%
def extract_job_id(sbatch_output):
    """
    Extract job ID from sbatch output
    """
    if isinstance(sbatch_output, bytes):
        sbatch_output = sbatch_output.decode('utf-8')
    
    # Split by lines, get the last line, then get the last word
    lines = sbatch_output.strip().split('\n')
    if lines:
        last_line = lines[-1]
        words = last_line.split()
        if words:
            return words[-1] 
    return None


# %%
# Set parameters for generating forecast files
predict_script = resource_filename(__name__, os.path.join('batch_predict.py'))
task_str = f"{gt_id} {horizon} -t {target_dates} -l {loss}"
if overwrite:
    task_str += " --overwrite"

# Set parameters for generating metrics files
metrics_prefix = cmd_prefix if metrics_only else "src/batch/batch_python.sh -m 10 -c 1 -h 2"
metrics_script = os.path.join('src', 'models', "batch_metrics.py")
metrics = "wtd_mse"
metrics_suffix = '_mo' if metrics_only else ''

# If not overwriting and skipping existing submodels, check if submodel forecasts already exist
if (not overwrite and skip_existing) and os.path.exists(f'models/{model_name}/submodel_forecasts'):
    submodels_existing = sorted(os.listdir(f'models/{model_name}/submodel_forecasts'))
else:
    submodels_existing = []

# Store metrics job IDs 
metrics_ids = []

# Seconds to sleep between job submissions
sleep_time = .0

i, i_run = 0, 0
# Iterate over parallel leads arrays
for ii in range(len(first_leads)):
    first_lead = first_leads[ii]
    last_lead = last_leads[ii]
    for day in days:
        for margin in margins:
            if with_metrics: 
                predict_job_name = f"--name {i}_{model_name}_{measurement_variable}{horizon}_{years}_{margin}_{day}_{first_lead}_{last_lead}".replace('+', '')
                metrics_job_name = f"--name {i}_{model_name}_mo_{measurement_variable}{horizon}_{years}_{margin}_{day}_{first_lead}_{last_lead}".replace('+', '')
            else: 
                job_name = f"--name {i}_{model_name}{metrics_suffix}_{measurement_variable}{horizon}_{years}_{margin}_{day}_{first_lead}_{last_lead}".replace('+', '')
            # Run batch predict for this configuration
            param_str = f"-y {years} -i {fit_intercept} -m {margin} -d {day} -fl {first_lead} -ll {last_lead}"
            # Get submodel name
            submodel_name = get_submodel_name(fit_intercept=fit_intercept, years=years,
                                                margin_in_days=margin, days=day, loss=loss,
                                                first_lead=first_lead, last_lead=last_lead)
            i += 1
            i_run += 1
            if metrics_only:
                metrics_filepath = os.path.join('eval', 'metrics', model_name, 'submodel_forecasts', submodel_name, f'{gt_id}_{horizon}', f'{metrics}-{gt_id}_{horizon}-{target_dates}.zarr')
                if skip_existing and os.path.isfile(metrics_filepath):
                    printf(f"\nSkipping -- metrics already exist\n{metrics_filepath}")
                    continue
                # Generate metrics for this submodel
                if 'python' in metrics_prefix:
                    metrics_cmd = f"{metrics_prefix} {metrics_script} {gt_id} {horizon} -mn {model_name} -sn {submodel_name} -t {target_dates} -m {metrics}"
                else:
                    metrics_cmd = f"{metrics_prefix} {job_name} \"python {metrics_script} {gt_id} {horizon} -mn {model_name} -sn {submodel_name} -t {target_dates} -m {metrics}\" &"
                printf(f"\n{job_name.replace('--name ', '')}\nGenerate submodel metrics:\n{metrics_cmd}")
                if not preview:
                    subprocess.call(metrics_cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
                    print(f"Sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)
            elif with_metrics: 
                predict_job_id = None
                # First, run prediction script
                if submodel_name in submodels_existing:
                    tasks_existing = sorted(os.listdir(f'models/{model_name}/submodel_forecasts/{submodel_name}'))
                    if task in tasks_existing: 
                        printf(f"\n{i} - Skipping {submodel_name}; forecasts already exist")
                        continue
                predict_cmd = f"{cmd_prefix} {predict_script} {task_str} {param_str}"
                printf(f"\n{predict_job_name.replace('--name ', '')}\nGenerate submodel predictions:\n{predict_cmd}")
                if not preview:
                    result = subprocess.run(predict_cmd, shell=True, capture_output=True, text=True)
                    predict_job_id = extract_job_id(result.stdout)
                    if predict_job_id: 
                        printf(f"Prediction job started with ID: {predict_job_id}")
                    else: 
                        printf(f"Warning: Failed to extract job ID from output: {result.stdout}")
                    print(f"Sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)
                # Run metrics script with predict job ID as dependency
                metrics_filepath = os.path.join('eval', 'metrics', model_name, 'submodel_forecasts', submodel_name, f'{gt_id}_{horizon}', f'{metrics}-{gt_id}_{horizon}-{target_dates}.zarr')
                # Generate metrics for this submodel with dependency
                dependency_str = ""
                if predict_job_id: 
                    dependency_str = f"-d afterok:{predict_job_id}"
                metrics_cmd = f"{metrics_prefix} {dependency_str} {metrics_script} {gt_id} {horizon} -mn {model_name} -sn {submodel_name} -t {target_dates} -m {metrics}"
                printf(f"\n{metrics_job_name.replace('--name ', '')}\nGenerate submodel metrics:\n{metrics_cmd}")
                if not preview:
                    result = subprocess.run(metrics_cmd, shell=True, capture_output=True, text=True)
                    metrics_job_id = extract_job_id(result.stdout)
                    if metrics_job_id: 
                        metrics_ids.append(metrics_job_id)
                        printf(f"Metrics job started with ID: {metrics_job_id}")
                    else: 
                        printf(f"Warning: Failed to extract job ID from output: {result.stdout}")
                    print(f"Sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)
            else:
                if submodel_name in submodels_existing:
                    tasks_existing = sorted(os.listdir(f'models/{model_name}/submodel_forecasts/{submodel_name}'))
                    if task in tasks_existing: 
                        printf(f"\n{i} - Skipping {submodel_name}; forecasts already exist")
                        continue
                if 'python' in cmd_prefix:
                    cmd = f"{cmd_prefix} {predict_script} {task_str} {param_str}"
                else:
                    cmd = f"{cmd_prefix} {job_name} \"python {predict_script} {task_str} {param_str}\" &"
                printf(f"\n{job_name.replace('--name ', '')}\nGenerate submodel predictions:\n{cmd}")
                if not preview:
                    subprocess.call(cmd, shell=True) 
                    print(f"Sleeping for {sleep_time} seconds")
                    time.sleep(sleep_time)

# Output metrics job IDs
if with_metrics and metrics_ids:
    metrics_ids_string = ":".join(metrics_ids)
    print(f"METRICS_JOB_IDS:{metrics_ids_string}")


# %%
