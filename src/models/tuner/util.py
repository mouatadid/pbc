# Util functions for tuner model

import importlib
import os
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
from models.utils.experiments_util import get_first_year, get_start_delta

MODEL_NAME = "tuner"
ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
ONE_YEAR = timedelta(days=365)

""
    #Data related util functions
""
def get_model_selected_submodel_name(model_name="ecmwfpp",
                                     gt_id="f1_tas",
                                     target_horizon="19"):
    """Returns the string selected submodel name for a given model, gt_id and target_horizon.

    Args:
      model_name: string model name
      gt_id: f1_tas or contest_precip
      target_horizon: 19 or 26
    """
    get_selected_submodel_name = getattr(importlib.import_module("src.models."+model_name+".attributes"), "get_selected_submodel_name")
    return get_selected_submodel_name(gt_id, target_horizon)
    
def get_tuned_model_selected_submodel_name(output_model_name="tuned_ecmwfpp",
                                     gt_id="f1_tas",
                                     target_horizon="19"):
    """Returns the string selected submodel name for a given model, gt_id and target_horizon.

    Args:
      model_name: string model name
      gt_id: f1_tas or contest_precip
      target_horizon: 19 or 26
    """
    get_selected_submodel_name = getattr(importlib.import_module("src.models."+output_model_name+".attributes"), "get_selected_submodel_name")
    return get_selected_submodel_name(gt_id, target_horizon)    
    
def get_tuner_submodel_name(output_model_name="tuned_ecmwfpp", num_years="all", margin_in_days=0):
    """Returns submodel name for a given setting of model parameters
    """
    return f"{output_model_name}_on_years{num_years}_margin{margin_in_days}"    
    
    
def get_target_dates_all(gt_id="era5-tas"):
    start_year = get_first_year(gt_id)
    start_date = datetime.strptime(f"{start_year}0101", "%Y%m%d")
    return pd.date_range(start = start_date, end = datetime.today()).to_pydatetime().tolist()


def load_metric_df(gt_id="era5-tas", 
                   target_horizon="19", 
                   model_name="ecmwfpp", 
                   metric="wtd_mse",
                   metric_file_regex="std_tune", 
                   first_year=2007):
    #STEP 1: get predict_rmse selected submodel
    # Store submodel or model performances in dataframe\n,
    task = f"{gt_id}_{target_horizon}"
    # Get metrics dataframe 
    metric_df = pd.DataFrame(index=get_target_dates_all(gt_id=gt_id))
    metric_df = metric_df[metric_df.index.year>=first_year]
    index_cols = 'time'
    # Identify the submodels with stored metric for this task
    filenames = []
    submodel_names = []
    # Identify the submodels with stored metric for this task
    eval_dir = os.path.join("eval", "metrics", model_name, "submodel_forecasts")
    models_dir = glob(f"{eval_dir}/*/{task}")
    submodel_names = [d.split(os.path.sep)[-2] for d in models_dir] 
    for submodel_name in submodel_names:
        filenames = glob(os.path.join(eval_dir, submodel_name, task, f"{metric}-{task}-{metric_file_regex}.zarr"))
        if filenames:
            for f in filenames:
                dfi = xr.open_dataset(f, engine='zarr', decode_timedelta=True).to_dataframe().rename(columns={metric:submodel_name})
                df = dfi if filenames.index(f)==0 else df.append(dfi)
            df = df.reset_index().sort_values(by="time").set_index(index_cols)
            df = df[~df.index.duplicated(keep='first')]
            metric_df = metric_df.join(df)
    metric_df = metric_df.dropna(axis=0, how="all").drop_duplicates()
    return metric_df


def get_tuning_dates(gt_id, horizon, target_date_obj, num_years, margin_in_days, X):
    """Returns indicator of whether each index element of X should be used for tuning
    
    Args:
        target_date_obj - target date datetime object
        X - dataframe with index equal to all relevant target dates and columns
            ["delta", "dividend", "remainder"]; warning: X will be modified!
    """
    start_delta = timedelta(days=get_start_delta(horizon, gt_id))
    last_train_date = target_date_obj - start_delta
    days_per_year = 365.242199

    X['delta'] = (target_date_obj - X.index).days
    X['remainder'] = np.floor(X.delta % days_per_year) 

    # Restrict data based on training date, dividend, and remainder
    indic = (X.index <= last_train_date)
    if margin_in_days is not None:
        indic &= ((X.remainder <= margin_in_days) | (X.remainder >= 365-margin_in_days))
    if num_years != "all":
        X['dividend'] = np.floor(X.delta / days_per_year)
        indic &= (X.dividend < num_years)

    return indic


def get_tuning_dates_selected_submodel(metric_df, tuning_dates):
    
    return metric_df[metric_df.index.isin(tuning_dates)].mean().idxmin() 

