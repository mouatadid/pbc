# Model attributes
import json
import os
from importlib.resources import files

MODEL_NAME="aifspp"
SELECTED_SUBMODEL_PARAMS_FILE = files(f"models.{MODEL_NAME}").joinpath("selected_submodel.json")

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier 
      target_horizon: string in {"19", "26"}
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args)

def get_submodel_name(fit_intercept=True, years=20, margin_in_days=None, 
                      days=1, loss="mse", first_lead=0, last_lead = 29):
    """
    Returns submodel name for a given setting of model parameters
    """
    submodel_name = f"{MODEL_NAME}-debias{fit_intercept}_years{years}_margin{margin_in_days}_days{days}_leads{first_lead}-{last_lead}_loss{loss}"
    return submodel_name