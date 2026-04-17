# Model attributes
import json, os, re
from importlib.resources import files
# Follow the submodel naming of ecmwfpp
from models.ecmwfpp.attributes import get_submodel_name as ecmwfpp_submodel_name

MODEL_NAME="ecmwf"
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

def get_submodel_name(first_lead=1, last_lead=1):
    return ecmwfpp_submodel_name(
        fit_intercept=False, years=20, margin_in_days=0, 
        days=1, loss="mse", first_lead=first_lead, 
        last_lead=last_lead)