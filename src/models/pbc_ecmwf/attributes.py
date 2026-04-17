# Model attributes
import json
import os
from pkg_resources import resource_filename


MODEL_NAME = "pbc_ecmwf"
SELECTED_SUBMODEL_PARAMS_FILE = os.path.join('src', 'models', MODEL_NAME,"selected_submodel.json")


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

def get_submodel_name(train_years='all', margin_in_days=None, equal=False, simplex=False):
    """Returns submodel name for a given setting of model parameters
    """
    base_name = f"{MODEL_NAME}-years{train_years}_margin{margin_in_days}"
    if equal:
        base_name += "_equal"
    elif simplex:
        base_name += "_simplex"
    return base_name


