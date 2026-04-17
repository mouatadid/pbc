# Model attributes
import json
import os
from pkg_resources import resource_filename


MODEL_NAME = "perpp_debias_aifs"
SELECTED_SUBMODEL_PARAMS_FILE = os.path.join("src", "models", MODEL_NAME, "selected_submodel.json")


def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      measurement_variable: e.g. f1_tas, f1_pr, f1_mslp
      target_horizon: string in {"19", "26"}
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args)


def get_submodel_name(train_years="all", margin_in_days=None, clim_years=20):
    """Returns submodel name for a given setting of model parameters
    """
    submodel_name = f"{MODEL_NAME}-years{train_years}_margin{margin_in_days}_clim{clim_years}"

    return submodel_name
