# Model attributes
from importlib import import_module
BASE_MODEL_NAME = "tuned_ecmwfpp"

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: string measurement variable, e.g. "f1_tas", "f1_pr", "f1_mslp"
      target_horizon: string in {"34w", "56w"}
    """
    # Derive submodel name from base model's selected submodel name
    attr = import_module(f"models.{BASE_MODEL_NAME}.attributes")
    return f"proj_{attr.get_selected_submodel_name(gt_id, target_horizon)}"
