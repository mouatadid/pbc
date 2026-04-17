MODEL_NAME = "pbc_ecmwf_combo"

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier 
      target_horizon: string in {"19", "26"}
    """
    return MODEL_NAME

def get_submodel_name():
    return MODEL_NAME