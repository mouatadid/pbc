MODEL_NAME = "duet"

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"global_tmp2m_1.5x1.5", "global_precip_1.5x1.5"}
      target_horizon: string in {"34w", "56w"}
    """
    return MODEL_NAME

def get_submodel_name():
    """
    Duet submodel name is always duet
    """
    return MODEL_NAME