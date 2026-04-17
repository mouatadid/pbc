# Model attributes
MODEL_NAME="debiased_ecmwf"

def get_selected_submodel_name(gt_id, target_horizon): 
    """Returns name of selected submodel
    
    Debiased ECMWF submodel is always "debiased_ecmwf"
    """
    return MODEL_NAME

def get_submodel_name(gt_id, target_horizon): 
    """Returns name of submodel

    debiased ECMWF submodel is always "debiased_ecmwf"
    """
    return MODEL_NAME