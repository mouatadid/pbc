# Model attributes
import json, os, re
from importlib.resources import files

MODEL_NAME="fuxi"

def get_selected_submodel_name(gt_id, target_horizon): 
    """Returns name of selected submodel
    
    Fuxi submodel is always "fuxi"
    """
    return MODEL_NAME

def get_submodel_name(gt_id, target_horizon): 
    """Returns name of submodel

    Fuxi submodel is always "fuxi"
    """
    return MODEL_NAME