
from AI_WQ_package import retrieve_evaluation_data
from pathlib import Path
import logging
import os

from data.helpers.config import load_config
from utils.data_io import get_zarr_store_path, save_to_zarr, check_zarr_exists

logger = logging.getLogger(__name__)

CONFIG = load_config()
BASE_DIR = Path(CONFIG['data_dir'])
SOURCE_CONFIG = CONFIG['sources']['aiwq']
PASSWORD = SOURCE_CONFIG['password']


def initial_download(start_date: str = None, end_date: str = None, force: bool = False) -> None:
    """Performs an initial download of the land sea mask."""

    logger.info("===== Starting Land Sea Mask Download =====")
    try:
        store_path = get_zarr_store_path(BASE_DIR, "lsmask")
        
        # Check if dataset already exists
        if check_zarr_exists(store_path) and not force:
            logger.error(f"Dataset {store_path} already exists. Use --force to overwrite.")
            return

        logger.info(f"Downloading land sea mask to {store_path}")
        land_sea_mask = retrieve_evaluation_data.retrieve_land_sea_mask(password=PASSWORD, local_destination="data/")

        save_to_zarr(land_sea_mask, store_path)
        land_sea_mask.close()

        logger.info("Deleting temporary file for land sea mask data.")
        os.remove('data/land_sea_mask_1pt5DEG.nc')
            
    except Exception as e:
        logger.error(f"Failed download. Error: {e}", exc_info=True)


    logger.info("===== Finished Land Sea Mask Download =====")


def update(start_date: str = None, end_date: str = None, force: bool = False) -> None:
    logger.error("Land sea mask does not require updates. Use initial_download to download the dataset.")
    return 
