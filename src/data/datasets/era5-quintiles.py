# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: aiwqd
#     language: python
#     name: python3
# ---

# %%
# Generate and save climatological quintiles from ERA5 data
#
# Args:
#     measurement (str): The measurement variable to process (e.g., "tas", "pr", "mslp");
#       defaults to "tas" if not provided.
#
# Example usage: python src/data/datasets/era5-quintiles.py tas
# Note: top reported 31.4% memory usage for tas when run on a node with 128GB of RAM.
# Note: ran to completion for tas with 120GB and 8 cores and for mslp with 132GB and 8 cores
from AI_WQ_package.compute_20yr_quintile_climatology import complete_20yr_quintiles
from utils.data_io import load_data, save_data
import sys

# %%
# Select measurement variable
if len(sys.argv) > 1:
    measurement = sys.argv[1]
else:
    # Default measurement if not provided
    measurement = "tas"

# %%
# Load ERA5 data
print(f"Loading ERA5 data for measurement: {measurement}")
era5 = load_data(f"era5-{measurement}").load()
print(f"Loading AIWQ data for measurement: {measurement}")
aiwq = load_data(f"aiwq-{measurement}").load()
print(f"Using AIWQ data when available and ERA5 otherwise.")
era5 = aiwq.combine_first(era5)
del aiwq

# %%
# Compute climatological quintiles
# Use 'none' for rolling_operation as data is already aggregated
print(f"Computing climatological quintiles for {measurement}...")
era5_quintiles = complete_20yr_quintiles(era5, rolling_operation='none')

# %%
# Make measurement variable float32 to save space
# Chunk data by quantile and latitude; leave other dimensions unchunked
print(f"Converting {measurement} data to float32 and chunking...")
era5_quintiles = era5_quintiles.astype('float32').chunk({"quantile": 1, "latitude": 1, "longitude": -1, "time": -1})

# %%
# Save the quintiles dataset
print(f"Saving quintiles dataset for {measurement}...")
save_data(era5_quintiles, f"era5-quintiles-{measurement}")
