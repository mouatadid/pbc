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
"""
Generates CDF indicator datasets (f1, f2, f3, f4) by comparing forecasted ERA5 values to corresponding climatological quintiles.
Saves each CDF level as a separate Zarr file for each variable.

Args:
    measurement (str): The measurement variable to process (e.g., "tas", "pr", "mslp");
      defaults to "tas" if not provided.

Example usages: 
  python src/data/datasets/era5-cdf.py tas
  for var in tas pr mslp; do 
    python src/data/datasets/era5-cdf.py $var
  done
"""
import sys
import xarray as xr
import numpy as np
from utils.data_io import load_data, save_data
from utils.timing import tic, toc

# %%
# Select measurement variable
if len(sys.argv) > 1:
    measurement = sys.argv[1]
else:
    # Default measurement if not provided
    measurement = "tas"


# %%
# Load ERA5 data and quintiles
print(f"Loading ERA5 and AIWQ data for measurement: {measurement}")
era5 = load_data(f"era5-{measurement}").load()
aiwq = load_data(f"aiwq-{measurement}").load()
print(f"Using AIWQ data when available and ERA5 otherwise.")
era5 = aiwq.combine_first(era5)
del aiwq

print(f"Loading ERA5 quintiles for measurement: {measurement}")
quintiles = load_data(f"era5-quintiles-{measurement}")

# %%
# Get dates overlap between ERA5 data and quintiles
common_times = np.intersect1d(era5.time.values, quintiles.time.values)
era5_sel = era5.sel(time=common_times)
quintiles_sel = quintiles.sel(time=common_times)

# %%
# Broadcast to match shapes
tic()
var_era5 = era5_sel[measurement]
var_era5_expanded = var_era5.expand_dims(dim={"quantile": quintiles.sizes["quantile"]}).load()
var_quantiles = quintiles_sel[measurement].load()
toc()

# %%
# Compare forecast to quantiles and compute indicators
print(f"Computing CDF indicators for {measurement}...")
tic()
cdf_cumulative = (var_era5_expanded < var_quantiles).astype("float32")
toc()


# %%
# Extract each cumulative indicator (f1 to f4) and save datasets
cdf_levels = [f'f{i}_{measurement}' for i in range(1, 5)]
for i, label in enumerate(cdf_levels):
    fk = cdf_cumulative.isel(quantile=i)
    fk = fk.drop_vars('quantile')
    ds = xr.Dataset({label: fk})
    print(f"Rechunking {label}...")
    ds = ds.chunk({"latitude": 1, "longitude": -1, "time": -1})
    print(f"Saving {label}...")
    tic()
    save_data(ds, f"era5-{label}")
    toc()

