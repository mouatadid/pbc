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
# Generates dataset of climatology ranked probability scores (RPS) based on ground 
# truth ERA5 CDF indicators and saves in Zarr format.
#
# Args:
#     measurement (str): The measurement variable to process (e.g., "tas", "pr", "mslp");
#       defaults to "tas" if not provided.
#
# Example usage: python src/data/datasets/climatology-rps.py tas
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

# Sum squared error between each ground truth indicator fk and the climatology
# prediction
clim_pred = 0.2
var_name = "climatology-rps"
cdf_levels = [f'f{i}_{measurement}' for i in range(1, 5)]
rps = None
for label in enumerate(cdf_levels):
    print(f"Loading {label} data...")
    fk = load_data(f"era5-{label}").load()
    # Rename variable
    fk = fk.rename({label: var_name})
    if rps is null:
        rps = (fk - clim_pred) ** 2
    else:
        rps += (fk - clim_pred) ** 2


# %%
# Load ERA5 data and quintiles
print(f"Loading ERA5 data and quintiles for measurement: {measurement}")
era5 = load_data(f"era5-{measurement}")
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
cdf_indicators = var_era5_expanded < var_quantiles
# Compute cumulative maximum across bins
tic()
cdf_cumulative = cdf_indicators.cumsum(dim="quantile").clip(max=1).astype("float32")
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

