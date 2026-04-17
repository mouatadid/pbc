# Check deterministic ecmwf NetCDF files for ensemble degeneracy.
#
# Example usage:
#   python src/data/tests/degeneracy.py -v tas -m ecmwf_cy41-47 -f reforecast

import xarray as xr
import os
import argparse
import sys

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model name (e.g., ecmwf_cy41-47)")
parser.add_argument("-f", "--forecast_type", required=True, help="Forecast type (e.g., forecast or reforecast)")
parser.add_argument("-v", "--measurement_variable", required=True, choices=["tas", "mslp", "pr"], help="Measurement variable (e.g., tas, mslp, pr)")
args = parser.parse_args()

# Assign arguments to variables
model = args.model
forecast_type = args.forecast_type
measurement_variable = args.measurement_variable
dir = f"data/ecmwf/{model}-{forecast_type}-{measurement_variable}"
print(f"Searching for ensemble degeneracy in directory: {dir}", flush=True)

# Iterate over every perturbed nc file in the directory
name_map = {"tas": "2t", "mslp": "msl", "pr": "tp"}
forecast_var = name_map[measurement_variable]
for filename in os.listdir(dir):
    if filename.endswith("perturbed.nc"):
        print(f"Processing file: {filename}", flush=True)
        filepath = os.path.join(dir, filename)
        if forecast_type == "reforecast":
            ds = xr.open_dataset(filepath, decode_times=False)
        else:
            ds = xr.open_dataset(filepath, decode_timedelta=True)
        ds19 = ds.sel(lead=ds.lead[19])[forecast_var].load()
        if (ds19.max(dim="M") == ds19.min(dim="M")).all():
            print(f"Warning: All ensemble members are identical in {filename}.", 
                  flush=True, file=sys.stderr)
print(f"Finished!", flush=True)