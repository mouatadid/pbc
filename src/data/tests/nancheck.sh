#!/bin/bash
#
# Check for NaN values in NetCDF files in a specified subdirectory of data/ecmwf
#
# Example usage:
#   src/data/tests/nancheck.sh ecmwf_cy49-reforecast-tas

# Base directory containing subdirectories
base_directory="data/ecmwf"
dir=$base_directory/$1

# Loop over each subdirectory in the base directory
if [[ -d "$dir" ]]; then
  echo "Processing directory: $dir"
  
  # Loop over each .nc file in the current directory
  for file in "$dir"/*.nc; do
    if [[ -f "$file" ]]; then
      echo "  Checking file: $file"
      ncks --chk_nan "$file"
    else
      echo "  No .nc files found in $dir"
    fi
  done
fi
