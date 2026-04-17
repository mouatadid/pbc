#!/bin/bash
# 
# Run degeneracy.py over all variables, ecmwf models, and forecast types
#
# Example usage:
#   src/data/tests/alldegeneracy.sh

# Make output directory for storing log files if it does not yet exist
output_dir="tests/degeneracy"
mkdir -p $output_dir

# Run all jobs in the background and save output and error logs
for var in "tas" "pr" "mslp"; do
  for model in "ecmwf_cy49"; do
    for forecast_type in "forecast" "reforecast"; do
      echo "Processing variable: $var, model: $model, forecast type: $forecast_type"
      python src/data/tests/degeneracy.py -v $var -m $model -f $forecast_type > $output_dir/$model-$forecast_type-$var.out 2> $output_dir/$model-$forecast_type-$var.err &
    done
  done
done