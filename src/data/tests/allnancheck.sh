#!/bin/bash
#
# Run nancheck.sh over all subdirectories of data/ecmwf
#
# Example usage:
#   src/data/tests/allnancheck.sh

# Base directory containing subdirectories
base_directory="data/ecmwf"

# Make output directory for storing log files if it does not yet exist
output_dir="tests/nans"
mkdir -p $output_dir

# Loop over each subdirectory in the base directory
# and store output and error logs
for dir in "$base_directory"/*; do
  dir=$(basename $dir)
  echo $dir
  src/data/tests/nancheck.sh $dir > $output_dir/$dir.out 2> $output_dir/$dir.err &
done
