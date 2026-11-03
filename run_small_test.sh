#!/usr/bin/env bash
# End-to-end smoke test reproducing the README forecasting pipeline on a small sample dataset.
# Usage:
#   ./run_small_test.sh

set -euo pipefail

# Ensure we run from the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "0. Generating sample .zarr datasets in data/"
echo "============================================================"
python src/data/tests/generate_sample_zarr.py

TUNE_DATES="20231201,20231205,20231210"
TEST_DATE="20240110"

echo ""
echo "============================================================"
echo "1. Climatology Baseline"
echo "============================================================"
python src/models/climatology/batch_predict.py era5-f1_mslp 19 -t "${TEST_DATE}"

echo ""
echo "============================================================"
echo "2. Debias++-ECMWF (submodels -> metrics -> tuner -> projector)"
echo "============================================================"
for k in 1 2 3 4; do
  python src/models/ecmwfpp/bulk_batch_predict.py "era5-f${k}_mslp" 19 -t "${TUNE_DATES},${TEST_DATE}" -wm -o
  python src/models/tuner/batch_predict.py "era5-f${k}_mslp" 19 -mn ecmwfpp -t "${TEST_DATE}"
done
python src/models/projector/batch_predict.py era5-mslp 19 -mn tuned_ecmwfpp -t "${TEST_DATE}" -o

echo ""
echo "============================================================"
echo "3. Persistence++-ECMWF (rolling regression -> projector)"
echo "============================================================"
for k in 1 2 3 4; do
  python src/models/perpp_ecmwf/bulk_batch_predict.py "era5-f${k}_mslp" 19 -t "${TEST_DATE}"
done
python src/models/projector/batch_predict.py era5-mslp 19 -mn perpp_ecmwf -t "${TEST_DATE}" -o

echo ""
echo "============================================================"
echo "4. PBC-ECMWF (ensemble Debias++ & Persistence++ -> projector)"
echo "============================================================"
for k in 1 2 3 4; do
  python src/models/pbc_ecmwf/bulk_batch_predict.py "era5-f${k}_mslp" 19 -t "${TEST_DATE}" -o
done
python src/models/projector/batch_predict.py era5-mslp 19 -mn pbc_ecmwf -t "${TEST_DATE}" -o

echo ""
echo "============================================================"
echo "5. Generated Forecast NetCDF Files"
echo "============================================================"
ls -lh \
  models/climatology/submodel_forecasts/climatology/era5-f1_mslp_19/*"${TEST_DATE}".nc \
  models/proj_tuned_ecmwfpp/submodel_forecasts/proj_tuned_ecmwfpp_on_years3_marginNone/era5-f*_mslp_19/*"${TEST_DATE}".nc \
  models/proj_perpp_ecmwf/submodel_forecasts/proj_perpp_ecmwf-yearsall_marginNone_clim20/era5-f*_mslp_19/*"${TEST_DATE}".nc \
  models/proj_pbc_ecmwf/submodel_forecasts/proj_pbc_ecmwf-yearsall_marginNone_equal/era5-f*_mslp_19/*"${TEST_DATE}".nc

echo ""
echo "All pipeline steps completed successfully!"
