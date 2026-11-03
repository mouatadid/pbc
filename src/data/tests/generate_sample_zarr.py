#!/usr/bin/env python
"""Generate lightweight synthetic .zarr datasets in `data/` for testing the PBC pipeline.

Creates minimal valid Zarr stores matching the exact schema expected by:
  - Debias++-ECMWF (`src/models/ecmwfpp/`)
  - Tuner (`src/models/tuner/`)
  - Projector (`src/models/projector/`)
  - Persistence++-ECMWF (`src/models/perpp_ecmwf/`)
  - PBC-ECMWF (`src/models/pbc_ecmwf/`)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from utils.data_io import save_to_zarr


def generate_sample_zarr(data_dir: str = "data", var: str = "mslp") -> None:
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Small spatial grid (16x16 instead of 121x240) for fast local smoke testing
    lats = np.linspace(-90.0, 90.0, 16, dtype=np.float32)
    lons = np.linspace(0.0, 345.0, 16, dtype=np.float32)

    # Daily ground-truth dates covering lagged features + prior-year window + test dates
    gt_times = pd.date_range("2022-10-01", "2024-01-15", freq="D")

    # Issuance dates for the 3 ECMWF model cycles (concatenated along issuance_date by open_mfdataset)
    issuance_splits = {
        "cy41-47": pd.date_range("2022-11-01", "2023-01-15", freq="D"),
        "cy48": pd.date_range("2023-10-01", "2023-11-30", freq="D"),
        "cy49": pd.date_range("2023-12-01", "2024-01-05", freq="D"),
    }
    leads = pd.timedelta_range("0 days", "29 days", freq="1D")
    year_deltas = np.array([-2, -1], dtype=np.int64)

    # Generate land-sea mask (lsmask.zarr) in case tas/pr is also tested
    lsm_da = xr.DataArray(
        np.ones((len(lats), len(lons)), dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": lats, "longitude": lons},
        name="lsm",
    )
    save_to_zarr(lsm_da.to_dataset(name="lsm"), base_dir / "lsmask.zarr", mode="w")

    for k in range(1, 5):
        var_name = f"f{k}_{var}"
        nominal_prob = k * 0.2

        # 1. Ground truth: data/era5-f{k}_{var}.zarr
        gt_vals = np.clip(
            nominal_prob + rng.normal(0.0, 0.08, size=(len(gt_times), len(lats), len(lons))),
            0.0,
            1.0,
        ).astype(np.float32)
        gt_ds = xr.Dataset(
            {var_name: (("time", "latitude", "longitude"), gt_vals)},
            coords={"time": gt_times, "latitude": lats, "longitude": lons},
        )
        save_to_zarr(gt_ds, base_dir / f"era5-{var_name}.zarr", mode="w")

        # 2. ECMWF forecasts & reforecasts across cy41-47, cy48, cy49
        for model_cycle, iss_dates in issuance_splits.items():
            # Forecast: (issuance_date, lead, latitude, longitude)
            fc_vals = np.clip(
                nominal_prob
                + rng.normal(0.0, 0.08, size=(len(iss_dates), len(leads), len(lats), len(lons))),
                0.0,
                1.0,
            ).astype(np.float32)
            fc_ds = xr.Dataset(
                {
                    var_name: (
                        ("issuance_date", "lead", "latitude", "longitude"),
                        fc_vals,
                    )
                },
                coords={
                    "issuance_date": iss_dates,
                    "lead": leads,
                    "latitude": lats,
                    "longitude": lons,
                },
            )
            save_to_zarr(
                fc_ds,
                base_dir / f"ecmwf_{model_cycle}-forecast-{var_name}.zarr",
                mode="w",
            )

            # Reforecast: (issuance_date, year_delta, lead, latitude, longitude)
            refc_vals = np.clip(
                nominal_prob
                + rng.normal(
                    0.0,
                    0.08,
                    size=(
                        len(iss_dates),
                        len(year_deltas),
                        len(leads),
                        len(lats),
                        len(lons),
                    ),
                ),
                0.0,
                1.0,
            ).astype(np.float32)
            refc_ds = xr.Dataset(
                {
                    var_name: (
                        (
                            "issuance_date",
                            "year_delta",
                            "lead",
                            "latitude",
                            "longitude",
                        ),
                        refc_vals,
                    )
                },
                coords={
                    "issuance_date": iss_dates,
                    "year_delta": year_deltas,
                    "lead": leads,
                    "latitude": lats,
                    "longitude": lons,
                },
            )
            save_to_zarr(
                refc_ds,
                base_dir / f"ecmwf_{model_cycle}-reforecast-{var_name}.zarr",
                mode="w",
            )

    print(f"Successfully generated sample .zarr datasets in {base_dir.resolve()}/")


if __name__ == "__main__":
    generate_sample_zarr()
