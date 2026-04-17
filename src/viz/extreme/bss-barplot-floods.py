"""
Yearly Brier Skill Score bar plots for GDACS flood events.

Reads flood events from a GDACS JSON file and computes Brier Skill Scores
(BSS) for each event date over the event's bounding-box region (lat/lon
from the JSON).  Results are aggregated by year and plotted as side-by-side
bars (ECMWF vs Debiased ECMWF), one figure per horizon.

BSS is defined relative to a climatological forecast that always assigns
probability 0.2 to the extreme quintile:
    BSS_year = 1 − mean(BS_model) / mean(BS_clim)

By default only the std_test dates (Mondays & Fridays, 2016-2024) are used.
Pass --all_dates to use every event date regardless of the std_test filter.

Generates two figures:
  - eval/viz/bss-barplot-floods/bss_barplot-floods-19d.pdf
  - eval/viz/bss-barplot-floods/bss_barplot-floods-26d.pdf

The Brier score evaluates the binary extreme-quintile event:
  • flood (precip) → extreme = Q5 (highest quintile)

The land-sea mask is always applied.

Example usage:
    python src/viz/extreme/bss-barplot-floods.py 
"""
import argparse
import json
import os
import sys

from utils.file_io import set_file_permissions
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from models.ecmwf.attributes import (
    MODEL_NAME as ECMWF_MODEL,
    get_selected_submodel_name as get_raw_ecmwf_model,
)
from models.debiased_ecmwf.attributes import (
    MODEL_NAME as ECMWF_DEB_MODEL,
    get_selected_submodel_name as get_ecmwf_model,
)
from models.pbc_ecmwf.attributes import (
    MODEL_NAME as PBC_PRECIP_MODEL,
    get_selected_submodel_name as get_pbc_precip_model,
)
from models.utils.models_util import get_task_forecast_dir
from models.utils.data_utils import get_measurement_variable
from models.utils.eval_util import get_target_dates
from utils.data_io import load_data

HORIZONS = ("19", "26")

PRECIP_GT_ID = "era5-pr"

MIN_YEAR = 2022  # drop years with too few events (e.g. 2021)
MAX_YEAR = 2025  # last year to include

CLIM_PROB = 0.2  # climatology always predicts 0.2 for any quintile

MODEL_LABELS = ["Raw ECMWF", "Debiased ECMWF", "PBC-ECMWF"]
MODEL_COLORS = {"Raw ECMWF": "#75147C", "Debiased ECMWF": "#D4A2D9", "PBC-ECMWF": "#F9D949"}

MODEL_CONFIGS = {
    ECMWF_MODEL: {"label": "Raw ECMWF", "color": "#75147C", "get_submodel": get_raw_ecmwf_model},
    ECMWF_DEB_MODEL: {"label": "Debiased ECMWF", "color": "#D4A2D9", "get_submodel": get_ecmwf_model},
    PBC_PRECIP_MODEL: {"label": "PBC-ECMWF", "color": "#F9D949", "get_submodel": get_pbc_precip_model},
}

DEFAULT_EVENTS_JSON = Path("eval/viz/bss-barplot-floods/data/flood-events-gdacs.json")
DEFAULT_CACHE_FILE = Path("eval/viz/bss-barplot-floods/data/bss_cache.json")
OUTPUT_DIR = Path("eval/viz/bss-barplot-floods")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all_dates", action="store_true",
        help="Use every event date instead of restricting to std_test.",
    )
    parser.add_argument(
        "--events-json", type=Path, default=DEFAULT_EVENTS_JSON,
        help="Path to the GDACS flood-events JSON file.",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recomputation even if a cached BSS file exists.",
    )
    parser.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
        help="Path to the BSS cache JSON file.",
    )
    args = parser.parse_args()
    use_all_dates: bool = args.all_dates

    suffix = "-all_dates" if use_all_dates else ""
    date_label = "all days" if use_all_dates else "std_test"
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file: Path = args.cache_file
    if use_all_dates:
        cache_file = cache_file.with_name(
            cache_file.stem + "-all_dates" + cache_file.suffix
        )

    # Try to load cached BSS values
    if not args.recompute and cache_file.exists():
        print(f"Loading cached BSS values from {cache_file}")
        yearly_bss_raw = _load_cache(cache_file)
    else:
        events = load_events(args.events_json)
        print(f"Loaded {len(events)} flood events from {args.events_json}")

        print("Computing global BSS for GDACS flood events …")
        yearly_bss_raw = collect_yearly_bss(
            events, MODEL_CONFIGS, PRECIP_GT_ID, extreme="high",
            use_all_dates=use_all_dates,
        )
        _save_cache(yearly_bss_raw, cache_file)

    for horizon in HORIZONS:
        save_yearly_bss_barplot(
            yearly_bss_raw,
            output_dir / f"bss_barplot-floods-{horizon}d{suffix}.pdf",
            "Precipitation (Floods)",
            horizon,
            date_label=date_label,
        )

    print("Done.")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _save_cache(
    yearly_raw: Dict[int, Dict[str, Dict[str, List[float]]]],
    path: Path,
) -> None:
    """Persist yearly BSS values to a JSON file."""
    # JSON keys must be strings; convert int years.
    serialisable = {
        str(year): {
            model: {h: vals for h, vals in horizons.items()}
            for model, horizons in models.items()
        }
        for year, models in yearly_raw.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    set_file_permissions(path)
    print(f"  Cached BSS values → {path}")


def _load_cache(
    path: Path,
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """Load yearly BSS values from a previously saved JSON cache."""
    with open(path) as fh:
        data = json.load(fh)
    # Convert string year keys back to ints, values back to plain lists.
    return {
        int(year): {
            model: {h: [float(v) for v in vals] for h, vals in horizons.items()}
            for model, horizons in models.items()
        }
        for year, models in data.items()
    }


def load_events(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the GDACS flood-events JSON produced by download-flood-events.py.

    Expected format::

        {
          "event-slug": {"date": "YYYY-MM-DD", "lat": [lo, hi], "lon": [lo, hi]},
          ...
        }

    The ``date`` field selects which forecast/obs files to load, and the
    ``lat``/``lon`` arrays define the bounding box over which the Brier
    score is computed.
    """
    with open(path) as fh:
        data = json.load(fh)
    return data


def collect_yearly_bss(
    events: Dict[str, dict],
    model_configs: Dict,
    gt_id: str,
    extreme: str,
    *,
    use_all_dates: bool = False,
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """Compute per-event BSS values and aggregate by year.

    For each event date, the BSS is:
        BSS = 1 − BS_model / BS_clim

    where both Brier scores are area-weighted global means.

    Parameters
    ----------
    extreme : {"high", "low"}
        Fixed extreme direction for all events.

    Returns
    -------
    yearly_raw : {year: {model_label: {horizon: [bss_per_event]}}}
    """
    # When not using all dates, restrict to the std_test set
    if not use_all_dates:
        std_test_dates = set(pd.Timestamp(d) for d in get_target_dates("std_test"))
    else:
        std_test_dates = None

    raw: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    obs_cache: Dict[str, xr.DataArray] = {}

    n_events = len(events)
    for idx, (event_name, info) in enumerate(events.items()):
        target_date = pd.Timestamp(info["date"])

        # std_test filter
        if std_test_dates is not None and target_date not in std_test_dates:
            continue

        year = target_date.year

        # Skip years outside [MIN_YEAR, MAX_YEAR]
        if year < MIN_YEAR or year > MAX_YEAR:
            continue

        lat_bounds = tuple(info["lat"])
        lon_bounds = tuple(info["lon"])

        for horizon in HORIZONS:
            # Load forecasts + obs for all models for this date/horizon
            predicted_probs = {}
            obs_binary = None
            weights = None

            for model_name, cfg in model_configs.items():
                try:
                    pred, obs = _load_forecast_and_obs(
                        model_name, cfg["get_submodel"], gt_id,
                        horizon, target_date, extreme, obs_cache,
                        lat_bounds, lon_bounds,
                    )
                    predicted_probs[cfg["label"]] = pred
                    if obs_binary is None:
                        obs_binary = obs
                        weights = _area_weights(obs)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    print(f"  {event_name} | {cfg['label']} {horizon}d → SKIPPED ({exc})")

            if obs_binary is None:
                continue

            # Climatology Brier score for this date
            clim_bs = _weighted_sum((CLIM_PROB - obs_binary) ** 2, weights)
            if clim_bs == 0.0:
                continue  # degenerate day, skip

            for label, pred in predicted_probs.items():
                model_bs = _weighted_sum((pred - obs_binary) ** 2, weights)
                bss = 1.0 - model_bs / clim_bs
                raw[year][label][horizon].append(bss)
                print(f"  {event_name} | {label} {horizon}d → BSS {bss:.4f}")

        # Progress logging
        if (idx + 1) % 50 == 0 or (idx + 1) == n_events:
            print(f"    processed {idx + 1}/{n_events} events")

    return dict(raw)


def _prob_gt_id(gt_id: str, extreme: str) -> str:
    """Map (gt_id, extreme) to the probabilistic gt_id used for file lookup."""
    dataset, variable = gt_id.split("-", 1)
    q_idx = 4 if extreme == "high" else 1
    return f"{dataset}-f{q_idx}_{variable}"


def _load_forecast_and_obs(
    model_name: str,
    get_submodel_fn,
    gt_id: str,
    horizon: str,
    target_date: pd.Timestamp,
    extreme: str,
    obs_cache: Dict[str, xr.DataArray],
    lat_bounds: Tuple[float, float] = None,
    lon_bounds: Tuple[float, float] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load forecast probability and observed binary for one date.

    Returns (predicted_prob, observed_binary), both 2-D DataArrays
    aligned on the same grid.  The land-sea mask is applied to observations.
    When *lat_bounds* / *lon_bounds* are given the arrays are subsetted
    to the event's bounding box before returning.
    """
    prob_gt_id = _prob_gt_id(gt_id, extreme)
    measurement_var = get_measurement_variable(prob_gt_id)

    # Forecast CDF
    submodel_key = prob_gt_id.split("-", 1)[1]
    submodel_name = get_submodel_fn(submodel_key, horizon)
    forecast_dir = Path(get_task_forecast_dir(
        model=model_name,
        submodel=submodel_name,
        gt_id=prob_gt_id,
        horizon=horizon,
    ))
    forecast_path = forecast_dir / f"{prob_gt_id}_{horizon}-{target_date:%Y%m%d}.nc"
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing forecast: {forecast_path}")

    with xr.open_dataset(forecast_path) as ds:
        fcst_cdf = ds[measurement_var].sel(
            time=np.datetime64(target_date),
        ).squeeze(drop=True).load()

    # Observed CDF (with land-sea mask, cached)
    cache_key = f"{prob_gt_id}_{target_date:%Y%m%d}"
    if cache_key in obs_cache:
        obs_cdf = obs_cache[cache_key]
    else:
        gt_ds = load_data(prob_gt_id, lsmask=True)
        obs_cdf = gt_ds[measurement_var].sel(
            time=np.datetime64(target_date),
        ).squeeze(drop=True).load()
        obs_cache[cache_key] = obs_cdf

    # Align grids
    fcst_cdf, obs_cdf_aligned = xr.align(fcst_cdf, obs_cdf, join="inner")

    # Subset to event bounding box
    if lat_bounds is not None and lon_bounds is not None:
        fcst_cdf = _subset_region(fcst_cdf, lat_bounds, lon_bounds)
        obs_cdf_aligned = _subset_region(obs_cdf_aligned, lat_bounds, lon_bounds)

    if extreme == "high":
        predicted_prob = 1.0 - fcst_cdf           # P(Q5) = 1 − F4
        observed_binary = 1.0 - obs_cdf_aligned   # 1 if obs exceeded Q4
    else:
        predicted_prob = fcst_cdf                  # P(Q1) = F1
        observed_binary = obs_cdf_aligned          # 1 if obs below Q1

    return predicted_prob, observed_binary


def _area_weights(da: xr.DataArray) -> xr.DataArray:
    """Return cos(latitude) weights matching *da*'s latitude coordinate."""
    return np.cos(np.deg2rad(da.latitude))


def _weighted_sum(field: xr.DataArray, weights: xr.DataArray) -> float:
    """Latitude-weighted sum over all grid points, ignoring NaNs."""
    w = weights.where(field.notnull())
    return float((field * w).sum(skipna=True).values)


def _subset_region(
    da: xr.DataArray,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
) -> xr.DataArray:
    """Select the spatial subset defined by *lat_bounds* / *lon_bounds*.

    Handles the case where *lon_bounds* straddle 360° (e.g. 358→376) by
    splitting the selection in two, shifting the low-end longitudes by
    +360, and concatenating them so the result is contiguous.
    """
    lat_sel = _slice_for_coord(da.latitude, lat_bounds)
    lo, hi = lon_bounds
    if hi > 360 and float(da.longitude.max()) <= 360:
        part_hi = da.sel(
            latitude=lat_sel,
            longitude=_slice_for_coord(da.longitude, (lo, 360.0)),
        )
        part_lo = da.sel(
            latitude=lat_sel,
            longitude=_slice_for_coord(da.longitude, (0.0, hi - 360.0)),
        )
        part_lo = part_lo.assign_coords(longitude=part_lo.longitude + 360)
        return xr.concat([part_hi, part_lo], dim="longitude")
    return da.sel(
        latitude=lat_sel,
        longitude=_slice_for_coord(da.longitude, lon_bounds),
    )


def _slice_for_coord(
    coord: xr.DataArray, bounds: Tuple[float, float],
) -> slice:
    """Return *slice(lo, hi)* respecting the coordinate's sort order."""
    lo, hi = bounds
    if lo > hi:
        lo, hi = hi, lo
    ascending = float(coord.values[-1]) >= float(coord.values[0])
    return slice(lo, hi) if ascending else slice(hi, lo)


def save_yearly_bss_barplot(
    yearly_raw: Dict[int, Dict[str, Dict[str, List[float]]]],
    output_path: Path,
    variable_label: str,
    horizon: str,
    *,
    date_label: str = "all days",
    n_boot: int = 5000,
    random_seed: int = 42,
) -> None:
    """Create and save a bar plot of yearly mean BSS with bootstrap CIs.

    Two series (ECMWF vs Debiased ECMWF) for a single forecast horizon.
    Style matches ``bss-pbc_plots-per_extreme.py``: red/pink colours,
    bold 20 pt fonts, 95 % bootstrap error bars, legend above the axes.
    """
    years = sorted(yearly_raw.keys())
    if not years:
        print(f"  No data to plot for {variable_label} {horizon}d.")
        return

    rng = np.random.default_rng(random_seed)

    x = np.arange(0, 1.5 * len(years), 1.5)
    width = 0.30

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(years)), 6))

    # Store all means for improvement computation
    all_means = {}
    for i, label in enumerate(MODEL_LABELS):
        means = []
        lower_errors = []
        upper_errors = []

        for y in years:
            data = np.asarray(yearly_raw.get(y, {}).get(label, {}).get(horizon, []))
            data = data[~np.isnan(data)]
            n = len(data)
            print(f"  Year {y}, model {label}, horizon {horizon}d: {n} target dates")

            if n == 0:
                means.append(float("nan"))
                lower_errors.append(0.0)
                upper_errors.append(0.0)
                continue

            mean_val = np.mean(data)

            # Bootstrap resampling for 95 % CI
            boot_means = np.empty(n_boot)
            for b in range(n_boot):
                sample = rng.choice(data, size=n, replace=True)
                boot_means[b] = np.mean(sample)

            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            means.append(mean_val)
            lower_errors.append(mean_val - ci_low)
            upper_errors.append(ci_high - mean_val)

        offset = (i - (len(MODEL_LABELS) - 1) / 2) * width
        ax.bar(
            x + offset, means,
            width=width,
            yerr=[lower_errors, upper_errors],
            capsize=4,
            color=MODEL_COLORS[label],
            label=label,
            zorder=3,
            error_kw=dict(ecolor="gray", elinewidth=1.5, capthick=1.5),
        )

        all_means[label] = np.array(means)

    print("% Improvements over Raw ECMWF:")
    print(100 * (all_means["PBC-ECMWF"] - all_means["Raw ECMWF"])/all_means["Raw ECMWF"])
    print("% Improvements over Debiased ECMWF:")
    print(100 * (all_means["PBC-ECMWF"] - all_means["Debiased ECMWF"])/all_means["Debiased ECMWF"])

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=20)
    ax.set_xlabel("Flood Forecasting", fontsize=20, fontweight="bold")
    ax.set_ylabel("Brier skill score", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)

    ax.legend(
        ncol=len(MODEL_LABELS),
        fontsize=20,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    ax.xaxis.grid(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    set_file_permissions(output_path)
    plt.close()
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
