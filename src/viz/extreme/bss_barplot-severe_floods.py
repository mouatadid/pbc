"""
Brier Skill Score bar plots for GDACS flood events.

Reads flood events from a GDACS JSON file and computes Brier Skill Scores
(BSS) for each event date over the event's bounding-box region (lat/lon
from the JSON).  Results are aggregated by severity and horizon.

The Brier score (BS) evaluates the cosine-latitude weighted mean squared error 
when predicting the binary extreme event of precipitation exceeding the input 
climatological percentile (default 95th percentile) in the bounding box 
surrounding a flood event (with a land-sea mask applied). 

BSS is defined relative to a climatological forecast that always predicts
the nominal probability of the upper quantile bin:
    BSS_severity = 1 − mean(BS_model) / mean(BS_clim)

By default only the std_test dates (Mondays & Fridays, 2016-2024) are used.
Pass --all_dates to use every event date regardless of the std_test filter.

Example usage:
    python src/viz/extreme/bss_barplot-severe_floods.py --all_dates
"""
import argparse
import json
import os
import sys

from utils.file_io import set_file_permissions, make_directories
from utils.logging import printf
from utils.timing import tic, toc
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from arch.bootstrap import StationaryBootstrap, optimal_block_length, IIDBootstrap

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

MODEL_LABELS = ["Raw ECMWF", "Debiased ECMWF", "PBC-ECMWF"]
MODEL_COLORS = {"Raw ECMWF": "purple", "Debiased ECMWF": "plum", "PBC-ECMWF": "gold"}

MODEL_CONFIGS = {
    ECMWF_MODEL: {"label": "Raw ECMWF", "color": "purple", "get_submodel": get_raw_ecmwf_model},
    ECMWF_DEB_MODEL: {"label": "Debiased ECMWF", "color": "plum", "get_submodel": get_ecmwf_model},
    PBC_PRECIP_MODEL: {"label": "PBC-ECMWF", "color": "gold", "get_submodel": get_pbc_precip_model},
}

DEFAULT_CACHE_DIR = Path("eval/viz/bss-barplot-floods/data")
DEFAULT_EVENTS_JSON = Path("eval/viz/bss-barplot-floods/data/flood-events-gdacs-2016_2026.json")
OUTPUT_DIR = Path("viz/pbc/extremes")

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
        help="Force recomputation even if a cached Brier score file exists.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Directory for storing Brier score cache JSON files.",
    )
    parser.add_argument(
        "--percentile", type=int, default=95,
        help="Percentile defining extreme events.",
    )
    args = parser.parse_args()
    use_all_dates: bool = args.all_dates
    percentile: int = args.percentile

    suffix = "-all_dates" if use_all_dates else ""
    date_label = "all days" if use_all_dates else "std_test"
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir: Path = args.cache_dir
    # Make cache directory with full permissions
    make_directories(cache_dir)

    # Try to load cached Brier scores
    suffix = "-all_dates" if use_all_dates else ""
    cache_file = cache_dir / f"F{percentile}_brier_severity_cache{suffix}.json"
    if not args.recompute and cache_file.exists():
        printf(f"Loading cached Brier scores from {cache_file}")
        severity_brier_raw = _load_cache(cache_file)
    else:
        events = load_events(args.events_json)
        printf(f"Loaded {len(events)} flood events from {args.events_json}")

        printf("Computing global Brier scores for GDACS flood events …")
        severity_brier_raw = collect_severity_brier(
            events, MODEL_CONFIGS, percentile,
            use_all_dates=use_all_dates,
        )
        _save_cache(severity_brier_raw, cache_file)

    # Reorganize floods into All and Severe groups
    grouped_brier_raw: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for sev in severity_brier_raw.keys():
        for model, horizons in severity_brier_raw.get(sev, {}).items():
            for horizon, values in horizons.items():
                grouped_brier_raw["All"][model][horizon].extend(values)
                # Non-green floods are considered Severe
                if sev != "Green":
                    grouped_brier_raw["Severe"][model][horizon].extend(values)

    save_severity_bss_barplot(
        grouped_brier_raw,
        output_dir / f"barplot_F{percentile}_bss_floods_severity{suffix}.pdf",
    )

    printf("Done.")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _save_cache(
    all_severities: Dict[str, Dict[str, Dict[str, List[float]]]],
    path: Path,
) -> None:
    """Persist severity Brier score values to a JSON file."""
    # JSON keys must be strings; convert int severities.
    serialisable = {
        severity: {
            model: {h: vals for h, vals in horizons.items()}
            for model, horizons in models.items()
        }
        for severity, models in all_severities.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    set_file_permissions(path)
    printf(f"  Cached Brier score values → {path}")


def _load_cache(
    path: Path,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Load severity Brier score values from a previously saved JSON cache."""
    with open(path) as fh:
        data = json.load(fh)
    # Convert string severity keys back to original format, values back to plain lists.
    return {
        severity: {
            model: {h: [float(v) for v in vals] for h, vals in horizons.items()}
            for model, horizons in models.items()
        }
        for severity, models in data.items()
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


def collect_severity_brier(
    events: Dict[str, dict],
    model_configs: Dict,
    percentile: int,
    *,
    use_all_dates: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Compute per-event cosine-latitude-weighted Brier scores values by severity.

    Returns
    -------
    all_severities : {severity: {model_label: {horizon: [brier_score_per_event]}}}
    """
    # When not using all dates, restrict to the std_test set
    if not use_all_dates:
        std_test_dates = set(pd.Timestamp(d) for d in get_target_dates("std_test"))
    else:
        std_test_dates = None

    raw: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    obs_cache: Dict[str, xr.DataArray] = {}

    n_events = len(events)
    gt_id = f"era5-F{percentile}_pr"
    # Since we load CDF forecasts, climatology prediction = percentile/100.0
    quantile = percentile / 100.0
    tic()
    for idx, (event_name, info) in enumerate(events.items()):
        target_date = pd.Timestamp(info["date"])

        # std_test filter
        if std_test_dates is not None and target_date not in std_test_dates:
            continue

        lat_bounds = tuple(info["lat"])
        lon_bounds = tuple(info["lon"])

        # Standardize to title case (e.g. "Red", "Orange", "Green")
        severity = info["alertlevel"].title()

        date = info["date"]

        for horizon in HORIZONS:
            # Load forecasts + obs for all models for this date/horizon
            predicted_probs = {}
            obs_binary = None
            weights = None
            skip = False

            for model_name, cfg in model_configs.items():
                try:
                    pred, obs = _load_forecast_and_obs(
                        model_name, cfg["get_submodel"], gt_id,
                        horizon, target_date, obs_cache,
                        lat_bounds, lon_bounds,
                    )
                    predicted_probs[cfg["label"]] = pred
                    if obs_binary is None:
                        obs_binary = obs
                        weights = _area_weights(obs)
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    printf(f"  {event_name}, {date} | {cfg['label']} {horizon}d → SKIPPED ({exc})")
                    skip = True
                    break

            # Skip events with missing forecasts or observations
            if skip or (obs_binary is None):
                continue

            # Climatology Brier score for this date
            clim_bs = _weighted_sum((quantile - obs_binary) ** 2, weights)
            if clim_bs == 0.0:
                continue  # degenerate day, skip
            raw[severity]["Climatology"][horizon].append(clim_bs)
            printf(f"  {event_name}, {date} | Climatology {horizon}d → BS {clim_bs:.4f}")

            for label, pred in predicted_probs.items():
                model_bs = _weighted_sum((pred - obs_binary) ** 2, weights)
                raw[severity][label][horizon].append(model_bs)
                printf(f"  {event_name}, {date} | {label} {horizon}d → BS {model_bs:.4f}")

        # Progress logging
        if (idx + 1) % 50 == 0 or (idx + 1) == n_events:
            printf(f"    processed {idx + 1}/{n_events} events"); toc(); tic()

    return dict(raw)


def _load_forecast_and_obs(
    model_name: str,
    get_submodel_fn,
    gt_id: str,
    horizon: str,
    target_date: pd.Timestamp,
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
    measurement_var = get_measurement_variable(gt_id)

    # Forecast CDF
    submodel_key = gt_id.split("-", 1)[1]
    submodel_name = get_submodel_fn(submodel_key, horizon)
    forecast_dir = Path(get_task_forecast_dir(
        model=model_name,
        submodel=submodel_name,
        gt_id=gt_id,
        horizon=horizon,
    ))
    forecast_path = forecast_dir / f"{gt_id}_{horizon}-{target_date:%Y%m%d}.nc"
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing forecast: {forecast_path}")

    with xr.open_dataset(forecast_path) as ds:
        fcst_cdf = ds[measurement_var].sel(
            time=np.datetime64(target_date),
        ).squeeze(drop=True).load()

    # Observed CDF (with land-sea mask, cached)
    cache_key = f"{gt_id}_{target_date:%Y%m%d}"
    if cache_key in obs_cache:
        obs_cdf = obs_cache[cache_key]
    else:
        gt_ds = load_data(gt_id, lsmask=True)
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

    return fcst_cdf, obs_cdf_aligned


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

def save_severity_bss_barplot(
    all_severities: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_path: Path,
    *,
    n_boot: int = 5000,
    random_seed: int = 42,
) -> None:
    """Create and save a bar plot of flood forecasting BSS for MODEL_LABELS. 
    Bar groups are grouped by severity and horizon.
    """
    severity_keys = sorted(all_severities.keys())
    if not severity_keys:
        printf(f"  No severity flood data to plot.")
        return

    rng = np.random.default_rng(random_seed)
    width = 0.30

    n_sev = len(severity_keys)
    n_horiz = len(HORIZONS)
    n_groups = n_sev * n_horiz
    
    scale_factor = 1.05
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_groups) * scale_factor, 6 * scale_factor))
    x_positions = np.arange(0, 1.5 * n_groups, 1.5)

    # Collect BSS values for each model
    bss = np.empty((len(severity_keys), len(HORIZONS), len(MODEL_LABELS)))

    # Use bias-corrected and accelerated (BCa) confidence intervals to test
    # for significant improvements
    confidence = 0.95; tail = 'lower'; method = 'bca'
    ci_low = np.full((len(severity_keys), len(HORIZONS), len(MODEL_LABELS)), np.inf)
    for jj, sev in enumerate(severity_keys):
        severity_data = all_severities.get(sev, {})
        for kk, horizon in enumerate(HORIZONS):
            # Identify the total number of events
            clim = np.asarray(severity_data.get("Climatology", {}).get(horizon, []))
            clim = clim[~np.isnan(clim)]
            n = len(clim)
            printf(f"  Severity {sev}, horizon {horizon}d: {n} target dates")

            pbc_data = np.asarray(severity_data.get("PBC-ECMWF", {}).get(horizon, []))
            pbc_data = pbc_data[~np.isnan(pbc_data)]
            inds = np.arange(n)
            for ii, label in enumerate(MODEL_LABELS):
                # Compute BSS for this severity, horizon, and model
                data = np.asarray(severity_data.get(label, {}).get(horizon, []))
                data = data[~np.isnan(data)]
                if len(data) > 0 and len(clim) > 0:
                    bss[jj, kk, ii] = 1 - (np.mean(data) / np.mean(clim))
                else:
                    bss[jj, kk, ii] = np.nan
                if label == "PBC-ECMWF":
                    continue
                # Compute BCa lower confidence bound for BSS difference between PBC and this model
                def ss_diff(inds):
                    return (data[inds].mean() - pbc_data[inds].mean())/clim[inds].mean()
                boot = IIDBootstrap(inds, seed=random_seed)
                ci_low[jj, kk, ii] = boot.conf_int(ss_diff, reps=n_boot, method=method, size=confidence, tail = tail)[0][0]
                printf(f"  {sev} | {horizon}d | PBC-ECMWF - {label} BSS {int(confidence * 100)}% {method} {tail} confidence bound: {ci_low[jj, kk, ii]}")

    # Store all centers for improvement computation
    all_centers = {}
    for ii, label in enumerate(MODEL_LABELS):
        centers = []
        hatch = []

        for jj, sev in enumerate(severity_keys):
            for kk, horizon in enumerate(HORIZONS):
                centers.append(bss[jj, kk, ii])
                # Store per-(severity, horizon) for improvement reporting
                all_centers[(sev, horizon, label)] = bss[jj, kk, ii]
                # Use hatch pattern if improvement is significant, i.e., if ci_low > 0
                hatch.append('x' if label == "PBC-ECMWF" and ci_low[jj, kk, :].min() > 0 else None)

        offset = (ii - (len(MODEL_LABELS) - 1) / 2) * width
        # First draw plot without hatch to get hatch-free legend entries
        # Then draw bars again with hatch but without legend entries
        for cc in range(2):
            ax.bar(
                x_positions + offset, centers,
                width=width,
                capsize=4,
                color=MODEL_COLORS[label],
                hatch=None if cc == 0 else hatch,
                label=label if cc == 0 else "_nolegend_",
                zorder=3,
            )

    # Report improvements per severity and horizon
    printf("\n% Improvements of PBC-ECMWF over baselines:")
    printf("-" * 80)
    for sev in severity_keys:
        for horizon in HORIZONS:
            week = "Week 3" if horizon == "19" else "Week 4"
            pbc_bss = all_centers.get((sev, horizon, "PBC-ECMWF"), np.nan)
            raw_bss = all_centers.get((sev, horizon, "Raw ECMWF"), np.nan)
            deb_bss = all_centers.get((sev, horizon, "Debiased ECMWF"), np.nan)
            
            if not np.isnan(pbc_bss) and not np.isnan(raw_bss):
                imp_raw = 100 * (pbc_bss - raw_bss) / raw_bss
            else:
                imp_raw = np.nan
                
            if not np.isnan(pbc_bss) and not np.isnan(deb_bss):
                imp_deb = 100 * (pbc_bss - deb_bss) / deb_bss
            else:
                imp_deb = np.nan
            
            printf(f"  {sev} ({week}): vs Raw ECMWF ({raw_bss}): {imp_raw:+.2f}% | vs Debiased ECMWF ({deb_bss}): {imp_deb:+.2f}%")

    # Set x-ticks to horizon labels within each severity group
    week_labels = ["Week 3", "Week 4"]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(week_labels * n_sev, fontsize=20)

    # Add severity labels centered under each group
    severity_names_display = {
        "All": "All Floods",
        "Severe": "Severe Floods",
    }
    sev_names = [severity_names_display.get(s, s) for s in severity_keys]
    for s in range(n_sev):
        # Calculate center based on actual x_positions for this severity
        severity_positions = x_positions[s * n_horiz:(s + 1) * n_horiz]
        center = (severity_positions[0] + severity_positions[-1]) / 2
        ax.text(
            center,
            -0.11,
            sev_names[s],
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylabel("Brier skill score", fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)

    ax.legend(
        ncol=len(MODEL_LABELS),
        fontsize=20,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
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
    printf(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
