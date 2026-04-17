"""

Generates two figures for a chosen target horizon (week 3 = 19d, or week 4 = 26d):
  - eval/viz/bss-pbc_plots-per_target/bss-bar_plot-week_N.pdf   — grouped bar plot
    of mean BSS per variable × extreme direction (ECMWF vs PBC)
  - eval/viz/bss-pbc_plots-per_target/bss-map_plot-week_N.pdf — 6-row × 3-col
    spatial grid (ECMWF | PBC | PBC−ECMWF diff), rows = 3 variables × 2 extremes

Variables: Temperature, Precipitation, Sea-level Pressure.

BSS is defined relative to a climatological forecast that always assigns
probability 0.2 to the extreme quintile.

Only std_test dates are used.

The Brier score evaluates binary extreme-quintile events:
  • extreme_high → Q5 (highest quintile)
  • extreme_low  → Q1 (lowest quintile)

The land-sea mask is applied to all variables except MSLP (sea-level pressure),
which is evaluated over the entire globe.

Computed metrics are cached to eval/viz/bss-pbc_plots-per_target/metrics/ so that
subsequent runs skip the expensive computation.  Pass --recompute to
force a fresh calculation.

Example usage:
    python src/viz/extreme/bss-pbc_plots-per_target_horizon.py
    python src/viz/extreme/bss-pbc_plots-per_target_horizon.py --horizon 26
"""
import argparse
import glob as globmod
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.file_io import set_file_permissions

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
from models.pbc_debias.attributes import (
    MODEL_NAME as PBC_TEMP_MODEL,
    get_selected_submodel_name as get_pbc_temp_model,
)
from models.utils.models_util import get_task_forecast_dir
from models.utils.data_utils import get_measurement_variable
from models.utils.eval_util import get_target_dates
from utils.data_io import load_data

EXTREMES = ("low", "high")

PRECIP_GT_ID = "era5-pr"
TEMP_GT_ID = "era5-tas"
MSLP_GT_ID = "era5-mslp"

MODEL_LABELS = ["Raw ECMWF", "Debiased ECMWF", "PBC-ECMWF"]

PRECIP_MODEL_CONFIGS = {
    ECMWF_MODEL: {"label": "Raw ECMWF", "color": "#75147C", "get_submodel": get_raw_ecmwf_model},
    ECMWF_DEB_MODEL: {"label": "Debiased ECMWF", "color": "#D4A2D9", "get_submodel": get_ecmwf_model},
    PBC_PRECIP_MODEL: {"label": "PBC-ECMWF", "color": "#F9D949", "get_submodel": get_pbc_precip_model},
}

TEMP_MODEL_CONFIGS = {
    ECMWF_MODEL: {"label": "Raw ECMWF", "color": "#75147C", "get_submodel": get_raw_ecmwf_model},
    ECMWF_DEB_MODEL: {"label": "Debiased ECMWF", "color": "#D4A2D9", "get_submodel": get_ecmwf_model},
    PBC_TEMP_MODEL: {"label": "PBC-ECMWF", "color": "#F9D949", "get_submodel": get_pbc_temp_model},
}

MSLP_MODEL_CONFIGS = {
    ECMWF_MODEL: {"label": "Raw ECMWF", "color": "#75147C", "get_submodel": get_raw_ecmwf_model},
    ECMWF_DEB_MODEL: {"label": "Debiased ECMWF", "color": "#D4A2D9", "get_submodel": get_ecmwf_model},
    PBC_TEMP_MODEL: {"label": "PBC-ECMWF", "color": "#F9D49", "get_submodel": get_pbc_temp_model},
}

CLIM_PROB = 0.2  # climatology always predicts 0.2 for any quintile

VARIABLE_SPECS = [
    # (display_name, gt_id, model_configs)
    ("Temperature", TEMP_GT_ID, TEMP_MODEL_CONFIGS),
    ("Precipitation", PRECIP_GT_ID, PRECIP_MODEL_CONFIGS),
    ("Sea Level Pressure", MSLP_GT_ID, MSLP_MODEL_CONFIGS),
]

EXTREME_DISPLAY = {
    "low": "Very Low",
    "high": "Very High",
}

HORIZON_DISPLAY = {
    "19": "Week 3",
    "26": "Week 4",
}

METRICS_DIR = Path("eval/viz/bss-pbc_plots-per_target/metrics")

# Regex to extract the date (YYYYMMDD) from forecast filenames
_DATE_RE = re.compile(r"-(\d{8})\.nc$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon", choices=["19", "26"], default="19",
        help="Target horizon in days: '19' for week 3, '26' for week 4 (default: 19).",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recomputation even if cached metrics exist.",
    )
    parser.add_argument(
        "--trial", action="store_true",
        help="Quick trial run: limit to 20 dates per variable/extreme.",
    )
    args = parser.parse_args()
    horizon: str = args.horizon
    recompute: bool = args.recompute
    max_dates: int | None = 20 if args.trial else None

    output_dir = Path("eval/viz/bss-pbc_plots-per_target")
    output_dir.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    week_tag = "week_3" if horizon == "19" else "week_4"

    barplot_cache = METRICS_DIR / f"barplot_bss_{week_tag}.json"

    if not recompute and barplot_cache.exists():
        print("Loading cached barplot BSS …")
        barplot_data = _load_barplot_cache(barplot_cache)
    else:
        print("=" * 60)
        print(f"Computing barplot BSS ({week_tag}, horizon {horizon}d) …")
        print("=" * 60)

        barplot_data: Dict[str, Dict[str, List[float]]] = {}

        for display_name, gt_id, model_configs in VARIABLE_SPECS:
            for extreme in EXTREMES:
                task_label = f"{display_name}\n{EXTREME_DISPLAY[extreme]}"
                bss_per_model = compute_barplot_bss(
                    model_configs, gt_id, extreme, horizon,
                    max_dates=max_dates,
                )
                barplot_data[task_label] = bss_per_model

        _save_barplot_cache(barplot_data, barplot_cache)

    plot_bss_barplot(barplot_data, output_dir / f"bss-bar_plot-{week_tag}.pdf", horizon)

    spatial_data: Dict[str, Dict[str, xr.DataArray]] = {}
    all_cached = True

    for display_name, gt_id, model_configs in VARIABLE_SPECS:
        for extreme in EXTREMES:
            task_key = f"{display_name}_{extreme}"
            if not recompute and _spatial_cache_exists(task_key, horizon):
                continue
            else:
                all_cached = False
                break
        if not all_cached:
            break

    if not recompute and all_cached:
        print("Loading cached spatial BSS …")
        for display_name, gt_id, model_configs in VARIABLE_SPECS:
            for extreme in EXTREMES:
                task_key = f"{display_name}_{extreme}"
                spatial_data[task_key] = _load_spatial_cache(task_key, horizon)
    else:
        print("=" * 60)
        print(f"Computing spatial BSS ({week_tag}, horizon {horizon}d) …")
        print("=" * 60)

        for display_name, gt_id, model_configs in VARIABLE_SPECS:
            for extreme in EXTREMES:
                task_key = f"{display_name}_{extreme}"
                bss_fields = compute_spatial_bss(
                    model_configs, gt_id, extreme, horizon,
                    max_dates=max_dates,
                )
                spatial_data[task_key] = bss_fields
                _save_spatial_cache(task_key, bss_fields, horizon)

    plot_bss_spatial_grid(spatial_data, output_dir / f"bss-map_plot-{week_tag}.pdf", horizon)

    print("Done.")


def compute_barplot_bss(
    model_configs: Dict,
    gt_id: str,
    extreme: str,
    horizon: str,
    max_dates: int | None = None,
) -> Dict[str, List[float]]:
    """Return {model_label: [bss_day1, bss_day2, …]} for one variable × horizon.

    BSS_t = 1 − BS_t / BS_clim_t, computed per std_test day.
    BS_t  = Σ_g w_g · (p_model − o)²
    BS_clim_t = Σ_g w_g · (0.2 − o)²
    """
    prob_gt_id = _prob_gt_id(gt_id, extreme)

    # Restrict to std_test dates
    std_test_dates = set(pd.Timestamp(d) for d in get_target_dates("std_test"))

    common_dates = _find_common_dates(model_configs, prob_gt_id, horizon)
    common_dates = [d for d in common_dates if d in std_test_dates]
    if max_dates is not None:
        common_dates = common_dates[:max_dates]

    print(f"  {prob_gt_id} {horizon}d: {len(common_dates)} common dates "
          f"({common_dates[0]:%Y-%m-%d} – {common_dates[-1]:%Y-%m-%d})"
          if common_dates else f"  {prob_gt_id} {horizon}d: no common dates")

    # Accumulators: {model_label: [bss_day1, bss_day2, …]}
    daily_bss: Dict[str, List[float]] = {cfg["label"]: [] for cfg in model_configs.values()}

    obs_cache: Dict[str, xr.DataArray] = {}

    for i, target_date in enumerate(common_dates):
        predicted_probs = {}
        obs_binary = None
        weights = None

        for model_name, cfg in model_configs.items():
            try:
                pred, obs = _load_forecast_and_obs(
                    model_name, cfg["get_submodel"], gt_id,
                    horizon, target_date, extreme, obs_cache,
                )
                predicted_probs[cfg["label"]] = pred
                if obs_binary is None:
                    obs_binary = obs
                    weights = _area_weights(obs)
            except (FileNotFoundError, KeyError, ValueError):
                pass

        if obs_binary is None:
            continue

        # Climatology Brier score for this day
        clim_bs = _weighted_sum((CLIM_PROB - obs_binary) ** 2, weights)
        if clim_bs == 0.0:
            continue  # degenerate day, skip

        for label, pred in predicted_probs.items():
            model_bs = _weighted_sum((pred - obs_binary) ** 2, weights)
            bss = 1.0 - model_bs / clim_bs
            daily_bss[label].append(bss)

        if (i + 1) % 50 == 0 or (i + 1) == len(common_dates):
            print(f"    {horizon}d: processed {i + 1}/{len(common_dates)} dates")

    return daily_bss


def compute_spatial_bss(
    model_configs: Dict,
    gt_id: str,
    extreme: str,
    horizon: str,
    max_dates: int | None = None,
) -> Dict[str, xr.DataArray]:
    """Return {model_label: DataArray of BSS_g} for one variable × horizon.

    BS_g      = (1/T) Σ_t (p_model − o)²
    BS_clim_g = (1/T) Σ_t (0.2 − o)²
    BSS_g     = 1 − BS_g / BS_clim_g
    """
    prob_gt_id = _prob_gt_id(gt_id, extreme)

    std_test_dates = set(pd.Timestamp(d) for d in get_target_dates("std_test"))

    common_dates = _find_common_dates(model_configs, prob_gt_id, horizon)
    common_dates = [d for d in common_dates if d in std_test_dates]
    if max_dates is not None:
        common_dates = common_dates[:max_dates]

    print(f"  {prob_gt_id} {horizon}d: {len(common_dates)} common dates "
          f"({common_dates[0]:%Y-%m-%d} – {common_dates[-1]:%Y-%m-%d})"
          if common_dates else f"  {prob_gt_id} {horizon}d: no common dates")

    # Accumulators per model: sum of squared errors and count
    model_sum: Dict[str, Optional[xr.DataArray]] = {cfg["label"]: None for cfg in model_configs.values()}
    model_count: Dict[str, Optional[xr.DataArray]] = {cfg["label"]: None for cfg in model_configs.values()}
    # Climatology accumulators (shared, same for all models)
    clim_sum: Optional[xr.DataArray] = None
    clim_count: Optional[xr.DataArray] = None

    obs_cache: Dict[str, xr.DataArray] = {}

    for i, target_date in enumerate(common_dates):
        predicted_probs = {}
        obs_binary = None

        for model_name, cfg in model_configs.items():
            try:
                pred, obs = _load_forecast_and_obs(
                    model_name, cfg["get_submodel"], gt_id,
                    horizon, target_date, extreme, obs_cache,
                )
                predicted_probs[cfg["label"]] = pred
                if obs_binary is None:
                    obs_binary = obs
            except (FileNotFoundError, KeyError, ValueError):
                pass

        if obs_binary is None:
            continue

        # Climatology squared error field for this date
        clim_sq = (CLIM_PROB - obs_binary) ** 2
        valid_mask = clim_sq.notnull()
        if clim_sum is None:
            clim_sum = clim_sq.fillna(0.0)
            clim_count = valid_mask.astype(float)
        else:
            clim_sum = clim_sum + clim_sq.fillna(0.0)
            clim_count = clim_count + valid_mask.astype(float)

        # Model squared error fields
        for label, pred in predicted_probs.items():
            sq = (pred - obs_binary) ** 2
            valid = sq.notnull()
            if model_sum[label] is None:
                model_sum[label] = sq.fillna(0.0)
                model_count[label] = valid.astype(float)
            else:
                model_sum[label] = model_sum[label] + sq.fillna(0.0)
                model_count[label] = model_count[label] + valid.astype(float)

        if (i + 1) % 50 == 0 or (i + 1) == len(common_dates):
            print(f"    {horizon}d: processed {i + 1}/{len(common_dates)} dates")

    # Compute BSS_g = 1 − mean_model_sq / mean_clim_sq
    result: Dict[str, xr.DataArray] = {}
    for label in model_sum:
        if model_sum[label] is not None and clim_sum is not None:
            mean_model = model_sum[label] / model_count[label].where(model_count[label] > 0)
            mean_clim = clim_sum / clim_count.where(clim_count > 0)
            bss = 1.0 - mean_model / mean_clim.where(mean_clim > 0)
            result[label] = bss
        else:
            result[label] = xr.DataArray()

    return result


def _prob_gt_id(gt_id: str, extreme: str) -> str:
    """Map (gt_id, extreme) to the probabilistic gt_id used for file lookup."""
    dataset, variable = gt_id.split("-", 1)
    q_idx = 4 if extreme == "high" else 1
    return f"{dataset}-f{q_idx}_{variable}"


def _find_common_dates(
    model_configs: Dict,
    prob_gt_id: str,
    horizon: str,
) -> List[pd.Timestamp]:
    """Return sorted dates for which *all* models have a forecast file."""
    date_sets = []
    for model_name, cfg in model_configs.items():
        submodel_key = prob_gt_id.split("-", 1)[1]
        submodel_name = cfg["get_submodel"](submodel_key, horizon)
        forecast_dir = get_task_forecast_dir(
            model=model_name,
            submodel=submodel_name,
            gt_id=prob_gt_id,
            horizon=horizon,
        )
        dates = _discover_dates(forecast_dir, prob_gt_id, horizon)
        date_sets.append(set(dates))

    if not date_sets:
        return []
    common = date_sets[0]
    for s in date_sets[1:]:
        common &= s
    return sorted(common)


def _discover_dates(
    forecast_dir: str,
    prob_gt_id: str,
    horizon: str,
) -> List[pd.Timestamp]:
    """List all available forecast dates by globbing the forecast directory."""
    pattern = os.path.join(forecast_dir, f"{prob_gt_id}_{horizon}-*.nc")
    dates = []
    for fpath in globmod.glob(pattern):
        m = _DATE_RE.search(fpath)
        if m:
            dates.append(pd.Timestamp(m.group(1)))
    return sorted(dates)


def _load_forecast_and_obs(
    model_name: str,
    get_submodel_fn,
    gt_id: str,
    horizon: str,
    target_date: pd.Timestamp,
    extreme: str,
    obs_cache: Dict[str, xr.DataArray],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load forecast probability and observed binary for one date.

    Returns (predicted_prob, observed_binary), both 2-D DataArrays
    aligned on the same grid.  The land-sea mask is applied to observations.
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

    # Observed CDF (cached); skip land-sea mask for MSLP
    use_lsmask = gt_id != MSLP_GT_ID
    cache_key = f"{prob_gt_id}_{target_date:%Y%m%d}"
    if cache_key in obs_cache:
        obs_cdf = obs_cache[cache_key]
    else:
        gt_ds = load_data(prob_gt_id, lsmask=use_lsmask)
        obs_cdf = gt_ds[measurement_var].sel(
            time=np.datetime64(target_date),
        ).squeeze(drop=True).load()
        obs_cache[cache_key] = obs_cdf

    # Align grids
    fcst_cdf, obs_cdf_aligned = xr.align(fcst_cdf, obs_cdf, join="inner")

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


def _sanitize_key(key: str) -> str:
    """Turn a task key into a filesystem-safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", key)


def _save_barplot_cache(
    barplot_data: Dict[str, Dict[str, float]],
    path: Path,
) -> None:
    """Persist barplot BSS values as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(barplot_data, f, indent=2)
    set_file_permissions(path)
    print(f"  Cached barplot BSS → {path}")


def _load_barplot_cache(path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load barplot BSS values from JSON.

    Handles the legacy cache format where values are single floats
    by wrapping them into single-element lists.
    """
    with open(path) as f:
        raw = json.load(f)
    for cat in raw:
        for label in raw[cat]:
            if not isinstance(raw[cat][label], list):
                raw[cat][label] = [raw[cat][label]]
    return raw


def _spatial_cache_path(task_key: str, model_label: str, horizon: str) -> Path:
    """Return the NetCDF path for a cached spatial BSS field."""
    safe_key = _sanitize_key(task_key)
    safe_label = _sanitize_key(model_label)
    return METRICS_DIR / f"spatial_bss_{safe_key}_{safe_label}_horizon_{horizon}.nc"


def _spatial_cache_exists(task_key: str, horizon: str) -> bool:
    """Check whether cached spatial fields exist for all models of a task."""
    for label in MODEL_LABELS:
        if not _spatial_cache_path(task_key, label, horizon).exists():
            return False
    return True


def _save_spatial_cache(
    task_key: str,
    bss_fields: Dict[str, xr.DataArray],
    horizon: str,
) -> None:
    """Save spatial BSS DataArrays as NetCDF files."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for label, da in bss_fields.items():
        path = _spatial_cache_path(task_key, label, horizon)
        if da.size > 0:
            da.to_netcdf(path)
            set_file_permissions(path)
    print(f"  Cached spatial BSS for {task_key}")


def _load_spatial_cache(task_key: str, horizon: str) -> Dict[str, xr.DataArray]:
    """Load spatial BSS DataArrays from NetCDF files."""
    result = {}
    for label in MODEL_LABELS:
        path = _spatial_cache_path(task_key, label, horizon)
        if path.exists():
            result[label] = xr.open_dataarray(path).load()
        else:
            result[label] = xr.DataArray()
    return result


def plot_bss_barplot(
    barplot_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    horizon: str,
    n_boot: int = 5000,
    random_seed: int = 42,
) -> None:
    """Grouped bar chart of mean BSS with 95% bootstrap CIs."""
    categories = list(barplot_data.keys())
    model_colors = {"Raw ECMWF": "#75147C", "Debiased ECMWF": "#D4A2D9", "PBC-ECMWF": "#F9D949"}
    rng = np.random.default_rng(random_seed)

    # ------- Compute means + bootstrap CIs -------
    means = np.zeros((len(MODEL_LABELS), len(categories)))
    lower_errors = np.zeros_like(means)
    upper_errors = np.zeros_like(means)

    # Keep track of all means for improvement computation
    all_means = {}
    for i, label in enumerate(MODEL_LABELS):
        for j, cat in enumerate(categories):
            data = np.asarray(barplot_data[cat].get(label, []))
            data = data[~np.isnan(data)]
            n = len(data)
            print(f"  Model {label}, category {cat}, horizon {horizon}d: {n} target dates")

            if n == 0:
                means[i, j] = float("nan")
                lower_errors[i, j] = 0.0
                upper_errors[i, j] = 0.0
                continue

            mean_val = np.mean(data)

            # Bootstrap resampling
            boot_means = np.empty(n_boot)
            for b in range(n_boot):
                sample = rng.choice(data, size=n, replace=True)
                boot_means[b] = np.mean(sample)

            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            means[i, j] = mean_val
            lower_errors[i, j] = mean_val - ci_low
            upper_errors[i, j] = ci_high - mean_val

        all_means[label] = means[i]

    # Display percentage improvements of PBC-ECMWF over ECMWF and Debiased ECMWF
    print(f"\n% Improvements of PBC-ECMWF over Raw ECMWF ({HORIZON_DISPLAY[horizon]}):")
    print(100 * (all_means["PBC-ECMWF"] - all_means["Raw ECMWF"]) / all_means["Raw ECMWF"])
    print(f"\n% Improvements of PBC-ECMWF over Debiased ECMWF ({HORIZON_DISPLAY[horizon]}):")
    print(100 * (all_means["PBC-ECMWF"] - all_means["Debiased ECMWF"]) / all_means["Debiased ECMWF"])


    # ------- Plot -------
    n_groups = len(categories)
    x = np.arange(n_groups)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, label in enumerate(MODEL_LABELS):
        ax.bar(
            x + (i - (len(MODEL_LABELS) - 1) / 2) * width,
            means[i],
            width,
            yerr=[lower_errors[i], upper_errors[i]],
            capsize=4,
            label=label,
            color=model_colors[label],
            zorder=3,
            error_kw=dict(ecolor="gray", elinewidth=1.5, capthick=1.5),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.split("\n")[1] for c in categories],
        fontsize=20,
    )

    # Variable group labels beneath the tick labels
    pair_centers = [0.5, 2.5, 4.5]
    var_names = [spec[0] for spec in VARIABLE_SPECS]
    for xc, var in zip(pair_centers, var_names):
        ax.text(
            xc, -0.12, var,
            ha="center", va="top",
            fontsize=20, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

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


def plot_bss_spatial_grid(
    spatial_data: Dict[str, Dict[str, xr.DataArray]],
    output_path: Path,
    horizon: str,
) -> None:
    """6-row × 4-col spatial BSS grid: ECMWF | Debiased ECMWF | PBC-ECMWF | diff.

    Rows: Temperature low, high; Precipitation low, high; MSLP low, high.
    """
    row_specs = [
        ("Temperature", "Temperature"),
        ("Precipitation", "Precipitation"),
        ("Sea Level Pressure", "Sea Level Pressure"),
    ]

    num_rows = len(row_specs) * len(EXTREMES)
    num_cols = len(MODEL_LABELS) + 1  # models + diff

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(18, 18),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )

    curr_row = 0
    im_bss, im_diff = None, None

    for display_name, plot_label in row_specs:
        for extreme in EXTREMES:
            task_key = f"{display_name}_{extreme}"
            bss_fields = spatial_data[task_key]

            pbc_bss = bss_fields.get("PBC-ECMWF", xr.DataArray())
            deb_bss = bss_fields.get("Debiased ECMWF", xr.DataArray())

            # Diff: PBC-ECMWF − Debiased ECMWF
            if pbc_bss.size > 0 and deb_bss.size > 0:
                diff_bss = pbc_bss - deb_bss
            else:
                diff_bss = xr.DataArray()

            row_axes = axes[curr_row]

            for ax in row_axes:
                ax.set_global()
                ax.coastlines(color="black", linewidth=0.6)
                ax.spines["geo"].set_linewidth(2.25)

            # Row label
            row_label = f"{plot_label}\n{EXTREME_DISPLAY[extreme]}"
            row_axes[0].text(
                -0.15, 0.5, row_label,
                va="center", ha="center",
                rotation=90,
                transform=row_axes[0].transAxes,
                fontsize=20,
            )

            # Plot each model column
            for i, label in enumerate(MODEL_LABELS):
                field = bss_fields.get(label, xr.DataArray())
                if field.size > 0:
                    im = field.plot(
                        ax=row_axes[i],
                        transform=ccrs.PlateCarree(),
                        cmap="RdBu_r",
                        vmin=-1, vmax=1,
                        add_colorbar=False,
                        rasterized=True,
                    )
                    im_bss = im

            # Diff column (last)
            if diff_bss.size > 0:
                im2 = diff_bss.plot(
                    ax=row_axes[-1],
                    transform=ccrs.PlateCarree(),
                    cmap="bwr",
                    vmin=-0.2, vmax=0.2,
                    add_colorbar=False,
                    rasterized=True,
                )
                im_diff = im2

                nonnull = diff_bss.notnull()
                pct_improved = float(
                    ((nonnull & (diff_bss > 0)).sum() / nonnull.sum()).values
                )
                print(f"  {task_key}: % grid cells improved: {pct_improved:.2%}")

            # Column titles (first row only)
            if curr_row == 0:
                for i, label in enumerate(MODEL_LABELS):
                    row_axes[i].set_title(label, fontsize=20, pad=30)
                row_axes[-1].set_title(
                    f"{MODEL_LABELS[-1]} − {MODEL_LABELS[-2]}",
                    fontsize=20, pad=30,
                )
            else:
                for ax in row_axes:
                    ax.set_title("")

            curr_row += 1

    # Colorbars
    if im_bss is not None:
        cbar1 = fig.colorbar(
            im_bss, ax=axes[:, :len(MODEL_LABELS)],
            orientation="horizontal",
            fraction=0.02, pad=0.02, aspect=40,
        )
        cbar1.set_label("Brier skill score (BSS)", fontsize=20, labelpad=10)
        cbar1.ax.tick_params(labelsize=20)

    if im_diff is not None:
        cbar2 = fig.colorbar(
            im_diff, ax=axes[:, -1],
            orientation="horizontal",
            fraction=0.02, pad=0.02, aspect=20,
        )
        cbar2.set_label("BSS difference", fontsize=20, labelpad=10)
        cbar2.ax.tick_params(labelsize=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    set_file_permissions(output_path)
    plt.close()
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()
