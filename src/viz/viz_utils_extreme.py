from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from arch.bootstrap import IIDBootstrap

from utils.file_io import set_file_permissions, make_directories
from utils.logging import printf
from utils.timing import tic, toc

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


# ============================================================================
# Configuration
# ============================================================================

HORIZONS = ("19", "26")

MODEL_LABELS = [
    "Raw ECMWF",
    "Debiased ECMWF",
    "PBC-ECMWF",
]

MODEL_COLORS = {
    "Raw ECMWF": "purple",
    "Debiased ECMWF": "plum",
    "PBC-ECMWF": "gold",
}

MODEL_CONFIGS = {
    ECMWF_MODEL: {
        "label": "Raw ECMWF",
        "color": "purple",
        "get_submodel": get_raw_ecmwf_model,
    },
    ECMWF_DEB_MODEL: {
        "label": "Debiased ECMWF",
        "color": "plum",
        "get_submodel": get_ecmwf_model,
    },
    PBC_PRECIP_MODEL: {
        "label": "PBC-ECMWF",
        "color": "gold",
        "get_submodel": get_pbc_precip_model,
    },
}

DEFAULT_CACHE_DIR = Path(
    "eval/viz/bss-barplot-floods/data"
)

DEFAULT_EVENTS_JSON = Path(
    "eval/viz/bss-barplot-floods/data/"
    "flood-events-gdacs-2016_2026.json"
)

DEFAULT_OUTPUT_DIR = Path(
    "viz/pbc/extremes"
)

SRC_DATA_DIR = os.path.join("viz", "pbc", "source_data")


# ============================================================================
# Public function
# ============================================================================

def generate_flood_bss_barplot(
    events_json: Path = DEFAULT_EVENTS_JSON,
    output_path: Optional[Path] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    percentile: int = 95,
    use_all_dates: bool = False,
    recompute: bool = False,
    n_boot: int = 5000,
    random_seed: int = 42,
    save_fig: bool = True,
    show_fig: bool = False,
    source_data: bool = False,
    source_data_filename: str = "fig_5-bss_week3.xlsx",
    source_data_sheet_prefix: str = "Fig_5b",
):
    """
    Generate the Brier Skill Score bar plot for GDACS flood events.

    Parameters
    ----------
    events_json : Path
        Path to the GDACS flood-events JSON file.

    output_path : Path, optional
        Output path for the figure. If None, a default path is generated.

    cache_dir : Path
        Directory containing cached event-level Brier scores.

    percentile : int
        Precipitation percentile defining the extreme event.

    use_all_dates : bool
        If False, restrict events to std_test dates.
        If True, use all available event dates.

    recompute : bool
        If True, recompute Brier scores even when a cache exists.

    n_boot : int
        Number of bootstrap samples for confidence intervals.

    random_seed : int
        Random seed for bootstrap confidence intervals.

    save_fig : bool
        Whether to save the resulting figure.

    show_fig : bool
        Whether to display the figure.

    source_data : bool
        Whether to save source data for the figure to an XLSX file.

    source_data_filename : str
        XLSX filename. The file is written to SRC_DATA_DIR.

    source_data_sheet_prefix : str
        Prefix used for source-data sheet names. This allows multiple
        subfigures to safely share the same workbook.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    bss : np.ndarray
        BSS values with dimensions:
        (severity, horizon, model).

    all_severities : dict
        Event-level Brier score data used to generate the figure.
    """

    events_json = Path(events_json)
    cache_dir = Path(cache_dir)

    # --------------------------------------------------------------
    # Default output path
    # --------------------------------------------------------------
    if output_path is None:

        suffix = (
            "-all_dates"
            if use_all_dates
            else ""
        )

        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"barplot_F{percentile}_"
            f"bss_floods_severity{suffix}.pdf"
        )

    output_path = Path(output_path)

    # --------------------------------------------------------------
    # Cache path
    # --------------------------------------------------------------
    suffix = (
        "-all_dates"
        if use_all_dates
        else ""
    )

    cache_file = (
        cache_dir
        / f"F{percentile}_"
        f"brier_severity_cache{suffix}.json"
    )

    make_directories(cache_dir)

    # --------------------------------------------------------------
    # Load or calculate Brier scores
    # --------------------------------------------------------------
    if (
        not recompute
        and cache_file.exists()
    ):

        printf(
            f"Loading cached Brier scores "
            f"from {cache_file}"
        )

        severity_brier_raw = _load_brier_cache(
            cache_file
        )

    else:

        events = load_events(
            events_json
        )

        printf(
            f"Loaded {len(events)} flood events "
            f"from {events_json}"
        )

        printf(
            "Computing Brier scores for "
            "GDACS flood events ..."
        )

        severity_brier_raw = collect_severity_brier(
            events=events,
            model_configs=MODEL_CONFIGS,
            percentile=percentile,
            use_all_dates=use_all_dates,
        )

        _save_brier_cache(
            severity_brier_raw,
            cache_file,
        )

    # --------------------------------------------------------------
    # Reorganize into All / Severe
    # --------------------------------------------------------------
    grouped_brier_raw = (
        _group_flood_brier_scores(
            severity_brier_raw
        )
    )

    # --------------------------------------------------------------
    # Create plot
    # --------------------------------------------------------------
    fig, bss = plot_flood_bss_barplot(
        grouped_brier_raw,
        n_boot=n_boot,
        random_seed=random_seed,
        output_path=output_path,
        save_fig=save_fig,
        show_fig=show_fig,
        source_data=source_data,
        source_data_filename=source_data_filename,
        source_data_sheet_prefix=source_data_sheet_prefix,
    )

    return (
        fig,
        bss,
        grouped_brier_raw,
    )



# ============================================================================
# Event loading
# ============================================================================

def load_events(
    path: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Read GDACS flood events from JSON.
    """

    with open(path) as fh:
        return json.load(fh)


# ============================================================================
# Cache helpers
# ============================================================================

def _save_brier_cache(
    all_severities: Dict[
        str,
        Dict[str, Dict[str, List[float]]]
    ],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as fh:
        json.dump(
            all_severities,
            fh,
            indent=2,
        )

    set_file_permissions(path)

    printf(
        f"  Cached Brier score values "
        f"→ {path}"
    )


def _load_brier_cache(
    path: Path,
) -> Dict[
    str,
    Dict[str, Dict[str, List[float]]]
]:

    with open(path) as fh:
        data = json.load(fh)

    return {
        severity: {
            model: {
                horizon: [
                    float(v)
                    for v in values
                ]
                for horizon, values
                in horizons.items()
            }
            for model, horizons
            in models.items()
        }
        for severity, models
        in data.items()
    }


# ============================================================================
# Group All / Severe floods
# ============================================================================

def _group_flood_brier_scores(
    severity_brier_raw,
):

    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for severity in severity_brier_raw:

        for model, horizons in (
            severity_brier_raw[
                severity
            ].items()
        ):

            for horizon, values in (
                horizons.items()
            ):

                # All floods
                grouped[
                    "All"
                ][model][horizon].extend(
                    values
                )

                # Everything except Green = Severe
                if severity != "Green":

                    grouped[
                        "Severe"
                    ][model][horizon].extend(
                        values
                    )

    return grouped


# ============================================================================
# Brier score calculation
# ============================================================================

def collect_severity_brier(
    events: Dict[str, dict],
    model_configs: Dict,
    percentile: int,
    *,
    use_all_dates: bool = False,
):
    """
    Compute event-level Brier scores grouped by GDACS severity.
    """

    if not use_all_dates:

        std_test_dates = set(
            pd.Timestamp(d)
            for d in get_target_dates(
                "std_test"
            )
        )

    else:

        std_test_dates = None

    raw = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    obs_cache = {}

    n_events = len(events)

    gt_id = (
        f"era5-F{percentile}_pr"
    )

    quantile = percentile / 100.0

    tic()

    for idx, (
        event_name,
        info,
    ) in enumerate(events.items()):

        target_date = pd.Timestamp(
            info["date"]
        )

        # ----------------------------------------------------------
        # std_test filtering
        # ----------------------------------------------------------
        if (
            std_test_dates is not None
            and target_date not in std_test_dates
        ):
            continue

        lat_bounds = tuple(
            info["lat"]
        )

        lon_bounds = tuple(
            info["lon"]
        )

        severity = info[
            "alertlevel"
        ].title()

        date = info["date"]

        # ----------------------------------------------------------
        # Horizons
        # ----------------------------------------------------------
        for horizon in HORIZONS:

            predicted_probs = {}

            obs_binary = None
            weights = None

            skip = False

            # ------------------------------------------------------
            # Load all model forecasts
            # ------------------------------------------------------
            for model_name, cfg in (
                model_configs.items()
            ):

                try:

                    pred, obs = (
                        _load_forecast_and_obs(
                            model_name=model_name,
                            get_submodel_fn=cfg[
                                "get_submodel"
                            ],
                            gt_id=gt_id,
                            horizon=horizon,
                            target_date=target_date,
                            obs_cache=obs_cache,
                            lat_bounds=lat_bounds,
                            lon_bounds=lon_bounds,
                        )
                    )

                    predicted_probs[
                        cfg["label"]
                    ] = pred

                    if obs_binary is None:

                        obs_binary = obs

                        weights = (
                            _area_weights(obs)
                        )

                except (
                    FileNotFoundError,
                    KeyError,
                    ValueError,
                ) as exc:

                    printf(
                        f"  {event_name}, {date} | "
                        f"{cfg['label']} "
                        f"{horizon}d → SKIPPED "
                        f"({exc})"
                    )

                    skip = True
                    break

            if (
                skip
                or obs_binary is None
            ):
                continue

            # ------------------------------------------------------
            # Climatological Brier score
            # ------------------------------------------------------
            clim_bs = _weighted_sum(
                (
                    quantile
                    - obs_binary
                ) ** 2,
                weights,
            )

            if clim_bs == 0.0:
                continue

            raw[
                severity
            ][
                "Climatology"
            ][
                horizon
            ].append(
                clim_bs
            )

            printf(
                f"  {event_name}, {date} | "
                f"Climatology {horizon}d → "
                f"BS {clim_bs:.4f}"
            )

            # ------------------------------------------------------
            # Model Brier scores
            # ------------------------------------------------------
            for label, pred in (
                predicted_probs.items()
            ):

                model_bs = _weighted_sum(
                    (
                        pred
                        - obs_binary
                    ) ** 2,
                    weights,
                )

                raw[
                    severity
                ][
                    label
                ][
                    horizon
                ].append(
                    model_bs
                )

                printf(
                    f"  {event_name}, {date} | "
                    f"{label} {horizon}d → "
                    f"BS {model_bs:.4f}"
                )

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------
        if (
            (idx + 1) % 50 == 0
            or (idx + 1) == n_events
        ):

            printf(
                f"    processed "
                f"{idx + 1}/{n_events} events"
            )

            toc()
            tic()

    return dict(raw)


# ============================================================================
# Forecast / observation loading
# ============================================================================

def _load_forecast_and_obs(
    model_name: str,
    get_submodel_fn,
    gt_id: str,
    horizon: str,
    target_date: pd.Timestamp,
    obs_cache,
    lat_bounds=None,
    lon_bounds=None,
):
    """
    Load forecast CDF and observed CDF for one event.
    """

    measurement_var = (
        get_measurement_variable(
            gt_id
        )
    )

    # --------------------------------------------------------------
    # Forecast
    # --------------------------------------------------------------
    submodel_key = gt_id.split(
        "-",
        1,
    )[1]

    submodel_name = (
        get_submodel_fn(
            submodel_key,
            horizon,
        )
    )

    forecast_dir = Path(
        get_task_forecast_dir(
            model=model_name,
            submodel=submodel_name,
            gt_id=gt_id,
            horizon=horizon,
        )
    )

    forecast_path = (
        forecast_dir
        / f"{gt_id}_{horizon}-"
        f"{target_date:%Y%m%d}.nc"
    )

    if not forecast_path.exists():

        raise FileNotFoundError(
            f"Missing forecast: "
            f"{forecast_path}"
        )

    with xr.open_dataset(
        forecast_path
    ) as ds:

        fcst_cdf = (
            ds[measurement_var]
            .sel(
                time=np.datetime64(
                    target_date
                ),
            )
            .squeeze(drop=True)
            .load()
        )

    # --------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------
    cache_key = (
        f"{gt_id}_"
        f"{target_date:%Y%m%d}"
    )

    if cache_key in obs_cache:

        obs_cdf = obs_cache[
            cache_key
        ]

    else:

        gt_ds = load_data(
            gt_id,
            lsmask=True,
        )

        obs_cdf = (
            gt_ds[measurement_var]
            .sel(
                time=np.datetime64(
                    target_date
                ),
            )
            .squeeze(drop=True)
            .load()
        )

        obs_cache[
            cache_key
        ] = obs_cdf

    # --------------------------------------------------------------
    # Align grids
    # --------------------------------------------------------------
    (
        fcst_cdf,
        obs_cdf_aligned,
    ) = xr.align(
        fcst_cdf,
        obs_cdf,
        join="inner",
    )

    # --------------------------------------------------------------
    # Subset event region
    # --------------------------------------------------------------
    if (
        lat_bounds is not None
        and lon_bounds is not None
    ):

        fcst_cdf = _subset_region(
            fcst_cdf,
            lat_bounds,
            lon_bounds,
        )

        obs_cdf_aligned = (
            _subset_region(
                obs_cdf_aligned,
                lat_bounds,
                lon_bounds,
            )
        )

    return (
        fcst_cdf,
        obs_cdf_aligned,
    )


# ============================================================================
# Spatial utilities
# ============================================================================

def _area_weights(
    da: xr.DataArray,
) -> xr.DataArray:

    return np.cos(
        np.deg2rad(
            da.latitude
        )
    )


def _weighted_sum(
    field: xr.DataArray,
    weights: xr.DataArray,
) -> float:

    w = weights.where(
        field.notnull()
    )

    return float(
        (
            field * w
        ).sum(
            skipna=True
        ).values
    )


def _subset_region(
    da: xr.DataArray,
    lat_bounds: Tuple[
        float,
        float,
    ],
    lon_bounds: Tuple[
        float,
        float,
    ],
) -> xr.DataArray:

    lat_sel = _slice_for_coord(
        da.latitude,
        lat_bounds,
    )

    lo, hi = lon_bounds

    if (
        hi > 360
        and float(
            da.longitude.max()
        ) <= 360
    ):

        part_hi = da.sel(
            latitude=lat_sel,
            longitude=_slice_for_coord(
                da.longitude,
                (lo, 360.0),
            ),
        )

        part_lo = da.sel(
            latitude=lat_sel,
            longitude=_slice_for_coord(
                da.longitude,
                (0.0, hi - 360.0),
            ),
        )

        part_lo = (
            part_lo.assign_coords(
                longitude=(
                    part_lo.longitude
                    + 360
                )
            )
        )

        return xr.concat(
            [
                part_hi,
                part_lo,
            ],
            dim="longitude",
        )

    return da.sel(
        latitude=lat_sel,
        longitude=_slice_for_coord(
            da.longitude,
            lon_bounds,
        ),
    )


def _slice_for_coord(
    coord: xr.DataArray,
    bounds: Tuple[
        float,
        float,
    ],
) -> slice:

    lo, hi = bounds

    if lo > hi:
        lo, hi = hi, lo

    ascending = (
        float(coord.values[-1])
        >= float(coord.values[0])
    )

    if ascending:
        return slice(lo, hi)

    return slice(hi, lo)


# ============================================================================
# Plotting
# ============================================================================

def plot_flood_bss_barplot(
    all_severities,
    *,
    n_boot: int = 5000,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
    save_fig: bool = True,
    show_fig: bool = False,
    source_data: bool = False,
    source_data_filename: str = "fig_5-bss_week3.xlsx",
    source_data_sheet_prefix: str = "Fig_5b",
):
    """
    Create the BSS bar plot for GDACS flood events.

    Parameters
    ----------
    all_severities : dict
        Event-level Brier score data grouped by severity, model and horizon.

    n_boot : int
        Number of bootstrap samples for confidence intervals.

    random_seed : int
        Random seed for bootstrap confidence intervals.

    output_path : Path, optional
        Figure output path.

    save_fig : bool
        Whether to save the figure.

    show_fig : bool
        Whether to display the figure.

    source_data : bool
        Whether to save source data to XLSX.

    source_data_filename : str
        XLSX filename. The file is written to SRC_DATA_DIR.

    source_data_sheet_prefix : str
        Prefix used for source-data sheet names.

    Returns
    -------
    fig : matplotlib.figure.Figure

    bss : np.ndarray
        BSS values with dimensions
        (severity, horizon, model).
    """

    severity_keys = sorted(
        all_severities.keys()
    )

    if not severity_keys:

        printf(
            "  No severity flood data to plot."
        )

        return None, None

    # --------------------------------------------------------------
    # Figure geometry
    # --------------------------------------------------------------
    n_sev = len(
        severity_keys
    )

    n_horiz = len(
        HORIZONS
    )

    n_groups = (
        n_sev
        * n_horiz
    )

    scale_factor = 1.05

    fig_width = max(
        10,
        1.5 * n_groups,
    ) * scale_factor

    fig_height = (
        6
        * scale_factor
    )

    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            fig_height,
        )
    )

    x_positions = np.arange(
        0,
        1.5 * n_groups,
        1.5,
    )

    width = 0.30

    # --------------------------------------------------------------
    # BSS arrays
    # --------------------------------------------------------------
    bss = np.full(
        (
            n_sev,
            n_horiz,
            len(MODEL_LABELS),
        ),
        np.nan,
    )

    # --------------------------------------------------------------
    # Bootstrap confidence intervals
    # --------------------------------------------------------------
    confidence = 0.95
    tail = "lower"
    method = "bca"

    ci_low = np.full(
        (
            n_sev,
            n_horiz,
            len(MODEL_LABELS),
        ),
        np.inf,
    )

    # --------------------------------------------------------------
    # Calculate BSS
    # --------------------------------------------------------------
    for jj, sev in enumerate(
        severity_keys
    ):

        severity_data = (
            all_severities.get(
                sev,
                {},
            )
        )

        for kk, horizon in enumerate(
            HORIZONS
        ):

            clim = np.asarray(
                severity_data
                .get(
                    "Climatology",
                    {},
                )
                .get(
                    horizon,
                    [],
                )
            )

            clim = clim[
                ~np.isnan(clim)
            ]

            n = len(clim)

            printf(
                f"  Severity {sev}, "
                f"horizon {horizon}d: "
                f"{n} target dates"
            )

            pbc_data = np.asarray(
                severity_data
                .get(
                    "PBC-ECMWF",
                    {},
                )
                .get(
                    horizon,
                    [],
                )
            )

            pbc_data = pbc_data[
                ~np.isnan(pbc_data)
            ]

            # ------------------------------------------------------
            # Calculate model BSS
            # ------------------------------------------------------
            model_data = {}

            for ii, label in enumerate(
                MODEL_LABELS
            ):

                data = np.asarray(
                    severity_data
                    .get(
                        label,
                        {},
                    )
                    .get(
                        horizon,
                        [],
                    )
                )

                data = data[
                    ~np.isnan(data)
                ]

                model_data[label] = data

                if (
                    len(data) > 0
                    and len(clim) > 0
                ):

                    bss[
                        jj,
                        kk,
                        ii,
                    ] = (
                        1
                        - np.mean(data)
                        / np.mean(clim)
                    )

                else:

                    bss[
                        jj,
                        kk,
                        ii,
                    ] = np.nan

            # ------------------------------------------------------
            # Bootstrap significance
            # ------------------------------------------------------
            if (
                len(clim) == 0
                or len(pbc_data) == 0
            ):
                continue

            for ii, label in enumerate(
                MODEL_LABELS
            ):

                if label == "PBC-ECMWF":
                    continue

                data = model_data[
                    label
                ]

                n_common = min(
                    len(data),
                    len(clim),
                    len(pbc_data),
                )

                if n_common == 0:
                    continue

                paired_inds = np.arange(
                    n_common
                )

                def ss_diff(
                    inds,
                    data=data,
                ):
                    return (
                        data[inds].mean()
                        - pbc_data[inds].mean()
                    ) / clim[inds].mean()

                boot = IIDBootstrap(
                    paired_inds,
                    seed=random_seed,
                )

                ci = boot.conf_int(
                    ss_diff,
                    reps=n_boot,
                    method=method,
                    size=confidence,
                    tail=tail,
                )

                ci_low[
                    jj,
                    kk,
                    ii,
                ] = ci[0][0]

                printf(
                    f"  {sev} | "
                    f"{horizon}d | "
                    f"PBC-ECMWF - "
                    f"{label} BSS "
                    f"{int(confidence * 100)}% "
                    f"{method} {tail} "
                    f"confidence bound: "
                    f"{ci_low[jj, kk, ii]}"
                )

    # --------------------------------------------------------------
    # Plot bars
    # --------------------------------------------------------------
    all_centers = {}

    for ii, label in enumerate(
        MODEL_LABELS
    ):

        centers = []
        hatches = []

        for jj, sev in enumerate(
            severity_keys
        ):

            for kk, horizon in enumerate(
                HORIZONS
            ):

                value = bss[
                    jj,
                    kk,
                    ii,
                ]

                centers.append(
                    value
                )

                all_centers[
                    (
                        sev,
                        horizon,
                        label,
                    )
                ] = value

                # PBC gets an x hatch if it significantly
                # improves over both baseline models.
                if (
                    label == "PBC-ECMWF"
                    and np.all(
                        ci_low[
                            jj,
                            kk,
                            :
                        ][
                            :len(MODEL_LABELS) - 1
                        ] > 0
                    )
                ):

                    hatches.append("x")

                else:

                    hatches.append(None)

        offset = (
            ii
            - (
                len(MODEL_LABELS)
                - 1
            ) / 2
        ) * width

        # Draw bars without hatch so that the legend
        # remains clean.
        ax.bar(
            x_positions + offset,
            centers,
            width=width,
            color=MODEL_COLORS[label],
            label=label,
            zorder=3,
        )

        # Overlay hatching without creating legend entries.
        for jj, hatch in enumerate(
            hatches
        ):

            if hatch is not None:

                ax.bar(
                    x_positions[jj] + offset,
                    centers[jj],
                    width=width,
                    color="none",
                    edgecolor="black",
                    linewidth=0,
                    hatch=hatch,
                    label="_nolegend_",
                    zorder=4,
                )

    # --------------------------------------------------------------
    # Report improvements
    # --------------------------------------------------------------
    printf(
        "\n% Improvements of "
        "PBC-ECMWF over baselines:"
    )

    printf("-" * 80)

    for sev in severity_keys:

        for horizon in HORIZONS:

            week = (
                "Week 3"
                if horizon == "19"
                else "Week 4"
            )

            pbc_bss = all_centers.get(
                (
                    sev,
                    horizon,
                    "PBC-ECMWF",
                ),
                np.nan,
            )

            raw_bss = all_centers.get(
                (
                    sev,
                    horizon,
                    "Raw ECMWF",
                ),
                np.nan,
            )

            deb_bss = all_centers.get(
                (
                    sev,
                    horizon,
                    "Debiased ECMWF",
                ),
                np.nan,
            )

            if (
                not np.isnan(pbc_bss)
                and not np.isnan(raw_bss)
                and raw_bss != 0
            ):

                imp_raw = (
                    100
                    * (
                        pbc_bss
                        - raw_bss
                    )
                    / raw_bss
                )

            else:

                imp_raw = np.nan

            if (
                not np.isnan(pbc_bss)
                and not np.isnan(deb_bss)
                and deb_bss != 0
            ):

                imp_deb = (
                    100
                    * (
                        pbc_bss
                        - deb_bss
                    )
                    / deb_bss
                )

            else:

                imp_deb = np.nan

            printf(
                f"  {sev} ({week}): "
                f"vs Raw ECMWF "
                f"({raw_bss}): "
                f"{imp_raw:+.2f}% | "
                f"vs Debiased ECMWF "
                f"({deb_bss}): "
                f"{imp_deb:+.2f}%"
            )

    # --------------------------------------------------------------
    # X axis
    # --------------------------------------------------------------
    week_labels = [
        "Week 3",
        "Week 4",
    ]

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        week_labels * n_sev,
        fontsize=20,
    )

    severity_names_display = {
        "All": "All Floods",
        "Severe": "Severe Floods",
    }

    sev_names = [
        severity_names_display.get(
            s,
            s,
        )
        for s in severity_keys
    ]

    for s in range(n_sev):

        severity_positions = (
            x_positions[
                s * n_horiz:
                (s + 1) * n_horiz
            ]
        )

        center = (
            severity_positions[0]
            + severity_positions[-1]
        ) / 2

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

    # --------------------------------------------------------------
    # Axes
    # --------------------------------------------------------------
    ax.set_ylabel(
        "Brier skill score",
        fontsize=20,
        fontweight="bold",
    )

    ax.tick_params(
        axis="y",
        labelsize=16,
    )

    ax.xaxis.grid(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    # --------------------------------------------------------------
    # Legend
    # --------------------------------------------------------------
    ax.legend(
        ncol=len(MODEL_LABELS),
        fontsize=20,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    plt.tight_layout()

    # ==============================================================
    # Save source data
    # ==============================================================
    if source_data:

        # ----------------------------------------------------------
        # Source-data output path
        # ----------------------------------------------------------
        fig_filename = os.path.join(
            SRC_DATA_DIR,
            source_data_filename,
        )

        fig_filename = Path(
            fig_filename
        )

        fig_filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------------
        # BSS source data
        #
        # One row per severity / horizon / model.
        # This directly contains the values represented by bars.
        # ----------------------------------------------------------
        bss_rows = []

        for jj, severity in enumerate(
            severity_keys
        ):

            for kk, horizon in enumerate(
                HORIZONS
            ):

                week = (
                    "Week 3"
                    if horizon == "19"
                    else "Week 4"
                )

                for ii, model in enumerate(
                    MODEL_LABELS
                ):

                    bss_rows.append({
                        "Severity": severity,
                        "Severity_Display": severity_names_display.get(
                            severity,
                            severity,
                        ),
                        "Horizon": horizon,
                        "Week": week,
                        "Model": model,
                        "BSS": bss[
                            jj,
                            kk,
                            ii,
                        ],
                    })

        bss_df = pd.DataFrame(
            bss_rows
        )

        # ----------------------------------------------------------
        # Significance source data
        #
        # Contains the lower confidence bounds used to determine
        # whether the PBC bar receives an x hatch.
        # ----------------------------------------------------------
        significance_rows = []

        for jj, severity in enumerate(
            severity_keys
        ):

            for kk, horizon in enumerate(
                HORIZONS
            ):

                week = (
                    "Week 3"
                    if horizon == "19"
                    else "Week 4"
                )

                baseline_significant = []

                for ii, model in enumerate(
                    MODEL_LABELS
                ):

                    if model == "PBC-ECMWF":
                        continue

                    lower_bound = ci_low[
                        jj,
                        kk,
                        ii,
                    ]

                    significant = (
                        np.isfinite(lower_bound)
                        and lower_bound > 0
                    )

                    baseline_significant.append(
                        significant
                    )

                    significance_rows.append({
                        "Severity": severity,
                        "Severity_Display": severity_names_display.get(
                            severity,
                            severity,
                        ),
                        "Horizon": horizon,
                        "Week": week,
                        "Target_Model": "PBC-ECMWF",
                        "Baseline_Model": model,
                        "Lower_Confidence_Bound": (
                            lower_bound
                            if np.isfinite(lower_bound)
                            else np.nan
                        ),
                        "Confidence_Level": confidence,
                        "Bootstrap_Method": method,
                        "Tail": tail,
                        "N_Boot": n_boot,
                        "Random_Seed": random_seed,
                        "Significant": significant,
                    })

                # This is exactly the condition used for the
                # cross-hatched PBC bar.
                significance_rows.append({
                    "Severity": severity,
                    "Severity_Display": severity_names_display.get(
                        severity,
                        severity,
                    ),
                    "Horizon": horizon,
                    "Week": week,
                    "Target_Model": "PBC-ECMWF",
                    "Baseline_Model": "ALL",
                    "Lower_Confidence_Bound": np.nan,
                    "Confidence_Level": confidence,
                    "Bootstrap_Method": method,
                    "Tail": tail,
                    "N_Boot": n_boot,
                    "Random_Seed": random_seed,
                    "Significant": bool(
                        baseline_significant
                        and all(
                            baseline_significant
                        )
                    ),
                })

        significance_df = pd.DataFrame(
            significance_rows
        )

        # ----------------------------------------------------------
        # Underlying Brier scores
        #
        # These are the actual event-level values from which the
        # plotted BSS values were calculated.
        # ----------------------------------------------------------
        brier_rows = []

        for severity in severity_keys:

            severity_data = all_severities.get(
                severity,
                {},
            )

            for model in (
                ["Climatology"]
                + MODEL_LABELS
            ):

                model_data = severity_data.get(
                    model,
                    {},
                )

                for horizon in HORIZONS:

                    values = model_data.get(
                        horizon,
                        [],
                    )

                    for event_index, value in enumerate(
                        values
                    ):

                        brier_rows.append({
                            "Severity": severity,
                            "Severity_Display": severity_names_display.get(
                                severity,
                                severity,
                            ),
                            "Horizon": horizon,
                            "Week": (
                                "Week 3"
                                if horizon == "19"
                                else "Week 4"
                            ),
                            "Model": model,
                            "Event_Index": event_index,
                            "Brier_Score": value,
                        })

        brier_df = pd.DataFrame(
            brier_rows
        )

        # ----------------------------------------------------------
        # Write / append workbook
        #
        # If the workbook exists, only sheets belonging to this
        # function are replaced. Other figure source-data sheets
        # remain untouched.
        # ----------------------------------------------------------
        sheet_names = {
            "bss": (
                f"{source_data_sheet_prefix}_bss"
            ),
            "significance": (
                f"{source_data_sheet_prefix}_significance"
            ),
            "brier_scores": (
                f"{source_data_sheet_prefix}_brier_scores"
            ),
        }

        if fig_filename.exists():

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:

                bss_df.to_excel(
                    writer,
                    sheet_name=sheet_names["bss"],
                    index=False,
                )

                significance_df.to_excel(
                    writer,
                    sheet_name=sheet_names["significance"],
                    index=False,
                )

                brier_df.to_excel(
                    writer,
                    sheet_name=sheet_names["brier_scores"],
                    index=False,
                )

        else:

            with pd.ExcelWriter(
                fig_filename,
                engine="openpyxl",
                mode="w",
            ) as writer:

                bss_df.to_excel(
                    writer,
                    sheet_name=sheet_names["bss"],
                    index=False,
                )

                significance_df.to_excel(
                    writer,
                    sheet_name=sheet_names["significance"],
                    index=False,
                )

                brier_df.to_excel(
                    writer,
                    sheet_name=sheet_names["brier_scores"],
                    index=False,
                )

        # Match the behavior of the figure/source-data functions
        # elsewhere in the codebase.
        set_file_permissions(
            fig_filename
        )

        printf(
            f"  Source data saved → "
            f"{fig_filename}"
        )

    # --------------------------------------------------------------
    # Save figure
    # --------------------------------------------------------------
    if save_fig:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        eps_output_path = output_path.with_suffix(".eps")

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        set_file_permissions(
            output_path
        )

        fig.savefig(
            eps_output_path,
            dpi=300,
            bbox_inches="tight",
        )

        set_file_permissions(
            eps_output_path
        )

        printf(
            f"  Saved {output_path}"
        )

    # --------------------------------------------------------------
    # Display / close
    # --------------------------------------------------------------
    if show_fig:

        plt.show()

    else:

        plt.close(fig)

    return fig, bss
