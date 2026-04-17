#!/usr/bin/env python3
"""
Download flood events from the GDACS (Global Disaster Alert and Coordination
System) API and save them in the EVENTS dict format used by event-precip.py.

GDACS provides event-level flood data (not individual satellite tiles), so
queries return in seconds.  Uses only the Python standard library + no auth.

Usage:
    python download-flood-events.py \\
        --start 2022-01-01 --end 2026-12-31 \\
        --out-json eval/viz/extreme-brier-barplot-floods/data/flood-events-gdacs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.file_io import set_file_permissions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDACS_API = (
    "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    "?eventlist={event_types}"
    "&fromDate={start}"
    "&toDate={end}"
    "&alertlevel={alert_levels}"
    "&pagenumber={page}"
)

PAGE_SIZE = 100  # GDACS default max per page

DATA_DIR = Path("eval/viz/extreme-brier-barplot-floods/data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Lowercase, replace spaces/special chars with hyphens."""
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch in (" ", "_", ","):
            if out and out[-1] != "-":
                out.append("-")
    return "".join(out).strip("-")


def _lon_to_360(lon: float) -> float:
    """Convert longitude from [-180, 180] to [0, 360) convention."""
    return lon % 360


# ---------------------------------------------------------------------------
# Fetch events from GDACS
# ---------------------------------------------------------------------------


def fetch_gdacs_events(
    start: str,
    end: str,
    alert_levels: str = "Green;Orange;Red",
    event_types: str = "FL",
) -> List[Dict[str, Any]]:
    """
    Query the GDACS REST API for disaster events, paginating automatically.

    The API returns at most 100 events per page. This function loops through
    pages until an empty result is returned.
    """
    all_features: List[Dict[str, Any]] = []
    page = 1

    while True:
        url = GDACS_API.format(
            event_types=event_types,
            start=start,
            end=end,
            alert_levels=alert_levels,
            page=page,
        )
        if page == 1:
            print(f"Querying GDACS API …\n  {url}", file=sys.stderr)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        features = data.get("features", [])
        print(
            f"  page {page}: {len(features)} events", file=sys.stderr,
        )

        if not features:
            break

        all_features.extend(features)

        if len(features) < PAGE_SIZE:
            # Last page (fewer than max results)
            break

        page += 1

    print(f"  total: {len(all_features)} events", file=sys.stderr)
    return all_features


# ---------------------------------------------------------------------------
# Convert GDACS features → EVENTS dict
# ---------------------------------------------------------------------------


def features_to_events(
    features: List[Dict[str, Any]],
    padding: float = 10.0,
    event_types: Optional[set] = None,
) -> Dict[str, Dict]:
    """
    Convert GDACS GeoJSON features into the EVENTS dict format.

    Parameters
    ----------
    features : list of GeoJSON Feature dicts
    padding  : degrees to add around the centroid for the bounding box
    event_types : optional set of GDACS event types to keep (e.g. {"FL"})
    """
    events: Dict[str, Dict] = {}
    slug_counts: Dict[str, int] = {}

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        etype = props.get("eventtype", "")
        if event_types and etype not in event_types:
            continue

        # Centroid coordinates
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        center_lon, center_lat = coords[0], coords[1]

        # Date: use fromdate as the event start
        from_date_str = props.get("fromdate", "")
        if not from_date_str:
            continue
        try:
            from_dt = datetime.fromisoformat(from_date_str)
        except (ValueError, TypeError):
            continue

        date_str = from_dt.strftime("%Y-%m-%d")
        year = from_dt.year

        # Build slug from country name + year
        country = props.get("country", "unknown")
        event_name = props.get("eventname", "")
        if event_name:
            base_slug = f"{_slugify(event_name)}"
        else:
            base_slug = f"flood-{_slugify(country)}-{year}"

        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        if slug_counts[base_slug] == 1:
            slug = base_slug
        else:
            slug = f"{base_slug}-{slug_counts[base_slug]}"

        # Build bounding box with padding, in 0-360 longitude convention
        lat_lo = max(center_lat - padding, -90.0)
        lat_hi = min(center_lat + padding, 90.0)
        lon_lo = _lon_to_360(center_lon - padding)
        lon_hi = _lon_to_360(center_lon + padding)

        # If the box crosses the antimeridian in 0-360, keep lon_hi > 360
        if lon_hi < lon_lo:
            lon_hi += 360

        events[slug] = {
            "date": date_str,
            "lat": (round(lat_lo, 1), round(lat_hi, 1)),
            "lon": (round(lon_lo, 1), round(lon_hi, 1)),
        }

    return events


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_json(events: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        k: {"date": v["date"], "lat": list(v["lat"]), "lon": list(v["lon"])}
        for k, v in events.items()
    }
    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    set_file_permissions(path)
    print(f"Saved {len(events)} events → {path}", file=sys.stderr)


def load_existing_events(path: Path) -> Dict:
    """Load previously saved events JSON, returning empty dict if absent."""
    if not path.exists():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {
        k: {"date": v["date"], "lat": tuple(v["lat"]), "lon": tuple(v["lon"])}
        for k, v in data.items()
    }


def print_python_literal(events: Dict) -> None:
    print("\n# ---- copy-paste into event-precip.py ----")
    print("EVENTS = {")
    for slug, info in events.items():
        lat, lon = info["lat"], info["lon"]
        print(
            f'    "{slug}":'
            f' {{"date": "{info["date"]}",'
            f' "lat": ({lat[0]}, {lat[1]}),'
            f' "lon": ({lon[0]}, {lon[1]})}},',
        )
    print("}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download flood events from the GDACS API and save in the "
            "EVENTS dict format used by event-precip.py."
        ),
    )
    parser.add_argument(
        "--start", default="2022-01-01",
        help="Start date YYYY-MM-DD (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end", default="2026-12-31",
        help="End date YYYY-MM-DD (default: 2026-12-31)",
    )
    parser.add_argument(
        "--alert-levels", default="Green;Orange;Red",
        help="GDACS alert levels to include (default: Green;Orange;Red)",
    )
    parser.add_argument(
        "--padding", type=float, default=10.0,
        help="Padding (°) to add around the event centroid (default: 10)",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=DATA_DIR / "flood-events-gdacs.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing JSON output instead of overwriting.",
    )
    parser.add_argument(
        "--print-python", action="store_true", default=True,
        help="Print events as Python dict literal (default: True)",
    )
    parser.add_argument(
        "--no-print-python", action="store_false", dest="print_python",
    )

    args = parser.parse_args()

    # --- Fetch flood events ---
    features = fetch_gdacs_events(
        start=args.start,
        end=args.end,
        alert_levels=args.alert_levels,
        event_types="FL",
    )

    if not features:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    # Filter to FL only (API may return other types if query includes them)
    events = features_to_events(
        features,
        padding=args.padding,
        event_types={"FL"},
    )

    if not events:
        print("No flood events after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(events)} flood events extracted", file=sys.stderr)

    # --- Merge with existing if --append ---
    if args.append:
        existing = load_existing_events(args.out_json)
        n_before = len(existing)
        existing.update(events)
        events = existing
        print(
            f"Appended {len(events) - n_before} new events "
            f"(total now: {len(events)})",
            file=sys.stderr,
        )

    # --- Output ---
    save_json(events, args.out_json)
    if args.print_python:
        print_python_literal(events)


if __name__ == "__main__":
    main()
