#!/usr/bin/env python3
"""Validate the machine-readable frontier snapshot and report staleness."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "观测站-observatory" / "frontier-snapshot.json"

REQUIRED_ITEM_FIELDS = {
    "id",
    "category",
    "name",
    "released",
    "source",
    "facts",
    "evidence_boundary",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-age",
        action="store_true",
        help="return non-zero when the snapshot age exceeds revalidate_after_days",
    )
    args = parser.parse_args()

    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    as_of = date.fromisoformat(data["as_of"])
    threshold = int(data["revalidate_after_days"])
    age_days = (date.today() - as_of).days
    items = data["items"]

    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        missing = REQUIRED_ITEM_FIELDS - item.keys()
        item_id = str(item.get("id", "<missing-id>"))
        if missing:
            errors.append(f"{item_id}: missing fields {sorted(missing)}")
            continue
        if item_id in seen:
            errors.append(f"{item_id}: duplicate id")
        seen.add(item_id)

        source = str(item["source"])
        parsed = urlparse(source)
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"{item_id}: source must be an absolute https URL")
        if not isinstance(item["facts"], list) or not item["facts"]:
            errors.append(f"{item_id}: facts must be a non-empty list")
        if not str(item["evidence_boundary"]).strip():
            errors.append(f"{item_id}: evidence_boundary must be non-empty")

    print("# Frontier snapshot audit")
    print()
    print(f"As of: {as_of.isoformat()}")
    print(f"Age: {age_days} days")
    print(f"Revalidate after: {threshold} days")
    print(f"Items: {len(items)}")
    print(f"Freshness: {'STALE' if age_days > threshold else 'fresh'}")
    print()

    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    if args.strict_age and age_days > threshold:
        print("Snapshot age exceeds the configured revalidation window.")
        return 2

    print("Frontier snapshot schema: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
