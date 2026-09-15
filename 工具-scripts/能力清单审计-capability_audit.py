#!/usr/bin/env python3
"""Validate the machine-readable Theory ↔ Code ↔ Test ↔ Evidence manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "能力清单-CAPABILITIES.json"

REQUIRED_FIELDS = {
    "id",
    "track",
    "title",
    "status",
    "maturity",
    "theory",
    "code",
    "tests",
    "evidence",
    "next",
}


def _paths(value: object, field: str, capability_id: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{capability_id}: {field} must be a list of strings")
    return value


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    statuses = set(data["status_values"])
    maturities = set(data["maturity_values"])
    tracks = set(data["tracks"])
    capabilities = data["capabilities"]

    if not isinstance(capabilities, list):
        raise ValueError("capabilities must be a list")

    seen: set[str] = set()
    errors: list[str] = []
    status_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()

    for item in capabilities:
        if not isinstance(item, dict):
            errors.append("capability entry is not an object")
            continue
        capability_id = str(item.get("id", "<missing-id>"))
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(f"{capability_id}: missing fields {sorted(missing)}")
            continue
        if capability_id in seen:
            errors.append(f"{capability_id}: duplicate id")
        seen.add(capability_id)

        status = item["status"]
        maturity = item["maturity"]
        track = item["track"]
        if status not in statuses:
            errors.append(f"{capability_id}: invalid status {status!r}")
        if maturity not in maturities:
            errors.append(f"{capability_id}: invalid maturity {maturity!r}")
        if track not in tracks:
            errors.append(f"{capability_id}: invalid track {track!r}")
        status_counts[str(status)] += 1
        track_counts[str(track)] += 1

        theory = _paths(item["theory"], "theory", capability_id)
        code = _paths(item["code"], "code", capability_id)
        tests = _paths(item["tests"], "tests", capability_id)
        evidence = _paths(item["evidence"], "evidence", capability_id)
        _paths(item["next"], "next", capability_id)

        for field, paths in (("theory", theory), ("code", code), ("tests", tests)):
            for relative in paths:
                if not (ROOT / relative).exists():
                    errors.append(f"{capability_id}: {field} path does not exist: {relative}")

        if status in {"implemented", "validated"} and not code:
            errors.append(f"{capability_id}: {status} capability must reference code")
        if status == "validated" and not tests:
            errors.append(f"{capability_id}: validated capability must reference tests")
        if status == "validated" and not evidence:
            errors.append(f"{capability_id}: validated capability must contain evidence")

    print("# Capability manifest audit")
    print()
    print(f"Capabilities: {len(capabilities)}")
    print("Status counts:", dict(sorted(status_counts.items())))
    print("Track counts:", dict(sorted(track_counts.items())))
    print()

    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Capability manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
