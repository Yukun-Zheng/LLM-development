#!/usr/bin/env python3
"""Validate machine-readable Theory ↔ Code ↔ Test ↔ Evidence manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "能力清单-CAPABILITIES.json"
OVERLAY_GLOB = "能力清单-CAPABILITIES-*.json"

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


def _load_manifests() -> tuple[dict[str, object], list[tuple[Path, dict[str, object]]]]:
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    overlays: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(ROOT.glob(OVERLAY_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        extends = data.get("extends")
        if extends not in {None, BASE_MANIFEST.name}:
            raise ValueError(f"{path.name}: unsupported extends target {extends!r}")
        overlays.append((path, data))
    return base, overlays


def main() -> int:
    data, overlays = _load_manifests()
    statuses = set(data["status_values"])
    maturities = set(data["maturity_values"])
    tracks = set(data["tracks"])

    sources: list[tuple[str, object]] = [
        (BASE_MANIFEST.name, data.get("capabilities")),
        *[(path.name, overlay.get("capabilities")) for path, overlay in overlays],
    ]

    capabilities: list[tuple[str, object]] = []
    errors: list[str] = []
    for source_name, entries in sources:
        if not isinstance(entries, list):
            errors.append(f"{source_name}: capabilities must be a list")
            continue
        capabilities.extend((source_name, item) for item in entries)

    seen: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()

    for source_name, item in capabilities:
        if not isinstance(item, dict):
            errors.append(f"{source_name}: capability entry is not an object")
            continue
        capability_id = str(item.get("id", "<missing-id>"))
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(
                f"{source_name}:{capability_id}: missing fields {sorted(missing)}"
            )
            continue
        if capability_id in seen:
            errors.append(
                f"{capability_id}: duplicate id in {seen[capability_id]} and {source_name}"
            )
        seen[capability_id] = source_name

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
                    errors.append(
                        f"{capability_id}: {field} path does not exist: {relative}"
                    )

        if status in {"implemented", "validated"} and not code:
            errors.append(f"{capability_id}: {status} capability must reference code")
        if status == "validated" and not tests:
            errors.append(f"{capability_id}: validated capability must reference tests")
        if status == "validated" and not evidence:
            errors.append(f"{capability_id}: validated capability must contain evidence")

    print("# Capability manifest audit")
    print()
    print("Manifest files:")
    for source_name, _ in sources:
        print(f"- {source_name}")
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

    print("Capability manifests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
