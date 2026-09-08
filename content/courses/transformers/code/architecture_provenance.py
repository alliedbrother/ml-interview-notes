"""Check the architecture evidence ledger and pinned vendor JSON snapshots offline.

Run from a repository checkout: python code/architecture_provenance.py
No model loading, remote-code execution, authentication, or network requests.
Implementation reviews and vendor reports are evidence records, not executable
proofs of model behavior. This checker does not independently recount weights.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent / "provenance"
CELL_NAMES = {"size", "attention", "moe", "shared_expert", "norm", "position"}
STATES = {"config-backed", "implementation-reviewed", "vendor-reported",
          "partial", "unverified", "not-applicable"}


def json_pointer(document, pointer):
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Expected an RFC 6901 JSON pointer")
    value = document
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise ValueError("Invalid JSON-pointer escape")
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            value = value[int(key)]
        else:
            raise ValueError(f"Cannot traverse pointer component {key!r}")
    return value


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Snapshot path escapes evidence directory")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def check_ledger(root=ROOT):
    root = Path(root)
    ledger = json.loads((root / "ledger.json").read_text())
    if ledger.get("format_version") != 1:
        raise ValueError("Unsupported ledger format")
    configs = ledger["configs"]
    references = ledger["references"]
    config_ids = {source["id"] for source in configs}
    reference_ids = {source["id"] for source in references}
    if len(config_ids) != len(configs) or len(reference_ids) != len(references):
        raise ValueError("Duplicate source identity")
    facts_checked = 0
    downloaded = 0
    for source in configs:
        revision = source.get("revision", "")
        if not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError("Config revision must be a full immutable commit SHA")
        prefix = f"https://huggingface.co/{source['repository']}/resolve/{revision}/"
        if source.get("source_url") not in (prefix + "config.json", prefix + "params.json"):
            raise ValueError("Unexpected config source URL")
        if source["status"] == "unavailable":
            if source.get("facts") or source.get("snapshot"):
                raise ValueError("Unavailable config cannot claim checked facts or a snapshot")
            continue
        if source["status"] != "downloaded":
            raise ValueError("Unknown config status")
        raw = within(root, source["snapshot"]).read_bytes()
        if len(raw) != source["bytes"] or hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError(f"Snapshot integrity mismatch: {source['id']}")
        document = json.loads(raw)
        pointers = set()
        for fact in source["facts"]:
            if fact["pointer"] in pointers:
                raise ValueError("Duplicate field assertion")
            pointers.add(fact["pointer"])
            actual = json_pointer(document, fact["pointer"])
            if canonical(actual) != canonical(fact["expected"]):
                raise ValueError(f"Field mismatch: {source['id']} {fact['pointer']}")
            facts_checked += 1
        downloaded += 1
    for source in references:
        if not re.search(r"/[0-9a-f]{40}/", source["url"]):
            raise ValueError("Reference URL must contain an immutable revision")
        if not re.fullmatch("[0-9a-f]{64}", source["sha256"]):
            raise ValueError("Missing recorded reference hash")
        if source.get("offline_semantics_rechecked") is not False:
            raise ValueError("Offline checker does not revalidate reference semantics")
        for locator in source.get("locators", []):
            if not locator["symbol"] or not 0 < locator["start_line"] <= locator["end_line"]:
                raise ValueError("Invalid reviewed symbol locator")
    cell_states = Counter()
    seen_rows = set()
    for row in ledger["rows"]:
        if row["id"] in seen_rows:
            raise ValueError("Duplicate comparison row")
        seen_rows.add(row["id"])
        if set(row["cells"]) != CELL_NAMES:
            raise ValueError("Every comparison column needs an evidence status")
        if not set(row["config_ids"]) <= config_ids or not row["notes"]:
            raise ValueError("Row needs valid config identities and scope notes")
        for cell in row["cells"].values():
            if cell["status"] not in STATES or not cell.get("scope"):
                raise ValueError("Invalid cell evidence state")
            if not set(cell.get("source_ids", [])) <= config_ids:
                raise ValueError("Unknown config reference")
            if not set(cell.get("reference_ids", [])) <= reference_ids:
                raise ValueError("Unknown implementation/report reference")
            if cell["status"] == "config-backed":
                selected = [source for source in configs if source["id"] in cell.get("source_ids", [])]
                if not selected or any(source["status"] != "downloaded" for source in selected):
                    raise ValueError("Config-backed cell requires available pinned bytes")
                evidence = cell.get("field_evidence", [])
                if {item["source_id"] for item in evidence} != {source["id"] for source in selected}:
                    raise ValueError("Config-backed cell must identify its field evidence")
                for item in evidence:
                    source = next(source for source in selected if source["id"] == item["source_id"])
                    known = {fact["pointer"] for fact in source["facts"]}
                    if not item.get("pointers") or not set(item["pointers"]) <= known:
                        raise ValueError("Cell field pointer is not a checked fact")
            cell_states[cell["status"]] += 1
    return {"rows": len(ledger["rows"]), "cells": sum(cell_states.values()),
            "config_attempts": len(configs), "public_json_snapshots": downloaded,
            "unavailable_configs": len(configs) - downloaded, "field_assertions": facts_checked,
            "reference_records": len(references), "cell_states": dict(sorted(cell_states.items())),
            "network_requests": 0, "model_weights_loaded": 0,
            "scope": "Checks pinned JSON values and ledger structure, not model execution or report truth."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(check_ledger(args.root), indent=2))
