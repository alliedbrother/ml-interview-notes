"""Deterministic educational scheduler and paged-KV ownership model (stdlib only).

Times are integer simulation ticks, NOT measured milliseconds. A dispatch consumes
one tick regardless of its work. This is an accounting model, not an engine replica.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import heapq
import json
from pathlib import Path
from typing import Any


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass
class Page:
    owners: set[str] = field(default_factory=set)
    pins: set[str] = field(default_factory=set)
    tokens: list[int] = field(default_factory=list)


class PagedKV:
    """One cache group with shared pages, synchronous CoW, and explicit DMA pins.

    Token integers stand in for KV rows. A pin retains storage, not a request name.
    No cached zero-reference eviction list: ownerless, unpinned pages are free.
    """

    def __init__(self, capacity: int, block_size: int):
        self.capacity = positive_int(capacity, "capacity")
        self.block_size = positive_int(block_size, "block_size")
        self.pages: dict[int, Page] = {}
        self.tables: dict[str, list[int]] = {}
        self.free = list(range(capacity))
        self.transfers: dict[str, tuple[int, ...]] = {}

    def create(self, owner: str) -> None:
        if owner in self.tables:
            raise ValueError("owner already exists")
        self.tables[owner] = []

    def _allocate(self, owner: str, tokens: list[int] | None = None) -> int:
        if not self.free:
            raise MemoryError("no reusable KV page")
        page_id = heapq.heappop(self.free)
        self.pages[page_id] = Page({owner}, set(), list(tokens or []))
        return page_id

    def _collect(self, page_id: int) -> None:
        page = self.pages[page_id]
        if not page.owners and not page.pins:
            del self.pages[page_id]
            heapq.heappush(self.free, page_id)

    def fork(self, source: str, target: str) -> None:
        if target in self.tables:
            raise ValueError("target already exists")
        table = self.tables[source]
        self.tables[target] = list(table)
        for page_id in table:
            self.pages[page_id].owners.add(target)

    def read(self, owner: str) -> list[int]:
        return [token for p in self.tables[owner] for token in self.pages[p].tokens]

    def append(self, owner: str, token: int) -> None:
        table = self.tables[owner]
        if not table or len(self.pages[table[-1]].tokens) == self.block_size:
            table.append(self._allocate(owner))
        else:
            old_id = table[-1]
            old = self.pages[old_id]
            # Pins freeze a transfer's source view even when there is one owner.
            if len(old.owners) > 1 or old.pins:
                new_id = self._allocate(owner, old.tokens)
                table[-1] = new_id
                old.owners.remove(owner)
                self._collect(old_id)
        self.pages[table[-1]].tokens.append(token)

    def pin(self, owner: str, transfer: str) -> tuple[int, ...]:
        if transfer in self.transfers:
            raise ValueError("transfer already exists")
        pages = tuple(self.tables[owner])
        self.transfers[transfer] = pages
        for page_id in pages:
            self.pages[page_id].pins.add(transfer)
        return pages

    def unpin(self, transfer: str) -> None:
        pages = self.transfers.pop(transfer)
        for page_id in pages:
            self.pages[page_id].pins.remove(transfer)
            self._collect(page_id)

    def release(self, owner: str) -> None:
        for page_id in self.tables.pop(owner):
            self.pages[page_id].owners.remove(owner)
            self._collect(page_id)

    def check(self) -> None:
        assert len(self.free) == len(set(self.free))
        assert set(self.free).isdisjoint(self.pages)
        assert set(self.free) | set(self.pages) == set(range(self.capacity))
        for owner, table in self.tables.items():
            assert len(table) == len(set(table))
            for offset, page_id in enumerate(table):
                page = self.pages[page_id]
                assert owner in page.owners
                assert 1 <= len(page.tokens) <= self.block_size
                if offset < len(table) - 1:
                    assert len(page.tokens) == self.block_size
        for page_id, page in self.pages.items():
            assert page.owners or page.pins
            assert page.owners == {o for o, t in self.tables.items() if page_id in t}
            assert page.pins == {key for key, ids in self.transfers.items() if page_id in ids}


@dataclass
class Request:
    name: str
    arrival: int
    prompt: int
    output: int
    status: str = "queued"
    prefetched: int = 0
    generated: int = 0
    admitted: int | None = None
    first_token: int | None = None
    end: int | None = None
    token_times: list[int] = field(default_factory=list)

    def blocks(self, block_size: int) -> int:
        # The final sampled token need not be fed back through the model.
        return (self.prompt + self.output - 2 + block_size) // block_size


def validate_trace(events: list[dict]) -> list[dict]:
    if not isinstance(events, list):
        raise ValueError("trace must be a list of events")
    names: dict[str, int] = {}
    cleaned = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("each event must be an object")
        kind = event.get("event")
        expected = {"event", "time", "id"}
        if kind == "arrival":
            expected |= {"prompt_tokens", "output_tokens"}
        elif kind != "cancel":
            raise ValueError("event must be arrival or cancel")
        if set(event) != expected:
            raise ValueError(f"invalid fields for {kind}: expected {sorted(expected)}")
        positive_int(event["time"], "time", 0)
        name = event["id"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("id must be a nonempty string")
        if kind == "arrival":
            if name in names:
                raise ValueError("request IDs must be unique across the entire trace")
            names[name] = event["time"]
            positive_int(event["prompt_tokens"], "prompt_tokens")
            positive_int(event["output_tokens"], "output_tokens")
        cleaned.append(dict(event))
    for event in cleaned:
        if event["event"] == "cancel":
            if event["id"] not in names or event["time"] < names[event["id"]]:
                raise ValueError("cancellation must name a request arriving at or before its time")
    return sorted(cleaned, key=lambda e: (e["time"], e["event"] == "cancel"))


def simulate(events: list[dict], *, capacity: int = 8, block_size: int = 4,
             token_budget: int = 4, max_active: int = 4, prefill_chunk: int = 2,
             release_delay: int = 1, policy: str = "continuous") -> dict:
    for name, value in (("token_budget", token_budget), ("max_active", max_active),
                        ("prefill_chunk", prefill_chunk)):
        positive_int(value, name)
    positive_int(release_delay, "release_delay", 0)
    if policy not in {"serial", "continuous"}:
        raise ValueError("policy must be serial or continuous")
    trace = validate_trace(events)
    pool = PagedKV(capacity, block_size)
    requests: dict[str, Request] = {}
    waiting: list[str] = []
    active: list[str] = []
    releases: list[tuple[int, str]] = []
    log: list[dict] = []
    now = cursor = dispatches = processed = 0
    decode_turn = prefill_turn = 0
    limit = 1 if policy == "serial" else max_active

    def emit(kind: str, **details: Any) -> None:
        log.append({"time": now, "event": kind, **details})

    def retire(request: Request, status: str) -> None:
        request.status, request.end = status, now
        if request.name in active:
            active.remove(request.name)
            heapq.heappush(releases, (now + release_delay, request.name))
        if request.name in waiting:
            waiting.remove(request.name)
        emit(status, id=request.name)

    def accounting() -> dict:
        # Draining pages cannot be reused; active credits include future growth.
        reserved = sum(requests[name].blocks(block_size) for name in active)
        draining = sum(len(pool.tables[name]) for _, name in releases)
        return {"reserved_blocks": reserved, "draining_blocks": draining,
                "allocated_blocks": len(pool.pages), "free_blocks": len(pool.free),
                "admission_headroom": capacity - reserved - draining,
                "block_tables": {name: list(table) for name, table in pool.tables.items()}}

    while cursor < len(trace) or waiting or active or releases:
        while cursor < len(trace) and trace[cursor]["time"] == now:
            event = trace[cursor]
            cursor += 1
            name = event["id"]
            if event["event"] == "arrival":
                requests[name] = Request(name, now, event["prompt_tokens"], event["output_tokens"])
                waiting.append(name)
                emit("arrival", id=name)
            else:
                request = requests[name]
                if request.status in {"queued", "prefill", "decode"}:
                    retire(request, "cancelled")
                else:
                    emit("cancel_ignored", id=name, status=request.status)
        while releases and releases[0][0] <= now:
            _, name = heapq.heappop(releases)
            pool.release(name)
            emit("release", id=name)
        while waiting and len(active) < limit:
            request = requests[waiting[0]]
            need = request.blocks(block_size)
            if need > capacity:
                retire(request, "rejected")
                continue
            if need > accounting()["admission_headroom"]:
                emit("admission_blocked", id=request.name, required_blocks=need)
                break
            waiting.pop(0)
            active.append(request.name)
            pool.create(request.name)
            request.admitted, request.status = now, "prefill"
            emit("admit", id=request.name, reserved_blocks=need)

        budget = token_budget
        work: list[tuple[str, str, int]] = []
        # Decode rows get one input token each. Rotate the first considered row.
        decode = [name for name in active if requests[name].status == "decode"]
        if decode:
            start = decode_turn % len(decode)
            decode = decode[start:] + decode[:start]
            for name in decode[:budget]:
                work.append((name, "decode", 1))
                budget -= 1
            decode_turn += min(len(decode), token_budget)
        prefill = [name for name in active if requests[name].status == "prefill"]
        if prefill and budget:
            start = prefill_turn % len(prefill)
            prefill = prefill[start:] + prefill[:start]
            served = 0
            for name in prefill:
                count = min(prefill_chunk, requests[name].prompt - requests[name].prefetched, budget)
                if not count:
                    break
                work.append((name, "prefill", count))
                budget -= count
                served += 1
            prefill_turn += served

        pool.check()
        snapshot = accounting()
        assert snapshot["admission_headroom"] >= 0
        assert snapshot["allocated_blocks"] <= snapshot["reserved_blocks"] + snapshot["draining_blocks"]
        emit("snapshot", active=list(active), queued=list(waiting), **snapshot)
        if not work:
            next_times = ([trace[cursor]["time"]] if cursor < len(trace) else [])
            next_times += [time for time, _ in releases]
            if not next_times:
                assert not waiting and not active, "no forward progress"
                break
            now = min(next_times)
            continue
        dispatches += 1
        emit("dispatch", work=[{"id": n, "phase": phase, "input_tokens": count}
                               for n, phase, count in work], unused_budget=budget)
        now += 1
        for name, phase, count in work:
            request = requests[name]
            cached = request.prefetched + max(request.generated - 1, 0)
            for offset in range(count):
                pool.append(name, cached + offset)
            processed += count
            if phase == "prefill":
                request.prefetched += count
                if request.prefetched < request.prompt:
                    continue
                request.status, request.first_token = "decode", now
            request.generated += 1
            request.token_times.append(now)
            emit("token", id=name, output_index=request.generated)
            if request.generated == request.output:
                retire(request, "finished")
        pool.check()

    pool.check()
    assert not pool.pages and not pool.tables
    result = {}
    for name, r in requests.items():
        result[name] = {"status": r.status, "arrival": r.arrival,
                        "admitted": r.admitted, "end": r.end,
                        "output_tokens": r.generated, "token_times": r.token_times,
                        "queue_ticks": None if r.admitted is None else r.admitted - r.arrival,
                        "ttft_ticks": None if r.first_token is None else r.first_token - r.arrival,
                        "itl_ticks": [b - a for a, b in zip(r.token_times, r.token_times[1:])]}
    return {"schema_version": 1, "time_unit": "synthetic_tick",
            "config": {"capacity": capacity, "block_size": block_size,
                       "token_budget": token_budget, "max_active": max_active,
                       "prefill_chunk": prefill_chunk, "release_delay": release_delay,
                       "policy": policy}, "requests": result, "events": log,
            "summary": {"drained_at_tick": now, "dispatches": dispatches,
                        "processed_input_tokens": processed, "free_blocks": len(pool.free),
                        "finished": sum(r.status == "finished" for r in requests.values()),
                        "cancelled": sum(r.status == "cancelled" for r in requests.values()),
                        "rejected": sum(r.status == "rejected" for r in requests.values())}}


def demo_trace() -> list[dict]:
    return [{"time": 0, "event": "arrival", "id": "long", "prompt_tokens": 8, "output_tokens": 4},
            {"time": 0, "event": "arrival", "id": "short", "prompt_tokens": 2, "output_tokens": 2},
            {"time": 1, "event": "arrival", "id": "cancel-me", "prompt_tokens": 6, "output_tokens": 4},
            {"time": 2, "event": "cancel", "id": "cancel-me"},
            {"time": 2, "event": "arrival", "id": "late", "prompt_tokens": 3, "output_tokens": 2}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="JSONL arrival/cancel events; otherwise built-in demo")
    parser.add_argument("--output", type=Path, help="write full JSON report; otherwise stdout")
    parser.add_argument("--policy", choices=("continuous", "serial"), default="continuous")
    for name, default in (("capacity", 8), ("block-size", 4), ("token-budget", 4),
                          ("max-active", 4), ("prefill-chunk", 2), ("release-delay", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    try:
        events = demo_trace() if args.trace is None else [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
        config = {k: v for k, v in vars(args).items() if k not in {"trace", "output"}}
        report = simulate(events, **config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
