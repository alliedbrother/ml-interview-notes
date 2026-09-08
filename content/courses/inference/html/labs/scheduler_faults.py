"""CPU lifecycle exercises, not a scheduler policy or a GPU transport emulator.

Tokens stand for KV rows. Call order stands for serialized control-plane events;
no wall-clock latency, performance prediction, or actual DMA is modeled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from scheduler_sim import PagedKV


class RecomputeRequest:
    """Rebuild committed input rows without emitting sampled outputs twice."""

    def __init__(self, pool: PagedKV, owner: str, prompt: list[int]):
        if not owner or not prompt or any(type(t) is not int for t in prompt):
            raise ValueError("provide an owner and nonempty integer-token prompt")
        if owner in pool.tables:
            raise ValueError("owner already exists")
        self.pool, self.owner = pool, owner
        self.prompt = list(prompt)
        self.outputs: list[int] = []
        self.state = "queued"
        self.recomputed_rows = 0
        self.events: list[dict] = []

    @property
    def committed(self) -> list[int]:
        # The newest sampled token has not been forwarded through the model yet.
        return self.prompt + self.outputs[:-1]

    def resume(self) -> bool:
        if self.state not in {"queued", "preempted"}:
            raise ValueError("only queued or preempted requests can resume")
        if self.owner in self.pool.tables:
            raise ValueError("owner namespace was reused while request was suspended")
        self.pool.create(self.owner)
        try:
            for token in self.committed:
                self.pool.append(self.owner, token)
        except MemoryError:
            self.pool.release(self.owner)
            self.events.append({"event": "resume_blocked", "rows": len(self.committed)})
            self.check()
            return False
        replay = len(self.committed) if self.state == "preempted" else 0
        self.recomputed_rows += replay
        self.state = "active"
        self.events.append({"event": "resumed", "recomputed_rows": replay})
        self.check()
        return True

    def sample(self, token: int) -> None:
        """Caller supplies a hypothetical model result; this is not a sampler."""
        if self.state != "active" or type(token) is not int:
            raise ValueError("sampling requires an active request and integer token")
        if self.outputs:
            self.pool.append(self.owner, self.outputs[-1])
        self.outputs.append(token)
        self.events.append({"event": "emit", "index": len(self.outputs) - 1, "token": token})
        self.check()

    def preempt(self) -> None:
        if self.state != "active":
            raise ValueError("only active requests can be preempted")
        self.pool.release(self.owner)
        self.state = "preempted"
        self.events.append({"event": "preempted", "retained_tokens": len(self.committed)})
        self.check()

    def cancel(self) -> None:
        if self.state == "cancelled":
            return
        if self.state == "active":
            self.pool.release(self.owner)
        self.state = "cancelled"
        self.events.append({"event": "cancelled"})
        self.check()

    def check(self) -> None:
        self.pool.check()
        if self.state == "active":
            assert self.pool.read(self.owner) == self.committed
        else:
            assert self.owner not in self.pool.tables
        assert [e["token"] for e in self.events if e["event"] == "emit"] == self.outputs


@dataclass
class Transfer:
    key: str
    room: str
    owner: str
    source_pin: str
    destination_pin: str
    snapshot: tuple[int, ...]
    destination_pages: tuple[int, ...]
    state: str = "in_flight"
    reason: str | None = None


class TransferManager:
    """Explicit completion/drain acknowledgements and generation-qualified rooms.

    Source and destination share a pool only to make ownership observable. Real
    workers have separate pools and need transport-specific completion guarantees.
    One manager owns the `_transfer/` owner/pin namespace for a pool. Do not mutate
    its staging owners directly. Completed records are retained as tombstones.
    """

    def __init__(self, pool: PagedKV):
        self.pool = pool
        self.generation = 0
        self.records: dict[str, Transfer] = {}
        self.rooms: dict[str, str] = {}
        self.events: list[dict] = []

    def begin(self, source: str, room: str) -> str:
        if not isinstance(room, str) or not room.strip() or room in self.rooms:
            raise ValueError("room must be nonempty and have no live transfer/output")
        snapshot = tuple(self.pool.read(source))
        if not snapshot:
            raise ValueError("cannot transfer empty KV")
        self.generation += 1
        key = f"_transfer/{self.generation}"
        source_pin, destination_pin = key + "/source", key + "/destination"
        if key in self.pool.tables or any(p in self.pool.transfers for p in (source_pin, destination_pin)):
            raise ValueError("reserved transfer namespace collision")
        self.pool.pin(source, source_pin)
        self.pool.create(key)
        try:
            # Reserve private destination pages; zeros are not published KV.
            for _ in snapshot:
                self.pool.append(key, 0)
        except MemoryError:
            self.pool.release(key)
            self.pool.unpin(source_pin)
            self.pool.check()
            raise
        pages = self.pool.pin(key, destination_pin)
        self.records[key] = Transfer(key, room, key, source_pin, destination_pin, snapshot, pages)
        self.rooms[room] = key
        self.events.append({"event": "begin", "key": key, "room": room, "rows": len(snapshot)})
        self.check()
        return key

    def complete(self, key: str) -> bool:
        record = self.records.get(key)
        if record is None or record.state != "in_flight":
            self.events.append({"event": "completion_ignored", "key": key})
            return False
        offset = 0
        for page_id in record.destination_pages:
            page = self.pool.pages[page_id]
            count = len(page.tokens)
            page.tokens[:] = record.snapshot[offset:offset + count]
            offset += count
        self.pool.unpin(record.source_pin)
        self.pool.unpin(record.destination_pin)
        record.state = "ready"
        self.events.append({"event": "ready", "key": key})
        self.check()
        return True

    def abort(self, key: str, reason: str = "cancel") -> bool:
        if reason not in {"cancel", "timeout", "producer_failure", "consumer_failure"}:
            raise ValueError("unknown abort reason")
        record = self.records.get(key)
        if record is None or record.state != "in_flight":
            return False
        record.state, record.reason = "draining", reason
        self.pool.release(record.owner)
        self.events.append({"event": "abort", "key": key, "reason": reason})
        self.check()
        return True

    def acknowledge_drained(self, key: str) -> bool:
        record = self.records.get(key)
        if record is None or record.state != "draining":
            return False
        # The caller asserts no writer can access either allocation after this.
        self.pool.unpin(record.source_pin)
        self.pool.unpin(record.destination_pin)
        record.state = "drained"
        del self.rooms[record.room]
        self.events.append({"event": "drained", "key": key})
        self.check()
        return True

    def read(self, key: str) -> list[int]:
        record = self.records[key]
        if record.state != "ready":
            raise ValueError("only completed KV is visible to decode")
        return self.pool.read(record.owner)

    def release_output(self, key: str) -> bool:
        record = self.records.get(key)
        if record is None or record.state != "ready":
            return False
        self.pool.release(record.owner)
        record.state = "released"
        del self.rooms[record.room]
        self.events.append({"event": "output_released", "key": key})
        self.check()
        return True

    def check(self) -> None:
        self.pool.check()
        for key, record in self.records.items():
            live = record.state in {"in_flight", "draining", "ready"}
            assert (self.rooms.get(record.room) == key) == live
            pinned = record.state in {"in_flight", "draining"}
            for pin in (record.source_pin, record.destination_pin):
                assert (pin in self.pool.transfers) == pinned
            assert (record.owner in self.pool.tables) == (record.state in {"in_flight", "ready"})
            if pinned:
                assert self.pool.transfers[record.destination_pin] == record.destination_pages
                assert tuple(t for p in self.pool.transfers[record.source_pin]
                             for t in self.pool.pages[p].tokens) == record.snapshot
            if record.state == "ready":
                assert tuple(self.read(key)) == record.snapshot


def demo() -> dict:
    pool = PagedKV(8, 2)
    request = RecomputeRequest(pool, "request", [10, 11, 12])
    assert request.resume()
    request.sample(20)
    request.sample(21)
    transfers = TransferManager(pool)
    old = transfers.begin("request", "room-a")
    request.preempt()
    transfers.abort(old, "timeout")
    retained_before_ack = len(pool.pages)
    assert not transfers.complete(old)
    transfers.acknowledge_drained(old)
    assert request.resume()
    request.sample(22)
    new = transfers.begin("request", "room-a")
    assert not transfers.complete(old)
    transfers.complete(new)
    received = transfers.read(new)
    transfers.release_output(new)
    request.cancel()
    return {"schema_version": 1, "model": "CPU lifecycle accounting; no timing model",
            "request_events": request.events, "transfer_events": transfers.events,
            "outputs": request.outputs, "received_rows": received,
            "recomputed_rows": request.recomputed_rows,
            "retained_pages_before_drain_ack": retained_before_ack,
            "free_pages_at_end": len(pool.free)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(demo(), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
