"""CPU recompute, asynchronous cancellation, and generation-fencing contracts."""

from html.parser import HTMLParser
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "content/courses/inference/html/labs"
sys.path.insert(0, str(LABS))
from scheduler_faults import PagedKV, RecomputeRequest, TransferManager, demo


class RecomputeTests(unittest.TestCase):
    def test_replay_preserves_emission_and_excludes_unconsumed_sample(self):
        pool = PagedKV(4, 2)
        request = RecomputeRequest(pool, "r", [1, 2, 3])
        self.assertTrue(request.resume())
        request.sample(7)
        request.sample(8)
        request.preempt()
        self.assertEqual(len(pool.free), 4)
        self.assertTrue(request.resume())
        self.assertEqual(pool.read("r"), [1, 2, 3, 7])
        request.sample(9)
        self.assertEqual(request.outputs, [7, 8, 9])
        self.assertEqual(request.recomputed_rows, 4)
        self.assertEqual(pool.read("r"), [1, 2, 3, 7, 8])

    def test_replay_allocation_failure_rolls_back(self):
        pool = PagedKV(2, 2)
        request = RecomputeRequest(pool, "r", [1, 2, 3])
        request.resume()
        request.preempt()
        pool.create("other")
        pool.append("other", 0)
        self.assertFalse(request.resume())
        self.assertEqual(request.state, "preempted")
        self.assertEqual(request.recomputed_rows, 0)
        self.assertEqual(pool.tables, {"other": [0]})
        pool.release("other")
        self.assertTrue(request.resume())
        self.assertEqual(request.recomputed_rows, 3)

    def test_failed_decode_does_not_emit(self):
        pool = PagedKV(1, 2)
        request = RecomputeRequest(pool, "r", [1, 2])
        request.resume()
        request.sample(3)
        with self.assertRaises(MemoryError):
            request.sample(4)
        self.assertEqual(request.outputs, [3])
        request.check()

    def test_repeated_cancel_and_invalid_state(self):
        request = RecomputeRequest(PagedKV(1, 2), "r", [1])
        request.cancel()
        request.cancel()
        self.assertEqual(len(request.events), 1)
        for action in (request.resume, request.preempt, lambda: request.sample(2)):
            with self.assertRaises(ValueError):
                action()
        with self.assertRaises(ValueError):
            RecomputeRequest(PagedKV(1, 2), "r", [True])

    def test_seeded_replay_against_emission_oracle(self):
        for seed in range(40):
            rng = random.Random(seed)
            pool = PagedKV(30, rng.randint(1, 4))
            prompt = [rng.randrange(100) for _ in range(3)]
            request = RecomputeRequest(pool, "r", prompt)
            request.resume()
            expected, replayed = [], 0
            for _ in range(50):
                if rng.random() < 0.45:
                    request.preempt()
                    replayed += len(prompt) + max(0, len(expected) - 1)
                    # This is bounded below pool capacity, so eventual replay is guaranteed.
                    self.assertTrue(request.resume())
                elif len(expected) < 20:
                    token = rng.randrange(1000)
                    request.sample(token)
                    expected.append(token)
                self.assertEqual(request.outputs, expected)
                self.assertEqual(pool.read("r"), prompt + expected[:-1])
                self.assertEqual(request.recomputed_rows, replayed)
                request.check()
            request.cancel()
            self.assertEqual(len(pool.free), pool.capacity)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.pool = PagedKV(8, 2)
        self.pool.create("source")
        for token in [10, 11, 12]:
            self.pool.append("source", token)
        self.manager = TransferManager(self.pool)

    def test_success_uses_frozen_source_and_requires_publication(self):
        key = self.manager.begin("source", "room")
        with self.assertRaises(ValueError):
            self.manager.read(key)
        self.pool.append("source", 13)
        self.pool.release("source")
        self.assertTrue(self.manager.complete(key))
        self.assertEqual(self.manager.read(key), [10, 11, 12])
        self.assertFalse(self.manager.complete(key))
        self.assertFalse(self.manager.abort(key))
        self.assertTrue(self.manager.release_output(key))
        self.assertFalse(self.manager.release_output(key))
        self.assertEqual(len(self.pool.free), 8)

    def test_all_abort_reasons_retain_both_sides_until_ack(self):
        for reason in ("cancel", "timeout", "producer_failure", "consumer_failure"):
            with self.subTest(reason=reason):
                pool = PagedKV(4, 2)
                pool.create("p")
                for token in [1, 2, 3]:
                    pool.append("p", token)
                manager = TransferManager(pool)
                key = manager.begin("p", "r")
                pool.release("p")
                manager.abort(key, reason)
                self.assertEqual(len(pool.free), 0)
                self.assertFalse(manager.complete(key))
                self.assertFalse(manager.abort(key, reason))
                self.assertEqual(len(pool.free), 0)
                manager.acknowledge_drained(key)
                self.assertEqual(len(pool.free), 4)
                self.assertFalse(manager.acknowledge_drained(key))

    def test_duplicate_room_and_late_callback_are_generation_fenced(self):
        old = self.manager.begin("source", "r")
        with self.assertRaises(ValueError):
            self.manager.begin("source", "r")
        self.manager.abort(old)
        with self.assertRaises(ValueError):
            self.manager.begin("source", "r")
        self.manager.acknowledge_drained(old)
        new = self.manager.begin("source", "r")
        before = list(self.pool.free)
        self.assertNotEqual(new, old)
        self.assertFalse(self.manager.complete(old))
        self.assertFalse(self.manager.acknowledge_drained(old))
        self.assertFalse(self.manager.complete("unknown"))
        self.assertEqual(self.pool.free, before)
        self.assertEqual(self.manager.records[new].state, "in_flight")
        self.manager.complete(new)
        self.assertEqual(self.manager.read(new), [10, 11, 12])

    def test_begin_failure_unpins_and_removes_partial_destination(self):
        pool = PagedKV(3, 2)
        pool.create("source")
        for token in [1, 2, 3]:
            pool.append("source", token)
        manager = TransferManager(pool)
        with self.assertRaises(MemoryError):
            manager.begin("source", "r")
        self.assertEqual(pool.transfers, {})
        self.assertEqual(list(pool.tables), ["source"])
        self.assertEqual(len(pool.free), 1)
        self.assertEqual(manager.rooms, {})
        self.assertEqual(manager.records, {})

    def test_invalid_reason_cannot_change_transfer(self):
        key = self.manager.begin("source", "r")
        with self.assertRaises(ValueError):
            self.manager.abort(key, "pretend_dma_stopped")
        self.assertEqual(self.manager.records[key].state, "in_flight")
        self.assertFalse(self.manager.acknowledge_drained(key))
        self.manager.check()

    def test_source_name_reuse_does_not_change_old_snapshot(self):
        key = self.manager.begin("source", "room")
        old_pages = set(self.pool.tables["source"])
        self.pool.release("source")
        self.pool.create("source")
        self.pool.append("source", 99)
        self.assertTrue(old_pages.isdisjoint(self.pool.tables["source"]))
        self.manager.complete(key)
        self.assertEqual(self.manager.read(key), [10, 11, 12])
        self.assertEqual(self.pool.read("source"), [99])
        self.manager.release_output(key)
        self.pool.release("source")
        self.assertEqual(len(self.pool.free), 8)

    def test_reserved_namespace_collision_leaves_source_untouched(self):
        self.pool.create("_transfer/1")
        self.pool.append("_transfer/1", 99)
        before = {owner: self.pool.read(owner) for owner in self.pool.tables}
        with self.assertRaises(ValueError):
            self.manager.begin("source", "room")
        self.assertEqual(self.pool.transfers, {})
        self.assertEqual({owner: self.pool.read(owner) for owner in self.pool.tables}, before)
        self.manager.check()

    def test_preemption_cannot_reuse_transport_retained_pages(self):
        pool = PagedKV(4, 2)
        request = RecomputeRequest(pool, "r", [1, 2, 3])
        request.resume()
        manager = TransferManager(pool)
        key = manager.begin("r", "room")
        request.preempt()
        manager.abort(key)
        self.assertFalse(request.resume())
        self.assertEqual(len(pool.free), 0)
        manager.acknowledge_drained(key)
        self.assertTrue(request.resume())
        self.assertEqual(pool.read("r"), [1, 2, 3])

    def test_seeded_callbacks_and_namespace_reuse(self):
        for seed in range(30):
            rng = random.Random(seed)
            pool = PagedKV(24, 3)
            pool.create("source")
            expected = [1, 2]
            for token in expected:
                pool.append("source", token)
            manager = TransferManager(pool)
            snapshots = {}
            for _ in range(100):
                action = rng.choice(("begin", "complete", "abort", "drain", "release", "append"))
                if action == "begin":
                    room = str(rng.randrange(4))
                    try:
                        key = manager.begin("source", room)
                        snapshots[key] = list(expected)
                    except (MemoryError, ValueError):
                        pass
                elif action == "append" and len(expected) < 9:
                    token = rng.randrange(100)
                    try:
                        pool.append("source", token)
                        expected.append(token)
                    except MemoryError:
                        pass
                else:
                    key = rng.choice(list(manager.records) + ["unknown"])
                    {"complete": manager.complete, "abort": manager.abort,
                     "drain": manager.acknowledge_drained,
                     "release": manager.release_output}.get(action, lambda _: None)(key)
                manager.check()
                self.assertEqual(pool.read("source"), expected)
                for key, record in manager.records.items():
                    if record.state == "ready":
                        self.assertEqual(manager.read(key), snapshots[key])
            for key in manager.records:
                manager.abort(key)
                manager.acknowledge_drained(key)
                manager.release_output(key)
            pool.release("source")
            manager.check()
            self.assertEqual(len(pool.free), pool.capacity)


class PublishedExamples(unittest.TestCase):
    def test_demo_and_cli(self):
        report = demo()
        self.assertEqual(report["outputs"], [20, 21, 22])
        self.assertEqual(report["received_rows"], [10, 11, 12, 20, 21])
        self.assertEqual(report["recomputed_rows"], 4)
        self.assertEqual(report["retained_pages_before_drain_ack"], 4)
        self.assertEqual(report["free_pages_at_end"], 8)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace.json"
            subprocess.run([sys.executable, str(LABS / "scheduler_faults.py"), "--output", str(target)], check=True)
            self.assertEqual(json.loads(target.read_text()), report)

    def test_printed_examples_execute(self):
        class Examples(HTMLParser):
            def __init__(self):
                super().__init__()
                self.current = None
                self.blocks = []

            def handle_starttag(self, tag, attrs):
                if tag == "code" and "data-inference-fault-example" in dict(attrs):
                    self.current = []

            def handle_data(self, data):
                if self.current is not None:
                    self.current.append(data)

            def handle_endtag(self, tag):
                if tag == "code" and self.current is not None:
                    self.blocks.append("".join(self.current))
                    self.current = None

        parser = Examples()
        parser.feed((LABS / "03-continuous-batching-effects/README.html").read_text())
        self.assertEqual(len(parser.blocks), 2)
        for block in parser.blocks:
            subprocess.run([sys.executable, "-c", block], cwd=LABS, check=True)


if __name__ == "__main__":
    unittest.main()
