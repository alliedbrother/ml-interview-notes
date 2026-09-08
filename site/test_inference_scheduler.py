"""CPU scheduler traces and randomized paged-KV ownership contracts."""

import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "content/courses/inference/html/labs/scheduler_sim.py"
spec = importlib.util.spec_from_file_location("scheduler_sim", SCRIPT)
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


def arrival(name="a", time=0, prompt=3, output=3):
    return {"event": "arrival", "id": name, "time": time,
            "prompt_tokens": prompt, "output_tokens": output}


class PagedKVTests(unittest.TestCase):
    def test_block_boundary_tail_waste(self):
        for count, expected in ((1, 15), (15, 1), (16, 0), (17, 15)):
            pool = sim.PagedKV(8, 16)
            pool.create("a")
            for token in range(count):
                pool.append("a", token)
                pool.check()
            self.assertEqual(len(pool.tables["a"]) * 16 - count, expected)
            self.assertEqual(pool.read("a"), list(range(count)))
            pool.release("a")
            self.assertEqual(len(pool.free), 8)

    def test_shared_partial_tail_copies_before_write(self):
        pool = sim.PagedKV(3, 4)
        pool.create("a")
        pool.append("a", 10)
        pool.fork("a", "b")
        pool.append("b", 20)
        self.assertEqual(pool.read("a"), [10])
        self.assertEqual(pool.read("b"), [10, 20])
        self.assertNotEqual(pool.tables["a"], pool.tables["b"])
        pool.release("a")
        pool.check()
        self.assertEqual(pool.read("b"), [10, 20])

    def test_transfer_pin_retains_frozen_source_after_cancel(self):
        pool = sim.PagedKV(2, 4)
        pool.create("a")
        pool.append("a", 10)
        old, = pool.pin("a", "dma-1")
        pool.append("a", 20)
        self.assertEqual(pool.pages[old].tokens, [10])
        pool.release("a")
        self.assertIn(old, pool.pages)
        self.assertNotIn(old, pool.free)
        pool.create("a")  # New owner generation does not inherit the old table.
        pool.append("a", 30)
        self.assertNotEqual(pool.tables["a"], [old])
        pool.unpin("dma-1")
        pool.check()
        self.assertIn(old, pool.free)

    def test_failed_copy_on_write_is_atomic(self):
        pool = sim.PagedKV(1, 4)
        pool.create("a")
        pool.append("a", 10)
        pool.fork("a", "b")
        with self.assertRaises(MemoryError):
            pool.append("b", 20)
        self.assertEqual(pool.read("a"), [10])
        self.assertEqual(pool.read("b"), [10])
        pool.check()

    def test_duplicate_transfer_and_owner_rejected(self):
        pool = sim.PagedKV(1, 4)
        pool.create("a")
        pool.pin("a", "dma")
        with self.assertRaises(ValueError):
            pool.pin("a", "dma")
        with self.assertRaises(ValueError):
            pool.create("a")
        with self.assertRaises(ValueError):
            pool.fork("a", "a")

    def test_seeded_allocator_reference_model(self):
        for seed in range(25):
            rng = random.Random(seed)
            pool = sim.PagedKV(12, 4)
            expected = {}
            pinned = {}
            serial = 0
            for _ in range(200):
                operation = rng.choice(("new", "append", "fork", "pin", "unpin", "release"))
                serial += 1
                key = f"r{serial}"
                if operation == "new":
                    pool.create(key)
                    expected[key] = []
                elif operation == "unpin" and pinned:
                    transfer = rng.choice(list(pinned))
                    pool.unpin(transfer)
                    del pinned[transfer]
                elif expected:
                    owner = rng.choice(list(expected))
                    if operation == "append":
                        try:
                            pool.append(owner, serial)
                        except MemoryError:
                            pass
                        else:
                            expected[owner].append(serial)
                    elif operation == "fork":
                        pool.fork(owner, key)
                        expected[key] = list(expected[owner])
                    elif operation == "release":
                        pool.release(owner)
                        del expected[owner]
                    elif operation == "pin":
                        ids = pool.pin(owner, key)
                        pinned[key] = {i: list(pool.pages[i].tokens) for i in ids}
                pool.check()
                for owner, tokens in expected.items():
                    self.assertEqual(pool.read(owner), tokens)
                for retained in pinned.values():
                    for page_id, frozen_tokens in retained.items():
                        self.assertEqual(pool.pages[page_id].tokens, frozen_tokens)
            for owner in list(expected):
                pool.release(owner)
            for transfer in list(pinned):
                pool.unpin(transfer)
            pool.check()
            self.assertEqual(len(pool.free), pool.capacity)


class SchedulerTests(unittest.TestCase):
    def test_displayed_examples_match_download(self):
        class Examples(HTMLParser):
            def __init__(self):
                super().__init__()
                self.collect = False
                self.examples = []

            def handle_starttag(self, tag, attrs):
                if tag == "code" and "data-scheduler-example" in dict(attrs):
                    self.collect = True
                    self.examples.append("")

            def handle_data(self, data):
                if self.collect:
                    self.examples[-1] += data

            def handle_endtag(self, tag):
                if tag == "code":
                    self.collect = False

        parser = Examples()
        base = ROOT / "content/courses/inference/html"
        for relative in ("02-memory-and-kv-cache/02-02-pagedattention.html",
                         "labs/03-continuous-batching-effects/README.html"):
            parser.feed((base / relative).read_text())
        self.assertEqual(len(parser.examples), 2)
        for example in parser.examples:
            exec(compile(example, "published-scheduler-example", "exec"), {})

    def test_first_token_from_final_prefill_and_n_minus_one_decodes(self):
        report = sim.simulate([arrival()], prefill_chunk=2, release_delay=0)
        row = report["requests"]["a"]
        self.assertEqual(row["token_times"], [2, 3, 4])
        self.assertEqual(row["ttft_ticks"], 2)
        self.assertEqual(row["itl_ticks"], [1, 1])
        self.assertEqual(report["summary"]["processed_input_tokens"], 5)

    def test_one_token_output_needs_no_decode_or_extra_cache_slot(self):
        report = sim.simulate([arrival(prompt=4, output=1)], capacity=1, block_size=4,
                              prefill_chunk=4, release_delay=0)
        self.assertEqual(report["requests"]["a"]["token_times"], [1])
        self.assertEqual(report["summary"]["processed_input_tokens"], 4)

    def test_deferred_release_is_not_admission_headroom(self):
        report = sim.simulate([arrival("a", prompt=1, output=1), arrival("b", prompt=1, output=1)],
                              capacity=1, release_delay=3)
        self.assertEqual(report["requests"]["a"]["end"], 1)
        self.assertEqual(report["requests"]["b"]["admitted"], 4)
        held = next(e for e in report["events"] if e["event"] == "snapshot" and e["time"] == 1)
        self.assertEqual(held["draining_blocks"], 1)
        self.assertEqual(held["admission_headroom"], 0)

    def test_conservative_reservation_blocks_even_with_unallocated_pages(self):
        report = sim.simulate([arrival("a", prompt=1, output=8), arrival("b", prompt=1, output=1)],
                              capacity=2, block_size=4)
        first = next(e for e in report["events"] if e["event"] == "snapshot")
        self.assertEqual(first["free_blocks"], 2)
        self.assertEqual(first["reserved_blocks"], 2)
        self.assertEqual(first["queued"], ["b"])

    def test_cancellation_queued_and_active_then_late_duplicate(self):
        trace = [arrival("a", prompt=8, output=4), arrival("b"),
                 {"time": 0, "event": "cancel", "id": "b"},
                 {"time": 1, "event": "cancel", "id": "a"},
                 {"time": 9, "event": "cancel", "id": "a"}]
        report = sim.simulate(trace, max_active=1)
        self.assertEqual(report["requests"]["b"]["admitted"], None)
        self.assertEqual(report["requests"]["a"]["output_tokens"], 0)
        self.assertEqual(report["summary"]["cancelled"], 2)
        self.assertEqual(report["summary"]["free_blocks"], 8)
        self.assertEqual(report["events"][-2]["event"], "cancel_ignored")

    def test_completion_precedes_same_tick_cancel(self):
        report = sim.simulate([arrival(prompt=1, output=1),
                               {"time": 1, "event": "cancel", "id": "a"}])
        self.assertEqual(report["requests"]["a"]["status"], "finished")
        self.assertTrue(any(e["event"] == "cancel_ignored" for e in report["events"]))

    def test_oversize_rejected_and_idle_time_jumps(self):
        report = sim.simulate([arrival("large", time=1000, prompt=33, output=1),
                               arrival("small", time=1001, prompt=1, output=1)], capacity=8)
        self.assertEqual(report["requests"]["large"]["status"], "rejected")
        self.assertEqual(report["requests"]["small"]["end"], 1002)
        self.assertLess(len(report["events"]), 20)

    def test_serial_and_continuous_same_work_different_wait(self):
        trace = [arrival("a", prompt=2, output=4), arrival("b", prompt=2, output=1)]
        serial = sim.simulate(trace, policy="serial", release_delay=0)
        continuous = sim.simulate(trace, policy="continuous", release_delay=0)
        self.assertEqual(serial["summary"]["processed_input_tokens"],
                         continuous["summary"]["processed_input_tokens"])
        self.assertLess(continuous["requests"]["b"]["ttft_ticks"],
                        serial["requests"]["b"]["ttft_ticks"])

    def test_strict_input_contracts(self):
        for trace in ([arrival(), arrival()],
                      [arrival(prompt=0)], [arrival(output=True)],
                      [{"event": "cancel", "id": "missing", "time": 0}],
                      [arrival(time=3), {"event": "cancel", "id": "a", "time": 2}],
                      [{**arrival(), "unknown": 1}]):
            with self.assertRaises(ValueError):
                sim.simulate(trace)
        for config in ({"capacity": 0}, {"token_budget": 0}, {"release_delay": -1},
                       {"max_active": False}, {"policy": "magic"}):
            with self.assertRaises(ValueError):
                sim.simulate([], **config)

    def test_seeded_scheduler_progress_accounting_and_replay(self):
        for seed in range(60):
            rng = random.Random(seed)
            trace = []
            for index in range(15):
                name, time = str(index), rng.randrange(20)
                trace.append(arrival(name, time, rng.randrange(1, 18), rng.randrange(1, 10)))
                if rng.random() < .4:
                    trace.append({"event": "cancel", "id": name, "time": time + rng.randrange(12)})
            config = {"capacity": rng.randrange(2, 12), "block_size": rng.randrange(1, 6),
                      "token_budget": rng.randrange(1, 7), "max_active": rng.randrange(1, 7),
                      "prefill_chunk": rng.randrange(1, 8), "release_delay": rng.randrange(4)}
            report = sim.simulate(trace, **config)
            self.assertEqual(report, sim.simulate(json.loads(json.dumps(trace)), **config))
            self.assertEqual(report["summary"]["free_blocks"], config["capacity"])
            self.assertEqual(sum(report["summary"][key] for key in ("finished", "cancelled", "rejected")), 15)
            for event in report["events"]:
                if event["event"] == "dispatch":
                    self.assertLessEqual(sum(w["input_tokens"] for w in event["work"]), config["token_budget"])
                if event["event"] == "snapshot":
                    self.assertLessEqual(event["reserved_blocks"] + event["draining_blocks"], config["capacity"])
                    self.assertEqual(event["allocated_blocks"] + event["free_blocks"], config["capacity"])
            for row in report["requests"].values():
                self.assertEqual(row["token_times"], sorted(set(row["token_times"])))
                self.assertGreaterEqual(row["end"], row["arrival"])

    def test_cli_jsonl_replay_matches_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            trace = path / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(e) for e in sim.demo_trace()) + "\n")
            output = path / "report.json"
            subprocess.run([sys.executable, str(SCRIPT), "--trace", str(trace), "--output", str(output)], check=True)
            self.assertEqual(json.loads(output.read_text()), sim.simulate(sim.demo_trace()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
