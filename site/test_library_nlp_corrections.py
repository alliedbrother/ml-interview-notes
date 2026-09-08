"""Run the explicitly independent, offline library and NLP chapter examples."""

from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "content/notes/libraries").glob("*.md")) + sorted(
    (ROOT / "content/notes/nlp").glob("*.md")
)


class CurriculumCorrections(unittest.TestCase):
    def test_offline_examples(self):
        count = 0
        with tempfile.TemporaryDirectory(prefix="library-nlp-labs-") as cache:
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                       TF_CPP_MIN_LOG_LEVEL="3", MPLCONFIGDIR=cache,
                       HF_HOME=cache, TOKENIZERS_PARALLELISM="false")
            for page in PAGES:
                blocks = re.findall(r"^```python runnable\n(.*?)^```\s*$",
                                    page.read_text(), re.MULTILINE | re.DOTALL)
                self.assertTrue(blocks, f"No independent lab in {page}")
                for number, code in enumerate(blocks, 1):
                    count += 1
                    with self.subTest(page=page.name, example=number):
                        compile(code, str(page), "exec")
                        started = time.monotonic()
                        result = subprocess.run([sys.executable, "-c", code],
                            cwd=cache, env=env, capture_output=True, text=True,
                            timeout=60)
                        self.assertEqual(result.returncode, 0,
                            result.stdout + "\n" + result.stderr)
                        print(f"PASS {page.parent.name}/{page.name} #{number} "
                              f"{time.monotonic() - started:.2f}s", flush=True)
        self.assertGreaterEqual(count, 21)


if __name__ == "__main__":
    unittest.main()
