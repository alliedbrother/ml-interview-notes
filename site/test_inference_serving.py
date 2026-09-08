"""Independent CPU checks for the serving/benchmark correction wave."""
from pathlib import Path
import codecs
import math
import subprocess
import unittest

from bs4 import BeautifulSoup
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'content/courses/inference/html'


class Corrections(unittest.TestCase):
    def test_corrected_source_contracts(self):
        cases = {
            '09-serving-system/09-01-openai-api-surface.html':
                ['first event, NOT first content token', 'max_tokens=512'],
            '09-serving-system/09-03-request-lifecycle.html':
                ['not a byte-memory bound', 'pickle by default (msgpack when configured)'],
            '09-serving-system/09-05-observability-and-failure.html':
                ['80$ KiB/token', '430{,}178', '7.6%'],
            '10-benchmarking/10-01-workload-characterization.html':
                ['25{,}165', "f''(S)=-400/(S+200)^3&lt;0"],
            '10-benchmarking/10-02-benchmark-harnesses.html':
                ['pooled median is 1.875 ms', 'no fixed RTT increment'],
            '10-benchmarking/10-03-honest-benchmarking.html':
                ['78.62', 'VLLM_SERVER_DEV_MODE=1 vllm bench sweep'],
            '10-benchmarking/10-04-batch-invariance.html':
                ['nor <code>sorted=True</code> guarantees stable tied indices'],
            '10-benchmarking/10-05-profiling-and-capacity.html':
                ['5.0964', '67.27', '0.51725', '48 replicas', 'USD 0.743', 'USD 4.34'],
        }
        for relative, fragments in cases.items():
            with self.subTest(page=relative):
                source = (BASE / relative).read_text()
                for fragment in fragments:
                    self.assertIn(fragment, source)

    def test_all_ten_pages_preserve_ids_and_sections(self):
        pages = sorted((BASE / '09-serving-system').glob('*.html')) + sorted(
            (BASE / '10-benchmarking').glob('*.html'))
        self.assertEqual(len(pages), 10)
        for page in pages:
            with self.subTest(page=page.name):
                old = subprocess.check_output(['git', 'show', f'HEAD:{page.relative_to(ROOT)}'],
                                              cwd=ROOT, text=True)
                before, after = BeautifulSoup(old, 'html.parser'), BeautifulSoup(page.read_text(), 'html.parser')
                self.assertEqual([x.get('id') for x in before.select('main section')],
                                 [x.get('id') for x in after.select('main section')])
                self.assertEqual([x.get_text() for x in before.select('main h2')],
                                 [x.get_text() for x in after.select('main h2')])
                self.assertEqual(len(before.select('script.diagram-source')),
                                 len(after.select('script.diagram-source')))

    def test_kv_rank_arithmetic(self):
        slots = 32.82 * 1024**2 / 80
        self.assertAlmostEqual(slots, 430178.304, places=3)
        self.assertLess(abs(32768 / slots - .0762), .0001)
        self.assertLess(abs(8192 / slots - .019), .0001)

    def test_workload_root_and_concavity(self):
        cp, cq, cd = 40.6e-6, 6.63e-10, 4.96e-8
        root = brentq(lambda s: cp*s+cq*s*s-9*128*cd*(s+64), 1, 1e6)
        self.assertAlmostEqual(root, 25165.182809, places=4)
        pre, dec = cp*root+cq*root**2, 128*cd*(root+64)
        self.assertAlmostEqual(pre/(pre+dec), .9)
        self.assertLess(.99*500/700+.01*24000/24200, 735/935)

    def test_decode_regression(self):
        service = .0311 + 1.2*.0396
        self.assertAlmostEqual(service, .07862)
        self.assertAlmostEqual(1/service, 12.719409819)

    def test_frame_weighting_and_utf8_flush(self):
        intervals = np.r_[np.full(3, 7.5/3), np.full(4, 7.5/4)]
        self.assertEqual(float(np.median(intervals)), 1.875)
        self.assertAlmostEqual(float(intervals.mean()), 15/7)
        # One-token frames: pooled mean and unweighted request TPOT still differ.
        self.assertNotEqual(np.mean([1, 9, 9, 9]), np.mean([1, 9]))
        text = '\u1000\u1001'
        decoder = codecs.getincrementaldecoder('utf8')()
        deltas = [decoder.decode(bytes([b]), final=False) for b in text.encode()]
        deltas.append(decoder.decode(b'', final=True))
        self.assertEqual(''.join(deltas), text)
        self.assertEqual(sum(bool(x) for x in deltas), 2)

    def test_capacity_screen_and_closure(self):
        s, p = .0912, .0704
        rho = brentq(lambda r: s*math.log(100*r)/(1-r)+p-.9, .010001, .999)
        self.assertAlmostEqual(rho, .5578967058)
        self.assertEqual(math.ceil(240/(rho/s)), 40)
        def service(lam):
            batch = 13.2*lam
            step = (15.01e9+batch*1910*131072)/3.35e12
            return p + 220*step/batch
        lam = brentq(lambda l: service(l)*math.log(100*l*service(l))/(1-l*service(l))+p-.9, 5, 6)
        self.assertAlmostEqual(lam, 5.0964006019)
        batch = 13.2*lam
        self.assertLess(batch, 224)
        self.assertAlmostEqual(batch/(220*lam), .06)
        self.assertLess(lam*service(lam), 1)
        self.assertEqual(math.ceil(240/lam), 48)
        self.assertAlmostEqual(1e6*3/(3600*lam*220), .7432458874)


if __name__ == '__main__':
    unittest.main()
