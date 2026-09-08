"""Focused browser regression checks for the linear algebra figures."""

import argparse
import math
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('**/assets/pages/linear-algebra.js', lambda route: route.fulfill(
            path=str(root / 'content/notes/_assets/linear-algebra.js'), content_type='text/javascript'))
        page.goto(args.base_url + '/notes/math/linear-algebra/', wait_until='networkidle')
        page.locator('#dot-product-viz svg').wait_for()
        assert page.locator('#dot-product-val').inner_text() == '11.00'
        page.locator('#vec-a-x').fill('0')
        page.locator('#vec-a-y').fill('0')
        assert page.locator('#angle-val').inner_text() == 'Undefined'
        assert 'undefined' in page.locator('#dot-product-explanation').inner_text()
        for field, value in [('vec-a-x', '1'), ('vec-b-x', '0'), ('vec-b-y', '1')]:
            page.locator('#' + field).fill(value)
        assert page.locator('#dot-product-val').inner_text() == '0.00'
        assert page.locator('#angle-val').inner_text() == '90.0°'
        page.wait_for_timeout(250)
        vectors = page.locator('#dot-product-viz line.la-vector').evaluate_all('''lines => lines.map(l => ({
            x: +l.getAttribute('x2') - +l.getAttribute('x1'),
            y: +l.getAttribute('y2') - +l.getAttribute('y1')
        }))''')
        assert math.isclose(vectors[0]['x'], -vectors[1]['y'], rel_tol=1e-8)
        wedge = page.locator('#dot-product-viz svg path').evaluate_all(
            "paths => paths.map(p => p.getAttribute('d')).filter(d => d.includes('A30'))")
        assert len(wedge) == 1 and wedge[0].startswith('M30,0'), wedge
        page.locator('#scalar-multiply-btn').click()
        assert page.locator('.widget__cell').first.inner_text() == '2.4'
        page.locator('#transpose-btn').click()
        assert page.locator('.widget__mrow').count() == 3
        page.locator('#reset-matrix-btn').click()
        page.locator('#scalar-input').fill('')
        page.locator('#scalar-multiply-btn').click()
        assert page.locator('.widget__cell').first.inner_text() == '1.2'
        assert not page.locator('#scalar-input').evaluate('(input) => input.validity.valid')
        assert 'Correct: 60/60' in page.locator('#classifier-readout').inner_text()
        page.locator('#classifier-angle').fill('135')
        assert 'Correct: 0/60' in page.locator('#classifier-readout').inner_text()
        page.locator('#classifier-angle').fill('-45')
        readout = page.locator('#pca-readout')
        first = float(readout.get_attribute('data-lambda1'))
        second = float(readout.get_attribute('data-lambda2'))
        assert first > second > 0
        assert 'sample variance' in readout.inner_text()
        components = page.locator('#pca-visualization line[marker-end]').evaluate_all('''lines => lines.map(l => ({
            x: +l.getAttribute('x2') - +l.getAttribute('x1'),
            y: +l.getAttribute('y2') - +l.getAttribute('y1')
        }))''')
        a, b = components[0], components[2]
        assert abs(a['x'] * b['x'] + a['y'] * b['y']) < 1e-8
        assert math.isclose(math.hypot(a['x'], a['y']) / math.hypot(b['x'], b['y']), math.sqrt(first / second))
        screenshots = Path('/tmp/linear-algebra-widgets')
        screenshots.mkdir(exist_ok=True)
        for width in (1440, 390, 320):
            page.set_viewport_size({'width': width, 'height': 1000})
            page.wait_for_timeout(300)
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
            for widget in ('interactive-matrix', 'dot-product-viz', 'interactive-classifier', 'pca-visualization', 'svd-visualization'):
                page.locator('#' + widget).screenshot(path=str(screenshots / f'{width}-{widget}.png'))
        assert not errors, errors
        browser.close()
        print('OK: matrix validation, zero-vector angle, equal scales, angle wedge, classifier, sample PCA, and responsive figures')


if __name__ == '__main__':
    main()
