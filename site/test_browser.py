"""Optional browser checks. Requires Playwright and a running local preview."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshots", default="/tmp/ml-notes-design")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--linear-algebra", action="store_true", help="Check the full linear algebra chapter and its figures")
    parser.add_argument("--interactions-only", action="store_true", help="Skip layout screenshots")
    parser.add_argument("--curriculum", action="store_true", help="Verify every ML/DL chapter, math, diagrams and downloads")
    parser.add_argument("--subjects", action="store_true", help="Verify math, libraries, NLP, Transformers and inference pages")
    parser.add_argument("--only-path", help="Limit page checks to one published route")
    parser.add_argument("--start-at", help="Resume the selected route inventory at this path")
    parser.add_argument("--width", type=int, help="Limit layout checks to one viewport width")
    args = parser.parse_args()
    screenshots = Path(args.screenshots)
    screenshots.mkdir(exist_ok=True, parents=True)
    paths = ["/", "/courses/", "/notes/", "/status/", "/notes/math/",
             "/notes/math/linear-algebra/", "/notes/ml/linear-models/",
             "/courses/transformers/03-self-attention-from-scratch/",
             "/courses/inference/", "/courses/inference/02-memory-and-kv-cache/02-02-pagedattention.html",
             "/courses/inference/labs/02-kv-cache-sizing/README.html"]
    if args.quick:
        paths = paths[:1]
    if args.linear_algebra:
        paths = ['/notes/math/linear-algebra/']
    if args.interactions_only:
        paths = []
    if args.curriculum:
        notes = Path(__file__).resolve().parent.parent / 'content/notes'
        paths = [f'/notes/{category}/' for category in ('ml', 'deep-learning')]
        paths += [f'/notes/{category}/{source.stem}/'
                  for category in ('ml', 'deep-learning')
                  for source in sorted((notes / category).glob('*.md'))]
    if args.subjects:
        root = Path(__file__).resolve().parent.parent
        notes = root / 'content/notes'
        categories = ('math', 'libraries', 'nlp')
        paths = [f'/notes/{category}/' for category in categories]
        paths += [f'/notes/{category}/{source.stem}/' for category in categories
                  for source in sorted((notes / category).glob('*.md'))]
        paths += ['/courses/transformers/']
        paths += [f'/courses/transformers/{source.stem}/'
                  for source in sorted((root / 'content/courses/transformers').glob('[0-9][0-9]-*.md'))
                  if source.name != '00-README.md']
        inference = root / 'content/courses/inference/html'
        paths += ['/courses/inference/']
        paths += ['/courses/inference/' + str(source.relative_to(inference))
                  for source in sorted(inference.rglob('*.html'))
                  if source not in (inference / 'index.html', inference / 'README.html')]
    if args.only_path:
        paths = [args.only_path]
    if args.start_at:
        if args.start_at not in paths:
            parser.error('--start-at must be a route in the selected inventory')
        paths = paths[paths.index(args.start_at):]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(color_scheme="light", permissions=["clipboard-read", "clipboard-write"])
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in ((args.width,) if args.width else (1440, 390, 320)):
            page.set_viewport_size({"width": width, "height": 1000 if width > 600 else 844})
            for i, path in enumerate(paths):
                page.goto(args.base_url + path, wait_until="networkidle")
                page.locator('.topbar svg').first.wait_for()
                if path == '/':
                    assert page.locator('.course-cover img').evaluate_all('(images) => images.every(i => i.complete && i.naturalWidth > 0)')
                overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
                if overflow:
                    print(page.evaluate('''() => Array.from(document.querySelectorAll('main *'))
                        .filter(el => !el.closest('.katex-mathml, .math-display, .table-scroll, pre, .mermaid-viewport') && el.getBoundingClientRect().right > innerWidth + 1)
                        .slice(0, 15).map(el => ({tag: el.tagName, cls: el.className,
                            width: el.getBoundingClientRect().width, text: el.textContent.slice(0, 120)}))'''), flush=True)
                    page.screenshot(path=str(screenshots / f'{width}-{i}-overflow.png'))
                assert not overflow, f"Horizontal overflow: {width} {path}"
                assert page.locator('table.dt th, table.dt td').evaluate_all('''cells => cells.every(cell =>
                    cell.getBoundingClientRect().width >= 126 || cell.getBoundingClientRect().width === 0
                )'''), f"Unreadably narrow table column: {width} {path}"
                assert page.locator('.topbar__l').evaluate_all('(links) => links.every(link => link.offsetHeight < 45)'), f"Wrapped navigation: {width} {path}"
                page.screenshot(path=str(screenshots / f"{width}-{i}.png"), full_page=(path == '/'))
                if width <= 600 and page.locator('#navtoggle').count():
                    page.locator('#navtoggle').click()
                    assert page.locator('#nav').evaluate('(nav) => nav.classList.contains("is-open")')
                    assert page.url.endswith(path)
                    page.screenshot(path=str(screenshots / f"{width}-{i}-menu.png"))
                    assert page.locator('html').get_attribute('data-focus') == 'false'
                    page.locator('#navtoggle').click()
                if args.curriculum:
                    assert page.locator('.katex-error').count() == 0, f'Invalid math: {path}'
                    if page.locator('.math-inline, .math-display').count():
                        assert page.locator('.katex').count() > 0, f'Math not rendered: {path}'
                    if page.locator('.diagram-shell').count():
                        page.wait_for_function('''() => Array.from(document.querySelectorAll('.diagram-shell'))
                            .every(shell => shell.querySelector('.mermaid-canvas svg') ||
                                shell.querySelector('.zoom-label')?.textContent.startsWith('Error:'))''')
                        assert page.locator('.zoom-label').evaluate_all(
                            '(nodes) => nodes.every(n => !n.textContent.startsWith("Error:"))'), f'Diagram error: {path}'
                    if path not in ('/notes/ml/', '/notes/deep-learning/'):
                        downloads = page.locator('a.code-download')
                        assert downloads.count() > 0, f'No runnable example download: {path}'
                        first = downloads.first
                        first.scroll_into_view_if_needed()
                        assert first.locator('svg').count() == 1, f'Missing download icon: {path}'
                        assert first.evaluate('''el => {
                            const copy = el.closest('.code-file').querySelector('.code-copy');
                            if (!copy) return false;
                            const a = el.getBoundingClientRect(), b = copy.getBoundingClientRect();
                            return a.right <= b.left && a.width >= 28 && a.height >= 28;
                        }'''), f'Overlapping code controls: {path}'
                        if width == 1440:
                            with page.expect_download() as event:
                                first.click()
                            download = event.value
                            assert download.failure() is None, f'Download failed: {path}'
                            program = Path(download.path()).read_text()
                            assert program.startswith('# Source: content/notes/')
                            compile(program, download.suggested_filename, 'exec')
                        if path in ('/notes/ml/trees-and-ensembles/', '/notes/deep-learning/attention-and-transformers/'):
                            page.screenshot(path=str(screenshots / f'{width}-{i}-example.png'))
                if args.subjects:
                    assert page.locator('.katex-error').count() == 0, f'Invalid math: {path}: ' + str(page.locator('.katex-error').all_text_contents())
                    if page.locator('.math-inline, .math-display').count():
                        assert page.locator('.katex').count() > 0, f'Math not rendered: {path}'
                    if page.locator('script.diagram-source').count():
                        page.wait_for_function('''() => Array.from(document.querySelectorAll('script.diagram-source'))
                            .every(source => {
                                const shell = source.closest('.diagram-shell');
                                return shell?.querySelector('.mermaid-canvas svg') ||
                                    shell?.querySelector('.zoom-label')?.textContent.startsWith('Error:');
                            })''')
                        assert page.locator('.zoom-label').evaluate_all(
                            '(nodes) => nodes.every(n => !n.textContent.startsWith("Error:"))'), f'Diagram error: {path}'
                    for link in page.locator('a[download]').all():
                        href = link.get_attribute('href')
                        if href and not href.startswith(('http:', 'https:', 'data:')):
                            response = context.request.get(page.url.rsplit('/', 1)[0] + '/' + href if not href.startswith('/') else args.base_url + href)
                            assert response.ok, f'Missing download {path}: {href}'
                if args.linear_algebra:
                    assert page.locator('.katex').count() > 200, 'Math did not render'
                    assert page.locator('.katex-error').count() == 0, 'Invalid math expression'
                    assert page.locator('figure.widget').count() == 5
                    assert page.locator('figure.widget .widget__stage > svg').count() == 4
                    for figure in page.locator('figure.widget').all():
                        figure.scroll_into_view_if_needed()
                        stage_id = figure.locator('.widget__stage').get_attribute('id')
                        figure.screenshot(path=str(screenshots / f'{width}-{stage_id}.png'))
                print(f"OK {width} {path}", flush=True)
        if args.subjects:
            assert not errors, '\n'.join(errors)
            print(f'OK: {len(paths)} corrected-subject pages; layout, math, diagrams and downloads checked')
            browser.close()
            return
        if args.curriculum:
            page.set_viewport_size({'width': 1440, 'height': 1000})
            page.goto(args.base_url + '/notes/ml/', wait_until='networkidle')
            page.locator('a[href="/notes/ml/trees-and-ensembles/#xgboost-second-order-trees-and-honest-early-stopping"]').first.click()
            assert page.url.endswith('#xgboost-second-order-trees-and-honest-early-stopping')
            assert page.locator('#xgboost-second-order-trees-and-honest-early-stopping').count() == 1
            page.evaluate("localStorage.setItem('ml:preferences', JSON.stringify({theme: 'dark'}))")
            page.emulate_media(color_scheme='dark')
            page.goto(args.base_url + '/notes/deep-learning/attention-and-transformers/', wait_until='networkidle')
            assert page.locator('html').get_attribute('data-theme') == 'dark'
            page.screenshot(path=str(screenshots / 'curriculum-dark.png'))
            assert not errors, '\n'.join(errors)
            print(f'OK: {len(paths)} ML/DL pages at the selected viewport widths; downloads and links checked')
            browser.close()
            return
        if args.quick:
            browser.close()
            return
        page.set_viewport_size({"width":1440,"height":1000})
        page.goto(args.base_url + '/notes/math/linear-algebra/', wait_until='networkidle')
        assert page.locator('#interactive-matrix .widget__cell').count() == 12
        page.locator('#scalar-multiply-btn').click()
        assert page.locator('.widget__cell').first.inner_text() == '2.4'
        page.locator('#transpose-btn').click()
        assert page.locator('.widget__mrow').count() == 3
        page.locator('#reset-matrix-btn').click()
        page.locator('#vec-a-x').fill('0')
        assert page.locator('#dot-product-val').inner_text() == '8.00'
        if args.linear_algebra:
            page.locator('#vec-a-y').fill('0')
            assert page.locator('#angle-val').text_content() == 'Undefined'
            assert page.locator('#dot-product-val').inner_text() == '0.00'
            page.locator('#vec-a-x').fill('3')
            page.locator('#vec-a-y').fill('2')
            page.locator('#classifier-angle').fill('0')
            page.locator('#classifier-offset').fill('2')
            assert 'b = -3.00' in page.locator('#classifier-readout').inner_text()
            assert page.locator('.la-normal').evaluate('(line) => Math.abs(+line.getAttribute("y2") - +line.getAttribute("y1")) < 0.001')
            assert page.locator('#pca-readout').evaluate('(el) => +el.dataset.lambda1 > +el.dataset.lambda2 && +el.dataset.lambda2 > 0')
        page.locator('[data-bookmark]').click()
        assert page.locator('[data-bookmark]').get_attribute('aria-pressed') == 'true'
        page.locator('button[data-focus]').click()
        assert page.locator('.nav').is_hidden()
        page.locator('button[data-focus]').click()
        page.locator('[data-open-search]').click()
        page.locator('#search-input').fill('KV cache')
        page.wait_for_function('document.querySelector("#search-status").textContent.includes("results")')
        assert page.locator('.search-result').count() > 0
        page.locator('[data-filter="labs"]').click()
        page.wait_for_function('Array.from(document.querySelectorAll(".search-result")).every(a => a.pathname.includes("/labs/"))')
        page.locator('#search-input').fill('')
        page.locator('[data-filter="saved"]').click()
        page.wait_for_function('document.querySelector("#search-status").textContent.includes("saved pages")')
        assert page.locator('.search-result').count() == 1
        page.locator('#search-dialog [data-close-dialog]').click()
        page.locator('[data-open-settings]').click()
        page.locator('[data-theme-choice="dark"]').click()
        page.locator('#reading-size').evaluate('(input) => input.value = "20"')
        page.locator('#reading-size').dispatch_event('input')
        page.locator('#settings-dialog [data-close-dialog]').click()
        assert page.locator('html').get_attribute('data-theme') == 'dark'
        page.screenshot(path=str(screenshots/'dark-note.png'))
        page.reload(wait_until='networkidle')
        assert page.locator('html').get_attribute('data-theme') == 'dark'
        assert page.locator('#reading-size').input_value() == '20'
        page.emulate_media(media='print')
        assert page.locator('.topbar').is_hidden()
        assert page.locator('body').evaluate('(el) => getComputedStyle(el).backgroundColor') == 'rgb(255, 255, 255)'
        page.emulate_media(media='screen')
        page.goto(args.base_url + '/courses/transformers/03-self-attention-from-scratch/', wait_until='networkidle')
        page.locator('.mermaid-canvas svg').first.wait_for(timeout=30000)
        assert page.locator('.katex').count() > 0
        page.locator('.code-copy').first.click()
        assert page.evaluate('navigator.clipboard.readText()')
        page.evaluate('window.scrollTo(0, 1500)')
        page.wait_for_timeout(200)
        page.goto(args.base_url + '/', wait_until='networkidle')
        assert page.locator('#continue-reading').is_visible()
        page.screenshot(path=str(screenshots/'dark-home.png'))
        page.set_viewport_size({'width':320,'height':720})
        page.locator('[data-open-search]').click()
        page.locator('#search-input').fill('attention')
        page.wait_for_function('document.querySelector("#search-status").textContent.includes("results")')
        page.screenshot(path=str(screenshots/'mobile-search.png'))
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth + 1')
        page.keyboard.press('Escape')
        page.locator('#search-dialog').wait_for(state='hidden')
        page.locator('[data-open-settings]').click()
        page.screenshot(path=str(screenshots/'mobile-settings.png'))
        page.locator('#reset-preferences').click()
        assert page.locator('html').get_attribute('data-theme') == 'light'
        assert page.locator('#reading-size').input_value() == '18'
        page.keyboard.press('Escape')
        assert not errors, json.dumps(errors, indent=2)
        browser.close()
        print('OK: search, saved pages, focus, preferences, widgets, math, diagrams, copy, and reading history')


if __name__ == '__main__':
    main()
