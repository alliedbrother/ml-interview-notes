import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

const config = {
  fitPadding: 26, minHeight: 250, maxHeightPx: 940, maxHeightVh: 0.82,
  maxInitialZoom: 1.8, minZoom: 0.08, maxZoom: 6.5, zoomStep: 0.14,
  readabilityFloor: 0.40
};
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
let activeDrag = null;
addEventListener('mousemove', e => activeDrag?.onMove(e));
addEventListener('mouseup', () => { activeDrag?.onEnd(); activeDrag = null; });

const isDark = matchMedia('(prefers-color-scheme: dark)').matches;

/* ELK is fetched from a CDN at view time. A static import cannot be caught, so
   one failed request would leave mermaid configured for a layout engine it does
   not have — and mermaid reports that by writing "Syntax error in text" into the
   page, blaming content that is perfectly valid. Load it dynamically and fall
   back to the built-in dagre layout instead: a diagram laid out differently
   beats a red box claiming the diagram is broken. */
let layoutEngine = 'elk';
try {
  const elkLayouts = (await import(
    'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk/dist/mermaid-layout-elk.esm.min.mjs'
  )).default;
  mermaid.registerLayoutLoaders(elkLayouts);
} catch (err) {
  layoutEngine = 'dagre';
  console.warn('[mermaid] ELK layout unavailable, falling back to dagre:', err);
}
mermaid.initialize({
  startOnLoad: false, theme: 'base', look: 'classic', layout: layoutEngine,
  themeVariables: {
    fontFamily: "'IBM Plex Sans', system-ui, sans-serif",
    fontSize: '17px',
    primaryColor:         isDark ? '#12363a' : '#dcecea',
    primaryBorderColor:   isDark ? '#4dbdb4' : '#0d6a6e',
    primaryTextColor:     isDark ? '#dde7e9' : '#16242a',
    secondaryColor:       isDark ? '#382519' : '#f7e9dc',
    secondaryBorderColor: isDark ? '#e0955a' : '#a8551c',
    secondaryTextColor:   isDark ? '#dde7e9' : '#16242a',
    tertiaryColor:        isDark ? '#25301a' : '#edf2df',
    tertiaryBorderColor:  isDark ? '#a8c95e' : '#55701c',
    tertiaryTextColor:    isDark ? '#dde7e9' : '#16242a',
    lineColor:            isDark ? '#8ea2a8' : '#5c6b70',
    clusterBkg:           isDark ? '#1b262b' : '#eae7e0',
    clusterBorder:        isDark ? 'rgba(160,200,205,0.28)' : 'rgba(28,50,54,0.28)',
    noteBkgColor:         isDark ? '#382519' : '#f7e9dc',
    noteTextColor:        isDark ? '#dde7e9' : '#16242a',
    noteBorderColor:      isDark ? '#e0955a' : '#a8551c'
  }
});

function initDiagram(shell) {
  const wrap = shell.querySelector('.mermaid-wrap');
  const viewport = shell.querySelector('.mermaid-viewport');
  const canvas = shell.querySelector('.mermaid-canvas');
  const source = shell.querySelector('.diagram-source');
  const label = shell.querySelector('.zoom-label');
  if (!wrap || !viewport || !canvas || !source || !label) return;

  let zoom = 1, fitMode = 'contain', panX = 0, panY = 0, svgW = 0, svgH = 0;
  let sx = 0, sy = 0, spx = 0, spy = 0, touchDist = 0, touchCx = 0, touchCy = 0;

  function constrainPan() {
    const vpW = viewport.clientWidth, vpH = viewport.clientHeight;
    const rW = svgW * zoom, rH = svgH * zoom, pad = config.fitPadding;
    panX = (rW + pad * 2 <= vpW) ? (vpW - rW) / 2 : clamp(panX, vpW - rW - pad, pad);
    panY = (rH + pad * 2 <= vpH) ? (vpH - rH) / 2 : clamp(panY, vpH - rH - pad, pad);
  }
  function applyTransform() {
    const svg = canvas.querySelector('svg');
    if (!svg || !svgW) return;
    constrainPan();
    svg.style.width = (svgW * zoom) + 'px';
    svg.style.height = (svgH * zoom) + 'px';
    canvas.style.transform = `translate(${panX}px, ${panY}px)`;
    label.textContent = Math.round(zoom * 100) + '% — ' + fitMode;
  }
  function canPan() {
    return svgW * zoom + config.fitPadding * 2 > viewport.clientWidth
        || svgH * zoom + config.fitPadding * 2 > viewport.clientHeight;
  }
  function computeSmartFit() {
    const vpW = viewport.clientWidth, vpH = viewport.clientHeight;
    const aW = Math.max(80, vpW - config.fitPadding * 2);
    const aH = Math.max(80, vpH - config.fitPadding * 2);
    const contain = Math.min(aW / svgW, aH / svgH);
    let z = contain, mode = 'contain';
    if (contain < config.readabilityFloor) {
      if (svgH / svgW >= vpH / Math.max(vpW, 1)) { z = aW / svgW; mode = 'width-priority'; }
      else { z = aH / svgH; mode = 'height-priority'; }
    }
    return { zoom: clamp(z, config.minZoom, config.maxInitialZoom), mode };
  }
  function fitDiagram() {
    if (!svgW) return;
    const fit = computeSmartFit();
    zoom = fit.zoom; fitMode = fit.mode;
    panX = (viewport.clientWidth - svgW * zoom) / 2;
    panY = (viewport.clientHeight - svgH * zoom) / 2;
    applyTransform();
  }
  function setOneToOne() {
    zoom = clamp(1, config.minZoom, config.maxZoom); fitMode = '1:1';
    panX = (viewport.clientWidth - svgW * zoom) / 2;
    panY = (viewport.clientHeight - svgH * zoom) / 2;
    applyTransform();
  }
  function zoomAround(factor, cx, cy) {
    const next = clamp(zoom * factor, config.minZoom, config.maxZoom);
    const ratio = next / zoom;
    panX = cx - ratio * (cx - panX); panY = cy - ratio * (cy - panY);
    zoom = next; fitMode = 'custom'; applyTransform();
  }
  function readSvgNaturalSize(svg) {
    let w = 0, h = 0;
    if (svg.viewBox?.baseVal?.width > 0) { w = svg.viewBox.baseVal.width; h = svg.viewBox.baseVal.height; }
    if (!w) { w = parseFloat(svg.getAttribute('width')) || 0; h = parseFloat(svg.getAttribute('height')) || 0; }
    if (!w) { const b = svg.getBBox(); w = b.width; h = b.height; }
    if (!w) { const r = svg.getBoundingClientRect(); w = r.width || 1000; h = r.height || 700; }
    if (!svg.getAttribute('viewBox')) svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    return { w, h };
  }
  function setAdaptiveHeight() {
    if (!svgW) return;
    const usableW = Math.max(280, wrap.getBoundingClientRect().width - 2);
    const idealH = (svgH / svgW) * usableW + config.fitPadding * 2;
    const maxVp = Math.floor(innerHeight * config.maxHeightVh);
    const hardMax = Math.min(config.maxHeightPx, Math.max(config.minHeight + 40, maxVp));
    wrap.style.height = Math.round(clamp(idealH, config.minHeight, hardMax)) + 'px';
  }
  function openInNewTab() {
    const svg = canvas.querySelector('svg');
    if (!svg) return;
    const clone = svg.cloneNode(true);
    clone.style.width = ''; clone.style.height = '';
    const bg = isDark ? '#0e1417' : '#f3f1ec';
    const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Diagram</title><style>
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:${bg};padding:40px;box-sizing:border-box}svg{max-width:100%;max-height:90vh;height:auto}
</style></head><body>${clone.outerHTML}</body></html>`;
    open(URL.createObjectURL(new Blob([html], { type: 'text/html' })), '_blank');
  }
  async function render() {
    try {
      const code = source.textContent.trim();
      if (!code) { label.textContent = 'Error: empty source'; return; }
      const id = 'd-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
      const { svg } = await mermaid.render(id, code);
      canvas.innerHTML = svg;
      const svgNode = canvas.querySelector('svg');
      if (!svgNode) { label.textContent = 'Error: no SVG'; return; }
      const size = readSvgNaturalSize(svgNode);
      svgW = size.w; svgH = size.h;
      svgNode.removeAttribute('width'); svgNode.removeAttribute('height');
      svgNode.style.maxWidth = 'none'; svgNode.style.display = 'block';
      setAdaptiveHeight(); fitDiagram();
    } catch (err) {
      console.error('Mermaid render failed:', err);
      label.textContent = 'Error: ' + (err.message || 'render failed');
    }
  }
  const actions = {
    'zoom-in': () => zoomAround(1 + config.zoomStep, viewport.clientWidth / 2, viewport.clientHeight / 2),
    'zoom-out': () => zoomAround(1 / (1 + config.zoomStep), viewport.clientWidth / 2, viewport.clientHeight / 2),
    'zoom-fit': fitDiagram, 'zoom-one': setOneToOne, 'zoom-expand': openInNewTab
  };
  Object.entries(actions).forEach(([a, h]) => wrap.querySelector(`[data-action="${a}"]`)?.addEventListener('click', h));
  viewport.addEventListener('dblclick', fitDiagram);
  viewport.addEventListener('wheel', e => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      zoomAround(e.deltaY < 0 ? 1 + config.zoomStep : 1 / (1 + config.zoomStep), e.clientX - rect.left, e.clientY - rect.top);
      return;
    }
    if (canPan()) { e.preventDefault(); panX -= e.deltaX; panY -= e.deltaY; applyTransform(); }
  }, { passive: false });
  viewport.addEventListener('mousedown', e => {
    if (e.target.closest('.zoom-controls') || !canPan()) return;
    wrap.classList.add('is-panning');
    sx = e.clientX; sy = e.clientY; spx = panX; spy = panY;
    e.preventDefault();
    activeDrag = {
      onMove: ev => { panX = spx + (ev.clientX - sx); panY = spy + (ev.clientY - sy); applyTransform(); },
      onEnd: () => wrap.classList.remove('is-panning')
    };
  });
  viewport.addEventListener('touchstart', e => {
    if (e.touches.length === 1) { sx = e.touches[0].clientX; sy = e.touches[0].clientY; spx = panX; spy = panY; }
    else if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX, dy = e.touches[0].clientY - e.touches[1].clientY;
      touchDist = Math.sqrt(dx * dx + dy * dy);
      const r = viewport.getBoundingClientRect();
      touchCx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - r.left;
      touchCy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - r.top;
    }
  }, { passive: true });
  viewport.addEventListener('touchmove', e => {
    if (e.touches.length === 1 && canPan()) {
      if (touchDist > 0) { sx = e.touches[0].clientX; sy = e.touches[0].clientY; spx = panX; spy = panY; touchDist = 0; }
      e.preventDefault();
      panX = spx + (e.touches[0].clientX - sx); panY = spy + (e.touches[0].clientY - sy); applyTransform();
    } else if (e.touches.length === 2 && touchDist > 0) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX, dy = e.touches[0].clientY - e.touches[1].clientY;
      const d = Math.sqrt(dx * dx + dy * dy);
      zoomAround(d / touchDist, touchCx, touchCy); touchDist = d;
    }
  }, { passive: false });
  new ResizeObserver(() => { if (svgW) { setAdaptiveHeight(); fitDiagram(); } }).observe(wrap);
  render();
}
document.querySelectorAll('.diagram-shell').forEach(initDiagram);
