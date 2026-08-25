/* ================================================================
   Interactive linear-algebra figures
   ----------------------------------------------------------------
   Five D3 v7 widgets that attach themselves to whatever containers
   happen to be on the page:

     #interactive-matrix     scalar multiply / transpose / reset
     #dot-product-viz        draggable vectors a and b
     #interactive-classifier drag a separating line through 2 clouds
     #pca-visualization      a cloud and its principal components
     #svd-visualization      A = U S V^T as four blocks

   Every widget is optional: if its container is missing the widget
   simply never initialises. Colours are read back from the CSS
   custom properties in the site theme (never hardcoded), so the
   figures follow the light/dark palette; they are redrawn when the
   colour scheme flips and when the container width changes.

   Requires the d3 v7 global to have been loaded first.
   ================================================================ */

(function () {
  'use strict';

  /* ---------------------------------------------------------- utils */

  function byId(id) { return document.getElementById(id); }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* Read one design token. getComputedStyle resolves var() chains in
     modern engines, but a couple hand back the raw "var(--x)" token
     stream, so unwrap it defensively. Body first: the per-track
     --accent override lives on <body data-track>. */
  function cssVar(name, fallback) {
    var host = document.body || document.documentElement;
    var value = '';
    try { value = getComputedStyle(host).getPropertyValue(name); } catch (err) { value = ''; }
    value = (value || '').trim();
    var hops = 0;
    while (value.indexOf('var(') === 0 && hops < 4) {
      hops += 1;
      var inner = value.slice(4, value.lastIndexOf(')')).split(',')[0].trim();
      try { value = (getComputedStyle(host).getPropertyValue(inner) || '').trim(); }
      catch (err2) { value = ''; }
    }
    return value || fallback;
  }

  function palette() {
    return {
      text: cssVar('--text', '#14273d'),
      dim: cssVar('--text-dim', '#5b7186'),
      border: cssVar('--border', 'rgba(30,58,95,.13)'),
      borderBright: cssVar('--border-bright', 'rgba(30,58,95,.26)'),
      surface: cssVar('--surface', '#fbfcfd'),
      accent: cssVar('--accent', '#0b4f75'),
      blue: cssVar('--blue', '#0b4f75'),
      amber: cssVar('--amber', '#b45309'),
      sage: cssVar('--sage', '#4d7c0f'),
      rose: cssVar('--rose', '#9f1239'),
      steel: cssVar('--steel', '#475569')
    };
  }

  /* Logical drawing width: the container's own width, floored so the
     figure never collapses and (optionally) capped. */
  function widthOf(el, min, max) {
    var w = Math.round(el.getBoundingClientRect().width || el.clientWidth || 0);
    if (!w) w = min;
    if (max) w = Math.min(w, max);
    return Math.max(min, w);
  }

  /* Every redrawable figure registers itself here. */
  var figures = [];
  function register(name, draw) { figures.push({ name: name, draw: draw }); }
  function drawAll() {
    for (var i = 0; i < figures.length; i++) {
      try {
        figures[i].draw();
      } catch (err) {
        if (window.console && console.error) console.error('[linear-algebra] ' + figures[i].name, err);
      }
    }
  }

  /* One tooltip element, shared by every figure that wants one. */
  var tipEl = null;
  function tipNode() {
    if (!tipEl) {
      tipEl = document.createElement('div');
      tipEl.className = 'widget-tip';
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }
  function showTip(markup) {
    var t = tipNode();
    t.innerHTML = markup;
    t.style.visibility = 'visible';
  }
  function moveTip(event) {
    var t = tipNode();
    t.style.top = (event.pageY - 12) + 'px';
    t.style.left = (event.pageX + 14) + 'px';
  }
  function hideTip() { if (tipEl) tipEl.style.visibility = 'hidden'; }

  /* -------------------------------------------------- 1. the matrix */

  function initMatrix() {
    var host = byId('interactive-matrix');
    if (!host) return;

    var scalarInput = byId('scalar-input');
    var multiplyBtn = byId('scalar-multiply-btn');
    var transposeBtn = byId('transpose-btn');
    var resetBtn = byId('reset-matrix-btn');

    var original = [
      [1.2, 15.5, 0.8],
      [2.5, 22.1, 0.2],
      [0.8, 12.0, 0.9],
      [3.1, 25.3, 0.1]
    ];
    var current = original.map(function (row) { return row.slice(); });

    function render() {
      while (host.firstChild) host.removeChild(host.firstChild);
      var grid = document.createElement('div');
      grid.className = 'widget__matrix';
      current.forEach(function (rowData, i) {
        var row = document.createElement('div');
        row.className = 'widget__mrow';
        rowData.forEach(function (value, j) {
          var cell = document.createElement('div');
          cell.className = 'widget__cell';
          cell.setAttribute('data-row', String(i));
          cell.setAttribute('data-col', String(j));
          cell.textContent = value.toFixed(1);
          row.appendChild(cell);
        });
        grid.appendChild(row);
      });
      host.appendChild(grid);
    }

    function cells() { return host.querySelectorAll('.widget__cell'); }

    function cool() {
      var all = cells();
      for (var i = 0; i < all.length; i++) all[i].classList.remove('is-hot');
    }

    /* Delegated so it survives every re-render: hovering a cell lights
       up its whole row and its whole column. */
    host.addEventListener('mouseover', function (event) {
      var cell = event.target && event.target.closest ? event.target.closest('.widget__cell') : null;
      if (!cell || !host.contains(cell)) return;
      var row = cell.getAttribute('data-row');
      var col = cell.getAttribute('data-col');
      cool();
      var all = cells();
      for (var i = 0; i < all.length; i++) {
        if (all[i].getAttribute('data-row') === row || all[i].getAttribute('data-col') === col) {
          all[i].classList.add('is-hot');
        }
      }
    });
    host.addEventListener('mouseleave', cool);

    if (multiplyBtn) {
      multiplyBtn.addEventListener('click', function () {
        var scalar = scalarInput ? (parseFloat(scalarInput.value) || 0) : 0;
        current = current.map(function (row) {
          return row.map(function (v) { return v * scalar; });
        });
        render();
      });
    }

    if (transposeBtn) {
      transposeBtn.addEventListener('click', function () {
        if (!current.length || !current[0].length) return;
        current = current[0].map(function (_, col) {
          return current.map(function (row) { return row[col]; });
        });
        render();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        current = original.map(function (row) { return row.slice(); });
        render();
      });
    }

    render();
  }

  /* --------------------------------------------- 2. the dot product */

  function initDotProduct() {
    var host = byId('dot-product-viz');
    if (!host || typeof d3 === 'undefined') return;

    var inputs = {
      ax: byId('vec-a-x'), ay: byId('vec-a-y'),
      bx: byId('vec-b-x'), by: byId('vec-b-y')
    };
    if (!inputs.ax || !inputs.ay || !inputs.bx || !inputs.by) return;

    var dotOut = byId('dot-product-val');
    var angleOut = byId('angle-val');
    var noteOut = byId('dot-product-explanation');

    var MAX = 6;
    var view = null;
    var dragging = false;

    function read(id) {
      return { x: +inputs[id + 'x'].value || 0, y: +inputs[id + 'y'].value || 0 };
    }

    function build() {
      var c = palette();
      var width = widthOf(host, 280);
      var height = 320;
      var margin = { top: 18, right: 18, bottom: 18, left: 18 };
      var iw = width - margin.left - margin.right;
      var ih = height - margin.top - margin.bottom;

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', '0 0 ' + width + ' ' + height)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      var defs = svg.append('defs');
      [['a', c.blue], ['b', c.rose]].forEach(function (pair) {
        defs.append('marker')
          .attr('id', 'la-dp-arrow-' + pair[0])
          .attr('viewBox', '0 -5 10 10')
          .attr('refX', 9).attr('refY', 0)
          .attr('markerWidth', 5).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', pair[1]);
      });

      var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      var x = d3.scaleLinear().domain([-MAX, MAX]).range([0, iw]);
      var y = d3.scaleLinear().domain([-MAX, MAX]).range([ih, 0]);
      var cx = x(0), cy = y(0);

      var grid = g.append('g');
      d3.range(-MAX, MAX + 1).forEach(function (v) {
        grid.append('line')
          .attr('x1', x(v)).attr('x2', x(v)).attr('y1', 0).attr('y2', ih)
          .attr('stroke', c.border).attr('stroke-width', 1);
        grid.append('line')
          .attr('y1', y(v)).attr('y2', y(v)).attr('x1', 0).attr('x2', iw)
          .attr('stroke', c.border).attr('stroke-width', 1);
      });
      g.append('line')
        .attr('x1', 0).attr('x2', iw).attr('y1', cy).attr('y2', cy)
        .attr('stroke', c.borderBright).attr('stroke-width', 1.4);
      g.append('line')
        .attr('y1', 0).attr('y2', ih).attr('x1', cx).attr('x2', cx)
        .attr('stroke', c.borderBright).attr('stroke-width', 1.4);

      view = {
        c: c, x: x, y: y, cx: cx, cy: cy,
        angleG: g.append('g'),
        vectorG: g.append('g')
      };
      update();
    }

    function update() {
      if (!view) return;
      var c = view.c, x = view.x, y = view.y, cx = view.cx, cy = view.cy;

      var a = read('a');
      var b = read('b');
      var dot = a.x * b.x + a.y * b.y;
      var magA = Math.sqrt(a.x * a.x + a.y * a.y);
      var magB = Math.sqrt(b.x * b.x + b.y * b.y);
      var angleRad = (magA * magB === 0) ? 0 : Math.acos(clamp(dot / (magA * magB), -1, 1));
      var angleDeg = isNaN(angleRad) ? 0 : angleRad * (180 / Math.PI);

      if (dotOut) dotOut.textContent = dot.toFixed(2);
      if (angleOut) angleOut.textContent = angleDeg.toFixed(1) + '°';

      var cosTheta = dot / (magA * magB);
      if (isNaN(cosTheta)) cosTheta = 0;

      if (noteOut) {
        /* Cover the whole range: a blank readout for most of the input space
           teaches nothing, and the default vectors land in the middle. */
        var explanation;
        if (magA === 0 || magB === 0) explanation = 'A zero vector has no direction — the dot product is 0.';
        else if (cosTheta > 0.95) explanation = 'Almost the same direction — about as similar as it gets.';
        else if (cosTheta > 0.3) explanation = 'Pointing broadly the same way — positively similar.';
        else if (cosTheta > 0.1) explanation = 'Only loosely related — a small positive dot product.';
        else if (cosTheta >= -0.1) explanation = 'Near perpendicular: nothing in common, dot product near zero.';
        else if (cosTheta >= -0.95) explanation = 'Pointing broadly opposite ways — anti-similar.';
        else explanation = 'Almost exactly opposite — maximally anti-similar.';
        noteOut.textContent = explanation;
      }

      /* the wedge between a and b, swept the short way round */
      view.angleG.selectAll('*').remove();
      if (magA > 0 && magB > 0) {
        var angleA = Math.atan2(-a.y, a.x);
        var cross = a.x * b.y - a.y * b.x;
        var sweep = cross === 0 ? 1 : -Math.sign(cross);
        var arc = d3.arc().innerRadius(0).outerRadius(30)
          .startAngle(angleA)
          .endAngle(angleA + sweep * angleRad);
        view.angleG.append('path')
          .attr('d', arc)
          .attr('transform', 'translate(' + cx + ',' + cy + ')')
          .attr('fill', c.amber)
          .attr('opacity', 0.42);
      }

      var vectors = [
        { x: a.x, y: a.y, color: c.blue, id: 'a' },
        { x: b.x, y: b.y, color: c.rose, id: 'b' }
      ];

      var lines = view.vectorG.selectAll('line.la-vector').data(vectors, function (d) { return d.id; });
      var linesAll = lines.enter().append('line')
        .attr('class', 'la-vector')
        .attr('x1', cx).attr('y1', cy)
        .attr('marker-end', function (d) { return 'url(#la-dp-arrow-' + d.id + ')'; })
        .merge(lines)
        .attr('stroke', function (d) { return d.color; })
        .attr('stroke-width', 3.5)
        .attr('stroke-linecap', 'round');
      (dragging ? linesAll : linesAll.transition().duration(180))
        .attr('x2', function (d) { return x(d.x); })
        .attr('y2', function (d) { return y(d.y); });

      var handles = view.vectorG.selectAll('circle.la-handle').data(vectors, function (d) { return d.id; });
      var handlesAll = handles.enter().append('circle')
        .attr('class', 'la-handle')
        .attr('r', 9)
        .attr('cx', cx).attr('cy', cy)
        .style('cursor', 'grab')
        .call(d3.drag()
          /* the bound datum carries x/y in DATA units; without this the
             default subject accessor would offset every drag by them */
          .subject(function (event) { return { x: event.x, y: event.y }; })
          .on('start', function () { dragging = true; d3.select(this).style('cursor', 'grabbing'); })
          .on('drag', function (event, d) {
            inputs[d.id + 'x'].value = clamp(x.invert(event.x), -MAX, MAX).toFixed(1);
            inputs[d.id + 'y'].value = clamp(y.invert(event.y), -MAX, MAX).toFixed(1);
            update();
          })
          .on('end', function () { dragging = false; d3.select(this).style('cursor', 'grab'); }))
        .merge(handles)
        .attr('fill', function (d) { return d.color; });
      (dragging ? handlesAll : handlesAll.transition().duration(180))
        .attr('cx', function (d) { return x(d.x); })
        .attr('cy', function (d) { return y(d.y); });
    }

    Object.keys(inputs).forEach(function (k) {
      inputs[k].addEventListener('input', update);
    });

    register('dot-product', build);
  }

  /* ---------------------------------------------- 3. the classifier */

  function initClassifier() {
    var host = byId('interactive-classifier');
    if (!host || typeof d3 === 'undefined') return;

    /* Sampled once, so resizing or flipping the theme does not deal a
       whole new crowd of customers. */
    var cloud = null;
    function points() {
      if (cloud) return cloud;
      var hipsters = d3.range(30).map(function () {
        return { x: d3.randomNormal(2.5, 0.9)(), y: d3.randomNormal(7.5, 0.9)(), cls: 0, icon: '🥸' };
      });
      var techies = d3.range(30).map(function () {
        return { x: d3.randomNormal(7.5, 0.9)(), y: d3.randomNormal(2.5, 0.9)(), cls: 1, icon: '🚀' };
      });
      cloud = hipsters.concat(techies).filter(function (d) {
        return d.x >= 0 && d.x <= 10 && d.y >= 0 && d.y <= 10;
      });
      return cloud;
    }

    var angle = -Math.PI / 4;   /* survives redraws */

    function draw() {
      var c = palette();
      var width = widthOf(host, 300);
      var height = Math.round(clamp(width * 0.74, 300, 400));
      var margin = { top: 16, right: 22, bottom: 44, left: 48 };
      var iw = width - margin.left - margin.right;
      var ih = height - margin.top - margin.bottom;

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', '0 0 ' + width + ' ' + height)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      svg.append('text')
        .attr('class', 'widget__mono')
        .attr('text-anchor', 'middle')
        .attr('x', margin.left + iw / 2)
        .attr('y', height - 12)
        .attr('fill', c.dim)
        .attr('font-size', 9.5)
        .attr('font-weight', 600)
        .text('STARTUP PITCH ENTHUSIASM');

      svg.append('text')
        .attr('class', 'widget__mono')
        .attr('text-anchor', 'middle')
        .attr('transform', 'translate(13,' + (margin.top + ih / 2) + ') rotate(-90)')
        .attr('fill', c.dim)
        .attr('font-size', 9.5)
        .attr('font-weight', 600)
        .text('LOVE FOR OBSCURE INDIE MUSIC');

      var x = d3.scaleLinear().domain([0, 10]).range([0, iw]);
      var y = d3.scaleLinear().domain([0, 10]).range([ih, 0]);

      var xAxis = g.append('g')
        .attr('transform', 'translate(0,' + ih + ')')
        .call(d3.axisBottom(x).ticks(5).tickSize(-ih).tickPadding(8));
      var yAxis = g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iw).tickPadding(8));
      [xAxis, yAxis].forEach(function (axis) {
        axis.select('.domain').remove();
        axis.selectAll('.tick line').attr('stroke', c.border);
        axis.selectAll('.tick text')
          .attr('fill', c.dim).attr('font-size', 9.5).attr('class', 'widget__mono');
      });

      /* the separator sits under the crowd so the emoji stay readable,
         and is clipped to the plot so it never crosses the axis labels */
      var clipId = 'la-clf-clip';
      svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('x', 0).attr('y', 0).attr('width', iw).attr('height', ih);
      var line = g.append('g').attr('clip-path', 'url(#' + clipId + ')').append('line')
        .attr('stroke', c.text).attr('stroke-width', 2.5).attr('stroke-linecap', 'round');

      g.selectAll('text.la-emoji')
        .data(points())
        .enter().append('text')
        .attr('class', 'la-emoji')
        .attr('x', function (d) { return x(d.x); })
        .attr('y', function (d) { return y(d.y); })
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', 22)
        .style('cursor', 'pointer')
        .text(function (d) { return d.icon; })
        .on('mouseover', function (event, d) {
          d3.select(this).transition().duration(180).attr('font-size', 30);
          showTip('Indie Love: ' + d.y.toFixed(1) + '<br>Pitch Zeal: ' + d.x.toFixed(1));
        })
        .on('mousemove', moveTip)
        .on('mouseout', function () {
          d3.select(this).transition().duration(180).attr('font-size', 22);
          hideTip();
        });

      var handle = g.append('circle')
        .attr('r', 9)
        .attr('fill', c.accent)
        .attr('stroke', c.surface)
        .attr('stroke-width', 2)
        .style('cursor', 'move')
        .on('mouseover', function () { d3.select(this).transition().duration(180).attr('r', 11); })
        .on('mouseout', function () { d3.select(this).transition().duration(180).attr('r', 9); });

      function place(next) {
        angle = next;
        var reach = Math.max(width, height) * 2;
        var midX = x(5), midY = y(5);
        line
          .attr('x1', midX + reach * Math.cos(angle)).attr('y1', midY + reach * Math.sin(angle))
          .attr('x2', midX - reach * Math.cos(angle)).attr('y2', midY - reach * Math.sin(angle));
        handle
          .attr('cx', midX + (ih / 3.5) * Math.cos(angle + Math.PI / 2))
          .attr('cy', midY + (ih / 3.5) * Math.sin(angle + Math.PI / 2));
      }

      handle.call(d3.drag().on('drag', function (event) {
        place(Math.atan2(event.y - y(5), event.x - x(5)) - Math.PI / 2);
      }));

      place(angle);
    }

    register('classifier', draw);
  }

  /* ----------------------------------------------------- 4. the PCA */

  function initPCA() {
    var host = byId('pca-visualization');
    if (!host || typeof d3 === 'undefined') return;

    var SPREAD = [1, 0.4];            /* std dev along each axis */
    var TILT = -Math.PI / 4;          /* the cloud is rotated by this */
    var cloud = null;

    function points() {
      if (cloud) return cloud;
      cloud = d3.range(150).map(function () {
        var u = d3.randomNormal(0, SPREAD[0])();
        var v = d3.randomNormal(0, SPREAD[1])();
        return {
          x: u * Math.cos(TILT) - v * Math.sin(TILT),
          y: u * Math.sin(TILT) + v * Math.cos(TILT)
        };
      });
      return cloud;
    }

    function draw() {
      var c = palette();
      var width = widthOf(host, 300);
      var height = Math.round(clamp(width * 0.62, 250, 340));
      var margin = { top: 22, right: 26, bottom: 22, left: 26 };
      var iw = width - margin.left - margin.right;
      var ih = height - margin.top - margin.bottom;

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', '0 0 ' + width + ' ' + height)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      var defs = svg.append('defs');
      [['pc1', c.rose], ['pc2', c.blue]].forEach(function (pair) {
        defs.append('marker')
          .attr('id', 'la-pca-arrow-' + pair[0])
          .attr('viewBox', '0 -5 10 10')
          .attr('refX', 8).attr('refY', 0)
          .attr('markerWidth', 5).attr('markerHeight', 5)
          .attr('orient', 'auto')
          .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', pair[1]);
      });

      var g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      var data = points();
      var spanX = d3.max(data, function (d) { return Math.abs(d.x); }) * 1.08;
      var spanY = d3.max(data, function (d) { return Math.abs(d.y); }) * 1.15;
      var x = d3.scaleLinear().domain([-spanX, spanX]).range([0, iw]);
      var y = d3.scaleLinear().domain([-spanY, spanY]).range([ih, 0]);
      var cx = x(0), cy = y(0);

      g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', cy).attr('y2', cy)
        .attr('stroke', c.border).attr('stroke-width', 1);
      g.append('line').attr('y1', 0).attr('y2', ih).attr('x1', cx).attr('x2', cx)
        .attr('stroke', c.border).attr('stroke-width', 1);

      g.selectAll('circle.la-pt')
        .data(data)
        .enter().append('circle')
        .attr('class', 'la-pt')
        .attr('cx', function (d) { return x(d.x); })
        .attr('cy', function (d) { return y(d.y); })
        .attr('r', 2.6)
        .attr('fill', c.steel)
        .attr('opacity', 0.5);

      /* Both components are drawn through the scales, so they stay
         glued to the cloud whatever the aspect ratio, and their
         length is proportional to the variance they explain. */
      function component(theta, scale, colour, marker, label) {
        var reach = 2.3 * scale;
        var dx = Math.cos(theta) * reach;
        var dy = Math.sin(theta) * reach;
        [1, -1].forEach(function (sign) {
          g.append('line')
            .attr('x1', cx).attr('y1', cy)
            .attr('x2', x(dx * sign)).attr('y2', y(dy * sign))
            .attr('stroke', colour).attr('stroke-width', 2.6)
            .attr('stroke-linecap', 'round')
            .attr('marker-end', 'url(#la-pca-arrow-' + marker + ')');
        });
        g.append('text')
          .attr('class', 'widget__mono')
          .attr('x', x(dx) + (dx > 0 ? 8 : -8))
          .attr('y', y(dy) - 8)
          .attr('text-anchor', dx > 0 ? 'start' : 'end')
          .attr('fill', colour)
          .attr('font-size', 10)
          .attr('font-weight', 700)
          .text(label);
      }

      component(TILT, SPREAD[0], c.rose, 'pc1', 'PC1');
      component(TILT + Math.PI / 2, SPREAD[1], c.blue, 'pc2', 'PC2');
    }

    register('pca', draw);
  }

  /* ----------------------------------------------------- 5. the SVD */

  function initSVD() {
    var host = byId('svd-visualization');
    if (!host || typeof d3 === 'undefined') return;

    function draw() {
      var c = palette();
      var W = 476, H = 176;
      var size = 96, top = 26;

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', W)
        .attr('height', H)
        .attr('viewBox', '0 0 ' + W + ' ' + H)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      var blocks = [
        { key: 'A', label: 'A', colour: c.accent, x: 8, sub: 'm × n' },
        { key: 'U', label: 'U', colour: c.blue, x: 138, sub: 'm × r' },
        { key: 'S', label: 'Σ', colour: c.sage, x: 250, sub: 'r × r' },
        { key: 'V', label: 'Vᵀ', colour: c.rose, x: 362, sub: 'r × n' }
      ];

      blocks.forEach(function (b) {
        svg.append('rect')
          .attr('x', b.x).attr('y', top)
          .attr('width', size).attr('height', size).attr('rx', 8)
          .attr('fill', b.colour).attr('fill-opacity', 0.12)
          .attr('stroke', b.colour).attr('stroke-width', 1.4);
      });

      /* the singular values themselves, fading as they shrink: keep
         only the top few and you have the low-rank approximation */
      var cell = size / 3;
      [0.8, 0.45, 0.2].forEach(function (weight, i) {
        var sigma = blocks[2];
        svg.append('rect')
          .attr('x', sigma.x + i * cell).attr('y', top + i * cell)
          .attr('width', cell).attr('height', cell)
          .attr('fill', c.sage).attr('fill-opacity', weight);
        svg.append('text')
          .attr('class', 'widget__mono')
          .attr('x', sigma.x + i * cell + cell / 2)
          .attr('y', top + i * cell + cell / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .attr('fill', c.text)
          .attr('font-size', 9)
          .text('σ' + (i + 1));
      });

      blocks.forEach(function (b) {
        svg.append('text')
          .attr('x', b.x + size / 2)
          .attr('y', b.key === 'S' ? top - 9 : top + size / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', b.key === 'S' ? 'auto' : 'central')
          .attr('fill', b.colour)
          .attr('font-size', b.key === 'S' ? 15 : 30)
          .attr('font-weight', 700)
          .text(b.label);
        svg.append('text')
          .attr('class', 'widget__mono')
          .attr('x', b.x + size / 2)
          .attr('y', top + size + 20)
          .attr('text-anchor', 'middle')
          .attr('fill', c.dim)
          .attr('font-size', 9.5)
          .text(b.sub);
      });

      svg.append('text')
        .attr('x', 121).attr('y', top + size / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
        .attr('fill', c.dim).attr('font-size', 24).text('=');
      [242, 354].forEach(function (px) {
        svg.append('text')
          .attr('x', px).attr('y', top + size / 2)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
          .attr('fill', c.dim).attr('font-size', 20).text('·');
      });
    }

    register('svd', draw);
  }

  /* ------------------------------------------------------ lifecycle */

  function boot() {
    initMatrix();        /* pure DOM + CSS, so it needs no redraws */
    initDotProduct();
    initClassifier();
    initPCA();
    initSVD();
    if (!figures.length) return;

    drawAll();

    /* redraw on a real width change only: mobile browsers fire resize
       when the URL bar slides away, and that must not reshuffle
       anything the reader has dragged into place */
    var lastWidth = window.innerWidth;
    var timer = null;
    window.addEventListener('resize', function () {
      if (window.innerWidth === lastWidth) return;
      lastWidth = window.innerWidth;
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () { timer = null; drawAll(); }, 160);
    });

    /* repaint with the other palette when the OS theme flips */
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) mq.addEventListener('change', drawAll);
      else if (mq.addListener) mq.addListener(drawAll);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
