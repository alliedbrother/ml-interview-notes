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
      grid.setAttribute('role', 'table');
      grid.setAttribute('aria-label', current.length + ' by ' + current[0].length + ' matrix');
      current.forEach(function (rowData, i) {
        var row = document.createElement('div');
        row.className = 'widget__mrow';
        row.setAttribute('role', 'row');
        rowData.forEach(function (value, j) {
          var cell = document.createElement('div');
          cell.className = 'widget__cell';
          cell.setAttribute('role', 'cell');
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
        var scalar = scalarInput ? scalarInput.valueAsNumber : NaN;
        var valid = Number.isFinite(scalar) && current.every(function (row) {
          return row.every(function (v) { return Number.isFinite(v * scalar); });
        });
        if (scalarInput) {
          scalarInput.setCustomValidity(valid ? '' : 'Enter a finite scalar that keeps the matrix entries finite.');
          if (!valid) scalarInput.reportValidity();
        }
        if (!valid) return;
        current = current.map(function (row) {
          return row.map(function (v) { return v * scalar; });
        });
        render();
      });
    }

    if (scalarInput) scalarInput.addEventListener('input', function () { scalarInput.setCustomValidity(''); });

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
      if (Object.keys(inputs).some(function (key) { return !inputs[key].checkValidity(); })) return;
      var c = palette();
      var width = widthOf(host, 280);
      var height = 320;
      var margin = { top: 18, right: 18, bottom: 18, left: 18 };
      var iw = width - margin.left - margin.right;
      var ih = height - margin.top - margin.bottom;
      MAX = Math.max(6, Math.ceil(d3.max(Object.keys(inputs), function (key) {
        return Math.abs(+inputs[key].value || 0);
      }) + 1));

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', '0 0 ' + width + ' ' + height)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .attr('role', 'img').attr('aria-label', 'Vectors a and b with equal coordinate scales; values and angle are listed alongside.');

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

      var unit = Math.min(iw, ih) / (2 * MAX);
      var x = d3.scaleLinear().domain([-MAX, MAX]).range([iw / 2 - MAX * unit, iw / 2 + MAX * unit]);
      var y = d3.scaleLinear().domain([-MAX, MAX]).range([ih / 2 + MAX * unit, ih / 2 - MAX * unit]);
      var cx = x(0), cy = y(0);

      var grid = g.append('g');
      x.ticks(12).forEach(function (v) {
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
      if (angleOut) angleOut.textContent = magA === 0 || magB === 0 ? 'Undefined' : angleDeg.toFixed(1) + '°';

      var cosTheta = dot / (magA * magB);
      if (isNaN(cosTheta)) cosTheta = 0;

      if (noteOut) {
        /* Cover the whole range: a blank readout for most of the input space
           teaches nothing, and the default vectors land in the middle. */
        var explanation;
        if (magA === 0 || magB === 0) explanation = 'The zero vector has no direction: its dot product is 0, but its angle and cosine similarity are undefined.';
        else if (Math.abs(cosTheta) < 1e-10) explanation = 'The vectors are perpendicular: their dot product and cosine similarity are both 0. Orthogonality does not imply statistical independence.';
        else {
          var angleKind = cosTheta > 1 - 1e-10 ? 'zero' : cosTheta < -1 + 1e-10 ? 'straight' : dot > 0 ? 'acute' : 'obtuse';
          explanation = 'Cosine similarity = ' + cosTheta.toFixed(3) + '. The angle is ' + angleKind + '; the dot product also depends on both vector lengths.';
        }
        noteOut.textContent = explanation;
      }

      /* the wedge between a and b, swept the short way round */
      view.angleG.selectAll('*').remove();
      if (magA > 0 && magB > 0) {
        var angleA = Math.atan2(-a.y, a.x) + Math.PI / 2;
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
      inputs[k].min = '-1000000';
      inputs[k].max = '1000000';
      inputs[k].required = true;
      inputs[k].addEventListener('input', build);
    });

    register('dot-product', build);
  }

  /* ---------------------------------------------- 3. the classifier */

  function initClassifier() {
    var host = byId('interactive-classifier');
    if (!host || typeof d3 === 'undefined') return;

    /* Fixed-seed points make comparisons repeatable across visits. */
    var cloud = null;
    function points() {
      if (cloud) return cloud;
      var normal = d3.randomNormal.source(d3.randomLcg(0.42))(0, 0.9);
      var negative = d3.range(30).map(function () {
        return { x: 2.5 + normal(), y: 7.5 + normal(), cls: 0 };
      });
      var positive = d3.range(30).map(function () {
        return { x: 7.5 + normal(), y: 2.5 + normal(), cls: 1 };
      });
      cloud = negative.concat(positive).filter(function (d) {
        return d.x >= 0 && d.x <= 10 && d.y >= 0 && d.y <= 10;
      });
      return cloud;
    }

    var angle = -Math.PI / 4;   /* survives redraws */
    var offset = 0;

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
        .text('FEATURE x1');

      svg.append('text')
        .attr('class', 'widget__mono')
        .attr('text-anchor', 'middle')
        .attr('transform', 'translate(13,' + (margin.top + ih / 2) + ') rotate(-90)')
        .attr('fill', c.dim)
        .attr('font-size', 9.5)
        .attr('font-weight', 600)
        .text('FEATURE x2');

      var side = Math.min(iw, ih);
      var x = d3.scaleLinear().domain([0, 10]).range([(iw - side) / 2, (iw + side) / 2]);
      var y = d3.scaleLinear().domain([0, 10]).range([(ih + side) / 2, (ih - side) / 2]);

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

      /* Clip the separator so it cannot cross the axis labels. */
      var clipId = 'la-clf-clip';
      svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('x', 0).attr('y', 0).attr('width', iw).attr('height', ih);
      var line = g.append('g').attr('clip-path', 'url(#' + clipId + ')').append('line')
        .attr('stroke', c.text).attr('stroke-width', 2.5).attr('stroke-linecap', 'round');

      g.selectAll('path.la-sample')
        .data(points())
        .enter().append('path')
        .attr('class', 'la-sample')
        .attr('transform', function (d) { return 'translate(' + x(d.x) + ',' + y(d.y) + ')'; })
        .attr('d', d3.symbol().type(function (d) { return d.cls ? d3.symbolTriangle : d3.symbolCircle; }).size(65))
        .attr('fill', function (d) { return d.cls ? c.rose : c.blue; })
        .attr('opacity', 0.8)
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
          showTip('Class ' + (d.cls ? '+1' : '-1') + '<br>x1: ' + d.x.toFixed(2) + '<br>x2: ' + d.y.toFixed(2));
        })
        .on('mousemove', moveTip)
        .on('mouseout', function () {
          hideTip();
        });

      var normalLine = g.append('line').attr('class', 'la-normal')
        .attr('stroke', c.accent).attr('stroke-width', 2)
        .attr('x1', x(5)).attr('y1', y(5));
      var normalLabel = g.append('text').attr('fill', c.accent)
        .attr('font-size', 12).attr('font-weight', 600).text('w');
      var handle = g.append('circle')
        .attr('r', 9)
        .attr('fill', c.accent)
        .attr('stroke', c.surface)
        .attr('stroke-width', 2)
        .style('cursor', 'move')
        .on('mouseover', function () { d3.select(this).transition().duration(180).attr('r', 11); })
        .on('mouseout', function () { d3.select(this).transition().duration(180).attr('r', 9); });

      var controls = root.append('div').attr('class', 'widget__bar');
      controls.append('label').attr('class', 'widget__lbl').attr('for', 'classifier-angle').text('Normal angle');
      var angleInput = controls.append('input').attr('id', 'classifier-angle').attr('class', 'widget__num')
        .attr('type', 'number').attr('step', 5).attr('min', -180).attr('max', 180)
        .on('input', function () { if (Number.isFinite(this.valueAsNumber)) place(this.valueAsNumber * Math.PI / 180); });
      controls.append('label').attr('class', 'widget__lbl').attr('for', 'classifier-offset').text('Centered intercept');
      controls.append('input').attr('id', 'classifier-offset').attr('class', 'widget__num')
        .attr('type', 'number').attr('step', 0.25).attr('min', -8).attr('max', 8).property('value', offset)
        .on('input', function () {
          if (!Number.isFinite(this.valueAsNumber)) return;
          offset = clamp(this.valueAsNumber, -8, 8);
          this.value = offset;
          place(angle);
        });
      var readout = root.append('p').attr('class', 'widget__note').attr('id', 'classifier-readout').attr('aria-live', 'polite');
      svg.attr('role', 'img').attr('aria-label', 'Two labeled classes and a linear decision boundary. Circles are class -1; triangles are class +1.');

      function place(next) {
        angle = Math.atan2(Math.sin(next), Math.cos(next));
        var wx = Math.cos(angle), wy = Math.sin(angle);
        var reach = 30;
        var midX = 5 - offset * wx, midY = 5 - offset * wy;
        line
          .attr('x1', x(midX - reach * wy)).attr('y1', y(midY + reach * wx))
          .attr('x2', x(midX + reach * wy)).attr('y2', y(midY - reach * wx));
        handle
          .attr('cx', x(5 + wx)).attr('cy', y(5 + wy));
        normalLine.attr('x2', x(5 + wx)).attr('y2', y(5 + wy));
        normalLabel.attr('x', x(5 + wx) + 12).attr('y', y(5 + wy) - 10);
        var correct = points().filter(function (d) {
          return ((wx * (d.x - 5) + wy * (d.y - 5) + offset >= 0) ? 1 : 0) === d.cls;
        }).length;
        angleInput.property('value', (angle * 180 / Math.PI).toFixed(0));
        readout.text('w = [' + wx.toFixed(2) + ', ' + wy.toFixed(2) + '], b = ' + (offset - 5 * wx - 5 * wy).toFixed(2) + '. Correct: ' + correct + '/' + points().length + '.');
      }

      handle.call(d3.drag().on('drag', function (event) {
        place(Math.atan2(y.invert(event.y) - 5, x.invert(event.x) - 5));
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
      var normal = d3.randomNormal.source(d3.randomLcg(0.71))(0, 1);
      cloud = d3.range(150).map(function () {
        var u = normal() * SPREAD[0];
        var v = normal() * SPREAD[1];
        return {
          x: u * Math.cos(TILT) - v * Math.sin(TILT),
          y: u * Math.sin(TILT) + v * Math.cos(TILT)
        };
      });
      var meanX = d3.mean(cloud, function (d) { return d.x; });
      var meanY = d3.mean(cloud, function (d) { return d.y; });
      cloud.forEach(function (d) { d.x -= meanX; d.y -= meanY; });
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
      var xx = d3.sum(data, function (d) { return d.x * d.x; }) / (data.length - 1);
      var xy = d3.sum(data, function (d) { return d.x * d.y; }) / (data.length - 1);
      var yy = d3.sum(data, function (d) { return d.y * d.y; }) / (data.length - 1);
      // Closed-form eigensystem for this symmetric 2 by 2 sample covariance.
      var gap = Math.hypot(xx - yy, 2 * xy);
      var lambda1 = (xx + yy + gap) / 2;
      var lambda2 = Math.max(0, (xx + yy - gap) / 2);
      var theta = 0.5 * Math.atan2(2 * xy, xx - yy);
      var span = Math.max(2 * Math.sqrt(lambda1), d3.max(data, function (d) { return Math.max(Math.abs(d.x), Math.abs(d.y)); })) * 1.2;
      var unit = Math.min(iw, ih) / (2 * span);
      var x = d3.scaleLinear().domain([-span, span]).range([iw / 2 - span * unit, iw / 2 + span * unit]);
      var y = d3.scaleLinear().domain([-span, span]).range([ih / 2 + span * unit, ih / 2 - span * unit]);
      var cx = x(0), cy = y(0);
      svg.attr('role', 'img').attr('aria-label', 'Centered data with orthogonal sample principal-component directions. Arrow lengths are two standard deviations.');

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

      /* Equal axis scales preserve orthogonality; each ray is 2 sqrt(lambda). */
      function component(theta, scale, colour, marker, label) {
        var reach = 2 * scale;
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

      component(theta, Math.sqrt(lambda1), c.rose, 'pc1', 'PC1');
      component(theta + Math.PI / 2, Math.sqrt(lambda2), c.blue, 'pc2', 'PC2');
      root.append('p').attr('id', 'pca-readout').attr('class', 'widget__note')
        .attr('data-lambda1', lambda1).attr('data-lambda2', lambda2)
        .text('Sample eigenvalues: ' + lambda1.toFixed(3) + ' and ' + lambda2.toFixed(3) + '. PC1 retains ' + (100 * lambda1 / (lambda1 + lambda2)).toFixed(1) + '% of sample variance. Arrow lengths: 2 sqrt(lambda), not variance.');
    }

    register('pca', draw);
  }

  /* ----------------------------------------------------- 5. the SVD */

  function initSVD() {
    var host = byId('svd-visualization');
    if (!host || typeof d3 === 'undefined') return;

    function draw() {
      var c = palette();
      var W = 476, H = 198;
      var size = 96, top = 26;

      var root = d3.select(host);
      root.selectAll('*').remove();

      var svg = root.append('svg')
        .attr('width', W)
        .attr('height', H)
        .attr('viewBox', '0 0 ' + W + ' ' + H)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .attr('role', 'img').attr('aria-label', 'Compact singular value decomposition: A, m by n, equals U, m by r, times Sigma, r by r, times V transpose, r by n. Here r is rank A.');

      var blocks = [
        { key: 'A', label: 'A', colour: c.accent, x: 8, w: 84, h: 96, sub: 'm × n' },
        { key: 'U', label: 'U', colour: c.blue, x: 154, w: 64, h: 96, sub: 'm × r' },
        { key: 'S', label: 'Σ', colour: c.sage, x: 266, w: 64, h: 64, sub: 'r × r' },
        { key: 'V', label: 'Vᵀ', colour: c.rose, x: 362, w: 96, h: 64, sub: 'r × n' }
      ];

      blocks.forEach(function (b) {
        svg.append('rect')
          .attr('x', b.x).attr('y', top + (size - b.h) / 2)
          .attr('width', b.w).attr('height', b.h).attr('rx', 4)
          .attr('fill', b.colour).attr('fill-opacity', 0.12)
          .attr('stroke', b.colour).attr('stroke-width', 1.4);
      });

      /* Symbolic ordering, not measured singular values of a supplied matrix. */
      var cell = blocks[2].w / 3;
      ['σ1', '⋱', 'σr'].forEach(function (label, i) {
        var sigma = blocks[2];
        svg.append('rect')
          .attr('x', sigma.x + i * cell).attr('y', top + (size - sigma.h) / 2 + i * cell)
          .attr('width', cell).attr('height', cell)
          .attr('fill', c.sage).attr('fill-opacity', 0.35);
        svg.append('text')
          .attr('class', 'widget__mono')
          .attr('x', sigma.x + i * cell + cell / 2)
          .attr('y', top + (size - sigma.h) / 2 + i * cell + cell / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'central')
          .attr('fill', c.text)
          .attr('font-size', 9)
          .text(label);
      });

      blocks.forEach(function (b) {
        svg.append('text')
          .attr('x', b.x + b.w / 2)
          .attr('y', b.key === 'S' ? top - 9 : top + size / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', b.key === 'S' ? 'auto' : 'central')
          .attr('fill', b.colour)
          .attr('font-size', b.key === 'S' ? 15 : 30)
          .attr('font-weight', 700)
          .text(b.label);
        svg.append('text')
          .attr('class', 'widget__mono')
          .attr('x', b.x + b.w / 2)
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
      svg.append('text').attr('x', W / 2).attr('y', 180).attr('text-anchor', 'middle')
        .attr('fill', c.dim).attr('font-size', 10).text('Compact SVD: r = rank(A), σ1 ≥ ... ≥ σr > 0. Schematic, not numerical data.');
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
    window.addEventListener('reader:theme', drawAll);
    window.addEventListener('reader:resize', drawAll);

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
