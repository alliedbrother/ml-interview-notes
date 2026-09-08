(function () {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const read = (key, fallback) => {
    try { return JSON.parse(localStorage.getItem('ml:' + key)) ?? fallback; }
    catch (_) { return fallback; }
  };
  const write = (key, value) => {
    try { localStorage.setItem('ml:' + key, JSON.stringify(value)); } catch (_) { /* Private browsing remains usable. */ }
  };
  const icons = () => window.lucide?.createIcons({ attrs: { 'aria-hidden': 'true' } });
  const announce = text => { $('#reader-announcement').textContent = text; };
  const preferences = { theme: 'system', size: 18, focus: false, ...read('preferences', {}) };
  const osTheme = matchMedia('(prefers-color-scheme: dark)');

  function applyPreferences() {
    if (!['system', 'light', 'dark'].includes(preferences.theme)) preferences.theme = 'system';
    preferences.size = Math.min(22, Math.max(16, Number(preferences.size) || 18));
    const theme = preferences.theme === 'system' ? (osTheme.matches ? 'dark' : 'light') : preferences.theme;
    const oldTheme = document.documentElement.dataset.theme;
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.focus = String(!!preferences.focus);
    document.documentElement.style.setProperty('--reading-size', preferences.size + 'px');
    $$('[data-theme-choice]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.themeChoice === preferences.theme)));
    $$('button[data-focus]').forEach(button => {
      button.setAttribute('aria-pressed', String(!!preferences.focus));
      button.title = button.ariaLabel = preferences.focus ? 'Exit focus mode' : 'Focus mode';
    });
    $('#focus-setting').checked = !!preferences.focus;
    $('#reading-size').value = preferences.size;
    $('#size-output').textContent = preferences.size + ' px';
    if (oldTheme !== theme) dispatchEvent(new CustomEvent('reader:theme'));
    dispatchEvent(new Event('reader:resize'));
    write('preferences', preferences);
  }
  applyPreferences();
  osTheme.addEventListener('change', applyPreferences);
  $$('[data-theme-choice]').forEach(button => button.addEventListener('click', () => {
    preferences.theme = button.dataset.themeChoice;
    applyPreferences();
  }));
  $('#reading-size').addEventListener('input', event => { preferences.size = +event.target.value; applyPreferences(); });
  $('#focus-setting').addEventListener('change', event => { preferences.focus = event.target.checked; applyPreferences(); });
  $$('button[data-focus]').forEach(button => button.addEventListener('click', () => { preferences.focus = !preferences.focus; applyPreferences(); }));
  $('#reset-preferences').addEventListener('click', () => { Object.assign(preferences, { theme: 'system', size: 18, focus: false }); applyPreferences(); });
  $$('[data-print]').forEach(button => button.addEventListener('click', () => window.print()));

  let returnFocus = null;
  function showDialog(dialog) {
    returnFocus = document.activeElement;
    dialog.showModal();
    document.body.style.overflow = 'hidden';
  }
  $$('dialog').forEach(dialog => {
    $('[data-close-dialog]', dialog).addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target !== dialog) return;
      const r = dialog.getBoundingClientRect();
      if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
    });
    dialog.addEventListener('close', () => { document.body.style.overflow = ''; returnFocus?.focus(); });
  });
  $$('[data-open-settings]').forEach(button => button.addEventListener('click', () => showDialog($('#settings-dialog'))));

  let bookmarks = read('bookmarks', []);
  if (!Array.isArray(bookmarks)) bookmarks = [];
  const canonical = $('link[rel="canonical"]');
  const pageURL = canonical ? new URL(canonical.href).pathname : location.pathname;
  function syncBookmark() {
    const saved = bookmarks.includes(pageURL);
    $$('[data-bookmark]').forEach(button => {
      button.setAttribute('aria-pressed', String(saved));
      button.title = button.ariaLabel = saved ? 'Remove saved page' : 'Save this page';
      button.innerHTML = '<i data-lucide="' + (saved ? 'bookmark-check' : 'bookmark') + '"></i>';
    });
    icons();
  }
  $$('[data-bookmark]').forEach(button => button.addEventListener('click', () => {
    bookmarks = bookmarks.includes(pageURL) ? bookmarks.filter(url => url !== pageURL) : [...bookmarks, pageURL];
    write('bookmarks', bookmarks);
    syncBookmark();
    announce(bookmarks.includes(pageURL) ? 'Page saved' : 'Page removed from saved pages');
  }));
  syncBookmark();

  let indexPromise;
  let filter = 'all';
  let searchVersion = 0;
  function loadIndex() {
    if (!indexPromise) indexPromise = fetch('/assets/search.json').then(response => {
      if (!response.ok) throw new Error('Search unavailable');
      return response.json();
    }).then(rows => rows.map(row => ({ ...row, haystack: (row.title + ' ' + row.category + ' ' + row.text).toLowerCase() }))).catch(error => { indexPromise = null; throw error; });
    return indexPromise;
  }
  function safePath(path) { return typeof path === 'string' && path.startsWith('/') && !path.startsWith('//'); }
  function resultNode(row, query) {
    const link = document.createElement('a');
    link.className = 'search-result';
    link.href = row.url;
    let excerpt = row.description || row.text.slice(0, 170);
    if (query && !excerpt.toLowerCase().includes(query)) {
      const at = row.text.toLowerCase().indexOf(query);
      if (at >= 0) excerpt = (at > 60 ? '... ' : '') + row.text.slice(Math.max(0, at - 60), at + 145) + '...';
    }
    [['category', row.category], ['title', row.title], ['description', excerpt]].forEach(([kind, text]) => {
      const span = document.createElement('span');
      span.className = 'search-result__' + kind;
      span.textContent = text;
      link.append(span);
    });
    return link;
  }
  async function search() {
    const version = ++searchVersion;
    const query = $('#search-input').value.trim().toLowerCase();
    const terms = query.split(/\s+/).filter(Boolean);
    $('#search-status').textContent = 'Loading library...';
    try {
      const index = await loadIndex();
      if (version !== searchVersion) return;
      const results = index.filter(row => safePath(row.url)
        && (filter === 'all' || (filter === 'saved' ? bookmarks.includes(row.url) : row.kind === filter))
        && terms.every(term => row.haystack.includes(term)))
        .map(row => ({ row, score: !query ? (bookmarks.includes(row.url) ? 40 : row.url.split('/').length <= 4 ? 10 : 0)
          : (row.title.toLowerCase().includes(query) ? 100 : 0) + terms.reduce((score, term) => score + (row.title.toLowerCase().includes(term) ? 20 : 0) + (row.description.toLowerCase().includes(term) ? 3 : 0), 0) }))
        .sort((a, b) => b.score - a.score || a.row.title.localeCompare(b.row.title));
      const shown = results.slice(0, query || filter === 'saved' ? 40 : 12);
      $('#search-results').replaceChildren(...shown.map(({ row }) => resultNode(row, terms[0] || '')));
      $('#search-status').textContent = !results.length
        ? (filter === 'saved' && !query ? 'No saved pages yet.' : 'No matches. Try another concept or a broader term.')
        : query ? results.length + ' results' + (results.length > 40 ? ' · Showing the first 40' : '')
          : filter === 'saved' ? results.length + ' saved pages' : 'From the library';
    } catch (_) {
      if (version !== searchVersion) return;
      $('#search-results').replaceChildren();
      $('#search-status').textContent = 'Search could not load. Try again in a moment.';
    }
  }
  function openSearch() { showDialog($('#search-dialog')); $('#search-input').focus(); search(); }
  $$('[data-open-search]').forEach(button => button.addEventListener('click', openSearch));
  let searchTimer;
  $('#search-input').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(search, 130); });
  $$('[data-filter]').forEach(button => button.addEventListener('click', () => {
    filter = button.dataset.filter;
    $$('[data-filter]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    search();
  }));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      const dialog = $('dialog[open]');
      if (dialog) {
        event.preventDefault();
        dialog.close();
        return;
      }
    }
    const editing = event.target.closest('input, textarea, [contenteditable="true"]');
    if ((event.key === '/' && !editing) || ((event.metaKey || event.ctrlKey) && event.key === 'k')) {
      event.preventDefault();
      if (!$('#search-dialog').open && !$('#settings-dialog').open) openSearch();
    }
    if (event.key === 'Escape' && !$('dialog[open]') && preferences.focus) { preferences.focus = false; applyPreferences(); }
    if ($('#search-dialog').open && event.key === 'ArrowDown' && event.target === $('#search-input')) {
      const first = $('.search-result');
      if (first) { event.preventDefault(); first.focus(); }
    }
  });

  $$('main pre').forEach(pre => {
    const code = $('code', pre);
    if (!code || !code.textContent.trim()) return;
    const host = pre.closest('.code-file') || pre.parentElement;
    host.classList.add('copy-host');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'icon-button code-copy';
    button.title = button.ariaLabel = 'Copy code';
    button.innerHTML = '<i data-lucide="copy"></i>';
    host.append(button);
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.innerHTML = '<i data-lucide="check"></i>';
        button.title = button.ariaLabel = 'Copied';
        announce('Code copied');
        icons();
        setTimeout(() => { button.innerHTML = '<i data-lucide="copy"></i>'; button.title = button.ariaLabel = 'Copy code'; icons(); }, 1600);
      } catch (_) { announce('Copy unavailable. Select the code to copy it.'); }
    });
  });
  const diagramIcons = { 'zoom-in': 'plus', 'zoom-out': 'minus', 'zoom-fit': 'maximize', 'zoom-one': 'scan', 'zoom-expand': 'external-link' };
  $$('[data-action]').forEach(button => {
    if (!diagramIcons[button.dataset.action]) return;
    button.innerHTML = '<i data-lucide="' + diagramIcons[button.dataset.action] + '"></i>';
    button.setAttribute('aria-label', button.title);
  });
  icons();

  let history = read('history', []);
  if (!Array.isArray(history)) history = [];
  const resume = $('#continue-reading');
  if (resume) {
    const last = history.find(row => safePath(row.url));
    if (last) {
      const a = document.createElement('a');
      a.href = last.url + (last.ratio > .02 && last.ratio < .98 ? '#resume-reading' : '');
      a.innerHTML = '<i data-lucide="book-open"></i><span><small>Continue reading</small><strong></strong></span><i data-lucide="arrow-right"></i>';
      $('strong', a).textContent = last.title;
      resume.append(a);
      resume.hidden = false;
      icons();
    }
  }
  if (document.body.classList.contains('is-reading')) {
    const main = $('main');
    let ratio = 0;
    let pending = false;
    function progress() {
      const start = main.offsetTop;
      ratio = Math.max(0, Math.min(1, (scrollY - start) / Math.max(1, main.offsetHeight - innerHeight)));
      document.documentElement.style.setProperty('--progress', ratio * 100 + '%');
      pending = false;
    }
    function saveProgress() {
      const current = { url: pageURL, title: $('h1', main).textContent, ratio, time: Date.now() };
      const latest = read('history', []);
      write('history', [current, ...(Array.isArray(latest) ? latest : []).filter(row => row.url !== pageURL)].slice(0, 20));
    }
    addEventListener('scroll', () => { if (!pending) { pending = true; requestAnimationFrame(progress); } }, { passive: true });
    addEventListener('resize', progress);
    addEventListener('pagehide', saveProgress);
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') saveProgress(); });
    new ResizeObserver(progress).observe(main);
    if (location.hash === '#resume-reading') {
      const last = history.find(row => row.url === pageURL);
      const restore = () => {
        if (last) window.scrollTo({ top: main.offsetTop + last.ratio * Math.max(1, main.offsetHeight - innerHeight), behavior: 'instant' });
      };
      if (document.readyState === 'complete') restore(); else addEventListener('load', restore, { once: true });
    }
    progress();
  }
})();
