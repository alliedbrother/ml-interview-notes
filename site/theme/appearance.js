(function () {
  try {
    var prefs = JSON.parse(localStorage.getItem('ml:preferences') || '{}');
    var theme = ['light', 'dark'].includes(prefs.theme) ? prefs.theme : 'system';
    document.documentElement.dataset.theme = theme === 'system'
      ? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme;
    document.documentElement.style.setProperty('--reading-size', Math.max(16, Math.min(22, Number(prefs.size) || 18)) + 'px');
    if (prefs.focus) document.documentElement.dataset.focus = 'true';
  } catch (_) {
    document.documentElement.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
})();
