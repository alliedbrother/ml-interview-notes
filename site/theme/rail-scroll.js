/* Keep the left rail's scroll position across page loads.

   Every page here is a full document load, so the rail re-renders at the top
   each time. On a course with ~90 entries that means clicking a lab near the
   bottom drops you back at "Start here" — you lose your place on every click.

   The rail's scroll offset is remembered per course in sessionStorage (per tab,
   cleared when the tab closes). With nothing stored yet — a fresh tab, or a
   deep link from outside — the current item is centred instead, which is the
   right first impression rather than an arbitrary offset.

   sessionStorage can throw outright in some privacy modes, so every access is
   guarded and failure just means the feature is absent. */
(function () {
  function init() {
    var rail = document.querySelector('.nav');
    if (!rail) return;

    // key per course, so the two courses don't fight over one offset
    var parts = location.pathname.split('/').filter(Boolean);
    var key = 'railScroll:' + (parts.slice(0, 2).join('/') || 'root');

    function read() {
      try { return sessionStorage.getItem(key); } catch (e) { return null; }
    }
    function write() {
      try { sessionStorage.setItem(key, String(rail.scrollTop)); } catch (e) {}
    }

    var saved = read();
    if (saved !== null && saved !== '') {
      rail.scrollTop = parseInt(saved, 10) || 0;
    } else {
      // Centre the current entry WITHOUT scrollIntoView, which would also
      // scroll the window and yank the reader away from the top of the article.
      var cur = rail.querySelector('.nav__i.is-current');
      if (cur) {
        var target = cur.offsetTop - rail.clientHeight / 2 + cur.offsetHeight / 2;
        rail.scrollTop = target > 0 ? target : 0;
      }
    }

    var pending = null;
    rail.addEventListener('scroll', function () {
      if (pending) return;
      pending = setTimeout(function () { pending = null; write(); }, 120);
    }, { passive: true });

    // capture the offset at the moment of navigation, before the page unloads
    rail.addEventListener('click', write);
    window.addEventListener('pagehide', write);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
