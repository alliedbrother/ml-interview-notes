/* scroll-spy for the right-hand TOC + mobile nav toggle */
(function () {
  const toc = document.getElementById('toc');
  if (toc) {
    const links = [...toc.querySelectorAll('a')];
    const sections = links
      .map(l => ({ link: l, el: document.getElementById(l.getAttribute('href').slice(1)) }))
      .filter(s => s.el);
    if (sections.length) {
      const obs = new IntersectionObserver(entries => {
        entries.forEach(en => {
          if (!en.isIntersecting) return;
          links.forEach(l => l.classList.remove('active'));
          const m = sections.find(s => s.el === en.target);
          if (m) m.link.classList.add('active');
        });
      }, { rootMargin: '-8% 0px -78% 0px' });
      sections.forEach(s => obs.observe(s.el));
    }
  }
  const nav = document.getElementById('nav');
  const btn = document.getElementById('navtoggle');
  if (nav && btn) {
    btn.addEventListener('click', () => {
      const open = nav.classList.toggle('is-open');
      btn.textContent = open ? 'Close' : 'Modules';
      btn.setAttribute('aria-expanded', String(open));
    });
  }
})();
