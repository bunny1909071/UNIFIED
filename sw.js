// UNIFIED service worker: makes the app installable and the shell load instantly.
// Bump VERSION whenever you want every device to drop its old cached copy.
const VERSION = 'v6';
const CACHE = 'jash-shell-' + VERSION;
const SHELL = ['/', '/static/icons/icon-192.png', '/static/icons/icon-512.png'];
const CDN = ['fonts.googleapis.com', 'fonts.gstatic.com', 'cdnjs.cloudflare.com'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Never touch API calls or health checks: medical results must always be live.
  if (url.origin === location.origin && (url.pathname.startsWith('/api/') || url.pathname.startsWith('/health') || url.pathname.startsWith('/admin') || url.pathname.startsWith('/r/'))) return;

  // Page loads: network first (so updates arrive), cached shell when offline.
  // Only the home page is cached; shared-report pages (/r/...) and everything else go straight to the network.
  // Only the app shell is cached. Other pages (for example shared report links /r/...) always go to the network.
  if (req.mode === 'navigate' && url.pathname !== '/') return;
  if (req.mode === 'navigate') {
    if (url.origin !== location.origin || url.pathname !== '/') return;
    e.respondWith(
      fetch(req)
        .then(res => {
          if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put('/', copy)); }
          return res;
        })
        .catch(() => caches.match('/'))
    );
    return;
  }

  // Fonts + Chart.js + icons: serve from cache instantly, refresh in the background.
  if (url.origin === location.origin || CDN.includes(url.hostname)) {
    e.respondWith(
      caches.match(req).then(hit => {
        const net = fetch(req).then(res => {
          if (res && (res.ok || res.type === 'opaque')) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
          return res;
        }).catch(() => hit);
        return hit || net;
      })
    );
  }
});
