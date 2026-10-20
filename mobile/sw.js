// Service worker for the Paimon mobile PWA.
//
// Its ONLY job is installability + an offline app-shell. It deliberately does
// NOT cache any /api/* response (those must always hit the live PC backend) —
// it caches just the static shell (html/js/css/icons) with a network-first
// strategy so a fresh deploy is picked up immediately when online, and the app
// still opens (showing "offline") when the phone can't reach the PC.

const SHELL_CACHE = "paimon-shell-v1";
const SHELL = [
  "/m/",
  "/m/index.html",
  "/m/app.js",
  "/m/style.css",
  "/m/manifest.webmanifest",
  "/m/icons/icon-192.png",
  "/m/icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL).catch(() => {})));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== SHELL_CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // Never intercept API traffic, voice, deliveries, or anything outside /m/.
  if (e.request.method !== "GET" || !url.pathname.startsWith("/m/")) return;
  if (url.pathname.startsWith("/m/api")) return; // (none today, but be safe)

  // Network-first for the shell: fresh when online, cached when offline.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(SHELL_CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request).then((m) => m || caches.match("/m/index.html")))
  );
});
