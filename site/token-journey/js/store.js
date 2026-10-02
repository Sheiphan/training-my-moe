/* site/token-journey/js/store.js
 *
 * Owns which architecture variant is currently active and exposes read
 * accessors into that variant's trace. Every other module reads "the
 * current variant's data" at CALL time (not once at page load), so the
 * whole page re-renders when the user picks a different architecture.
 *
 * Variants are loaded ON DEMAND. `data/manifest.js` (a few KB) ships with
 * the page and lists what exists; each variant's real trace lives in its
 * own `data/trace_<key>.js` and is only fetched when first selected. That
 * matters here: the seven traces total ~13 MB, and the largest single one
 * (FINEGRAINED, 32 experts) is 3.6 MB -- loading all of them up front, as
 * the previous single-file export did, meant parsing every architecture's
 * weights to look at one.
 *
 * Loading uses an injected <script> tag rather than fetch() on purpose:
 * these pages are meant to open straight off the filesystem, and browsers
 * block fetch/XHR across file:// origins while still permitting a script
 * load. Same reason the data ships as `window.TRACE_VARIANTS[key] = {...}`
 * instead of as .json.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});
  const MANIFEST = window.TRACE_MANIFEST;
  window.TRACE_VARIANTS = window.TRACE_VARIANTS || {};

  let currentKey = MANIFEST.default;
  const pending = new Map();

  function loaded(key) {
    return Boolean(window.TRACE_VARIANTS[key]);
  }

  /** Resolves once `window.TRACE_VARIANTS[key]` exists. */
  function load(key) {
    const entry = MANIFEST.variants[key];
    if (!entry) return Promise.reject(new Error(`unknown trace variant: ${key}`));
    if (loaded(key)) return Promise.resolve(window.TRACE_VARIANTS[key]);
    if (pending.has(key)) return pending.get(key);

    const promise = new Promise((resolve, reject) => {
      const el = document.createElement("script");
      // The manifest sits in data/, and so do the trace files it names.
      el.src = "data/" + entry.file;
      el.async = false;
      el.onload = () =>
        loaded(key)
          ? resolve(window.TRACE_VARIANTS[key])
          : reject(new Error(`${entry.file} loaded but defined no variant '${key}'`));
      el.onerror = () => reject(new Error(`could not load data/${entry.file}`));
      document.head.appendChild(el);
    });
    pending.set(key, promise);
    return promise;
  }

  function data() {
    const d = window.TRACE_VARIANTS[currentKey];
    if (!d) throw new Error(`trace variant '${currentKey}' is not loaded yet`);
    return d;
  }

  TJ.store = {
    manifest: MANIFEST,
    variantKeys: Object.keys(MANIFEST.variants),
    meta(key) {
      return MANIFEST.variants[key || currentKey];
    },
    key() {
      return currentKey;
    },
    isLoaded: loaded,
    load,
    /** Loads if needed, then makes `key` current. */
    async set(key) {
      await load(key);
      currentKey = key;
      return data();
    },
    data,
    cfg() {
      return data().config;
    },
    weights() {
      return data().weights;
    },
    layers() {
      return data().layers;
    },
    layer0() {
      return data().layers[0];
    },
    target0() {
      return data().layers[0].target;
    },
  };
})();
