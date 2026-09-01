/* token_journey/js/store.js
 * Owns which (attention, positional) variant is currently active and
 * exposes read accessors into that variant's slice of window.TRACE_DATA.
 * Introduced so every other module can read "the current variant's data"
 * at CALL time (not just once at page load), letting the whole page
 * re-render when the user toggles attention/positional.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});
  const RAW = window.TRACE_DATA;
  let currentKey = RAW.default;

  function data() {
    return RAW.variants[currentKey];
  }

  TJ.store = {
    variantKeys: Object.keys(RAW.variants),
    key() {
      return currentKey;
    },
    set(key) {
      if (!RAW.variants[key]) throw new Error(`unknown trace variant: ${key}`);
      currentKey = key;
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
