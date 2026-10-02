/* site/token-journey/js/camera.js
 * D3-zoom-based "fit this real bounding box into the fixed visible window"
 * camera, plus dim/spotlight toggling. The SVG's own viewBox is the fixed
 * camera window (see diagram.js: `0 0 1000 700`); an inner <g id="viewport">
 * gets a translate+scale transform animated between steps.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  function unionBBox(ids) {
    let box = null;
    for (const id of ids) {
      const b = TJ.diagram.focusTargets[id];
      if (!b) continue;
      if (!box) {
        box = { x: b.x, y: b.y, width: b.width, height: b.height };
      } else {
        const x2 = Math.max(box.x + box.width, b.x + b.width);
        const y2 = Math.max(box.y + box.height, b.y + b.height);
        box.x = Math.min(box.x, b.x);
        box.y = Math.min(box.y, b.y);
        box.width = x2 - box.x;
        box.height = y2 - box.y;
      }
    }
    return box;
  }

  TJ.camera = {
    init(svgSelector) {
      this.svgEl = document.querySelector(svgSelector);
      this.viewport = d3.select(this.svgEl).select("#viewport");
      const vb = this.svgEl.viewBox.baseVal;
      this.size = { width: vb.width, height: vb.height };
    },

    /** focus: a focus-target id, or an array of ids to frame together. */
    goTo(focus, opts = {}) {
      const ids = Array.isArray(focus) ? focus : [focus];
      const bbox = unionBBox(ids);
      if (!bbox) return;

      const pad = opts.padding ?? 56;
      const maxScale = opts.maxScale ?? 2.6;
      const scale = Math.min((this.size.width - pad * 2) / bbox.width, (this.size.height - pad * 2) / bbox.height, maxScale);
      const tx = this.size.width / 2 - scale * (bbox.x + bbox.width / 2);
      const ty = this.size.height / 2 - scale * (bbox.y + bbox.height / 2);

      this.viewport
        .transition()
        .duration(opts.duration ?? 750)
        .ease(d3.easeCubicInOut)
        .attr("transform", `translate(${tx},${ty}) scale(${scale})`);

      this.spotlight(ids);
    },

    /** Dims every focusable node except the ones in `ids`. */
    spotlight(ids) {
      const idSet = new Set(ids);
      document.querySelectorAll("#diagram [data-focus]").forEach((el) => {
        const match = idSet.has(el.getAttribute("data-focus"));
        el.classList.toggle("dimmed", !match);
        el.classList.toggle("focused", match);
      });
    },

    clearSpotlight() {
      document.querySelectorAll("#diagram [data-focus]").forEach((el) => {
        el.classList.remove("dimmed", "focused");
      });
    },
  };
})();
