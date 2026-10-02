/* site/shared/svg-viewport.js
 * Pan, zoom and fullscreen for any <svg> with a viewBox. Drag to pan; pinch, or Ctrl/Cmd +
 * scroll, to zoom around the pointer. Plain scrolling is left alone so the page still scrolls.
 * Used by the model figure and the tensor walk.
 */
(function (root) {
  "use strict";

  function attach(svg, opts = {}) {
    let vb = svg.getAttribute("viewBox").split(/[\s,]+/).map(Number);
    const full = (opts.full || vb).slice();
    const set = v => { vb = v.slice(); svg.setAttribute("viewBox", v.map(x => +x.toFixed(2)).join(" ")); };
    const metrics = () => { const r = svg.getBoundingClientRect(); return { r, s: Math.max(vb[2] / r.width, vb[3] / r.height) }; };
    function toSvg(cx, cy) {
      const { r, s } = metrics();
      const ox = (r.width - vb[2] / s) / 2, oy = (r.height - vb[3] / s) / 2;   // preserveAspectRatio: meet
      return [vb[0] + (cx - r.left - ox) * s, vb[1] + (cy - r.top - oy) * s];
    }
    function zoomAt(f, cx, cy) {
      const [x, y] = toSvg(cx, cy);
      const w = Math.min(full[2] * 3, Math.max(full[2] / 14, vb[2] / f)), k = w / vb[2];
      set([x - (x - vb[0]) * k, y - (y - vb[1]) * k, vb[2] * k, vb[3] * k]);
    }
    const centre = () => { const r = svg.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; };
    const manual = () => opts.onManual && opts.onManual();

    const pts = new Map();
    let drag = null, moved = false, pinch = null;
    svg.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 1) { drag = { x: e.clientX, y: e.clientY, vb: vb.slice(), s: metrics().s }; moved = false; }
      if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), w: vb[2] }; }
    });
    svg.addEventListener("pointermove", e => {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pinch && pts.size === 2) {
        const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        zoomAt(vb[2] / (pinch.w * pinch.d / d), (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        moved = true; manual(); return;
      }
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!moved && Math.hypot(dx, dy) < 4) return;
      if (!moved) { moved = true; svg.setPointerCapture(e.pointerId); svg.style.cursor = "grabbing"; manual(); }
      set([drag.vb[0] - dx * drag.s, drag.vb[1] - dy * drag.s, drag.vb[2], drag.vb[3]]);
    });
    const end = e => { pts.delete(e.pointerId); if (pts.size < 2) pinch = null; if (!pts.size) { drag = null; svg.style.cursor = "grab"; } };
    svg.addEventListener("pointerup", end);
    svg.addEventListener("pointercancel", end);
    svg.addEventListener("click", e => { if (moved) { e.stopPropagation(); e.preventDefault(); moved = false; } }, true);
    svg.addEventListener("wheel", e => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      zoomAt(Math.exp(-e.deltaY * 0.0075), e.clientX, e.clientY);
      manual();
    }, { passive: false });
    svg.style.touchAction = "none";
    svg.style.cursor = "grab";

    return {
      get: () => vb.slice(),
      set,
      full: () => full.slice(),
      fit: () => set(full),
      zoomBy: f => { zoomAt(f, ...centre()); manual(); },
    };
  }

  /* returns a toggle for `el`; onChange(isFullscreen) fires on enter and exit */
  function fullscreen(el, onChange) {
    document.addEventListener("fullscreenchange", () => onChange && onChange(document.fullscreenElement === el));
    return () => {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      else if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
    };
  }

  const API = { attach, fullscreen };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.SvgViewport = API;
})(typeof window !== "undefined" ? window : globalThis);
