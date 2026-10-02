/* site/shared/vscode-links/content.js
 * Content script for the unpacked "Open code links in VS Code" extension (load this folder in
 * brave://extensions with Developer mode on, then allow file URLs). On file:// pages, clicking a link
 * to a code file opens it in VS Code at the linked line: path/to/file.py#L37 -> vscode://file/abs/path/to/file.py:37.
 */
(function () {
  "use strict";

  // Change to "vscode-insiders" if you use VS Code Insiders, or "cursor" for Cursor.
  const SCHEME = "vscode";

  // Files that should keep opening in the browser.
  const BROWSER_EXT = /\.(md|markdown|html?|svg|png|jpe?g|gif|webp|pdf|mp4|webm)$/i;

  /** file:///abs/path/file.py#L37 -> vscode://file/abs/path/file.py:37, or null if not a local code file. */
  function toEditorUrl(href, base) {
    let url;
    try { url = new URL(href, base); } catch (e) { return null; }
    if (url.protocol !== "file:" || BROWSER_EXT.test(url.pathname) || url.pathname.endsWith("/")) return null;
    const line = /^#L(\d+)/.exec(url.hash);
    return `${SCHEME}://file${url.pathname}${line ? ":" + line[1] : ""}`;
  }

  if (typeof module !== "undefined" && module.exports) { module.exports = { toEditorUrl }; return; }

  // Proof of life: a console line and a small badge for a few seconds, so you can tell the script runs on this page.
  console.info("[open-in-vscode] active on", location.href);
  window.addEventListener("DOMContentLoaded", () => {
    const b = document.createElement("div");
    b.textContent = "VS Code links: on";
    b.style.cssText = "position:fixed;right:12px;bottom:12px;z-index:2147483647;background:#1E2430;color:#fff;" +
      "font:600 12px system-ui,sans-serif;padding:6px 10px;border-radius:6px;opacity:.92;pointer-events:none";
    document.body.appendChild(b);
    setTimeout(() => b.remove(), 4000);
  });

  // Capture phase on window, so links added after load (the tensor walk builds its links live) are handled too.
  window.addEventListener("click", e => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a) return;
    const target = toEditorUrl(a.getAttribute("href"), document.baseURI);
    console.info("[open-in-vscode] click", a.getAttribute("href"), "->", target || "(left to the browser)");
    if (!target) return;
    e.preventDefault();
    e.stopPropagation();
    window.location.href = target;
  }, true);

  // Hovering shows where the link will go.
  window.addEventListener("mouseover", e => {
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a || a.dataset.vscodeTitled) return;
    const target = toEditorUrl(a.getAttribute("href"), document.baseURI);
    if (target) { a.title = (a.title ? a.title + " · " : "") + "opens in VS Code"; a.dataset.vscodeTitled = "1"; }
  }, true);
})();
