/* site/token-journey/js/matrix-viz.js
 * Reusable real-number matrix multiplication and elementwise views.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});
  const active = [];
  const CELL_W = 56;

  function fmt(value) {
    return (Math.round(value * 1000) / 1000).toFixed(3);
  }

  function tex(element, latex, displayMode) {
    if (!element) return;
    if (!latex) {
      element.innerHTML = "";
      return;
    }
    katex.render(latex, element, { throwOnError: false, displayMode: !!displayMode });
  }

  function cellDiv(value, className) {
    return `<div class="mm-cell ${className || ""}" style="width:${CELL_W}px">${fmt(value)}</div>`;
  }

  function rowCells(values, take, activeIndex) {
    return values
      .slice(0, take)
      .map((value, index) => cellDiv(value, index === activeIndex ? "active" : ""))
      .join("");
  }

  /**
   * opts: { formula, aLabel, a, bLabel, b, outLabel, colLabels,
   * rowWindow, colWindow, colIndices (optional meaningful output subset) }
   */
  function renderMatMul(container, opts) {
    const { aLabel, a, bLabel, b, outLabel, colLabels } = opts;
    const inputDim = a.length;
    const outputDim = b[0].length;
    const rowWindow = Math.min(opts.rowWindow ?? 8, inputDim);
    const colWindow = Math.min(opts.colWindow ?? 6, outputDim);
    const navigableCols = opts.colIndices
      ? [...opts.colIndices]
      : Array.from({ length: outputDim }, (_, index) => index);

    const output = new Array(outputDim).fill(0);
    for (let col = 0; col < outputDim; col++) {
      for (let row = 0; row < inputDim; row++) output[col] += a[row] * b[row][col];
    }

    let cursor = 0;
    let playing = false;
    let timer = null;

    function draw() {
      const activeCol = navigableCols[cursor];
      const windowStart = opts.colIndices ? 0 : Math.floor(cursor / colWindow) * colWindow;
      const visibleCols = opts.colIndices
        ? navigableCols.slice(0, colWindow)
        : navigableCols.slice(windowStart, windowStart + colWindow);
      const matrixRows = [];

      for (let row = 0; row < rowWindow; row++) {
        const cells = visibleCols
          .map((col) => cellDiv(b[row][col], col === activeCol ? "hi-col" : ""))
          .join("");
        matrixRows.push(`<div class="mm-mat-row">${cells}</div>`);
      }

      const outputCells = visibleCols
        .map((col) => cellDiv(output[col], col === activeCol ? "active" : ""))
        .join("");
      const hiddenRows = inputDim - rowWindow;
      const colName = colLabels ? colLabels[activeCol] : `${activeCol}`;

      container.innerHTML = `
        <div class="mm-formula-line"></div>
        <div class="mm-flow">
          <div class="mm-block">
            <div class="mm-label">${aLabel} <span class="mm-dim">(1\u00d7${inputDim})</span></div>
            <div class="mm-vec">${rowCells(a, rowWindow, -1)}</div>
            ${inputDim > rowWindow ? `<div class="mm-more">+${inputDim - rowWindow} more dims</div>` : `<div class="mm-more">&nbsp;</div>`}
          </div>
          <div class="mm-op">\u00d7</div>
          <div class="mm-block">
            <div class="mm-label">${bLabel} <span class="mm-dim">(${inputDim}\u00d7${outputDim})</span></div>
            <div class="mm-mat" style="grid-template-columns:repeat(${visibleCols.length},${CELL_W}px)">${matrixRows.join("")}</div>
            <div class="mm-more">${hiddenRows ? `${hiddenRows} more rows` : ""}${hiddenRows && outputDim > visibleCols.length ? " \u00b7 " : ""}${outputDim > visibleCols.length ? `${outputDim - visibleCols.length} other columns` : ""}${!hiddenRows && outputDim <= visibleCols.length ? "&nbsp;" : ""}</div>
          </div>
          <div class="mm-op">=</div>
          <div class="mm-block">
            <div class="mm-label">${outLabel} <span class="mm-dim">(1\u00d7${outputDim})</span></div>
            <div class="mm-vec">${outputCells}</div>
            ${outputDim > visibleCols.length ? `<div class="mm-more">showing ${visibleCols.length} of ${outputDim}</div>` : `<div class="mm-more">&nbsp;</div>`}
          </div>
        </div>
        <div class="mm-detail"></div>
        <div class="mm-controls">
          <button class="mm-prev" title="Previous output column">\u2190</button>
          <button class="mm-play">${playing ? "\u23f8 Pause" : "\u25b6 Animate"}</button>
          <button class="mm-next" title="Next output column">\u2192</button>
          <span class="mm-status">output ${cursor + 1} of ${navigableCols.length} \u00b7 column ${activeCol} \u00b7 ${colName}</span>
        </div>`;

      if (opts.formula) tex(container.querySelector(".mm-formula-line"), opts.formula, true);

      const terms = [];
      for (let row = 0; row < rowWindow; row++) terms.push(`(${fmt(a[row])})(${fmt(b[row][activeCol])})`);
      const detailLatex = `\\text{${outLabel}}[${activeCol}] = ${terms.join(" + ")}${
        hiddenRows > 0 ? ` + \\ldots\\ (${hiddenRows}\\text{ more real terms})` : ""
      } = \\mathbf{${fmt(output[activeCol])}}`;
      tex(container.querySelector(".mm-detail"), detailLatex, true);

      container.querySelector(".mm-prev").addEventListener("click", () => {
        stop();
        cursor = (cursor - 1 + navigableCols.length) % navigableCols.length;
        draw();
      });
      container.querySelector(".mm-next").addEventListener("click", () => {
        stop();
        cursor = (cursor + 1) % navigableCols.length;
        draw();
      });
      container.querySelector(".mm-play").addEventListener("click", () => (playing ? stop() : play()));
    }

    function play() {
      playing = true;
      timer = setInterval(() => {
        cursor = (cursor + 1) % navigableCols.length;
        draw();
      }, 1500);
      draw();
    }

    function stop() {
      playing = false;
      if (timer) clearInterval(timer);
      timer = null;
      draw();
    }

    draw();
    const handle = { stop };
    active.push(handle);
    return handle;
  }

  function renderElementwise(container, opts) {
    const take = opts.take ?? 6;
    const stepsHTML = opts.steps
      .map((step, index) => {
        const cells = rowCells(step.values, take, -1);
        const arrow = index < opts.steps.length - 1 ? `<div class="ew-arrow">\u2192</div>` : "";
        return `<div class="ew-step" style="--stage:${index}">
          <div class="ew-label">${step.label}</div>
          <div class="mm-vec">${cells}</div>
          <div class="mm-more">${step.values.length > take ? `+${step.values.length - take} more` : "&nbsp;"}</div>
        </div>${arrow}`;
      })
      .join("");
    container.innerHTML = `<div class="mm-formula-line"></div><div class="ew-flow">${stepsHTML}</div>`;
    if (opts.title) tex(container.querySelector(".mm-formula-line"), opts.title, true);
  }

  function stopAll() {
    while (active.length) active.pop().stop();
  }

  TJ.matrixViz = { renderMatMul, renderElementwise, stopAll, cellDiv, fmt, tex };
})();