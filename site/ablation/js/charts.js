/* site/ablation/js/charts.js
 *
 * D3 chart primitives for the ablation study page. Every value passed in is
 * real (read from window.ABLATION_DATA, itself written from real
 * `train.jsonl` files) -- these functions only lay out and animate numbers
 * the caller already has, they never derive a metric.
 *
 * Colours are read from CSS custom properties rather than hard-coded, so
 * the same chart code renders correctly in both light and dark themes.
 */
(function () {
  "use strict";
  const AB = (window.AB = window.AB || {});

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  const C = {
    get control() { return cssVar("--control"); },
    get variant() { return cssVar("--variant"); },
    get better() { return cssVar("--better"); },
    get worse() { return cssVar("--worse"); },
    get ink3() { return cssVar("--ink-3"); },
    get border() { return cssVar("--border"); },
    preset(name) {
      return { BASELINE: cssVar("--p-baseline"), BASELINE_V2: cssVar("--p-v2"), FINEGRAINED: cssVar("--p-fine") }[name] || cssVar("--control");
    },
  };

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dur = (ms) => (reduceMotion ? 0 : ms);

  // ---- shared tooltip -----------------------------------------------------
  let tipEl = null;
  function tip() {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "tooltip";
      document.body.appendChild(tipEl);
    }
    return tipEl;
  }
  function showTip(event, html) {
    const el = tip();
    el.innerHTML = html;
    el.classList.add("on");
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const r = el.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = event.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = event.clientY - r.height - pad;
    el.style.left = x + "px";
    el.style.top = y + "px";
  }
  function hideTip() {
    if (tipEl) tipEl.classList.remove("on");
  }
  AB.hideTip = hideTip;

  function svgIn(container, height) {
    container.innerHTML = "";
    const width = Math.max(520, container.clientWidth || 640);
    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`);
    return { svg, width };
  }

  /* ------------------------------------------------------------------ *
   * Training curves. `series` is a list of
   *   { label, color, steps, values, dashed?, band? }
   * where `band` (optional) is [lowerValues, upperValues] drawn as a
   * translucent envelope -- used for the multi-seed preset runs, where the
   * band is the real min/max across seeds and the line is their mean.
   * ------------------------------------------------------------------ */
  function renderCurves(container, opts) {
    const series = opts.series.filter((s) => s.steps && s.steps.length);
    if (!series.length) {
      container.innerHTML = '<p style="color:var(--ink-3);font-size:.86rem">No curve data.</p>';
      return;
    }
    const H = opts.height || 300;
    const { svg, width } = svgIn(container, H);
    const m = { top: 12, right: 16, bottom: 34, left: 52 };
    const plotW = width - m.left - m.right;
    const plotH = H - m.top - m.bottom;
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const allSteps = series.flatMap((s) => s.steps);
    const allVals = series.flatMap((s) => (s.band ? s.band[0].concat(s.band[1]) : s.values));
    const x = d3.scaleLinear().domain(d3.extent(allSteps)).range([0, plotW]);
    const yPad = (d3.max(allVals) - d3.min(allVals)) * 0.08 || 0.1;
    const y = d3
      .scaleLinear()
      .domain([d3.min(allVals) - yPad, d3.max(allVals) + yPad])
      .range([plotH, 0]);

    g.append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).ticks(5).tickSize(-plotW).tickFormat(""));
    g.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${plotH})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2f")));

    svg
      .append("text")
      .attr("class", "chart-label")
      .attr("x", m.left + plotW / 2)
      .attr("y", H - 3)
      .attr("text-anchor", "middle")
      .text(opts.xLabel || "training step");
    svg
      .append("text")
      .attr("class", "chart-label")
      .attr("transform", `translate(12,${m.top + plotH / 2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .text(opts.yLabel || "loss");

    series.forEach((s) => {
      if (s.band) {
        const area = d3
          .area()
          .x((_, i) => x(s.steps[i]))
          .y0((_, i) => y(s.band[0][i]))
          .y1((_, i) => y(s.band[1][i]))
          .curve(d3.curveMonotoneX);
        g.append("path")
          .datum(s.steps)
          .attr("d", area)
          .attr("fill", s.color)
          .attr("opacity", 0.14);
      }
      const line = d3
        .line()
        .x((_, i) => x(s.steps[i]))
        .y((_, i) => y(s.values[i]))
        .curve(d3.curveMonotoneX);
      const path = g
        .append("path")
        .datum(s.values)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", s.color)
        .attr("stroke-width", s.dashed ? 2 : 2.6)
        .attr("stroke-dasharray", s.dashed ? "5 4" : null)
        .attr("stroke-linecap", "round");
      if (!reduceMotion) {
        const len = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", s.dashed ? "5 4" : `${len} ${len}`)
          .attr("stroke-dashoffset", s.dashed ? 0 : len)
          .transition()
          .duration(dur(700))
          .ease(d3.easeCubicOut)
          .attr("stroke-dashoffset", 0);
      }
    });

    // hover: a vertical rule that reads every series at the nearest step
    const focus = g.append("g").style("display", "none");
    focus
      .append("line")
      .attr("y1", 0)
      .attr("y2", plotH)
      .attr("stroke", C.ink3)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "3 3");
    const dots = series.map((s) =>
      focus.append("circle").attr("r", 4).attr("fill", s.color).attr("stroke", "var(--surface)").attr("stroke-width", 2)
    );

    g.append("rect")
      .attr("width", plotW)
      .attr("height", plotH)
      .attr("fill", "transparent")
      .style("cursor", "crosshair")
      .on("mouseenter", () => focus.style("display", null))
      .on("mouseleave", () => {
        focus.style("display", "none");
        hideTip();
      })
      .on("mousemove", (event) => {
        const [mx] = d3.pointer(event);
        const step = x.invert(mx);
        let html = "";
        focus.attr("transform", null);
        series.forEach((s, i) => {
          const idx = d3.leastIndex(s.steps, (a, b) => Math.abs(a - step) - Math.abs(b - step));
          const sx = x(s.steps[idx]);
          const sy = y(s.values[idx]);
          dots[i].attr("cx", sx).attr("cy", sy);
          if (i === 0) focus.select("line").attr("x1", sx).attr("x2", sx);
          html += `<div><span style="color:${s.color}">&#9632;</span> ${s.label}: <b>${s.values[idx].toFixed(4)}</b></div>`;
        });
        const idx0 = d3.leastIndex(series[0].steps, (a, b) => Math.abs(a - step) - Math.abs(b - step));
        showTip(event, `<div style="opacity:.7;margin-bottom:3px">step ${series[0].steps[idx0]}</div>${html}`);
      });
  }

  /* ------------------------------------------------------------------ *
   * Expert utilization: grouped bars, one group per layer, one bar per
   * routed expert, with a dashed "perfectly balanced" reference line.
   * `groups` is [{ label, counts }].
   * ------------------------------------------------------------------ */
  function renderExpertBars(container, opts) {
    const groups = opts.groups.filter((gr) => gr.counts && gr.counts.length);
    if (!groups.length) {
      container.innerHTML = '<p style="color:var(--ink-3);font-size:.86rem">No expert-utilization data logged for this run.</p>';
      return;
    }
    const nLayers = groups[0].counts.length;
    const nExperts = groups[0].counts[0].length;
    const H = 78 * nLayers + 46;
    const { svg, width } = svgIn(container, H);
    const m = { top: 10, right: 14, bottom: 26, left: 74 };
    const plotW = width - m.left - m.right;
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);

    const rowH = 78;
    const groupW = plotW / groups.length;

    for (let layer = 0; layer < nLayers; layer++) {
      const rowY = layer * rowH;
      svg
        .append("text")
        .attr("class", "chart-label")
        .attr("x", m.left - 10)
        .attr("y", m.top + rowY + rowH / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .text(`layer ${layer}`);

      groups.forEach((gr, gi) => {
        const counts = gr.counts[layer];
        const total = counts.reduce((a, b) => a + b, 0) || 1;
        const ideal = 1 / counts.length;
        const barsY = rowY + 12;
        const barsH = rowH - 26;
        const x = d3
          .scaleBand()
          .domain(d3.range(counts.length))
          .range([gi * groupW + 8, (gi + 1) * groupW - 8])
          .padding(counts.length > 16 ? 0.12 : 0.24);
        const maxFrac = Math.max(d3.max(counts) / total, ideal * 1.6);
        const y = d3.scaleLinear().domain([0, maxFrac]).range([barsH, 0]);

        if (layer === 0) {
          svg
            .append("text")
            .attr("class", "chart-label")
            .attr("x", m.left + gi * groupW + groupW / 2)
            .attr("y", H - 8)
            .attr("text-anchor", "middle")
            .style("font-weight", "650")
            .style("fill", gr.color)
            .text(gr.label);
        }

        const layerG = g.append("g").attr("transform", `translate(0,${barsY})`);
        layerG
          .selectAll("rect")
          .data(counts)
          .join("rect")
          .attr("x", (_, i) => x(i))
          .attr("width", x.bandwidth())
          .attr("y", barsH)
          .attr("height", 0)
          .attr("rx", 1.5)
          .attr("fill", gr.color)
          .attr("opacity", (d) => 0.45 + 0.55 * Math.min(1, d / total / (ideal * 2)))
          .on("mousemove", (event, d) =>
            showTip(
              event,
              `${gr.label} &middot; layer ${layer}<br>expert <b>${counts.indexOf(d)}</b><br>` +
                `${d.toLocaleString()} tokens (<b>${((d / total) * 100).toFixed(1)}%</b>)<br>` +
                `<span style="opacity:.7">even share would be ${(ideal * 100).toFixed(1)}%</span>`
            )
          )
          .on("mouseleave", hideTip)
          .transition()
          .duration(dur(520))
          .delay((_, i) => dur(i * 8 + layer * 40))
          .attr("y", (d) => y(d / total))
          .attr("height", (d) => barsH - y(d / total));

        layerG
          .append("line")
          .attr("class", "ideal-line")
          .attr("x1", gi * groupW + 8)
          .attr("x2", (gi + 1) * groupW - 8)
          .attr("y1", y(ideal))
          .attr("y2", y(ideal));
      });
    }
    return { nExperts };
  }

  /* ------------------------------------------------------------------ *
   * Coefficient-of-variation summary: one dot per layer per variant on a
   * shared axis. CV = std(load)/mean(load); 0 is perfectly even routing.
   * ------------------------------------------------------------------ */
  function renderBalanceDots(container, opts) {
    const rows = opts.rows.filter((r) => r.cvs && r.cvs.length);
    if (!rows.length) {
      container.innerHTML = "";
      return;
    }
    const H = 34 * rows.length + 40;
    const { svg, width } = svgIn(container, H);
    const m = { top: 10, right: 22, bottom: 28, left: 118 };
    const plotW = width - m.left - m.right;
    const g = svg.append("g").attr("transform", `translate(${m.left},${m.top})`);
    const maxCv = Math.max(0.35, d3.max(rows.flatMap((r) => r.cvs)) * 1.15);
    const x = d3.scaleLinear().domain([0, maxCv]).range([0, plotW]);

    g.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${rows.length * 34})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".2f")));
    svg
      .append("text")
      .attr("class", "chart-label")
      .attr("x", m.left + plotW / 2)
      .attr("y", H - 2)
      .attr("text-anchor", "middle")
      .text("coefficient of variation across routed experts  (0 = perfectly even)");

    rows.forEach((r, i) => {
      const yy = i * 34 + 15;
      svg
        .append("text")
        .attr("class", "chart-label")
        .attr("x", m.left - 10)
        .attr("y", m.top + yy)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .style("font-weight", "600")
        .text(r.label);
      g.append("line")
        .attr("x1", 0)
        .attr("x2", plotW)
        .attr("y1", yy)
        .attr("y2", yy)
        .attr("stroke", C.border);
      g.selectAll(null)
        .data(r.cvs)
        .join("circle")
        .attr("cy", yy)
        .attr("r", 5)
        .attr("fill", r.color)
        .attr("opacity", 0.75)
        .attr("cx", 0)
        .on("mousemove", (event, d) =>
          showTip(event, `${r.label} &middot; layer ${r.cvs.indexOf(d)}<br>CV = <b>${d.toFixed(3)}</b>`)
        )
        .on("mouseleave", hideTip)
        .transition()
        .duration(dur(520))
        .delay((_, j) => dur(j * 50))
        .attr("cx", (d) => x(d));
    });
  }

  AB.charts = { renderCurves, renderExpertBars, renderBalanceDots, C, showTip, hideTip };
})();
