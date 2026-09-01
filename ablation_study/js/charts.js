/* ablation_study/js/charts.js
 * Reusable D3 chart primitives for the ablation study page. Every value
 * passed in is real (read from window.ABLATION_DATA) -- these functions
 * only lay out and animate numbers the caller already has.
 */
(function () {
  "use strict";
  const AB = (window.AB = window.AB || {});

  function fmt(value, digits) {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return value.toFixed(digits ?? 2);
  }

  function fmtInt(value) {
    return Math.round(value).toLocaleString();
  }

  /**
   * opts: { metrics: [{ key, label, baselineValue, variantValue, format,
   * lowerIsBetter }] } -- one grouped horizontal bar pair per metric.
   */
  function renderMetricBars(container, opts) {
    container.innerHTML = "";
    const width = Math.max(560, container.clientWidth || 560);
    const rowH = 56;
    const height = opts.metrics.length * rowH + 20;
    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
    const labelW = 150;
    const plotW = width - labelW - 90;
    const maxScale = d3.scaleLinear();

    const legend = d3.select(container).insert("div", "svg").attr("class", "chart-legend");
    legend.html(
      `<span><span class="sw" style="background:#94a3b8"></span>Baseline (${opts.baselineLabel})</span>` +
        `<span><span class="sw" style="background:#7c3aed"></span>${opts.variantLabel}</span>`
    );

    opts.metrics.forEach((metric, i) => {
      const y0 = i * rowH + 10;
      const format = metric.format || ((v) => fmt(v, 3));
      const maxVal = Math.max(metric.baselineValue, metric.variantValue, 1e-9) * 1.15;
      const scale = maxScale.domain([0, maxVal]).range([0, plotW]);
      const better =
        metric.variantValue === metric.baselineValue
          ? null
          : metric.lowerIsBetter
          ? metric.variantValue < metric.baselineValue
          : metric.variantValue > metric.baselineValue;

      const row = svg.append("g").attr("transform", `translate(0,${y0})`);
      row
        .append("text")
        .attr("class", "bar-label")
        .attr("x", 0)
        .attr("y", 16)
        .text(metric.label);

      const baseBar = row
        .append("rect")
        .attr("class", "bar-baseline")
        .attr("x", labelW)
        .attr("y", 0)
        .attr("height", 14)
        .attr("width", 0)
        .attr("rx", 3);
      baseBar.transition().duration(500).attr("width", scale(metric.baselineValue));
      row
        .append("text")
        .attr("class", "bar-value")
        .attr("x", labelW + scale(metric.baselineValue) + 6)
        .attr("y", 11)
        .text(format(metric.baselineValue));

      const varBar = row
        .append("rect")
        .attr("class", "bar-variant" + (better === null ? "" : better ? " better" : " worse"))
        .attr("x", labelW)
        .attr("y", 20)
        .attr("height", 14)
        .attr("width", 0)
        .attr("rx", 3);
      varBar.transition().duration(500).delay(80).attr("width", scale(metric.variantValue));
      row
        .append("text")
        .attr("class", "bar-value")
        .attr("x", labelW + scale(metric.variantValue) + 6)
        .attr("y", 31)
        .text(format(metric.variantValue));
    });
  }

  /**
   * opts: { baseline: {label, steps, train_loss, eval_steps, val_loss},
   * variant: {label, steps, train_loss, eval_steps, val_loss} }
   */
  function renderLossCurve(container, opts) {
    container.innerHTML = "";
    const width = Math.max(620, container.clientWidth || 620);
    const height = 260;
    const margin = { top: 12, right: 16, bottom: 26, left: 40 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    const legend = d3.select(container).insert("div", null).attr("class", "chart-legend");
    legend.html(
      `<span><span class="sw" style="background:#94a3b8"></span>${opts.baseline.label} (train)</span>` +
        `<span><span class="sw" style="background:#64748b"></span>${opts.baseline.label} (val)</span>` +
        `<span><span class="sw" style="background:#a78bfa"></span>${opts.variant.label} (train)</span>` +
        `<span><span class="sw" style="background:#4338ca"></span>${opts.variant.label} (val)</span>`
    );

    const allSteps = [...opts.baseline.steps, ...opts.variant.steps];
    const allLoss = [...opts.baseline.train_loss, ...opts.variant.train_loss, ...opts.baseline.val_loss, ...opts.variant.val_loss];
    const x = d3.scaleLinear().domain([0, d3.max(allSteps) || 1]).range([0, plotW]);
    const y = d3.scaleLinear().domain([0, d3.max(allLoss) * 1.08]).range([plotH, 0]);

    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    g.append("g").attr("class", "chart-axis").attr("transform", `translate(0,${plotH})`).call(d3.axisBottom(x).ticks(6));
    g.append("g").attr("class", "chart-axis").call(d3.axisLeft(y).ticks(5));

    function drawLine(xs, ys, cls) {
      const line = d3
        .line()
        .x((d, i) => x(xs[i]))
        .y((d) => y(d));
      const path = g.append("path").datum(ys).attr("class", cls).attr("d", line);
      const length = path.node().getTotalLength();
      path
        .attr("stroke-dasharray", `${length} ${length}`)
        .attr("stroke-dashoffset", length)
        .transition()
        .duration(700)
        .ease(d3.easeCubicOut)
        .attr("stroke-dashoffset", 0);
    }

    drawLine(opts.baseline.steps, opts.baseline.train_loss, "line-baseline");
    drawLine(opts.variant.steps, opts.variant.train_loss, "line-variant");
    drawLine(opts.baseline.eval_steps, opts.baseline.val_loss, "line-baseline-val");
    drawLine(opts.variant.eval_steps, opts.variant.val_loss, "line-variant-val");

    g.selectAll(".dot-baseline")
      .data(opts.baseline.val_loss)
      .enter()
      .append("circle")
      .attr("class", "dot-baseline")
      .attr("cx", (d, i) => x(opts.baseline.eval_steps[i]))
      .attr("cy", (d) => y(d))
      .attr("r", 0)
      .transition()
      .delay(700)
      .duration(250)
      .attr("r", 2.6);

    g.selectAll(".dot-variant")
      .data(opts.variant.val_loss)
      .enter()
      .append("circle")
      .attr("class", "dot-variant")
      .attr("cx", (d, i) => x(opts.variant.eval_steps[i]))
      .attr("cy", (d) => y(d))
      .attr("r", 0)
      .transition()
      .delay(700)
      .duration(250)
      .attr("r", 3);
  }

  /** opts: { baselineLabel, variantLabel, baselineCounts, variantCounts }
   * -- counts: [layer][expert] -> token count (final logged snapshot). */
  function renderExpertBars(container, opts) {
    container.innerHTML = "";
    const nLayers = opts.baselineCounts.length;
    const nExperts = opts.baselineCounts[0].length;
    const width = Math.max(560, container.clientWidth || 560);
    const layerH = 92;
    const barAreaH = 50;
    const height = nLayers * layerH + 12;

    const legend = d3.select(container).insert("div", null).attr("class", "chart-legend");
    legend.html(
      `<span><span class="sw" style="background:#94a3b8"></span>Baseline (${opts.baselineLabel})</span>` +
        `<span><span class="sw" style="background:#7c3aed"></span>${opts.variantLabel}</span>`
    );

    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
    const maxCount = d3.max([...opts.baselineCounts.flat(), ...opts.variantCounts.flat()]) || 1;
    const groupW = (width - 40) / nExperts;
    const y = d3.scaleLinear().domain([0, maxCount * 1.1]).range([0, barAreaH]);

    for (let layer = 0; layer < nLayers; layer++) {
      const g = svg.append("g").attr("transform", `translate(20,${layer * layerH + 20})`);
      g.append("text").attr("class", "expert-group-label").attr("x", 0).attr("y", -6).text(`layer ${layer}`);
      for (let expert = 0; expert < nExperts; expert++) {
        const gx = expert * groupW;
        const baseVal = opts.baselineCounts[layer][expert];
        const varVal = opts.variantCounts[layer][expert];
        const barW = Math.min(28, groupW / 2 - 6);

        g.append("text")
          .attr("class", "bar-label")
          .attr("x", gx + barW)
          .attr("y", barAreaH + 14)
          .attr("text-anchor", "middle")
          .text(`E${expert}`);

        const b1 = g
          .append("rect")
          .attr("class", "bar-baseline")
          .attr("x", gx)
          .attr("width", barW)
          .attr("y", barAreaH)
          .attr("height", 0)
          .attr("rx", 2);
        b1.transition().duration(500).attr("y", barAreaH - y(baseVal)).attr("height", y(baseVal));

        const b2 = g
          .append("rect")
          .attr("class", "bar-variant")
          .attr("x", gx + barW + 4)
          .attr("width", barW)
          .attr("y", barAreaH)
          .attr("height", 0)
          .attr("rx", 2);
        b2.transition().duration(500).delay(80).attr("y", barAreaH - y(varVal)).attr("height", y(varVal));
      }
    }
  }

  AB.charts = { renderMetricBars, renderLossCurve, renderExpertBars, fmt, fmtInt };
})();
