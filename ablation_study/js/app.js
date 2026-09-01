/* ablation_study/js/app.js
 * State + control wiring for the ablation study page. At most one axis
 * may differ from the SMOKE_TEST baseline at a time (that's what makes
 * each comparison a genuine single-variable experiment) -- picking a new
 * axis resets the others back to baseline automatically.
 */
(function () {
  "use strict";
  const DATA = window.ABLATION_DATA;

  const BASELINE_DEFAULTS = { attention: "gqa", positional: "rope", activation: "swiglu", moe_balancing: "aux_loss", tie: true };
  const AXIS_VARIANT = {
    attention: { mla: "ATTN_MLA" },
    positional: { nope: "POS_NOPE", alibi: "POS_ALIBI" },
    activation: { geglu: "ACT_GEGLU", gelu: "ACT_GELU" },
    moe_balancing: { loss_free: "MOE_LOSSFREE" },
  };
  const AXIS_FIELDS = {
    baseline: [],
    attention: ["attention_variant", "mla_latent_dim"],
    positional: ["positional_scheme"],
    activation: ["activation"],
    moe_balancing: ["moe_balancing"],
    tie_word_embeddings: ["tie_word_embeddings"],
  };

  const state = { preset: "SMOKE_TEST", attention: "gqa", positional: "rope", activation: "swiglu", moe_balancing: "aux_loss", tie: true };

  function resetAxes() {
    Object.assign(state, BASELINE_DEFAULTS);
  }

  function selectedVariantName() {
    if (!state.tie) return "TIE_FALSE";
    for (const [axis, map] of Object.entries(AXIS_VARIANT)) {
      if (state[axis] in map) return map[state[axis]];
    }
    return DATA.baseline;
  }

  function updateButtons() {
    document.querySelectorAll(".ctrl-group").forEach((group) => {
      const key = group.dataset.key;
      group.querySelectorAll("button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.value === state[key]);
      });
    });
    document.getElementById("tieChk").checked = state.tie;
  }

  function fmtField(field, value) {
    if (value === null || value === undefined) return "none";
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  function better(metric, baseValue, variantValue, lowerIsBetter) {
    if (variantValue === baseValue) return null;
    return lowerIsBetter ? variantValue < baseValue : variantValue > baseValue;
  }

  function renderStatsLine(baseline, variant, isBaselineOnly) {
    const el = document.getElementById("statsLine");
    if (isBaselineOnly) {
      el.innerHTML =
        `Showing the real <strong>${baseline.label}</strong> baseline itself (val_loss=<strong>${baseline.final_val_loss.toFixed(4)}</strong>, ` +
        `val_perplexity=<strong>${baseline.val_perplexity.toFixed(3)}</strong>, tokens/sec=<strong>${Math.round(baseline.tokens_per_sec).toLocaleString()}</strong>). ` +
        `Select an axis above to compare a real single-variable variant against it.`;
      return;
    }
    const fields = AXIS_FIELDS[variant.axis] || [];
    const diffHTML = fields
      .map((field) => `<code>${field}</code>: ${fmtField(field, baseline.config[field])} <span class="diff-arrow">→</span> ${fmtField(field, variant.config[field])}`)
      .join(", ");
    const lossBetter = better("val_loss", baseline.final_val_loss, variant.final_val_loss, true);
    const cls = lossBetter === null ? "" : lossBetter ? "better" : "worse";
    el.innerHTML =
      `<strong>${variant.label}</strong> vs baseline <strong>${baseline.label}</strong> — changed ${diffHTML} &nbsp;|&nbsp; ` +
      `val_loss: ${baseline.final_val_loss.toFixed(4)} <span class="diff-arrow">→</span> <span class="${cls}">${variant.final_val_loss.toFixed(4)}</span> ` +
      `&nbsp;·&nbsp; val_perplexity: ${baseline.val_perplexity.toFixed(3)} <span class="diff-arrow">→</span> ${variant.val_perplexity.toFixed(3)} ` +
      `&nbsp;·&nbsp; tokens/sec: ${Math.round(baseline.tokens_per_sec).toLocaleString()} <span class="diff-arrow">→</span> ${Math.round(variant.tokens_per_sec).toLocaleString()}`;
  }

  function renderComparison() {
    const baseline = DATA.variants[DATA.baseline];
    const variantName = selectedVariantName();
    const variant = DATA.variants[variantName];
    const isBaselineOnly = variantName === DATA.baseline;

    renderStatsLine(baseline, variant, isBaselineOnly);

    AB.charts.renderMetricBars(document.getElementById("metricBars"), {
      baselineLabel: baseline.label,
      variantLabel: variant.label,
      metrics: [
        { label: "val_loss", baselineValue: baseline.final_val_loss, variantValue: variant.final_val_loss, format: (v) => v.toFixed(4), lowerIsBetter: true },
        { label: "val_perplexity", baselineValue: baseline.val_perplexity, variantValue: variant.val_perplexity, format: (v) => v.toFixed(3), lowerIsBetter: true },
        { label: "tokens/sec", baselineValue: baseline.tokens_per_sec, variantValue: variant.tokens_per_sec, format: (v) => Math.round(v).toLocaleString(), lowerIsBetter: false },
      ],
    });

    AB.charts.renderLossCurve(document.getElementById("lossCurve"), {
      baseline: { label: baseline.label, ...baseline.curve },
      variant: { label: variant.label, ...variant.curve },
    });

    AB.charts.renderExpertBars(document.getElementById("expertBars"), {
      baselineLabel: baseline.label,
      variantLabel: variant.label,
      baselineCounts: baseline.curve.expert_token_counts.at(-1),
      variantCounts: variant.curve.expert_token_counts.at(-1),
    });

    const deepdiveSection = document.getElementById("deepdiveSection");
    const deepdivePanel = document.getElementById("deepdivePanel");
    const deepdiveTitle = document.getElementById("deepdiveTitle");
    if (!isBaselineOnly && AB.deepDive[variant.axis]) {
      deepdiveSection.style.display = "";
      deepdiveTitle.textContent = variant.axis === "moe_balancing" ? "Why Loss-Free balancing changes routing" : "Why MLA's KV-cache is smaller";
      deepdivePanel.innerHTML = "";
      AB.deepDive[variant.axis](deepdivePanel, DATA);
    } else {
      deepdiveSection.style.display = "none";
    }
  }

  function render() {
    updateButtons();
    const notice = document.getElementById("illustrativeNotice");
    const comparison = document.getElementById("comparison");
    if (state.preset === "BASELINE") {
      notice.style.display = "";
      notice.innerHTML =
        "<strong>BASELINE</strong> (d_model=256, n_layers=6, ~23.5M params) is an illustrative larger-scale preset, not part of this real ablation study -- " +
        "every ablation variant here deliberately extends the laptop-fast <code>SMOKE_TEST</code> config instead (see " +
        "<code>journey_of_a_model/015-research-comparison-framework.md</code>), so no real trained numbers exist for BASELINE. " +
        "Explore its architecture in <code>architecture_diagram.html</code>, or switch back to SMOKE_TEST above for real comparisons.";
      document.getElementById("statsLine").innerHTML = "";
      comparison.style.display = "none";
      return;
    }
    notice.style.display = "none";
    comparison.style.display = "";
    renderComparison();
  }

  function wireControls() {
    document.querySelectorAll(".ctrl-group").forEach((group) => {
      const key = group.dataset.key;
      group.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (key === "preset") {
            state.preset = btn.dataset.value;
          } else {
            resetAxes();
            state[key] = btn.dataset.value;
          }
          render();
        });
      });
    });
    document.getElementById("tieChk").addEventListener("change", (e) => {
      resetAxes();
      state.tie = e.target.checked;
      render();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    wireControls();
    render();
  });
})();
