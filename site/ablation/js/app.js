/* site/ablation/js/app.js
 *
 * State + rendering for the ablation study page.
 *
 * Two independent tiers share the page:
 *
 *  - "axes"    -- the single-variable study. Every variant changes exactly
 *                 ONE axis relative to SMOKE_TEST and trained on the same
 *                 fixed token slice for the same 600-step budget, so a
 *                 metric difference reflects only that axis.
 *  - "presets" -- the modernized head-to-head from journal entry 035:
 *                 BASELINE vs BASELINE_V2 vs FINEGRAINED, a different base
 *                 config entirely, trained over 2 seeds so a difference can
 *                 be checked against seed noise rather than eyeballed.
 *
 * They are deliberately never mixed: the two runs use different base
 * configs, different slices and different budgets, so a number from one
 * tier is not comparable with a number from the other.
 */
(function () {
  "use strict";
  const DATA = window.ABLATION_DATA;
  const AXES = DATA.axes || {};
  const PRESETS = DATA.presets || null;
  const BASE = DATA.baseline;

  // Which spec each axis came from -- purely for grouping the picker, so a
  // reader can see at a glance which knobs are new in moe-modernization.
  const ERA = {
    positional: "original",
    activation: "original",
    moe_balancing: "original",
    attention: "original",
    tie_word_embeddings: "original",
    router_scoring: "modern",
    qk_norm: "modern",
    mtp_depth: "modern",
    expert_granularity: "modern",
    optimizer: "modern",
    lr_schedule: "modern",
  };

  // Fields to diff in the explainer card, per axis.
  const AXIS_FIELDS = {
    attention: ["attention_variant", "mla_latent_dim"],
    positional: ["positional_scheme"],
    activation: ["activation"],
    moe_balancing: ["moe_balancing"],
    tie_word_embeddings: ["tie_word_embeddings"],
    router_scoring: ["router_scoring"],
    qk_norm: ["qk_norm"],
    mtp_depth: ["mtp_depth"],
    expert_granularity: ["n_routed_experts", "d_ff", "top_k", "n_shared_experts"],
  };

  const state = {
    tier: "axes",
    variant: BASE,
    sortKey: "delta",
    sortDir: 1,
  };

  const $ = (id) => document.getElementById(id);
  const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d));
  const int = (v) => Math.round(v).toLocaleString();
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function fmtField(value) {
    if (value === null || value === undefined) return "none";
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  /* ------------------------------------------------------------------ *
   * Axis tier
   * ------------------------------------------------------------------ */

  function variantRows() {
    const base = DATA.variants[BASE];
    return Object.entries(DATA.variants).map(([name, v]) => ({
      name,
      v,
      isControl: name === BASE,
      delta: v.final_val_loss - base.final_val_loss,
      speedRatio: v.tokens_per_sec / base.tokens_per_sec,
    }));
  }

  function renderPicker() {
    const rows = variantRows();
    const groups = [
      { key: "original", title: "Axes from the original spec" },
      { key: "modern", title: "Axes added by moe-modernization" },
    ];
    const host = $("picker");
    host.innerHTML = groups
      .map((grp) => {
        const chips = rows
          .filter((r) => (r.isControl ? grp.key === "original" : ERA[r.v.axis] === grp.key))
          .map((r) => {
            const d = r.delta;
            const cls = r.isControl ? "chip is-control" : "chip";
            const deltaTxt = r.isControl
              ? "control"
              : (d < 0 ? "−" : "+") + Math.abs(d).toFixed(3);
            return `<button class="${cls}" role="button" aria-pressed="${r.name === state.variant}" data-variant="${r.name}">
              ${esc(r.v.label)} <span class="delta">${deltaTxt}</span></button>`;
          })
          .join("");
        return `<div class="picker-group"><h3>${grp.title}</h3><div class="chips">${chips}</div></div>`;
      })
      .join("");
    host.querySelectorAll("[data-variant]").forEach((btn) =>
      btn.addEventListener("click", () => {
        state.variant = btn.dataset.variant;
        renderAxisTier();
      })
    );
  }

  function renderExplainer() {
    const v = DATA.variants[state.variant];
    const base = DATA.variants[BASE];
    const axis = AXES[v.axis] || { label: v.axis, question: "", why: "" };
    const isControl = state.variant === BASE;

    let changed = "";
    if (!isControl) {
      const fields = AXIS_FIELDS[v.axis] || [];
      const bits = fields
        .filter((f) => base.config[f] !== v.config[f])
        .map(
          (f) =>
            `<span><span class="field">${f}</span> ${fmtField(base.config[f])}<span class="arrow">→</span>${fmtField(v.config[f])}</span>`
        );
      Object.entries(v.train_overrides || {}).forEach(([f, val]) => {
        bits.push(
          `<span><span class="field">TrainConfig.${f}</span> ${fmtField(
            f === "optimizer" ? "adamw" : "cosine"
          )}<span class="arrow">→</span>${fmtField(val)}</span>`
        );
      });
      changed = `<div class="changed"><b style="font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3)">Changed</b>${bits.join(
        ""
      )}</div>`;
    }

    const paper = axis.paper
      ? `<a class="paper-link" href="https://arxiv.org/abs/${axis.paper}" target="_blank" rel="noopener">Read the paper &rarr; arXiv:${axis.paper}</a>`
      : "";

    $("explainer").innerHTML =
      `<h2>${esc(axis.label)}</h2>` +
      (axis.question ? `<p class="question">${esc(axis.question)}</p>` : "") +
      `<p class="why">${esc(axis.why)}</p>` +
      changed +
      paper;
  }

  function renderVerdict() {
    const v = DATA.variants[state.variant];
    const base = DATA.variants[BASE];
    const isControl = state.variant === BASE;
    const d = v.final_val_loss - base.final_val_loss;
    const lossCls = isControl ? "" : d < 0 ? "better" : "worse";
    const speedD = v.tokens_per_sec / base.tokens_per_sec - 1;
    const paramD = v.params.total - base.params.total;

    $("verdict").innerHTML = [
      `<div class="stat ${lossCls}">
        <div class="k">Final val loss</div>
        <div class="v">${fmt(v.final_val_loss, 4)}</div>
        <div class="sub">${isControl ? "the control" : `${d < 0 ? "−" : "+"}${Math.abs(d).toFixed(4)} vs ${BASE}`}</div>
      </div>`,
      `<div class="stat">
        <div class="k">Perplexity</div>
        <div class="v">${fmt(v.val_perplexity, 3)}</div>
        <div class="sub">${isControl ? "e^(val loss)" : `control ${fmt(base.val_perplexity, 3)}`}</div>
      </div>`,
      `<div class="stat">
        <div class="k">Throughput</div>
        <div class="v">${int(v.tokens_per_sec)}</div>
        <div class="sub">tok/s${isControl ? "" : ` &middot; ${speedD >= 0 ? "+" : ""}${(speedD * 100).toFixed(1)}%`}</div>
      </div>`,
      `<div class="stat">
        <div class="k">Parameters</div>
        <div class="v">${int(v.params.total)}</div>
        <div class="sub">${int(v.params.active)} active${isControl || paramD === 0 ? "" : ` &middot; ${paramD > 0 ? "+" : "−"}${int(Math.abs(paramD))}`}</div>
      </div>`,
    ].join("");
  }

  function renderBoard() {
    const rows = variantRows();
    const key = state.sortKey;
    const dir = state.sortDir;
    rows.sort((a, b) => {
      // the control always pins to the top: everything else is measured
      // against it, so it is a header row more than a ranked row.
      if (a.isControl) return -1;
      if (b.isControl) return 1;
      const get = {
        name: (r) => r.v.label.toLowerCase(),
        delta: (r) => r.delta,
        loss: (r) => r.v.final_val_loss,
        ppl: (r) => r.v.val_perplexity,
        speed: (r) => r.v.tokens_per_sec,
        params: (r) => r.v.params.total,
      }[key];
      const av = get(a);
      const bv = get(b);
      return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * dir;
    });

    const maxAbs = Math.max(...rows.map((r) => Math.abs(r.delta)), 1e-6);
    const cols = [
      ["name", "Variant"],
      ["delta", "Δ val loss vs control"],
      ["loss", "Val loss"],
      ["ppl", "Perplexity"],
      ["speed", "Tok/s"],
      ["params", "Params"],
    ];
    const head = cols
      .map(
        ([k, label]) =>
          `<th class="sortable" data-sort="${k}"${
            key === k ? ` aria-sort="${dir === 1 ? "ascending" : "descending"}"` : ""
          }>${label}<span class="caret">${key === k ? (dir === 1 ? "▲" : "▼") : "⇅"}</span></th>`
      )
      .join("");

    const body = rows
      .map((r) => {
        const pct = (Math.abs(r.delta) / maxAbs) * 50;
        const bar = r.isControl
          ? '<div class="dbar"><div class="zero"></div></div>'
          : `<div class="dbar"><div class="zero"></div><div class="fill ${
              r.delta > 0 ? "pos" : "neg"
            }" style="width:${pct}%"></div></div>`;
        const dCls = r.isControl ? "d-zero" : r.delta < 0 ? "d-better" : "d-worse";
        const dTxt = r.isControl
          ? "control"
          : `${r.delta < 0 ? "−" : "+"}${Math.abs(r.delta).toFixed(4)}`;
        return `<tr data-variant="${r.name}" class="${r.isControl ? "is-control" : ""}${
          r.name === state.variant ? " selected" : ""
        }">
          <td><div class="name">${esc(r.v.label)}</div><span class="axis-tag">${esc(
          (AXES[r.v.axis] || {}).label || r.v.axis
        )}</span></td>
          <td><div style="display:flex;align-items:center;gap:10px">${bar}<span class="num ${dCls}">${dTxt}</span></div></td>
          <td class="num">${fmt(r.v.final_val_loss, 4)}</td>
          <td class="num">${fmt(r.v.val_perplexity, 3)}</td>
          <td class="num">${int(r.v.tokens_per_sec)}</td>
          <td class="num">${int(r.v.params.total)}</td>
        </tr>`;
      })
      .join("");

    $("board").innerHTML = `<table class="board"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    $("board")
      .querySelectorAll("th[data-sort]")
      .forEach((th) =>
        th.addEventListener("click", () => {
          const k = th.dataset.sort;
          if (state.sortKey === k) state.sortDir *= -1;
          else {
            state.sortKey = k;
            state.sortDir = 1;
          }
          renderBoard();
        })
      );
    $("board")
      .querySelectorAll("tbody tr")
      .forEach((tr) =>
        tr.addEventListener("click", () => {
          state.variant = tr.dataset.variant;
          renderAxisTier();
        })
      );
  }

  function renderAxisCharts() {
    const v = DATA.variants[state.variant];
    const base = DATA.variants[BASE];
    const isControl = state.variant === BASE;
    const C = AB.charts.C;

    const series = [
      {
        label: `${base.label} · val`,
        color: C.control,
        steps: base.curve.eval_steps,
        values: base.curve.val_loss,
        dashed: true,
      },
    ];
    if (!isControl) {
      series.push({
        label: `${v.label} · val`,
        color: C.variant,
        steps: v.curve.eval_steps,
        values: v.curve.val_loss,
      });
    }
    AB.charts.renderCurves($("lossCurve"), { series, yLabel: "validation loss" });
    $("lossLegend").innerHTML = series
      .map((s) => `<span><span class="sw line" style="background:${s.color}"></span>${esc(s.label)}</span>`)
      .join("");

    const groups = [
      { label: base.label, color: C.control, counts: base.curve.expert_token_counts.at(-1) },
    ];
    if (!isControl) {
      groups.push({ label: v.label, color: C.variant, counts: v.curve.expert_token_counts.at(-1) });
    }
    AB.charts.renderExpertBars($("expertBars"), { groups });
    const nE = v.config.n_routed_experts;
    $("expertSub").innerHTML =
      `Real dispatched-token counts at step ${base.curve.expert_steps.at(-1)}, per layer. The dashed line is a perfectly even ` +
      `share (1/${nE} = ${((1 / nE) * 100).toFixed(1)}% for ${esc(v.label)}). Hover any bar for its real count.`;
  }

  function renderDeepDive() {
    const v = DATA.variants[state.variant];
    const section = $("deepdiveSection");
    const handler = AB.deepDive[v.axis];
    if (state.variant === BASE || !handler) {
      section.classList.add("hidden");
      return;
    }
    section.classList.remove("hidden");
    $("deepdiveTitle").textContent = handler.title || "How this actually works";
    const panel = $("deepdivePanel");
    panel.innerHTML = "";
    handler(panel, DATA, state.variant);
  }

  function renderAxisTier() {
    renderPicker();
    renderExplainer();
    renderVerdict();
    renderBoard();
    renderAxisCharts();
    renderDeepDive();
  }

  /* ------------------------------------------------------------------ *
   * Preset tier -- the modernized head-to-head (journal 035)
   * ------------------------------------------------------------------ */

  // What each modernized preset changed, and why. Prose only; every number
  // rendered alongside comes from PRESETS' real multi-seed run.
  const PRESET_CHANGES = {
    BASELINE_V2: [
      {
        field: "qk_norm",
        from: "False",
        to: "True",
        why: "A per-head RMSNorm on q and k before RoPE. Without it the attention logits inherit the residual stream's growth with depth; with it their scale is pinned, so softmax cannot saturate into a one-hot spike in the deeper layers.",
        entry: "022",
      },
      {
        field: "router_scoring",
        from: '"softmax"',
        to: '"sigmoid"',
        why: "Softmax shares one denominator across all experts, so scores compete directly. Sigmoid judges each expert independently and only renormalizes across the chosen top-k, giving flatter gate weights and higher routing entropy.",
        entry: "025",
      },
      {
        field: "init_std",
        from: "0.02",
        to: "√(2/(5·d_model)) ≈ 0.0395",
        why: "0.02 is GPT-2's constant, tuned for d_model=768. At d_model=256 that is far too small; the width-aware value restores the intended per-layer signal scale at this width.",
        entry: "023",
      },
    ],
    FINEGRAINED: [
      {
        field: "n_routed_experts",
        from: "8",
        to: "32",
        why: "DeepSeekMoE segmentation: split each expert rather than add capacity.",
        entry: "027",
      },
      {
        field: "d_ff",
        from: "512",
        to: "128",
        why: "Each expert narrows by the same factor the count grew, so total routed FFN parameters are unchanged (32×128 = 8×512).",
        entry: "027",
      },
      {
        field: "top_k",
        from: "2",
        to: "8",
        why: "Per-token active routed FFN width is also held constant (8×128 = 2×512) — what changes is that a token now picks 8 of 32 experts instead of 2 of 8, which is 10 million possible combinations instead of 28.",
        entry: "027",
      },
    ],
  };

  function presetStd(entry) {
    return entry.val_loss_std;
  }

  function renderPresetTier() {
    if (!PRESETS) {
      $("tier-presets").innerHTML =
        '<div class="panel"><p style="margin:0;color:var(--ink-2)">The modernized-preset run is not on disk. ' +
        "Run <code>notebooks/07_modernized_presets.ipynb</code>, then " +
        "<code>uv run python scripts/run_ablation_study.py --smoke --finalize-only</code> to populate this tier.</p></div>";
      return;
    }
    const C = AB.charts.C;
    const names = Object.keys(PRESETS.variants);
    const base = PRESETS.variants[PRESETS.baseline];

    // --- cards ---
    const roles = {
      BASELINE: "control · the original spec",
      BASELINE_V2: "recommended · 3 changes",
      FINEGRAINED: "experimental · + segmentation",
    };
    $("presetCards").innerHTML = names
      .map((n) => {
        const e = PRESETS.variants[n];
        const isBase = n === PRESETS.baseline;
        const d = e.delta_val_loss_vs_baseline;
        // "Real" means the gap clears the combined seed-noise band --
        // exactly the test run_comparison applies (`within_seed_noise`).
        const real = e.within_seed_noise === false;
        const pill = isBase
          ? '<span class="verdict-pill control">the control</span>'
          : `<span class="verdict-pill ${d < 0 ? "better" : "worse"}">${
              d < 0 ? "better" : "worse"
            } by ${Math.abs(d).toFixed(4)}${real ? " · beyond seed noise" : " · within seed noise"}</span>`;
        const changes = PRESET_CHANGES[n];
        const list = changes
          ? `<ul>${changes.map((c) => `<li><code>${c.field}</code> → ${esc(c.to)}</li>`).join("")}</ul>`
          : "<ul><li>Unmodified reference configuration</li></ul>";
        return `<div class="preset-card" style="--accent:${C.preset(n)}">
          <h3 style="color:${C.preset(n)}">${esc(n)}</h3>
          <div class="role">${esc(roles[n] || "")}</div>
          <div class="metric-label">final validation loss, mean of ${PRESETS.seeds.length} seeds</div>
          <div class="big">${e.val_loss_mean.toFixed(4)}<span class="pm"> ± ${presetStd(e).toFixed(4)}</span></div>
          ${pill}
          ${list}
        </div>`;
      })
      .join("");

    // --- change table ---
    const rows = [];
    ["BASELINE_V2", "FINEGRAINED"].forEach((n) => {
      if (!PRESETS.variants[n] || !PRESET_CHANGES[n]) return;
      PRESET_CHANGES[n].forEach((c, i) => {
        rows.push(`<tr>
          ${i === 0 ? `<td rowspan="${PRESET_CHANGES[n].length}" style="font-weight:650;color:${C.preset(n)}">${esc(n)}</td>` : ""}
          <td class="field-cell">${esc(c.field)}</td>
          <td class="field-cell" style="color:var(--ink-3)">${esc(c.from)} <span class="arrow">→</span> ${esc(c.to)}</td>
          <td class="why-cell">${esc(c.why)} <a class="paper-link" style="border:0;margin:0" href="../journey_of_a_model/">entry ${c.entry}</a></td>
        </tr>`);
      });
    });
    $("changeTable").innerHTML = `<table class="change-table"><thead><tr>
        <th>Preset</th><th>Field</th><th>Change</th><th>Why it should help</th>
      </tr></thead><tbody>${rows.join("")}</tbody></table>`;

    // --- curves, with a real min/max band across seeds ---
    const series = names
      .map((n) => {
        const e = PRESETS.variants[n];
        if (!e.curves || !e.curves.length) return null;
        const steps = e.curves[0].eval_steps;
        const perSeed = e.curves.map((c) => c.val_loss);
        const len = Math.min(steps.length, ...perSeed.map((s) => s.length));
        const mean = [];
        const lo = [];
        const hi = [];
        for (let i = 0; i < len; i++) {
          const vals = perSeed.map((s) => s[i]);
          mean.push(vals.reduce((a, b) => a + b, 0) / vals.length);
          lo.push(Math.min(...vals));
          hi.push(Math.max(...vals));
        }
        return {
          label: n,
          color: C.preset(n),
          steps: steps.slice(0, len),
          values: mean,
          band: perSeed.length > 1 ? [lo, hi] : null,
          dashed: n === PRESETS.baseline,
        };
      })
      .filter(Boolean);
    AB.charts.renderCurves($("presetCurve"), { series, yLabel: "validation loss", height: 330 });
    $("presetLegend").innerHTML =
      series
        .map((s) => `<span><span class="sw line" style="background:${s.color}"></span>${esc(s.label)}</span>`)
        .join("") +
      `<span style="color:var(--ink-3)">shaded band = real min–max across seeds ${PRESETS.seeds.join(" and ")}</span>`;

    // --- expert utilization + balance ---
    const groups = names
      .map((n) => {
        const e = PRESETS.variants[n];
        const counts = e.curves && e.curves[0] && e.curves[0].expert_token_counts.at(-1);
        return counts ? { label: n, color: C.preset(n), counts } : null;
      })
      .filter(Boolean);
    if (groups.length) {
      AB.charts.renderExpertBars($("presetExpertBars"), { groups });
      // CV = std(load)/mean(load): 0 when every expert gets an equal share.
      const rowsCv = groups.map((gr) => ({
        label: gr.label,
        color: gr.color,
        cvs: gr.counts.map((layer) => {
          const n = layer.length;
          const mean = layer.reduce((a, b) => a + b, 0) / n;
          if (!mean) return 0;
          const variance = layer.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
          return Math.sqrt(variance) / mean;
        }),
      }));
      AB.charts.renderBalanceDots($("presetBalance"), { rows: rowsCv });
    }

    // --- run provenance ---
    const cfg = PRESETS.config;
    $("presetProvenance").innerHTML =
      `<b>${esc(cfg.corpus)}</b> · ${esc(cfg.tokenizer)} tokenizer, vocab ${cfg.vocab_size} ` +
      `· identical fixed ${int(cfg.fixed_slice_tokens)}-token slice · ${int(cfg.total_tokens)} tokens ` +
      `(${cfg.steps} steps × batch ${cfg.batch_size} × seq ${cfg.seq_len}) · seeds ${PRESETS.seeds.join(", ")} · ${esc(cfg.device)}`;
  }

  /* ------------------------------------------------------------------ *
   * Tier switching + boot
   * ------------------------------------------------------------------ */

  function renderTier() {
    document.querySelectorAll(".tier-tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t.dataset.tier === state.tier))
    );
    $("tier-axes").classList.toggle("hidden", state.tier !== "axes");
    $("tier-presets").classList.toggle("hidden", state.tier !== "presets");
    AB.hideTip();
    if (state.tier === "axes") renderAxisTier();
    else renderPresetTier();
  }

  function boot() {
    const d = DATA.data_config;
    const t = DATA.train_config;
    $("axisProvenance").innerHTML =
      `<b>${esc(d.corpus)}</b> · ${esc(d.tokenizer_type)} tokenizer · identical fixed ${int(
        d.fixed_slice_tokens
      )}-token slice · ${int(t.total_tokens)} tokens (${t.total_steps} steps × batch ${
        t.batch_size
      } × seq ${t.seq_len}) · ${esc(t.device_requested)} · ${
        Object.keys(DATA.variants).length
      } real runs`;

    document.querySelectorAll(".tier-tab").forEach((t2) =>
      t2.addEventListener("click", () => {
        state.tier = t2.dataset.tier;
        renderTier();
      })
    );
    if (!PRESETS) {
      const tab = document.querySelector('.tier-tab[data-tier="presets"]');
      if (tab) tab.querySelector("small").textContent = "not on disk";
    }
    renderTier();

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.tier === "axes") renderAxisCharts();
        else renderPresetTier();
      }, 180);
    });
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
