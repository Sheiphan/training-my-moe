/* site/token-journey/js/app.js
 * Main controller: current-step state, camera/narrative/real-panel sync,
 * the right-side step rail, autoplay, and keyboard navigation.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  const state = {
    index: 0,
    playing: false,
    timer: null,
    steps: null,
    switching: false,
  };

  const FAMILIES = [
    {
      key: "classic",
      title: "Original architecture — one axis at a time",
      note:
        "Four real SMOKE_TEST-sized checkpoints (d_model 64, 2 layers) from the ablation study, covering every " +
        "attention × positional pairing that has real trained weights. MLA always applies its own decoupled " +
        "RoPE, so MLA × NoPE/ALiBi do not exist as distinct variants.",
    },
    {
      key: "modern",
      title: "Modernized presets — what the spec changed",
      note:
        "The three full-size presets (d_model 256, 6 layers) trained head to head in journal entry 035. " +
        "BASELINE is the control; BASELINE_V2 adds QK-norm, sigmoid routing and width-aware init; FINEGRAINED " +
        "further segments the experts. Switching between them changes the walkthrough itself, not just the numbers.",
    },
  ];

  function buildPicker() {
    const M = TJ.store.manifest;
    const host = document.getElementById("variant-picker");
    host.innerHTML = FAMILIES.map((fam) => {
      const cards = Object.entries(M.variants)
        .filter(([, m]) => m.family === fam.key)
        .map(([key, m]) => {
          const c = m.config;
          const bits = [
            `${c.n_routed_experts}&times;E / top-${c.top_k}`,
            c.qk_norm ? "QK-norm" : null,
            c.router_scoring === "sigmoid" ? "sigmoid routing" : null,
            c.attention_variant === "mla" ? "MLA" : null,
            c.positional_scheme !== "rope" ? c.positional_scheme.toUpperCase() : null,
          ].filter(Boolean);
          return `<button class="variant-card" data-variant="${key}" aria-pressed="false">
              <span class="vc-name">${m.label}</span>
              <span class="vc-blurb">${m.blurb}</span>
              <span class="vc-tags">${bits.map((b) => `<span class="vc-tag">${b}</span>`).join("")}</span>
              <span class="vc-size">${m.size_kb >= 1024 ? (m.size_kb / 1024).toFixed(1) + " MB" : Math.round(m.size_kb) + " KB"} trace</span>
            </button>`;
        })
        .join("");
      if (!cards) return "";
      return `<section class="variant-family">
          <h3>${fam.title}</h3>
          <p class="vf-note">${fam.note}</p>
          <div class="variant-cards">${cards}</div>
        </section>`;
    }).join("");

    host.querySelectorAll("[data-variant]").forEach((btn) =>
      btn.addEventListener("click", () => switchVariant(btn.dataset.variant))
    );
  }

  function updateVariantButtons() {
    const current = TJ.store.key();
    document.querySelectorAll("#variant-picker [data-variant]").forEach((btn) => {
      const on = btn.dataset.variant === current;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-pressed", String(on));
    });
  }

  /** A compact, always-visible readout of the architecture being walked. */
  function renderArchSummary() {
    const c = TJ.store.cfg();
    const m = TJ.store.meta();
    const rows = [
      ["shape", `d_model ${c.d_model} &middot; ${c.n_layers} layers &middot; ${c.n_heads}Q/${c.n_kv_heads}KV heads`],
      ["attention", c.attention_variant.toUpperCase() + (c.qk_norm ? " + QK-norm" : "")],
      ["position", c.positional_scheme.toUpperCase()],
      ["experts", `${c.n_routed_experts} routed &times; d_ff ${c.d_ff}, top-${c.top_k} &middot; ${c.n_shared_experts} shared`],
      ["routing", `${c.router_scoring} &middot; ${c.moe_balancing.replace("_", "-")}`],
      ["init_std", String(c.init_std)],
    ];
    document.getElementById("arch-summary").innerHTML =
      `<div class="as-title">${m.label}</div>` +
      rows
        .map(([k, v]) => `<div class="as-item"><span class="as-k">${k}</span><span class="as-v">${v}</span></div>`)
        .join("");
    document.getElementById("variant-note").innerHTML =
      `Trace captured from <code>${m.checkpoint}</code>` +
      (m.has_expert_weights
        ? ". The expert matrix-multiplication deep-dive is available for this variant."
        : ". This variant ships activations but not its per-expert weight matrices (they would add tens of MB " +
          "to explain a step the smaller variants already cover), so the expert matmul deep-dive is not shown for it.");
  }

  function setLoading(on, text) {
    const veil = document.getElementById("diagram-loading");
    if (text) document.getElementById("loading-text").textContent = text;
    veil.hidden = !on;
  }

  async function switchVariant(key) {
    if (state.switching || key === TJ.store.key()) return;
    stopAutoplay();
    state.switching = true;
    const meta = TJ.store.meta(key);
    if (!TJ.store.isLoaded(key)) {
      setLoading(true, `Loading ${meta.label} trace (${Math.round(meta.size_kb)} KB)…`);
    }
    try {
      await TJ.store.set(key);
    } catch (err) {
      setLoading(true, `Could not load ${meta.label}: ${err.message}`);
      state.switching = false;
      return;
    }
    setLoading(false);
    const data = TJ.store.data();
    state.steps = TJ.buildSteps(data);
    TJ.diagram.mount("#diagram", data);
    TJ.camera.init("#diagram");
    updateVariantButtons();
    renderArchSummary();
    buildRail();
    goToStep(Math.min(state.index, state.steps.length - 1));
    state.switching = false;
  }

  function chapterGroups() {
    const groups = [];
    for (const step of state.steps) {
      let g = groups[groups.length - 1];
      if (!g || g.chapter !== step.chapter) {
        g = { chapter: step.chapter, steps: [] };
        groups.push(g);
      }
      g.steps.push(step);
    }
    return groups;
  }

  function buildRail() {
    const rail = document.getElementById("rail-steps");
    const groups = chapterGroups();
    let globalIndex = 0;
    const html = [];
    for (const g of groups) {
      html.push(`<div class="rail-chapter">${g.chapter}</div>`);
      for (const step of g.steps) {
        html.push(`<button class="rail-step" data-index="${globalIndex}"><span class="rail-num">${globalIndex + 1}</span><span class="rail-title">${step.title}</span></button>`);
        globalIndex++;
      }
    }
    rail.innerHTML = html.join("");
    rail.querySelectorAll(".rail-step").forEach((btn) => {
      btn.addEventListener("click", () => {
        stopAutoplay();
        goToStep(parseInt(btn.dataset.index, 10));
      });
    });
  }

  // Deep-dives that re-run a real matmul through the expert FFNs, and so
  // need `weights.experts` -- which the large modern variants deliberately
  // do not ship (see scripts/export_trace.py). Gating here rather than
  // inside each dive keeps the requirement in one visible place.
  const NEEDS_EXPERT_WEIGHTS = new Set(["moe-experts", "moe-combine"]);

  function hasDeepDive(step) {
    if (!TJ.deepDive[step.id]) return false;
    if (NEEDS_EXPERT_WEIGHTS.has(step.id) && !TJ.store.weights().has_expert_weights) return false;
    return true;
  }

  function renderNarrative(step) {
    const card = document.querySelector(".narrative-card");
    card.classList.remove("step-enter");
    document.getElementById("step-chapter").textContent = step.chapter;
    document.getElementById("step-title").textContent = step.title;
    document.getElementById("step-narrative").textContent = step.narrative;

    const formulaEl = document.getElementById("step-formula");
    if (step.formula) {
      formulaEl.style.display = "";
      katex.render(step.formula, formulaEl, { throwOnError: false, displayMode: true });
    } else {
      formulaEl.style.display = "none";
      formulaEl.innerHTML = "";
    }

    const realEl = document.getElementById("step-real");
    if (step.real) {
      realEl.style.display = "";
      TJ.renderToken.renderReal(step.real, realEl);
    } else {
      realEl.style.display = "none";
      realEl.innerHTML = "";
    }

    TJ.matrixViz.stopAll();
    const panel = document.getElementById("deepdive-panel");
    const btn = document.getElementById("deepdive-btn");
    panel.style.display = "none";
    panel.innerHTML = "";
    btn.setAttribute("aria-expanded", "false");
    if (hasDeepDive(step)) {
      btn.style.display = "";
      panel.style.display = "";
      TJ.deepDive[step.id](panel);
      btn.textContent = "\u2191 Close math view";
      btn.setAttribute("aria-expanded", "true");
    } else {
      btn.style.display = "none";
    }
    void card.offsetWidth;
    card.classList.add("step-enter");
  }

  function updateRailActive() {
    document.querySelectorAll(".rail-step").forEach((btn) => {
      const active = parseInt(btn.dataset.index, 10) === state.index;
      btn.classList.toggle("active", active);
      if (active) btn.setAttribute("aria-current", "step");
      else btn.removeAttribute("aria-current");
    });
    const active = document.querySelector(".rail-step.active");
    if (active) active.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    document.getElementById("progress-label").textContent = `Step ${state.index + 1} of ${state.steps.length}`;
    document.getElementById("progress-fill").style.width = `${((state.index + 1) / state.steps.length) * 100}%`;
  }

  function goToStep(i) {
    state.index = Math.max(0, Math.min(state.steps.length - 1, i));
    const step = state.steps[state.index];
    TJ.camera.goTo(step.focus);
    TJ.renderToken.syncExpertHighlight();
    renderNarrative(step);
    updateRailActive();
    document.getElementById("prev-btn").disabled = state.index === 0;
    document.getElementById("next-btn").disabled = state.index === state.steps.length - 1;
  }

  function next() {
    if (state.index >= state.steps.length - 1) {
      stopAutoplay();
      return;
    }
    goToStep(state.index + 1);
  }
  function prev() {
    goToStep(state.index - 1);
  }

  function startAutoplay() {
    if (state.index >= state.steps.length - 1) goToStep(0);
    state.playing = true;
    document.getElementById("play-btn").textContent = "\u23f8 Pause";
    document.getElementById("play-btn").classList.add("playing");
    state.timer = setInterval(() => {
      if (state.index >= state.steps.length - 1) {
        stopAutoplay();
        return;
      }
      next();
    }, 5000);
  }
  function stopAutoplay() {
    state.playing = false;
    document.getElementById("play-btn").textContent = "\u25b6 Play";
    document.getElementById("play-btn").classList.remove("playing");
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }
  function toggleAutoplay() {
    if (state.playing) stopAutoplay();
    else startAutoplay();
  }

  function wireControls() {
    document.getElementById("next-btn").addEventListener("click", () => {
      stopAutoplay();
      next();
    });
    document.getElementById("prev-btn").addEventListener("click", () => {
      stopAutoplay();
      prev();
    });
    document.getElementById("play-btn").addEventListener("click", toggleAutoplay);
    document.getElementById("deepdive-btn").addEventListener("click", () => {
      const panel = document.getElementById("deepdive-panel");
      const button = document.getElementById("deepdive-btn");
      const step = state.steps[state.index];
      const showing = panel.style.display !== "none";
      if (showing) {
        TJ.matrixViz.stopAll();
        panel.style.display = "none";
        panel.innerHTML = "";
        button.textContent = "\uD83D\uDD0D Zoom into the math";
        button.setAttribute("aria-expanded", "false");
      } else {
        stopAutoplay();
        panel.style.display = "";
        TJ.deepDive[step.id](panel);
        button.textContent = "\u2191 Close math view";
        button.setAttribute("aria-expanded", "true");
        panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });

    window.addEventListener("keydown", (e) => {
      if (e.target.closest("button, a, input, select, textarea") || e.target.isContentEditable) return;
      if (e.key === "ArrowRight" || e.key === " ") {
        e.preventDefault();
        stopAutoplay();
        next();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stopAutoplay();
        prev();
      }
    });
  }

  /** `?variant=<key>` lets architecture.html hand a configuration straight
   * to this page, so building a shape on the schematic and then walking a
   * real token through it is one click rather than a re-selection. */
  function requestedVariant() {
    const wanted = new URLSearchParams(window.location.search).get("variant");
    return wanted && TJ.store.manifest.variants[wanted] ? wanted : null;
  }

  async function init() {
    buildPicker();
    wireControls();
    const first = requestedVariant() || TJ.store.key();
    setLoading(true, `Loading ${TJ.store.meta(first).label} trace…`);
    try {
      await TJ.store.set(first);
    } catch (err) {
      setLoading(true, `Could not load the default trace: ${err.message}`);
      return;
    }
    setLoading(false);
    const data = TJ.store.data();
    state.steps = TJ.buildSteps(data);
    TJ.diagram.mount("#diagram", data);
    TJ.camera.init("#diagram");
    updateVariantButtons();
    renderArchSummary();
    buildRail();
    goToStep(0);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
