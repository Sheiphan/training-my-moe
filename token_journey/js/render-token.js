/* token_journey/js/render-token.js
 * Per-step "real numbers" panel + the live expert-highlight sync. Every
 * value here is read from `TJ.store` (the CURRENT variant's slice of
 * window.TRACE_DATA, see store.js) at render time -- nothing is invented
 * client-side except the two "residual add" steps, which are an EXACT
 * sum of two other real captured vectors (addition needs no un-captured
 * weights, so it's still a real, correct number, not an approximation).
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  function chLabel(ch) {
    if (ch === "\n") return "\\n";
    if (ch === " ") return "\u2423";
    return ch;
  }

  function vectorChipsHTML(values, opts = {}) {
    const n = opts.take ?? 8;
    const slice = values.slice(0, n);
    const parts = slice.map(
      (v, i) => `<div class="chip"><span class="chip-idx">${opts.indexLabel ? opts.indexLabel(i) : i}</span><span class="chip-val">${v.toFixed(3)}</span></div>`
    );
    if (values.length > n) parts.push(`<div class="chip chip-more">+${values.length - n} more dims</div>`);
    return `${opts.label ? `<div class="real-label">${opts.label}</div>` : ""}<div class="real-vec">${parts.join("")}</div>`;
  }

  function addVectors(...vecs) {
    const len = vecs[0].length;
    const out = new Array(len).fill(0);
    for (const v of vecs) for (let i = 0; i < len; i++) out[i] += v[i];
    return out;
  }

  function matVec(vector, matrixT) {
    return matrixT[0].map((_, col) => vector.reduce((sum, value, row) => sum + value * matrixT[row][col], 0));
  }

  function attnHeatmapHTML() {
    const T = TJ.store.data();
    const target0 = TJ.store.target0();
    const rows = target0.attn_scores_post_softmax; // (n_heads, seq_len)
    const tokens = T.tokens.map(chLabel);
    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);
    let html = `<div class="real-label">Real attention weights \u2014 this token's row, every head</div><div class="heatmap">`;
    rows.forEach((row, h) => {
      html += `<div class="heat-row"><span class="heat-head">head ${h}</span>`;
      row.forEach((v, j) => {
        html += `<span class="heat-cell" style="background:${color(v)}" title="attends to '${tokens[j]}' with weight ${v.toFixed(3)}">${tokens[j]}</span>`;
      });
      html += `</div>`;
    });
    html += `</div>`;
    return html;
  }

  function topkHTML() {
    const target0 = TJ.store.target0();
    const idx = target0.topk_indices;
    const w = target0.topk_weights;
    let html = `<div class="real-label">This token's real chosen experts</div><div class="topk-list">`;
    idx.forEach((e, i) => {
      html += `<div class="topk-item"><span class="topk-expert">Expert ${e}</span><span class="topk-weight">weight ${w[i].toFixed(3)}</span></div>`;
    });
    html += `</div>`;
    return html;
  }

  function routerHTML() {
    const probs = TJ.store.target0().router_probs;
    const max = Math.max(...probs);
    let html = `<div class="real-label">This token's real router probabilities</div><div class="prob-bars">`;
    probs.forEach((p, i) => {
      const h = Math.max(4, (p / max) * 44);
      html += `<div class="prob-bar-wrap"><div class="prob-bar" style="height:${h}px" title="expert ${i}: ${p.toFixed(3)}"></div><span>E${i} \u00b7 ${p.toFixed(2)}</span></div>`;
    });
    html += `</div>`;
    return html;
  }

  function logitsHTML() {
    const T = TJ.store.data();
    const logits = T.logits[T.target_position];
    const top = logits
      .map((v, i) => ({ v, i }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 6);
    return vectorChipsHTML(
      top.map((t) => t.v),
      { label: "Top real logits (pre-softmax scores)", take: 6, indexLabel: (i) => `'${chLabel(T.vocab[top[i].i])}'` }
    );
  }

  function outputHTML() {
    const T = TJ.store.data();
    const probs = T.probs;
    const top = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);
    let html = `<div class="real-label">Real predicted next-character distribution</div><div class="real-vec">`;
    top.forEach(({ p, i }) => {
      const ch = T.vocab[i];
      const label = ch === "\n" ? "\\n (newline)" : ch === " " ? "\u2423 (space)" : `'${ch}'`;
      html += `<div class="chip"><span class="chip-idx">${label}</span><span class="chip-val">${(p * 100).toFixed(2)}%</span></div>`;
    });
    html += `</div>`;
    return html;
  }

  function rotateSlice(vec, angles) {
    const half = vec.length / 2;
    return vec.map((v, i) => v * Math.cos(angles[i]) + (i < half ? -vec[i + half] : vec[i - half]) * Math.sin(angles[i]));
  }

  const RENDERERS = {
    embed: () => vectorChipsHTML(TJ.store.data().target_embedding, { label: `Real embedding vector for '${chLabel(TJ.store.data().target_token)}' (first 8 of ${TJ.store.cfg().d_model} dims)` }),
    qkv: () => {
      const cfg = TJ.store.cfg();
      const target0 = TJ.store.target0();
      if (cfg.attention_variant === "mla") {
        const W = TJ.store.weights();
        const cKv = matVec(W.attn_norm_output_target, W.w_dkv);
        return vectorChipsHTML(cKv, { label: "Real shared latent c_kv (this is what gets cached)" }) + vectorChipsHTML(target0.q[0].slice(0, TJ.store.cfg().d_model / TJ.store.cfg().n_heads), { label: "Real content query, head 0" });
      }
      return vectorChipsHTML(target0.q[0], { label: "Real Q, head 0 (this token)" }) + vectorChipsHTML(target0.k[0], { label: "Real K, head 0 (this token)" });
    },
    mla_uproj: () => {
      const target0 = TJ.store.target0();
      const cfg = TJ.store.cfg();
      const headDim = cfg.d_model / cfg.n_heads;
      const ropeHeadDim = Math.floor(headDim / 2);
      return (
        vectorChipsHTML(target0.k[0].slice(0, headDim), { label: "Real up-projected content K, head 0" }) +
        vectorChipsHTML(target0.v[0], { label: "Real content V, head 0" })
      );
    },
    mla_rope: () => {
      const target0 = TJ.store.target0();
      return vectorChipsHTML(target0.rope_angles_or_alibi_bias || [], { label: "Real decoupled-RoPE rotation angles for this token's position (radians)" });
    },
    rope: () => vectorChipsHTML(TJ.store.target0().rope_angles_or_alibi_bias || [], { label: "Real RoPE rotation angles for this token's position (radians)" }),
    alibi_bias: () => vectorChipsHTML(TJ.store.target0().rope_angles_or_alibi_bias[0], { label: "Real ALiBi bias, head 0 (this token \u2192 every token)", take: 8 }),
    attn: attnHeatmapHTML,
    attnout: () => vectorChipsHTML(TJ.store.target0().attn_output, { label: "Real attention output (first 8 dims)" }),
    attnresid: () => vectorChipsHTML(addVectors(TJ.store.data().target_embedding, TJ.store.target0().attn_output), { label: "Real sum: embedding + attention output (first 8 dims)" }),
    router: routerHTML,
    topk: topkHTML,
    experts: () => topkHTML() + `<div class="real-note">+ ${TJ.store.cfg().n_shared_experts} shared expert(s), always active for every token.</div>`,
    combine: () => vectorChipsHTML(TJ.store.target0().moe_output, { label: "Real MoE output (first 8 dims)" }),
    moeresid: () =>
      vectorChipsHTML(addVectors(TJ.store.data().target_embedding, TJ.store.target0().attn_output, TJ.store.target0().moe_output), {
        label: "Real sum: embedding + attention output + MoE output (first 8 dims) \u2014 this is Block 1's real final output",
      }),
    finalnorm: () => vectorChipsHTML(TJ.store.data().target_final_norm_output, { label: "Real final-RMSNorm output (first 8 dims)" }),
    logits: logitsHTML,
    output: outputHTML,
  };

  TJ.renderToken = {
    renderReal(stepId, container) {
      const fn = RENDERERS[stepId];
      container.innerHTML = fn ? fn() : "";
    },

    /** Keeps the diagram's expert boxes glowing on the REAL chosen top-k,
     * regardless of which step is currently active (harmless when dimmed). */
    syncExpertHighlight() {
      const cfg = TJ.store.cfg();
      const chosen = new Set(TJ.store.target0().topk_indices);
      for (let i = 0; i < cfg.n_routed_experts; i++) {
        const el = document.getElementById(`expert-${i}`);
        if (el) el.classList.toggle("chosen", chosen.has(i));
      }
    },
  };
})();

