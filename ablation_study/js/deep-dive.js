/* ablation_study/js/deep-dive.js
 * Axis-specific "why did this change the numbers" explainers. Each entry
 * is a function(container, data) that renders real KaTeX formulas plus
 * real numbers already present in window.ABLATION_DATA -- nothing here
 * is fabricated or re-derived beyond simple, verifiable arithmetic.
 */
(function () {
  "use strict";
  const AB = (window.AB = window.AB || {});

  function tex(el, latex, displayMode) {
    if (!el) return;
    katex.render(latex, el, { throwOnError: false, displayMode: !!displayMode });
  }

  function noteEl(container, latex) {
    const el = document.createElement("div");
    el.className = "mm-note";
    container.appendChild(el);
    tex(el, latex, false);
  }

  AB.deepDive = {
    moe_balancing(container, data) {
      const demo = data.loss_free_demo;
      const rounds = [0, 10, 39];
      const colors = ["#ef4444", "#3b82f6", "#22c55e", "#eab308"];
      const cards = rounds
        .map((round, stage) => {
          const bias = demo.snapshots[String(round)];
          const bars = bias
            .map((v, expert) => {
              const height = Math.max(4, (v + 6) * 6);
              return `<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
                <div style="width:20px;height:${height}px;background:${colors[expert]};border-radius:2px 2px 0 0;"></div>
                <span style="font-size:0.68rem;color:#94a3b8;">E${expert}</span>
                <span style="font-size:0.66rem;color:#cbd5e1;font-family:'SF Mono',monospace;">${v.toFixed(2)}</span>
              </div>`;
            })
            .join("");
          return `<div class="bias-round" style="--stage:${stage}">
            <span class="round-label">round ${round}</span>
            <div style="display:flex;gap:8px;align-items:flex-end;height:70px;">${bars}</div>
          </div>`;
        })
        .join("");

      container.innerHTML = `<div class="mm-formula-line"></div><div class="bias-rounds">${cards}</div>`;
      tex(
        container.querySelector(".mm-formula-line"),
        String.raw`\text{selection\_score}_i = \text{logit}_i + b_i,\qquad b_i \leftarrow b_i + \eta\cdot\operatorname{sign}(\bar c - c_i)`,
        true
      );
      noteEl(
        container,
        String.raw`\text{Fixed skewed logits } [${demo.initial_logits.join(", ")}] \text{ would let expert 0 win every round. } ` +
          String.raw`\text{Its real win rate falls from } ${(demo.expert0_win_rate_early * 100).toFixed(0)}\% \text{ (rounds 0-4) to } ` +
          String.raw`${(demo.expert0_win_rate_late * 100).toFixed(0)}\% \text{ (rounds 35-39) purely from the bias nudge above -- no extra loss term, matching this run's real MOE\_LOSSFREE variant.}`
      );
    },

    attention(container, data) {
      const cfg = data.variants.ATTN_MLA.config;
      const headDim = cfg.d_model / cfg.n_heads;
      const ropeHeadDim = Math.floor(headDim / 2);
      const latentDim = cfg.mla_latent_dim;
      const gqaCache = cfg.n_kv_heads * headDim * 2; // K and V per kv-head
      const mlaCache = latentDim + ropeHeadDim;
      container.innerHTML = `<div class="mm-formula-line"></div>`;
      tex(
        container.querySelector(".mm-formula-line"),
        String.raw`\text{KV-cache/token: GQA} = 2\,n_{kv}d_h = ${gqaCache},\qquad \text{MLA} = d_{latent}+d_h^{R} = ${latentDim}+${ropeHeadDim}=${mlaCache}`,
        true
      );
      noteEl(
        container,
        String.raw`\text{MLA's real cache here is } ${((mlaCache / gqaCache) * 100).toFixed(1)}\% \text{ the size of GQA's, at a real val\_loss of } ` +
          String.raw`${data.variants.ATTN_MLA.final_val_loss.toFixed(4)} \text{ vs baseline's } ${data.variants.SMOKE_TEST.final_val_loss.toFixed(4)}.`
      );
    },
  };
})();
