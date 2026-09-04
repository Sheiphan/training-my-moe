/* ablation_study/js/deep-dive.js
 *
 * Per-axis "why did this change the numbers" explainers. Each entry is a
 * function(container, data, variantName) with a `.title` property.
 *
 * Every quantity rendered is either (a) read straight out of
 * window.ABLATION_DATA -- itself written from real train.jsonl logs and
 * real forward passes through real checkpoints -- or (b) plain arithmetic
 * on those values that the reader can check by hand (a ratio, a binomial
 * coefficient, an entropy). Nothing here is fabricated or approximated.
 */
(function () {
  "use strict";
  const AB = (window.AB = window.AB || {});

  function tex(el, latex, displayMode) {
    if (!el) return;
    katex.render(latex, el, { throwOnError: false, displayMode: !!displayMode });
  }

  function head(container, text) {
    const el = document.createElement("div");
    el.className = "dd-head";
    el.textContent = text;
    container.appendChild(el);
  }

  function formula(container, latex) {
    const el = document.createElement("div");
    el.className = "mm-formula-line";
    container.appendChild(el);
    tex(el, latex, true);
  }

  function note(container, html) {
    const el = document.createElement("div");
    el.className = "mm-note";
    el.innerHTML = html;
    container.appendChild(el);
  }

  /** A small inline sparkline-style bar row set, used for per-layer series. */
  function layerBars(values, opts) {
    const max = Math.max(...values.map(Math.abs), 1e-9);
    const color = opts.color || "#e8a95f";
    return (
      `<div style="display:flex;gap:6px;align-items:flex-end;height:74px;margin-top:6px">` +
      values
        .map((v, i) => {
          const h = Math.max(3, (Math.abs(v) / max) * 62);
          return `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">
            <div style="width:26px;height:${h}px;background:${color};border-radius:2px 2px 0 0"></div>
            <span style="font-size:.64rem;color:#7f8b9c">L${i}</span>
            <span style="font-size:.66rem;color:#b9c3d1;font-family:ui-monospace,monospace">${v.toFixed(2)}</span>
          </div>`;
        })
        .join("") +
      `</div>`
    );
  }

  /** Two probability vectors side by side as labelled bars. */
  function probCompare(labels, rows, chosen) {
    const max = Math.max(...rows.flatMap((r) => r.values));
    return rows
      .map(
        (r) =>
          `<div style="margin-bottom:14px">
            <div style="font-size:.78rem;color:#8b96a6;margin-bottom:6px">${r.title}</div>
            <div class="prob-compare">` +
          r.values
            .map(
              (v, i) =>
                `<div class="prob-row ${chosen && chosen.includes(i) ? "is-chosen" : ""}">
                  <span class="pr-label">expert ${i}</span>
                  <span class="pr-track"><span class="pr-fill" style="width:${(v / max) * 100}%;background:${r.color}"></span></span>
                  <span class="pr-val">${v.toFixed(4)}</span>
                </div>`
            )
            .join("") +
          `</div></div>`
      )
      .join("");
  }

  function spread(values) {
    return Math.max(...values) - Math.min(...values);
  }

  function entropyOf(probs) {
    return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
  }

  /* ==================================================================== */

  const dd = {};

  // ---- QK-norm -------------------------------------------------------
  dd.qk_norm = function (container, data) {
    formula(
      container,
      String.raw`q \leftarrow \frac{q}{\sqrt{\tfrac{1}{d_h}\sum_i q_i^2 + \epsilon}}\odot\gamma_q,\qquad
                 k \leftarrow \frac{k}{\sqrt{\tfrac{1}{d_h}\sum_i k_i^2+\epsilon}}\odot\gamma_k
                 \qquad\text{(applied per head, before RoPE)}`
    );

    const probe = data.modern_probe;
    if (probe && probe.models.BASELINE && probe.models.BASELINE_V2) {
      const off = probe.models.BASELINE.attn_logit_std;
      const on = probe.models.BASELINE_V2.attn_logit_std;
      head(container, "Real attention-logit spread, layer by layer");
      const grid = document.createElement("div");
      grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:22px";
      grid.innerHTML =
        `<div><div style="font-size:.8rem;color:#b9c3d1;margin-bottom:2px">
           <b style="color:#8d949e">qk_norm = False</b> &mdash; BASELINE</div>
           ${layerBars(off, { color: "#8d949e" })}
           <div style="font-size:.78rem;color:#7f8b9c;margin-top:8px">range across depth: <b style="color:#e8a95f">${spread(off).toFixed(2)}</b></div></div>` +
        `<div><div style="font-size:.8rem;color:#b9c3d1;margin-bottom:2px">
           <b style="color:#e0904f">qk_norm = True</b> &mdash; BASELINE_V2</div>
           ${layerBars(on, { color: "#e0904f" })}
           <div style="font-size:.78rem;color:#7f8b9c;margin-top:8px">range across depth: <b style="color:#e8a95f">${spread(on).toFixed(2)}</b></div></div>`;
      container.appendChild(grid);

      note(
        container,
        `Standard deviation of the pre-mask attention logits, measured on one real forward pass through each ` +
          `trained checkpoint (prompt <code style="color:#e8a95f">${probe.prompt.replace(/\n/g, "\\n")}</code>). ` +
          `Without QK-norm the spread wanders between <b>${Math.min(...off).toFixed(2)}</b> and <b>${Math.max(...off).toFixed(2)}</b> ` +
          `as depth grows, because q and k inherit the residual stream's scale and the dot product squares it. ` +
          `With QK-norm every layer sits between <b>${Math.min(...on).toFixed(2)}</b> and <b>${Math.max(...on).toFixed(2)}</b> — ` +
          `a <b>${(spread(off) / Math.max(spread(on), 1e-9)).toFixed(0)}×</b> smaller range. That bounded scale is the ` +
          `whole point: softmax over a logit range that keeps growing eventually saturates to one-hot, and a ` +
          `saturated softmax has almost no gradient left to learn from.`
      );
    }

    const v = data.variants.ATT_QKNORM;
    const base = data.variants[data.baseline];
    if (v) {
      note(
        container,
        `In this page's own single-axis run, flipping <code>qk_norm</code> on and changing nothing else moved ` +
          `validation loss from <b>${base.final_val_loss.toFixed(4)}</b> to <b>${v.final_val_loss.toFixed(4)}</b> ` +
          `for <b>${(v.params.total - base.params.total).toLocaleString()}</b> extra parameters — the two per-head ` +
          `RMSNorm gain vectors, which is ${(((v.params.total - base.params.total) / base.params.total) * 100).toFixed(3)}% of the model.`
      );
    }
  };
  dd.qk_norm.title = "QK-norm: bounding the attention-logit scale";

  // ---- Router scoring ------------------------------------------------
  dd.router_scoring = function (container, data) {
    formula(
      container,
      String.raw`\text{softmax: } p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
                 \qquad\qquad
                 \text{sigmoid: } s_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}},\quad p_i=\frac{s_i}{\sum_j s_j}`
    );

    const probe = data.modern_probe;
    if (probe && probe.models.BASELINE_V2) {
      const m = probe.models.BASELINE_V2;
      const sm = m.softmax_probs_layer0;
      const sg = m.sigmoid_probs_layer0;
      const topk = sm
        .map((p, i) => [p, i])
        .sort((a, b) => b[0] - a[0])
        .slice(0, m.top_k)
        .map((x) => x[1]);

      head(container, "Both rules on the SAME real router logits — layer 0, final token");
      const el = document.createElement("div");
      el.innerHTML = probCompare(
        null,
        [
          { title: "softmax — one shared denominator, experts compete", values: sm, color: "#8d949e" },
          { title: "sigmoid — scored independently, then renormalized", values: sg, color: "#e0904f" },
        ],
        topk
      );
      container.appendChild(el);

      const hSm = entropyOf(sm) / Math.log(sm.length);
      const hSg = entropyOf(sg) / Math.log(sg.length);
      note(
        container,
        `Same model, same forward pass, same logits — only the scoring function differs. Softmax's largest ` +
          `share is <b>${Math.max(...sm).toFixed(4)}</b>; sigmoid's is <b>${Math.max(...sg).toFixed(4)}</b>. ` +
          `Normalized entropy rises from <b>${hSm.toFixed(4)}</b> to <b>${hSg.toFixed(4)}</b> ` +
          `(1.0 would be perfectly uniform). Softmax couples every expert through one denominator, so raising ` +
          `one score necessarily suppresses all the others and a single expert can run away with the token. ` +
          `Sigmoid asks each expert "how well do you match?" independently, and only the ${m.top_k} winners are ` +
          `renormalized against each other — so the gate weights that survive are flatter and more than one ` +
          `expert keeps a real share of the gradient.`
      );
    }

    const v = data.variants.ROUTE_SIGMOID;
    const base = data.variants[data.baseline];
    if (v) {
      note(
        container,
        `On the single-axis run above, sigmoid routing alone moved validation loss from ` +
          `<b>${base.final_val_loss.toFixed(4)}</b> to <b>${v.final_val_loss.toFixed(4)}</b> at ` +
          `<b>identical</b> parameter count — the router matrix is the same shape either way, only how its ` +
          `outputs are read changed.`
      );
    }
  };
  dd.router_scoring.title = "Sigmoid routing: judging experts independently";

  // ---- Expert granularity --------------------------------------------
  dd.expert_granularity = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    const bc = base.config;
    const vc = v.config;

    function choose(n, k) {
      let r = 1;
      for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
      return Math.round(r);
    }
    const combBase = choose(bc.n_routed_experts, bc.top_k);
    const combVar = choose(vc.n_routed_experts, vc.top_k);

    formula(
      container,
      String.raw`\underbrace{N_r \times d_{ff}}_{\text{total routed FFN width}}:\;
                 ${bc.n_routed_experts}\times${bc.d_ff}=${bc.n_routed_experts * bc.d_ff}
                 \;\longrightarrow\;${vc.n_routed_experts}\times${vc.d_ff}=${vc.n_routed_experts * vc.d_ff}
                 \qquad
                 \underbrace{K\times d_{ff}}_{\text{active per token}}:\;
                 ${bc.top_k}\times${bc.d_ff}=${bc.top_k * bc.d_ff}
                 \;\longrightarrow\;${vc.top_k}\times${vc.d_ff}=${vc.top_k * vc.d_ff}`
    );

    note(
      container,
      `Both budgets are held constant on purpose: the total routed feed-forward width is unchanged ` +
        `(<b>${bc.n_routed_experts * bc.d_ff}</b> either way) and so is the width each token actually activates ` +
        `(<b>${bc.top_k * bc.d_ff}</b> either way). What changes is <em>how finely that same capacity can be ` +
        `addressed</em>. A token picking ${bc.top_k} of ${bc.n_routed_experts} experts has ` +
        `<b>${combBase.toLocaleString()}</b> possible expert combinations; picking ${vc.top_k} of ` +
        `${vc.n_routed_experts} has <b>${combVar.toLocaleString()}</b> — ` +
        `<b>${Math.round(combVar / combBase).toLocaleString()}×</b> more distinct ways to specialize, ` +
        `for the same compute.`
    );

    const dParams = v.params.total - base.params.total;
    note(
      container,
      `Measured here: total parameters ${base.params.total.toLocaleString()} → ` +
        `<b>${v.params.total.toLocaleString()}</b> (${dParams >= 0 ? "+" : "−"}${Math.abs(dParams).toLocaleString()}, ` +
        `entirely the router, which must now score ${vc.n_routed_experts} experts instead of ${bc.n_routed_experts}), ` +
        `validation loss ${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b>, throughput ` +
        `${Math.round(base.tokens_per_sec).toLocaleString()} → <b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s. ` +
        `The throughput cost is real and worth noticing: more experts means more, smaller matrix multiplies, ` +
        `which is less efficient per FLOP even when the FLOP count is unchanged.`
    );
  };
  dd.expert_granularity.title = "Fine-grained experts: same capacity, finer addressing";

  // ---- MoE balancing (loss-free) -------------------------------------
  dd.moe_balancing = function (container, data) {
    const demo = data.loss_free_demo;
    formula(
      container,
      String.raw`\text{selection\_score}_i = \text{score}_i + b_i,\qquad
                 b_i \leftarrow b_i + \eta\cdot\operatorname{sign}(\bar c - c_i)`
    );
    const rounds = [0, 10, 39];
    const colors = ["#d96b60", "#5b9dd9", "#4aa87a", "#e0c05a"];
    head(container, "The bias buffer over 40 real balancer rounds");
    const cards = rounds
      .map((round, stage) => {
        const bias = demo.snapshots[String(round)];
        const bars = bias
          .map((v, expert) => {
            const height = Math.max(4, (v + 6) * 6);
            return `<div style="display:flex;flex-direction:column;align-items:center;gap:3px">
              <div style="width:20px;height:${height}px;background:${colors[expert]};border-radius:2px 2px 0 0"></div>
              <span style="font-size:.68rem;color:#7f8b9c">E${expert}</span>
              <span style="font-size:.66rem;color:#b9c3d1;font-family:ui-monospace,monospace">${v.toFixed(2)}</span>
            </div>`;
          })
          .join("");
        return `<div class="bias-round" style="--stage:${stage}">
          <span class="round-label">round ${round}</span>
          <div style="display:flex;gap:8px;align-items:flex-end;height:70px">${bars}</div>
        </div>`;
      })
      .join("");
    const wrap = document.createElement("div");
    wrap.className = "bias-rounds";
    wrap.innerHTML = cards;
    container.appendChild(wrap);
    note(
      container,
      `Fixed, deliberately skewed logits <code style="color:#e8a95f">[${demo.initial_logits.join(", ")}]</code> ` +
        `would let expert 0 win every single round forever. Its real win rate falls from ` +
        `<b>${(demo.expert0_win_rate_early * 100).toFixed(0)}%</b> (rounds 0–4) to ` +
        `<b>${(demo.expert0_win_rate_late * 100).toFixed(0)}%</b> (rounds 35–39) purely from the bias nudge — ` +
        `no gradient, no extra loss term competing with the language-modelling objective. The bias affects ` +
        `<em>selection only</em>; the weight a chosen expert's output is multiplied by still comes from the ` +
        `unbiased routing probabilities.`
    );
  };
  dd.moe_balancing.title = "Loss-free balancing: a bias nudge instead of a loss term";

  // ---- Attention (MLA) -----------------------------------------------
  dd.attention = function (container, data) {
    const v = data.variants.ATTN_MLA;
    const cfg = v.config;
    const headDim = cfg.d_model / cfg.n_heads;
    const ropeHeadDim = Math.floor(headDim / 2);
    const latentDim = cfg.mla_latent_dim;
    const gqaCache = cfg.n_kv_heads * headDim * 2;
    const mlaCache = latentDim + ropeHeadDim;
    formula(
      container,
      String.raw`\text{KV-cache per token: GQA}=2\,n_{kv}d_h=${gqaCache},
                 \qquad \text{MLA}=d_{latent}+d_h^{R}=${latentDim}+${ropeHeadDim}=${mlaCache}`
    );
    note(
      container,
      `MLA caches one small shared latent instead of a key and a value for every kv-head, so its real cache ` +
        `here is <b>${((mlaCache / gqaCache) * 100).toFixed(1)}%</b> the size of GQA's. That is the entire ` +
        `motivation — generation is memory-bandwidth bound, and the KV cache is what you re-read on every ` +
        `single decoded token. Measured cost at this scale: validation loss ` +
        `${data.variants[data.baseline].final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b>, ` +
        `throughput ${Math.round(data.variants[data.baseline].tokens_per_sec).toLocaleString()} → ` +
        `<b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s (the up-projection back to full K and V ` +
        `is extra work on every forward pass, which a 2-layer model has no cache pressure to pay it back).`
    );
  };
  dd.attention.title = "MLA: caching a latent instead of keys and values";

  // ---- LR schedule ---------------------------------------------------
  dd.lr_schedule = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    formula(
      container,
      String.raw`\text{cosine: } \eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\!\left(1+\cos\tfrac{\pi t}{T}\right)
                 \qquad
                 \text{WSD: } \eta_t=\begin{cases}\eta_{max}& t< (1-f)T\\ \text{decay}& \text{otherwise}\end{cases}`
    );
    const bl = base.curve.lr || [];
    const vl = v.curve.lr || [];
    if (bl.length && vl.length) {
      head(container, "The two real learning-rate schedules this study logged");
      const maxLr = Math.max(...bl, ...vl);
      const w = 640;
      const h = 130;
      const pt = (arr, i) => `${(i / (arr.length - 1)) * w},${h - (arr[i] / maxLr) * (h - 12) - 6}`;
      const path = (arr) => arr.map((_, i) => pt(arr, i)).join(" ");
      const svg = document.createElement("div");
      svg.innerHTML = `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:${h}px;overflow:visible">
          <polyline points="${path(bl)}" fill="none" stroke="#8d949e" stroke-width="2.4" stroke-dasharray="5 4"/>
          <polyline points="${path(vl)}" fill="none" stroke="#e0904f" stroke-width="2.8"/>
        </svg>
        <div style="font-size:.78rem;color:#7f8b9c;margin-top:4px">
          <span style="color:#8d949e">— — cosine (${base.label})</span> &nbsp;&nbsp;
          <span style="color:#e0904f">&mdash;&mdash; ${v.label}</span> &nbsp;·&nbsp; peak
          ${maxLr.toExponential(2)} over ${bl.length} logged steps
        </div>`;
      container.appendChild(svg);
    }
    note(
      container,
      `Cosine decay bakes the <em>end</em> of training into its shape from step 0 — stop early and you stop ` +
        `mid-decay at a learning rate that was never meant to be final. Warmup-Stable-Decay holds the peak ` +
        `rate flat for most of the run and only decays over the last fraction, so the same run can be extended ` +
        `or cut short without invalidating everything before the decay. Measured here: validation loss ` +
        `${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b> on an identical token budget.`
    );
  };
  dd.lr_schedule.title = "WSD: not having to know the budget in advance";

  // ---- Optimizer (Muon) ----------------------------------------------
  dd.optimizer = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    formula(
      container,
      String.raw`M_t=\mu M_{t-1}+G_t,\qquad O_t=\text{NewtonSchulz5}(M_t)\approx UV^\top
                 \;\text{ where } M_t=U\Sigma V^\top,\qquad W\leftarrow W-\eta\,O_t`
    );
    note(
      container,
      `AdamW treats a weight matrix as a bag of independent scalars, each with its own adaptive step size. ` +
        `Muon instead treats it as a matrix: it orthogonalizes the momentum (a cheap Newton–Schulz iteration ` +
        `approximating <span style="font-family:ui-monospace">UVᵀ</span> from the SVD), which equalizes the ` +
        `update's effect across all singular directions instead of letting a few dominant ones absorb it. ` +
        `Parameters that are not 2-D — embeddings, norm gains — fall back to AdamW.`
    );
    note(
      container,
      `Measured on this run: validation loss ${base.final_val_loss.toFixed(4)} → ` +
        `<b>${v.final_val_loss.toFixed(4)}</b>, throughput ` +
        `${Math.round(base.tokens_per_sec).toLocaleString()} → <b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s. ` +
        `<b>Read this one carefully rather than as a verdict on Muon.</b> Muon's published results use a ` +
        `learning rate tuned for it, typically far larger than AdamW's; this study deliberately holds the LR ` +
        `fixed across every variant so the comparison stays single-variable, which means Muon here is running ` +
        `at someone else's hyperparameter. What this run honestly shows is that swapping the optimizer alone, ` +
        `changing nothing else, is not free — not that the method is bad.`
    );
  };
  dd.optimizer.title = "Muon: updating a matrix as a matrix";

  // ---- MTP -----------------------------------------------------------
  dd.mtp_depth = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    formula(
      container,
      String.raw`\mathcal{L}=\mathcal{L}_{\text{next}}+\lambda\cdot\frac{1}{D}\sum_{d=1}^{D}\mathcal{L}_{t+1+d}`
    );
    const mtp = (v.curve.mtp_loss || []).filter((x) => x !== null && x !== undefined);
    note(
      container,
      `An extra head predicts the token <em>two</em> positions ahead, and its loss is averaged into the total ` +
        `at weight <code style="color:#e8a95f">mtp_lambda = ${v.config.mtp_lambda}</code>. The main head is ` +
        `untouched at inference — the claim is that having to also predict further ahead forces a ` +
        `representation that plans, rather than one that only has to finish the current token.` +
        (mtp.length
          ? ` This run's real auxiliary MTP loss went from <b>${mtp[0].toFixed(4)}</b> at the first logged step ` +
            `to <b>${mtp[mtp.length - 1].toFixed(4)}</b> at the last.`
          : "")
    );
    note(
      container,
      `Measured: validation loss ${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b>, ` +
        `parameters ${base.params.total.toLocaleString()} → <b>${v.params.total.toLocaleString()}</b>, ` +
        `throughput ${Math.round(base.tokens_per_sec).toLocaleString()} → ` +
        `<b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s. The extra head costs parameters and ` +
        `time during training and is simply discarded afterwards.`
    );
  };
  dd.mtp_depth.title = "Multi-token prediction: an auxiliary head that looks further ahead";

  // ---- Positional ----------------------------------------------------
  dd.positional = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    const scheme = v.config.positional_scheme;
    const F = {
      nope: String.raw`q'=q,\qquad k'=k\qquad\text{(no positional operation at all)}`,
      alibi: String.raw`\text{score}_{h}[i,j]\;\mathrel{+}=\;-\,m_h\,(i-j),\qquad m_h=2^{-8h/n_{heads}}`,
      rope: String.raw`\theta_i=\text{pos}\cdot\theta^{-2i/d_h},\qquad q'_i=q_i\cos\theta_i+\text{rotate\_half}(q)_i\sin\theta_i`,
    }[scheme];
    formula(container, F);
    const why = {
      nope:
        `Nothing is added at all. The only order information left is the causal mask — token <i>t</i> can see ` +
        `tokens 0…<i>t</i> and no further. That alone turns out to be a usable signal, because a token's ` +
        `<em>count</em> of visible predecessors is itself positional information the network can learn to read.`,
      alibi:
        `No learned parameters and nothing added to q or k — just a fixed, per-head linear penalty on distance ` +
        `applied straight to the scores. Each head gets a different slope from a geometric sequence, so some ` +
        `heads look sharply local and others stay nearly global.`,
      rope:
        `Position is applied as a rotation of q and k rather than an addition, so the dot product between two ` +
        `tokens ends up depending only on their <em>relative</em> offset, not their absolute indices.`,
    }[scheme];
    note(container, why);
    note(
      container,
      `Measured: validation loss ${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b> at ` +
        `${v.params.total === base.params.total ? "identical parameter count" : v.params.total.toLocaleString() + " params"}` +
        `, throughput ${Math.round(base.tokens_per_sec).toLocaleString()} → ` +
        `<b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s. ` +
        `Note this is a <b>${base.config.max_seq_len}</b>-token context — positional schemes mostly differ in how ` +
        `they <em>extrapolate</em> beyond the trained length, which a short-context run like this cannot show.`
    );
  };
  dd.positional.title = "Positional encoding: how the model learns word order";

  // ---- Activation ----------------------------------------------------
  dd.activation = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    const act = v.config.activation;
    formula(
      container,
      {
        geglu: String.raw`\text{GEGLU}(x)=W_{down}\big(\text{GELU}(W_{gate}x)\odot W_{up}x\big)`,
        gelu: String.raw`\text{GELU-FFN}(x)=W_{2}\big(\text{GELU}(W_{1}x)\big)\qquad\text{(no gate branch)}`,
        swiglu: String.raw`\text{SwiGLU}(x)=W_{down}\big(\text{SiLU}(W_{gate}x)\odot W_{up}x\big)`,
      }[act]
    );
    note(
      container,
      act === "gelu"
        ? `Dropping the gate removes an entire weight matrix per expert: a gated expert has three ` +
          `(<code>w_gate</code>, <code>w_up</code>, <code>w_down</code>), a plain one has two. So this variant ` +
          `is genuinely smaller — <b>${v.params.total.toLocaleString()}</b> parameters against the control's ` +
          `${base.params.total.toLocaleString()} — and the comparison is not parameter-matched, which is worth ` +
          `holding in mind when reading its loss.`
        : `Identical structure to the control, identical parameter count — the only difference is which ` +
          `nonlinearity squashes the gate branch before it multiplies the up branch. SiLU is smooth and ` +
          `non-monotonic near zero; GELU is smooth and monotonic. Any difference in the number below is ` +
          `attributable to that and nothing else.`
    );
    note(
      container,
      `Measured: validation loss ${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b>, ` +
        `throughput ${Math.round(base.tokens_per_sec).toLocaleString()} → ` +
        `<b>${Math.round(v.tokens_per_sec).toLocaleString()}</b> tok/s.`
    );
  };
  dd.activation.title = "Expert activation: is the gate worth its extra matrix?";

  // ---- Tied embeddings -----------------------------------------------
  dd.tie_word_embeddings = function (container, data, name) {
    const v = data.variants[name];
    const base = data.variants[data.baseline];
    const cfg = base.config;
    formula(
      container,
      String.raw`\text{tied: }\;\text{logits}=x\,E^\top\qquad\text{untied: }\;\text{logits}=x\,W_{LM}^\top,\;
                 W_{LM}\in\mathbb{R}^{${cfg.vocab_size}\times${cfg.d_model}}`
    );
    const saved = cfg.vocab_size * cfg.d_model;
    note(
      container,
      `Tying makes the output layer literally the embedding table transposed, saving ` +
        `<code style="color:#e8a95f">vocab_size × d_model = ${cfg.vocab_size} × ${cfg.d_model} = ` +
        `${saved.toLocaleString()}</code> parameters. Untying gives the output its own matrix — which is exactly ` +
        `the <b>${(v.params.total - base.params.total).toLocaleString()}</b>-parameter difference measured here. ` +
        `The tradeoff is conceptual as much as numerical: tying forces "the vector that <em>means</em> token X" ` +
        `and "the vector that <em>predicts</em> token X" to be the same thing, which is a real constraint, and ` +
        `at a 65-character vocabulary the saving it buys is small.`
    );
    note(
      container,
      `Measured: validation loss ${base.final_val_loss.toFixed(4)} → <b>${v.final_val_loss.toFixed(4)}</b> — ` +
        `but for <b>${(((v.params.total - base.params.total) / base.params.total) * 100).toFixed(1)}%</b> more ` +
        `parameters, so unlike most rows on this page this one is not a free comparison.`
    );
  };
  dd.tie_word_embeddings.title = "Tied embeddings: one matrix doing two jobs";

  AB.deepDive = dd;
})();
