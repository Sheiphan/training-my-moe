/* token_journey/js/diagram.js
 * Draws the full pipeline SVG for ONE (attention, positional) variant's
 * real trained checkpoint (see scripts/export_trace.py). `mount()` takes
 * the variant's trace data explicitly and can be called again on a
 * variant switch to rebuild the SVG with a different attention
 * sub-diagram (GQA vs MLA) and positional-scheme label. Every focusable
 * region gets a data-focus attribute so camera.js can zoom the D3
 * viewport to its real rendered bounding box; the SAME 5 focus ids
 * (attn-qkv1/attn-rope1/attn-group1/attn-sdpa1/attn-out1) are reused by
 * both attention variants so camera.js and steps.js never need to know
 * which variant is active.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  const VIEW_W = 1000;
  const CX = 500;
  const INNER_W = 760;
  const INNER_X = CX - INNER_W / 2; // 120
  const SKIP_X = 965;
  const CONTAINER_X = 60;
  const CONTAINER_W = 880;

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function fmt(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(Math.round(n));
  }

  function textBlock(cx, cy, lines, opts = {}) {
    const size = opts.size || 13;
    const weight = opts.weight || 500;
    const lh = opts.lh || size + 5;
    const totalH = (lines.length - 1) * lh;
    const startY = cy - totalH / 2 + size * 0.35;
    const tspans = lines
      .map((l, i) => `<tspan x="${cx}" y="${startY + i * lh}">${esc(l)}</tspan>`)
      .join("");
    return `<text text-anchor="middle" font-size="${size}" font-weight="${weight}">${tspans}</text>`;
  }

  function box(x, y, w, h, lines, cls, opts = {}) {
    const cx = x + w / 2,
      cy = y + h / 2;
    const title = opts.title ? `<title>${esc(opts.title)}</title>` : "";
    const focusAttr = opts.focus ? ` data-focus="${opts.focus}"` : "";
    const idAttr = opts.id ? ` id="${opts.id}"` : "";
    return `<g class="node ${cls}"${focusAttr}${idAttr}>${title}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${
      opts.rx ?? 9
    }"/>${textBlock(cx, cy, lines, opts)}</g>`;
  }

  function caption(cx, y, text) {
    return `<text x="${cx}" y="${y}" text-anchor="middle" class="caption">${esc(text)}</text>`;
  }

  function arrow(x, y1, y2) {
    return `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" class="flow-arrow" marker-end="url(#arrowhead)"/>`;
  }

  function plusNode(cx, cy, opts = {}) {
    const title = opts.title ? `<title>${esc(opts.title)}</title>` : "";
    const focusAttr = opts.focus ? ` data-focus="${opts.focus}"` : "";
    return `<g class="node plus"${focusAttr}>${title}<circle cx="${cx}" cy="${cy}" r="15"/><text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central" font-size="18" font-weight="700">+</text></g>`;
  }

  function skipConnection(cx, yTop, yPlus, xRight) {
    return `<path d="M ${cx} ${yTop} L ${xRight} ${yTop} L ${xRight} ${yPlus} L ${
      cx + 15
    } ${yPlus}" class="skip-line" marker-end="url(#arrowhead-skip)"/>`;
  }

  function repeatContainer(top, bottom, nLayers) {
    const x = CONTAINER_X,
      w = CONTAINER_W;
    const ghost2 = `<rect x="${x + 16}" y="${top + 16}" width="${w}" height="${
      bottom - top
    }" rx="14" class="ghost"/>`;
    const ghost1 = `<rect x="${x + 8}" y="${top + 8}" width="${w}" height="${
      bottom - top
    }" rx="14" class="ghost"/>`;
    const main = `<rect x="${x}" y="${top}" width="${w}" height="${bottom - top}" rx="14" class="repeat-container"/>`;
    const tabW = 230,
      tabH = 26;
    const tabX = x + w - tabW - 14;
    const tab =
      `<rect x="${tabX}" y="${top - tabH / 2}" width="${tabW}" height="${tabH}" rx="6" class="repeat-tab"/>` +
      `<text x="${tabX + tabW / 2}" y="${top + 1}" text-anchor="middle" dominant-baseline="central" font-size="13" font-weight="700" fill="#334155">Repeated \u00d7 ${nLayers} layers</text>`;
    return `<g data-focus="block1-container"><title>The same steps run again for every other layer, with that layer's own learned weights.</title>${ghost2}${ghost1}${main}${tab}</g>`;
  }

  function defsMarkup() {
    return `<defs>
      <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#0f172a"/>
      </marker>
      <marker id="arrowhead-skip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/>
      </marker>
    </defs>`;
  }

  function buildAttentionGQA(y, cfg, parts) {
    const headDim = cfg.d_model / cfg.n_heads;
    const nRep = cfg.n_heads / cfg.n_kv_heads;
    const gap = 16;
    const qkvW = (INNER_W - 2 * gap) / 3;
    const items = [
      ["W_q \u2014 query", `${cfg.d_model} \u2192 ${cfg.n_heads}\u00d7${headDim}`],
      ["W_k \u2014 key", `${cfg.d_model} \u2192 ${cfg.n_kv_heads}\u00d7${headDim}`],
      ["W_v \u2014 value", `${cfg.d_model} \u2192 ${cfg.n_kv_heads}\u00d7${headDim}`],
    ];
    let x = INNER_X;
    for (const it of items) {
      parts.push(box(x, y, qkvW, 46, it, "attn-sub", { focus: "attn-qkv1", title: "linear projection, no bias", size: 12 }));
      x += qkvW + gap;
    }
    y += 46 + 14;

    // QK-norm sits between the projection and the positional transform,
    // and only appears at all when the checkpoint has it -- so the diagram
    // itself differs between BASELINE and BASELINE_V2.
    if (cfg.qk_norm) {
      parts.push(
        box(
          INNER_X,
          y,
          INNER_W,
          40,
          ["QK-norm \u2014 per-head RMSNorm on Q and K", `independent \u03b3 per head, ${headDim} dims each \u00b7 bounds the logit scale (Qwen3)`],
          "norm",
          { focus: "attn-qknorm1", title: "ModelConfig.qk_norm = True \u2014 applied before the positional transform", size: 12 }
        )
      );
      y += 40 + 14;
    }

    const posLabel = {
      rope: ["RoPE \u2014 rotates Q,K by \u03b8(pos) right here", `inv_freq_i = 1/${cfg.rope_theta}^(2i/${headDim})  (Su et al., 2021)`],
      nope: ["NoPE \u2014 no positional operation applied", "token order carried only by the causal mask"],
      alibi: ["ALiBi \u2014 positional info folds in as a score bias below", "fixed per-head slope, geometric series (Press et al., 2021)"],
    }[cfg.positional_scheme];
    parts.push(box(INNER_X, y, INNER_W, 40, posLabel, "pos-note", { focus: "attn-rope1", title: "ModelConfig.positional_scheme", size: 12 }));
    y += 40 + 14;

    const groupText =
      nRep > 1
        ? [`GQA grouping \u2014 repeat_interleave K,V \u00d7 ${nRep}`, `${cfg.n_kv_heads} kv-heads shared across ${cfg.n_heads} query heads`]
        : [`n_kv_heads = n_heads = ${cfg.n_heads} \u2192 plain multi-head attention`];
    parts.push(box(INNER_X, y, INNER_W, 40, groupText, "attn-sub", { focus: "attn-group1", title: "Grouped-Query Attention head sharing", size: 12 }));
    y += 40 + 14;

    parts.push(
      box(
        INNER_X,
        y,
        INNER_W,
        46,
        ["Scaled Dot-Product Attention (causal)", `softmax(QK\u1d40/\u221a${headDim}${cfg.positional_scheme === "alibi" ? " + ALiBi bias" : ""} + causal mask) \u00b7 V`],
        "attn-main",
        { focus: "attn-sdpa1", title: "causal mask = triu(-inf, diagonal=1)", size: 12 }
      )
    );
    y += 46 + 14;

    parts.push(
      box(INNER_X, y, INNER_W, 40, ["W_o \u2014 output projection", `${cfg.n_heads}\u00d7${headDim} \u2192 ${cfg.d_model}`], "attn-sub", {
        focus: "attn-out1",
        title: "concat all heads, project back to d_model",
        size: 12,
      })
    );
    y += 40;
    return y;
  }

  function buildAttentionMLA(y, cfg, parts) {
    const headDim = cfg.d_model / cfg.n_heads;
    const ropeHeadDim = Math.floor(headDim / 2);
    const latentDim = cfg.mla_latent_dim;
    const gqaEquivCache = cfg.n_kv_heads * headDim * 2;
    const mlaCache = latentDim + ropeHeadDim;
    const gap = 16;
    const halfW = (INNER_W - gap) / 2;

    parts.push(
      box(INNER_X, y, halfW, 46, ["W_dkv \u2014 down-project to latent", `${cfg.d_model} \u2192 ${latentDim}  (this is what gets cached)`], "attn-sub", {
        focus: "attn-qkv1",
        title: "low-rank joint K/V compression, DeepSeek-V2 Eq. 9-11",
        size: 11,
      })
    );
    parts.push(
      box(INNER_X + halfW + gap, y, halfW, 46, ["W_q \u2014 content query", `${cfg.d_model} \u2192 ${cfg.n_heads}\u00d7${headDim}`], "attn-sub", {
        focus: "attn-qkv1",
        title: "plain (uncompressed) query content path",
        size: 11,
      })
    );
    y += 46 + 14;

    parts.push(
      box(
        INNER_X,
        y,
        INNER_W,
        40,
        ["W_uk, W_uv \u2014 up-project latent per head", `${latentDim} \u2192 ${cfg.n_heads}\u00d7${headDim}  (content K,V \u2014 recomputed every forward pass)`],
        "attn-sub",
        { focus: "attn-rope1", title: "decompression happens every forward pass; only the latent is cached", size: 11 }
      )
    );
    y += 40 + 14;

    parts.push(
      box(
        INNER_X,
        y,
        INNER_W,
        46,
        [`W_qr, W_kr \u2014 decoupled RoPE slice (dim ${ropeHeadDim})`, "k_rope is ONE slice shared across all heads (MLA always applies this, regardless of positional_scheme)"],
        "pos-note",
        { focus: "attn-group1", title: "the only positional info MLA carries \u2014 decoupled from the compressed content K", size: 11 }
      )
    );
    y += 46 + 14;

    parts.push(
      box(
        INNER_X,
        y,
        INNER_W,
        46,
        ["Scaled Dot-Product Attention (causal)", `softmax(concat(Qc,Qr)\u00b7concat(Kc,Kr)\u1d40 / \u221a${headDim + ropeHeadDim} + mask) \u00b7 Vc`],
        "attn-main",
        { focus: "attn-sdpa1", title: "content and rope dims concatenated before the dot product", size: 12 }
      )
    );
    y += 46 + 14;

    parts.push(
      box(INNER_X, y, INNER_W, 40, ["W_o \u2014 output projection", `${cfg.n_heads}\u00d7${headDim} \u2192 ${cfg.d_model}`], "attn-sub", { focus: "attn-out1", size: 12 })
    );
    y += 40 + 14;

    parts.push(
      box(
        INNER_X,
        y,
        INNER_W,
        40,
        [`KV-cache/token = latent + rope-key = ${mlaCache} elements`, `vs GQA's ${gqaEquivCache} elements \u2192 ${((mlaCache / gqaEquivCache) * 100).toFixed(1)}% the size`],
        "note-good",
        { title: "MLAAttention.kv_cache_elements_per_token \u2014 real measured finding", size: 12 }
      )
    );
    y += 40;
    return y;
  }

  function buildAttention(y, cfg, parts) {
    return cfg.attention_variant === "mla" ? buildAttentionMLA(y, cfg, parts) : buildAttentionGQA(y, cfg, parts);
  }

  function buildMoE(y, cfg, parts) {
    parts.push(
      box(INNER_X, y, INNER_W, 40, ["Router \u2014 Linear (no bias)", `${cfg.d_model} \u2192 ${cfg.n_routed_experts} logits \u2192 ${cfg.router_scoring === "sigmoid" ? "sigmoid per expert \u2192 renormalize" : "softmax"} \u2192 router_probs`], "router-note", {
        focus: "moe-router1",
        title: "Router.gate",
      })
    );
    y += 40 + 14;

    const balText =
      cfg.moe_balancing === "aux_loss"
        ? ["Balancing: Aux-Loss  (Switch Transformer, 2021)", "differentiable loss added to training objective (weight 0.01)"]
        : ["Balancing: Loss-Free  (DeepSeek-V3, 2024)", "per-expert bias nudged \u00b1 update_rate each step, no gradient term"];
    parts.push(box(INNER_X, y, INNER_W, 40, balText, "pos-note", { focus: "moe-balance1", title: "ModelConfig.moe_balancing", size: 12 }));
    y += 40 + 14;

    parts.push(
      box(INNER_X, y, INNER_W, 40, [`Top-k select \u2014 k = ${cfg.top_k} of ${cfg.n_routed_experts} experts / token`, "gate weights renormalized to sum to 1"], "router-note", {
        focus: "moe-topk1",
        title: "select_top_k",
      })
    );
    y += 40 + 20;

    const n = cfg.n_routed_experts;
    const expGap = 10;
    const expW = (INNER_W - (n - 1) * expGap) / n;
    const actLabel = { swiglu: "SwiGLU", geglu: "GEGLU", gelu: "GELU" }[cfg.activation];
    const matCount = cfg.activation === "gelu" ? 2 : 3;
    let x = INNER_X;
    for (let i = 0; i < n; i++) {
      parts.push(
        box(x, y, expW, 54, [`E${i}`, actLabel, `${cfg.d_model}\u2192${cfg.d_ff}\u2192${cfg.d_model}`], "expert routed", {
          focus: "moe-experts1",
          title: `Routed expert ${i}: ${actLabel} FFN, ${matCount} weight matrices, ${fmt(matCount * cfg.d_model * cfg.d_ff)} params \u2014 active only when selected in the top-${cfg.top_k}`,
          size: n > 6 ? 10 : 12,
          id: `expert-${i}`,
        })
      );
      x += expW + expGap;
    }
    y += 54 + 10;
    parts.push(caption(CX, y, `${n} routed experts \u00b7 this token's real top-${cfg.top_k} choice glows once you reach that step`));
    y += 24;

    const ns = cfg.n_shared_experts;
    if (ns > 0) {
      const sGap = 10;
      const sW = 200;
      const totalSW = ns * sW + (ns - 1) * sGap;
      let sx = CX - totalSW / 2;
      for (let i = 0; i < ns; i++) {
        parts.push(
          box(sx, y, sW, 54, [`Shared E${i}`, actLabel, `${cfg.d_model}\u2192${cfg.d_ff}\u2192${cfg.d_model}`], "expert shared", {
            focus: "moe-shared1",
            title: `Shared expert ${i}: always active for every token, output added unweighted`,
            size: 12,
          })
        );
        sx += sW + sGap;
      }
      y += 54 + 10;
      parts.push(caption(CX, y, `${ns} shared expert(s) \u00b7 always active for every token`));
      y += 24;
    }

    parts.push(
      box(INNER_X, y, INNER_W, 40, ["Combine", "\u03a3 gate_i \u00b7 RoutedExpert_i(x)  +  \u03a3 SharedExpert_j(x)"], "combine", {
        focus: "moe-combine1",
        title: "MoELayer.forward \u2014 routed via index_add_, shared summed unweighted",
      })
    );
    y += 40;
    return y;
  }

  function build(trace) {
    const cfg = trace.config;
    const parts = [];
    let y = 34;

    const tokens = trace.tokens;
    const tokW = 42,
      tokGap = 8;
    const totalTokW = tokens.length * tokW + (tokens.length - 1) * tokGap;
    let tx = CX - totalTokW / 2;
    const tokenBoxes = [];
    tokens.forEach((ch, i) => {
      const label = ch === "\n" ? "\\n" : ch;
      tokenBoxes.push(
        box(tx, y, tokW, 36, [label], i === trace.target_position ? "token token-target" : "token", {
          focus: "tokens",
          title: i === trace.target_position ? `our followed token: "${label}" (position ${i})` : `token "${label}"`,
          size: 16,
          weight: 700,
        })
      );
      tx += tokW + tokGap;
    });
    parts.push(...tokenBoxes);
    y += 36 + 22;
    parts.push(caption(CX, y, `Real prompt "${trace.prompt}" \u00b7 seq_len = ${tokens.length} \u00b7 we're following the last token`));
    y += 16;

    parts.push(arrow(CX, y, y + 28));
    y += 28;

    const embW = 460;
    const embH = 56;
    const embX = CX - embW / 2;
    const embTopY = y;
    parts.push(
      box(embX, y, embW, embH, ["Embedding Table (nn.Embedding)", `${cfg.vocab_size} \u00d7 ${cfg.d_model} = ${fmt(cfg.vocab_size * cfg.d_model)} params`], "embed", {
        focus: "embedding",
        title: "init N(0, 0.02\u00b2)",
      })
    );
    y += embH;

    parts.push(arrow(CX, y, y + 26));
    y += 26;
    parts.push(box(CX - 165, y, 330, 34, [`Embeddings (seq_len \u00d7 ${cfg.d_model})`], "plain", { focus: "embeddings-vec" }));
    y += 34;

    parts.push(arrow(CX, y, y + 28));
    y += 28;

    const containerTop = y;
    y += 22;

    const attnBranchY = y - 10;
    parts.push(
      box(INNER_X, y, INNER_W, 36, ["RMSNorm", `x/RMS(x)\u00b7\u03b3  (\u03b3\u2208\u211d^${cfg.d_model}, \u03b5=1e-6)`], "norm", {
        focus: "attn-norm1",
        title: "pre-attention RMSNorm",
        size: 12,
      })
    );
    y += 36;

    parts.push(arrow(CX, y, y + 22));
    y += 22;
    y = buildAttention(y, cfg, parts);
    parts.push(arrow(CX, y, y + 22));
    y += 22;

    const attnPlusY = y + 15;
    parts.push(plusNode(CX, attnPlusY, { focus: "attn-plus1", title: "residual add: x = x + Attention(RMSNorm(x))" }));
    parts.push(skipConnection(CX, attnBranchY, attnPlusY, SKIP_X));
    y += 32;

    y += 22;
    const moeBranchY = y - 10;
    parts.push(
      box(INNER_X, y, INNER_W, 36, ["RMSNorm", `x/RMS(x)\u00b7\u03b3  (\u03b3\u2208\u211d^${cfg.d_model}, \u03b5=1e-6)`], "norm", {
        focus: "moe-norm1",
        title: "pre-MoE RMSNorm",
        size: 12,
      })
    );
    y += 36;

    parts.push(arrow(CX, y, y + 22));
    y += 22;
    y = buildMoE(y, cfg, parts);
    parts.push(arrow(CX, y, y + 22));
    y += 22;

    const moePlusY = y + 15;
    parts.push(plusNode(CX, moePlusY, { focus: "moe-plus1", title: "residual add: x = x + MoE(RMSNorm(x))" }));
    parts.push(skipConnection(CX, moeBranchY, moePlusY, SKIP_X));
    y += 32;

    const containerBottom = y + 14;
    parts.push(repeatContainer(containerTop, containerBottom, cfg.n_layers));
    y = containerBottom;

    y += 8;
    parts.push(arrow(CX, y, y + 26));
    y += 26;
    parts.push(
      box(CX - 190, y, 380, 36, ["Final RMSNorm", `x/RMS(x)\u00b7\u03b3  (\u03b3\u2208\u211d^${cfg.d_model})`], "norm", {
        focus: "final-norm",
        title: "applied once, after the last block",
        size: 12,
      })
    );
    y += 36;

    parts.push(arrow(CX, y, y + 26));
    y += 26;
    const lmW = 380,
      lmH = 46;
    const lmX = CX - lmW / 2;
    const lmTieY = y + lmH / 2;
    parts.push(
      box(lmX, y, lmW, lmH, ["LM Head \u2014 Linear", `${cfg.d_model} \u2192 ${cfg.vocab_size}${cfg.tie_word_embeddings ? "  (tied to embedding)" : ""}`], "embed", {
        focus: "lm-head",
        title: cfg.tie_word_embeddings ? "lm_head.weight = embedding.weight" : "independent lm_head.weight",
      })
    );
    if (cfg.tie_word_embeddings) {
      const leftX = 30;
      const embMidY = embTopY + embH / 2;
      parts.push(
        `<path d="M ${lmX} ${lmTieY} L ${leftX} ${lmTieY} L ${leftX} ${embMidY} L ${embX} ${embMidY}" class="tie-line" marker-end="url(#arrowhead-skip)"/>`
      );
    }
    y += lmH;

    parts.push(arrow(CX, y, y + 26));
    y += 26;
    parts.push(box(CX - 140, y, 280, 34, ["Softmax"], "output", { focus: "softmax", title: "softmax(logits[-1])" }));
    y += 34;

    parts.push(arrow(CX, y, y + 26));
    y += 26;

    const barBaseY = y + 100;
    const probs = trace.probs;
    const top = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 6);
    const maxP = top[0].p;
    const barW = 60,
      barGap = 14;
    const maxBarH = 100;
    const totalBarsW = top.length * barW + (top.length - 1) * barGap;
    let bx = CX - totalBarsW / 2;
    const barsMarkup = [`<g data-focus="output-bars">`];
    for (const { p, i } of top) {
      const hgt = Math.max(4, (p / maxP) * maxBarH);
      const rawCh = trace.vocab[i];
      const label = rawCh === "\n" ? "\\n" : rawCh === " " ? "\u2423" : rawCh;
      barsMarkup.push(
        `<rect x="${bx}" y="${barBaseY - hgt}" width="${barW}" height="${hgt}" rx="4" class="outbar"><title>real probability: ${label} at p=${p.toFixed(4)}</title></rect>`
      );
      barsMarkup.push(`<text x="${bx + barW / 2}" y="${barBaseY + 20}" text-anchor="middle" font-size="15" font-weight="700">${esc(label)}</text>`);
      barsMarkup.push(`<text x="${bx + barW / 2}" y="${barBaseY - hgt - 8}" text-anchor="middle" font-size="11" font-weight="700">${p.toFixed(3)}</text>`);
      bx += barW + barGap;
    }
    barsMarkup.push("</g>");
    parts.push(barsMarkup.join(""));
    y = barBaseY + 36;
    parts.push(caption(CX, y, `Real output distribution \u2014 every bar here is an actual measured probability, not illustrative`));
    y += 20;

    return { svg: parts.join(""), height: y + 24 };
  }

  TJ.diagram = {
    mount(svgSelector, trace) {
      const svg = document.querySelector(svgSelector);
      TJ.diagram.svgSelector = svgSelector;
      const { svg: inner, height } = build(trace);
      svg.setAttribute("viewBox", `0 0 ${VIEW_W} 700`);
      svg.innerHTML = defsMarkup() + `<g id="viewport">${inner}</g>`;

      const targets = {};
      svg.querySelectorAll("[data-focus]").forEach((el) => {
        const id = el.getAttribute("data-focus");
        const bbox = el.getBBox();
        // multiple elements can share one focus id (e.g. every token box
        // shares "tokens") -- accumulate a union bbox across all of them.
        if (!targets[id]) {
          targets[id] = { x: bbox.x, y: bbox.y, width: bbox.width, height: bbox.height };
        } else {
          const t = targets[id];
          const x2 = Math.max(t.x + t.width, bbox.x + bbox.width);
          const y2 = Math.max(t.y + t.height, bbox.y + bbox.height);
          t.x = Math.min(t.x, bbox.x);
          t.y = Math.min(t.y, bbox.y);
          t.width = x2 - t.x;
          t.height = y2 - t.y;
        }
      });
      targets.overview = { x: 0, y: 0, width: VIEW_W, height };
      TJ.diagram.focusTargets = targets;
      TJ.diagram.contentHeight = height;
      return targets;
    },
  };
})();

