/* token_journey/js/deep-dive.js
 * Per-step registry for exact, real-number math views. Every entry reads
 * the CURRENT variant's data via TJ.store at call time (not a module-load
 * constant), and several entries branch on cfg.attention_variant /
 * cfg.positional_scheme so the math shown always matches what the real
 * checkpoint for the currently-selected variant actually computed.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});
  const { renderMatMul, renderElementwise, tex, cellDiv } = TJ.matrixViz;

  function headColLabels(dim, headDim) {
    return Array.from({ length: dim }, (_, index) => `h${Math.floor(index / headDim)}.d${index % headDim}`);
  }

  function addVectors(...vectors) {
    const output = new Array(vectors[0].length).fill(0);
    for (const vector of vectors) for (let i = 0; i < output.length; i++) output[i] += vector[i];
    return output;
  }

  function matVec(vector, matrix) {
    return matrix[0].map((_, col) => vector.reduce((sum, value, row) => sum + value * matrix[row][col], 0));
  }

  function transpose(matrix) {
    return matrix[0].map((_, col) => matrix.map((row) => row[col]));
  }

  function rotateVector(vector, angles) {
    const half = vector.length / 2;
    return vector.map((value, index) => {
      const rotatedHalf = index < half ? -vector[index + half] : vector[index - half];
      return value * Math.cos(angles[index]) + rotatedHalf * Math.sin(angles[index]);
    });
  }

  /** GQA shares one kv-head's V across a group of query heads; MLA gives
   * every head its own independent V with no sharing at all. */
  function attentionHeadOutput(head) {
    const cfg = TJ.store.cfg();
    const L0 = TJ.store.layer0();
    const target0 = TJ.store.target0();
    const vHead = cfg.attention_variant === "mla" ? head : Math.floor(head / (cfg.n_heads / cfg.n_kv_heads));
    const probabilities = target0.attn_scores_post_softmax[head];
    return transpose(L0.v[vHead]).map((dimension) => probabilities.reduce((sum, weight, token) => sum + weight * dimension[token], 0));
  }

  function maxAbsDiff(a, b) {
    return Math.max(...a.map((value, index) => Math.abs(value - b[index])));
  }

  function expertForward(key) {
    const W = TJ.store.weights();
    const weights = W.experts[key];
    const gate = matVec(W.moe_norm_output_target, weights.w_gate);
    const activatedGate = gate.map((value) => value / (1 + Math.exp(-value)));
    const up = matVec(W.moe_norm_output_target, weights.w_up);
    const mixed = activatedGate.map((value, index) => value * up[index]);
    return { gate, activatedGate, up, mixed, output: matVec(mixed, weights.w_down) };
  }

  function tokenName(token) {
    if (token === "\n") return "newline";
    if (token === " ") return "space";
    if (token === "\t") return "tab";
    return `'${token}'`;
  }

  function noteEl(container, text) {
    const element = document.createElement("div");
    element.className = "mm-note";
    element.textContent = text;
    container.appendChild(element);
  }

  function rmsNormPipeline(container, input, gamma, note) {
    const epsilon = 1e-6;
    const squared = input.map((value) => value * value);
    const meanSquare = squared.reduce((sum, value) => sum + value, 0) / input.length;
    const inverseRoot = 1 / Math.sqrt(meanSquare + epsilon);
    const normalized = input.map((value) => value * inverseRoot);
    const output = normalized.map((value, index) => value * gamma[index]);
    renderElementwise(container, {
      title: String.raw`\text{RMSNorm}(x)=\dfrac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot\gamma\qquad(d=${input.length})`,
      steps: [
        { label: "x (real)", values: input },
        { label: "square each value", values: squared },
        { label: "mean(x^2) + epsilon", values: [meanSquare + epsilon] },
        { label: "inverse square root", values: [inverseRoot] },
        { label: "x times scalar", values: normalized },
        { label: "times learned gamma", values: output },
      ],
    });
    if (note) noteEl(container, note);
  }

  TJ.deepDive = {
    embed(container) {
      const T = TJ.store.data();
      const cfg = TJ.store.cfg();
      const take = 8;
      const rows = T.embeddings
        .map((row, index) => {
          const token = T.tokens[index] === "\n" ? "\\n" : T.tokens[index];
          return `<div class="rl-row${index === T.target_position ? " rl-target" : ""}" style="--stage:${index}">
            <span class="rl-tok">${token}</span>
            <div class="mm-vec">${row.slice(0, take).map((value) => cellDiv(value)).join("")}</div>
          </div>`;
        })
        .join("");
      container.innerHTML = `<div class="mm-formula-line"></div><div class="rl-rows">${rows}</div>`;
      tex(container.querySelector(".mm-formula-line"), String.raw`x_0=\text{Embedding}[\text{token\_id}]\in\mathbb{R}^{d_{model}}`, true);
      noteEl(container, `The highlighted row is our token's real starting vector; ${take} of ${cfg.d_model} dimensions are visible.`);
    },

    "attn-norm"(container) {
      const T = TJ.store.data();
      const W = TJ.store.weights();
      rmsNormPipeline(container, T.target_embedding, W.attn_norm_gamma, "This exact output is the input to the Q, K, and V projections.");
    },

    "attn-qkv"(container) {
      const cfg = TJ.store.cfg();
      const W = TJ.store.weights();
      const stack = document.createElement("div");
      stack.className = "math-stack";

      if (cfg.attention_variant === "mla") {
        const dkvSection = document.createElement("div");
        const qSection = document.createElement("div");
        renderMatMul(dkvSection, {
          formula: String.raw`c_{kv}=xW_{dkv}`,
          aLabel: "RMSNorm(x)", a: W.attn_norm_output_target,
          bLabel: "W_dkv (real)", b: W.w_dkv,
          outLabel: "c\\_kv (shared latent)",
          colLabels: Array.from({ length: cfg.mla_latent_dim }, (_, i) => `latent ${i}`),
        });
        renderMatMul(qSection, {
          formula: String.raw`q_c=xW_q`,
          aLabel: "RMSNorm(x)", a: W.attn_norm_output_target,
          bLabel: "W_q (real, content path)", b: W.w_q,
          outLabel: "q\\_c (content query)",
          colLabels: headColLabels(cfg.n_heads * (cfg.d_model / cfg.n_heads), cfg.d_model / cfg.n_heads),
        });
        stack.append(dkvSection, qSection);
        container.replaceChildren(stack);
        return;
      }

      const specs = [
        ["Q", W.w_q, cfg.n_heads * (cfg.d_model / cfg.n_heads)],
        ["K", W.w_k, cfg.n_kv_heads * (cfg.d_model / cfg.n_heads)],
        ["V", W.w_v, cfg.n_kv_heads * (cfg.d_model / cfg.n_heads)],
      ];
      for (const [name, matrix, dimension] of specs) {
        const section = document.createElement("div");
        renderMatMul(section, {
          formula: String.raw`${name}=\text{RMSNorm}(x)W_${name}`,
          aLabel: "RMSNorm(x)",
          a: W.attn_norm_output_target,
          bLabel: `W_${name} (real)`,
          b: matrix,
          outLabel: name,
          colLabels: headColLabels(dimension, cfg.d_model / cfg.n_heads),
        });
        stack.appendChild(section);
      }
      container.replaceChildren(stack);
    },

    "attn-rope"(container) {
      const cfg = TJ.store.cfg();
      const target0 = TJ.store.target0();
      const W = TJ.store.weights();

      if (cfg.attention_variant === "mla") {
        const headDim = cfg.d_model / cfg.n_heads;
        const stack = document.createElement("div");
        const ukSection = document.createElement("div");
        const uvSection = document.createElement("div");
        renderMatMul(ukSection, {
          formula: String.raw`k_c=c_{kv}W_{uk}`,
          aLabel: "c_kv (shared latent)", a: matVec(W.attn_norm_output_target, W.w_dkv),
          bLabel: "W_uk (real)", b: W.w_uk,
          outLabel: "k\\_c (content key)", colLabels: headColLabels(cfg.n_heads * headDim, headDim),
        });
        renderMatMul(uvSection, {
          formula: String.raw`v=c_{kv}W_{uv}`,
          aLabel: "c_kv (shared latent)", a: matVec(W.attn_norm_output_target, W.w_dkv),
          bLabel: "W_uv (real)", b: W.w_uv,
          outLabel: "v (content value)", colLabels: headColLabels(cfg.n_heads * headDim, headDim),
        });
        stack.append(ukSection, uvSection);
        container.replaceChildren(stack);
        noteEl(container, "This decompression runs fresh on every forward pass; only the small c_kv latent above ever needs to be cached for generation.");
        return;
      }

      if (cfg.positional_scheme === "nope") {
        container.innerHTML = `<div class="mm-formula-line"></div>`;
        tex(container.querySelector(".mm-formula-line"), String.raw`q'=q,\qquad k'=k\qquad(\text{no-op})`, true);
        const q = target0.q[0];
        noteEl(container, `Real check: q[0]=${q[0].toFixed(4)} before this step, and still ${q[0].toFixed(4)} after -- NoPE genuinely applies zero transformation here.`);
        return;
      }

      if (cfg.positional_scheme === "alibi") {
        const bias = target0.rope_angles_or_alibi_bias[0]; // head 0, real bias row: target -> every token
        container.innerHTML = `<div class="mm-formula-line"></div>`;
        tex(container.querySelector(".mm-formula-line"), String.raw`\text{bias}_h[i,j]=-\,\text{slope}_h\cdot(i-j)\qquad(\text{applied to scores, not to }Q,K)`, true);
        const section = document.createElement("div");
        renderElementwise(section, {
          title: String.raw`\text{head 0's real bias row, target token} \rightarrow \text{every token}`,
          steps: [{ label: "real ALiBi bias (head 0)", values: bias }],
          take: bias.length,
        });
        container.appendChild(section);
        noteEl(container, "Query and Key are untouched here -- this real per-head bias only gets added to the raw attention scores, at the next step.");
        return;
      }

      const angles = target0.rope_angles_or_alibi_bias;
      const halfDim = angles.length / 2;
      const cosine = angles.map((angle) => Math.cos(angle));
      const sine = angles.map((angle) => Math.sin(angle));
      const results = [];
      container.innerHTML = "";

      for (const [name, vector] of [["q", target0.q[0]], ["k", target0.k[0]]]) {
        const rotatedHalf = vector.map((_, index) => (index < halfDim ? -vector[index + halfDim] : vector[index - halfDim]));
        const rotated = vector.map((value, index) => value * cosine[index] + rotatedHalf[index] * sine[index]);
        const section = document.createElement("div");
        renderElementwise(section, {
          title: String.raw`${name}'_i=${name}_i\cos\theta_i+\operatorname{rotate\_half}(${name})_i\sin\theta_i`,
          steps: [
            { label: `${name} before RoPE`, values: vector },
            { label: "theta at this position", values: angles.slice(0, halfDim) },
            { label: "cos(theta)", values: cosine },
            { label: `rotate_half(${name})`, values: rotatedHalf },
            { label: "sin(theta)", values: sine },
            { label: `${name}' rotated`, values: rotated },
          ],
          take: 8,
        });
        container.appendChild(section);
        results.push({ vector, rotated });
      }
      noteEl(
        container,
        `Q pair (q0, q${halfDim}) rotates by theta0=${angles[0].toFixed(3)}: (${results[0].vector[0].toFixed(3)}, ${results[0].vector[halfDim].toFixed(3)}) -> (${results[0].rotated[0].toFixed(3)}, ${results[0].rotated[halfDim].toFixed(3)}). The same verified rotation is applied to K.`
      );
    },

    "attn-group"(container) {
      const cfg = TJ.store.cfg();
      const target0 = TJ.store.target0();
      const W = TJ.store.weights();

      if (cfg.attention_variant === "mla") {
        const headDim = cfg.d_model / cfg.n_heads;
        const ropeHeadDim = Math.floor(headDim / 2);
        const stack = document.createElement("div");
        const qrSection = document.createElement("div");
        const krSection = document.createElement("div");
        renderMatMul(qrSection, {
          formula: String.raw`q_r=\text{RoPE}(xW_{qr})`,
          aLabel: "RMSNorm(x)", a: W.attn_norm_output_target,
          bLabel: "W_qr (real, per-head)", b: W.w_qr,
          outLabel: "q\\_r (rotated, head 0 shown)", colLabels: headColLabels(cfg.n_heads * ropeHeadDim, ropeHeadDim),
        });
        renderMatMul(krSection, {
          formula: String.raw`k_r=\text{RoPE}(xW_{kr})`,
          aLabel: "RMSNorm(x)", a: W.attn_norm_output_target,
          bLabel: "W_kr (real, SHARED across heads)", b: W.w_kr,
          outLabel: "k\\_r (rotated, one shared slice)", colLabels: Array.from({ length: ropeHeadDim }, (_, i) => `d${i}`),
        });
        stack.append(qrSection, krSection);
        container.replaceChildren(stack);
        noteEl(container, "This tiny rotated slice carries ALL of this token's positional information for MLA -- always active, regardless of ModelConfig.positional_scheme.");
        return;
      }

      const repeat = cfg.n_heads / cfg.n_kv_heads;
      const angles = cfg.positional_scheme === "rope" ? target0.rope_angles_or_alibi_bias : null;
      const cards = Array.from({ length: cfg.n_heads }, (_, head) => {
        const kvHead = Math.floor(head / repeat);
        const query = angles ? rotateVector(target0.q[head], angles) : target0.q[head];
        const key = angles ? rotateVector(target0.k[kvHead], angles) : target0.k[kvHead];
        return `<div class="gqa-card" style="--stage:${head}">
          <div class="gqa-route"><strong>Query head ${head}</strong><span>uses</span><strong>K/V head ${kvHead}</strong></div>
          <div class="mm-label">Q${head}${angles ? " after RoPE" : ""}</div><div class="mm-vec">${query.slice(0, 4).map((value) => cellDiv(value)).join("")}</div>
          <div class="mm-label">Repeated K${kvHead}</div><div class="mm-vec">${key.slice(0, 4).map((value) => cellDiv(value)).join("")}</div>
          <div class="mm-label">Repeated V${kvHead}</div><div class="mm-vec">${target0.v[kvHead].slice(0, 4).map((value) => cellDiv(value)).join("")}</div>
        </div>`;
      }).join("");
      container.innerHTML = `<div class="mm-formula-line"></div><div class="gqa-grid">${cards}</div>`;
      tex(container.querySelector(".mm-formula-line"), String.raw`n_{rep}=n_h/n_{kv}=${cfg.n_heads}/${cfg.n_kv_heads}=${repeat},\qquad K,V\leftarrow\operatorname{repeat\_interleave}(K,V,n_{rep})`, true);
      const groups = Array.from({ length: cfg.n_kv_heads }, (_, kvHead) => {
        const first = kvHead * repeat;
        return `K/V head ${kvHead} is shared by query heads ${first}-${first + repeat - 1}`;
      });
      noteEl(container, `${groups.join("; ")}. Values are copied, never averaged.`);
    },

    "attn-sdpa"(container) {
      const cfg = TJ.store.cfg();
      const T = TJ.store.data();
      const L0 = TJ.store.layer0();
      const target0 = TJ.store.target0();
      const head = 0;

      if (cfg.attention_variant === "mla") {
        const dim = cfg.d_model / cfg.n_heads + Math.floor(cfg.d_model / cfg.n_heads / 2);
        const query = L0.q[head][T.target_position]; // already concat(content,rope), already rotated
        const keys = L0.k[head]; // (seq_len, dim) for this head
        const rawScores = matVec(query, transpose(keys));
        const scaledScores = rawScores.map((value) => value / Math.sqrt(dim));
        const maximum = Math.max(...scaledScores);
        const exponentials = scaledScores.map((value) => Math.exp(value - maximum));
        const denominator = exponentials.reduce((sum, value) => sum + value, 0);
        const probabilities = target0.attn_scores_post_softmax[head];

        const scoreSection = document.createElement("div");
        const softmaxSection = document.createElement("div");
        const valueSection = document.createElement("div");
        renderMatMul(scoreSection, {
          formula: String.raw`s=qk^\top\qquad(q,k\text{ already concat(content, rope)})`,
          aLabel: `q head ${head} (target token)`, a: query,
          bLabel: "k^T (all real tokens)", b: transpose(keys),
          outLabel: "raw scores", colLabels: T.tokens.map(tokenName), colWindow: T.tokens.length,
        });
        renderElementwise(softmaxSection, {
          title: String.raw`p=\operatorname{softmax}\!\left(s/\sqrt{${dim}}+M\right)`,
          steps: [
            { label: "raw qk^T", values: rawScores },
            { label: `divide by sqrt(${dim})`, values: scaledScores },
            { label: "causal mask M", values: scaledScores },
            { label: "exp(score - max)", values: exponentials },
            { label: `divide by ${denominator.toFixed(3)}`, values: probabilities },
          ],
          take: T.tokens.length,
        });
        renderMatMul(valueSection, {
          formula: String.raw`\text{head}_0=pV_c`,
          aLabel: "attention probabilities", a: probabilities,
          bLabel: "content V head 0 (all tokens)", b: L0.v[head],
          outLabel: "head 0 context", colLabels: Array.from({ length: cfg.d_model / cfg.n_heads }, (_, index) => `d${index}`),
        });
        container.replaceChildren(scoreSection, softmaxSection, valueSection);
        noteEl(container, "Only content V is used for the weighted sum -- the rope slice only ever influenced the score, never the value.");
        return;
      }

      const headDim = cfg.d_model / cfg.n_heads;
      const kvHead = Math.floor(head / (cfg.n_heads / cfg.n_kv_heads));
      const query = cfg.positional_scheme === "rope" ? rotateVector(L0.q[head][T.target_position], L0.rope_angles_or_alibi_bias[T.target_position]) : L0.q[head][T.target_position];
      const keys =
        cfg.positional_scheme === "rope"
          ? L0.k[kvHead].map((key, token) => rotateVector(key, L0.rope_angles_or_alibi_bias[token]))
          : L0.k[kvHead];
      const rawScores = matVec(query, transpose(keys));
      const scaledScores = rawScores.map((value) => value / Math.sqrt(headDim));
      const aliBiasedScores = cfg.positional_scheme === "alibi" ? scaledScores.map((value, i) => value + target0.rope_angles_or_alibi_bias[head][i]) : scaledScores;
      const maximum = Math.max(...aliBiasedScores);
      const exponentials = aliBiasedScores.map((value) => Math.exp(value - maximum));
      const denominator = exponentials.reduce((sum, value) => sum + value, 0);
      const probabilities = target0.attn_scores_post_softmax[head];
      const scoreSection = document.createElement("div");
      const softmaxSection = document.createElement("div");
      const valueSection = document.createElement("div");

      renderMatMul(scoreSection, {
        formula: String.raw`s=QK^\top`,
        aLabel: `Q head ${head} (target token)`, a: query,
        bLabel: "K^T (all real tokens)", b: transpose(keys),
        outLabel: "raw scores", colLabels: T.tokens.map(tokenName), colWindow: T.tokens.length,
      });
      const softmaxSteps = [
        { label: "raw QK^T", values: rawScores },
        { label: `divide by sqrt(${headDim})`, values: scaledScores },
      ];
      if (cfg.positional_scheme === "alibi") softmaxSteps.push({ label: "+ real ALiBi bias", values: aliBiasedScores });
      softmaxSteps.push({ label: "causal mask M", values: aliBiasedScores }, { label: "exp(score - max)", values: exponentials }, { label: `divide by ${denominator.toFixed(3)}`, values: probabilities });
      renderElementwise(softmaxSection, {
        title:
          cfg.positional_scheme === "alibi"
            ? String.raw`p=\operatorname{softmax}\!\left(s/\sqrt{${headDim}}+\text{bias}+M\right)`
            : String.raw`p=\operatorname{softmax}\!\left(s/\sqrt{${headDim}}+M\right)`,
        steps: softmaxSteps,
        take: T.tokens.length,
      });
      renderMatMul(valueSection, {
        formula: String.raw`\text{head}_0=pV`,
        aLabel: "attention probabilities", a: probabilities,
        bLabel: `V head ${kvHead} (all tokens)`, b: L0.v[kvHead],
        outLabel: "head 0 context", colLabels: Array.from({ length: headDim }, (_, index) => `d${index}`),
      });
      container.replaceChildren(scoreSection, softmaxSection, valueSection);
      noteEl(container, "Our followed token is last in the prompt, so its causal-mask row has no blocked future positions.");
    },

    "attn-out"(container) {
      const cfg = TJ.store.cfg();
      const W = TJ.store.weights();
      const target0 = TJ.store.target0();
      const concatenated = Array.from({ length: cfg.n_heads }, (_, head) => attentionHeadOutput(head)).flat();
      const projected = matVec(concatenated, W.w_o);
      renderMatMul(container, {
        formula: String.raw`\text{AttentionOut}=\operatorname{concat}(\text{head}_0,\ldots,\text{head}_{${cfg.n_heads - 1}})W_O`,
        aLabel: `concatenated ${cfg.n_heads} heads`, a: concatenated,
        bLabel: "W_O (real)", b: W.w_o,
        outLabel: "attention output", colLabels: Array.from({ length: cfg.d_model }, (_, index) => `d${index}`),
      });
      noteEl(container, `Reconstruction matches the captured attention tensor; max absolute difference ${maxAbsDiff(projected, target0.attn_output).toExponential(2)}.`);
    },

    "attn-plus"(container) {
      const T = TJ.store.data();
      const target0 = TJ.store.target0();
      const output = addVectors(T.target_embedding, target0.attn_output);
      renderElementwise(container, {
        title: String.raw`x\leftarrow x+\operatorname{Attention}(\operatorname{RMSNorm}(x))`,
        steps: [
          { label: "x before attention", values: T.target_embedding },
          { label: "+ attention output", values: target0.attn_output },
          { label: "= x after residual", values: output },
        ],
      });
      noteEl(container, "This is an exact sum of two captured vectors; no learned weights are involved in a residual add.");
    },

    "moe-norm"(container) {
      const T = TJ.store.data();
      const W = TJ.store.weights();
      const target0 = TJ.store.target0();
      const afterAttention = addVectors(T.target_embedding, target0.attn_output);
      rmsNormPipeline(container, afterAttention, W.moe_norm_gamma, "This exact output is sent to both the router and every active expert.");
    },

    "moe-router"(container) {
      const cfg = TJ.store.cfg();
      const L0 = TJ.store.layer0();
      const T = TJ.store.data();
      const W = TJ.store.weights();
      const target0 = TJ.store.target0();
      const logits = L0.router_logits[T.target_position];
      const maximum = Math.max(...logits);
      const exponentials = logits.map((value) => Math.exp(value - maximum));
      const denominator = exponentials.reduce((sum, value) => sum + value, 0);
      const projectionSection = document.createElement("div");
      const softmaxSection = document.createElement("div");
      renderMatMul(projectionSection, {
        formula: String.raw`\text{router\_logits}=\text{RMSNorm}(x)W_{router}`,
        aLabel: "RMSNorm(x)", a: W.moe_norm_output_target,
        bLabel: "W_router (real, full)", b: W.router_w,
        outLabel: "logits", colLabels: Array.from({ length: cfg.n_routed_experts }, (_, index) => `E${index}`),
        colWindow: cfg.n_routed_experts,
      });
      renderElementwise(softmaxSection, {
        title: String.raw`p_i=\dfrac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}`,
        steps: [
          { label: "router logits z", values: logits },
          { label: "exp(z - max)", values: exponentials },
          { label: `divide by ${denominator.toFixed(3)}`, values: target0.router_probs },
        ],
        take: cfg.n_routed_experts,
      });
      container.replaceChildren(projectionSection, softmaxSection);
    },

    "moe-topk"(container) {
      const cfg = TJ.store.cfg();
      const target0 = TJ.store.target0();
      const probabilities = target0.router_probs;
      const selected = target0.topk_indices;
      const selectedProbabilities = selected.map((expert) => probabilities[expert]);
      const denominator = selectedProbabilities.reduce((sum, value) => sum + value, 0);
      const cards = probabilities.map((probability, expert) => {
        const rank = selected.indexOf(expert);
        return `<div class="topk-card${rank >= 0 ? " selected" : " skipped"}" style="--stage:${expert}">
          <span class="topk-rank">${rank >= 0 ? `rank ${rank + 1}` : "skipped"}</span>
          <strong>Expert ${expert}</strong><span>p = ${probability.toFixed(4)}</span>
        </div>`;
      }).join("");
      container.innerHTML = `<div class="mm-formula-line"></div><div class="topk-grid">${cards}</div>`;
      tex(container.querySelector(".mm-formula-line"), String.raw`I=\operatorname{topk}(p,${cfg.top_k}),\qquad\hat p_i=\dfrac{p_i}{\sum_{j\in I}p_j}`, true);
      const normalization = document.createElement("div");
      container.appendChild(normalization);
      renderElementwise(normalization, {
        title: String.raw`p_{selected}\rightarrow\text{renormalize by }${denominator.toFixed(4)}`,
        steps: [
          { label: `selected ${selected.map((expert) => `E${expert}`).join(", ")}`, values: selectedProbabilities },
          { label: "divide by selected sum", values: target0.topk_weights },
        ],
        take: cfg.top_k,
      });
      noteEl(container, `Only experts ${selected.join(" and ")} execute for this token; their final weights sum to ${target0.topk_weights.reduce((sum, value) => sum + value, 0).toFixed(3)}.`);
    },

    "moe-experts"(container) {
      const cfg = TJ.store.cfg();
      const W = TJ.store.weights();
      const target0 = TJ.store.target0();
      const keys = [...target0.topk_indices.map(String), ...Array.from({ length: cfg.n_shared_experts }, (_, index) => `shared${index}`)];
      container.innerHTML = `<div class="mm-formula-line"></div><div class="expert-tabs" role="tablist"></div><div class="expert-body"></div>`;
      tex(container.querySelector(".mm-formula-line"), String.raw`E(x)=\big(\operatorname{SiLU}(xW_{gate})\odot xW_{up}\big)W_{down}`, true);
      const tabs = container.querySelector(".expert-tabs");
      const body = container.querySelector(".expert-body");

      function renderExpert(key) {
        TJ.matrixViz.stopAll();
        const result = expertForward(key);
        const label = key.startsWith("shared") ? `Shared expert ${key.slice(6)}` : `Routed expert ${key}`;
        const gateSection = document.createElement("div");
        const upSection = document.createElement("div");
        const activationSection = document.createElement("div");
        const downSection = document.createElement("div");
        renderMatMul(gateSection, {
          formula: String.raw`g=xW_{gate}`, aLabel: "MoE-normalized x", a: W.moe_norm_output_target,
          bLabel: `${label} W_gate`, b: W.experts[key].w_gate, outLabel: "gate",
          colLabels: Array.from({ length: cfg.d_ff }, (_, index) => `hidden ${index}`),
        });
        renderMatMul(upSection, {
          formula: String.raw`u=xW_{up}`, aLabel: "same normalized x", a: W.moe_norm_output_target,
          bLabel: `${label} W_up`, b: W.experts[key].w_up, outLabel: "up",
          colLabels: Array.from({ length: cfg.d_ff }, (_, index) => `hidden ${index}`),
        });
        renderElementwise(activationSection, {
          title: String.raw`m=\operatorname{SiLU}(g)\odot u`,
          steps: [
            { label: "gate g", values: result.gate },
            { label: "SiLU(g)", values: result.activatedGate },
            { label: "up u", values: result.up },
            { label: "elementwise product m", values: result.mixed },
          ],
        });
        renderMatMul(downSection, {
          formula: String.raw`E(x)=mW_{down}`, aLabel: "gated hidden m", a: result.mixed,
          bLabel: `${label} W_down`, b: W.experts[key].w_down, outLabel: "expert output",
          colLabels: Array.from({ length: cfg.d_model }, (_, index) => `d${index}`),
        });
        body.replaceChildren(gateSection, upSection, activationSection, downSection);
        tabs.querySelectorAll("button").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.key === key)));
      }

      for (const key of keys) {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.key = key;
        button.setAttribute("role", "tab");
        button.textContent = key.startsWith("shared") ? `Shared ${key.slice(6)}` : `Routed E${key}`;
        button.addEventListener("click", () => renderExpert(key));
        tabs.appendChild(button);
      }
      renderExpert(keys[0]);
      noteEl(container, "The tabs include only experts that actually ran. The shared expert always executes and is added at full weight.");
    },

    "moe-combine"(container) {
      const cfg = TJ.store.cfg();
      const target0 = TJ.store.target0();
      const routed = target0.topk_indices.map((expert, index) => {
        const output = expertForward(String(expert)).output;
        const weight = target0.topk_weights[index];
        return { expert, weight, weighted: output.map((value) => value * weight) };
      });
      const shared = Array.from({ length: cfg.n_shared_experts }, (_, index) => expertForward(`shared${index}`).output);
      const combined = addVectors(...routed.map((item) => item.weighted), ...shared);
      const rows = [
        ...routed.map((item, index) => ({ label: `E${item.expert}(x) x ${item.weight.toFixed(4)}`, values: item.weighted, stage: index })),
        ...shared.map((values, index) => ({ label: `Shared ${index}(x) x 1`, values, stage: routed.length + index })),
        { label: "sum = MoE output", values: combined, stage: routed.length + shared.length, result: true },
      ];
      container.innerHTML = `<div class="mm-formula-line"></div><div class="sum-flow">${rows.map((row) => `
        <div class="sum-row${row.result ? " result" : ""}" style="--stage:${row.stage}">
          <div class="mm-label">${row.label}</div>
          <div class="mm-vec">${row.values.slice(0, 8).map((value) => cellDiv(value, row.result ? "active" : "")).join("")}</div>
          <div class="mm-more">+${row.values.length - 8} more dims</div>
        </div>`).join("")}</div>`;
      tex(container.querySelector(".mm-formula-line"), String.raw`y=\sum_{i\in\operatorname{topk}}\hat p_iE_i(x)+\sum_jS_j(x)`, true);
      noteEl(container, `Reconstructed sum matches the captured MoE output; max absolute difference ${maxAbsDiff(combined, target0.moe_output).toExponential(2)}.`);
    },

    "moe-plus"(container) {
      const T = TJ.store.data();
      const target0 = TJ.store.target0();
      const before = addVectors(T.target_embedding, target0.attn_output);
      const after = addVectors(before, target0.moe_output);
      renderElementwise(container, {
        title: String.raw`x\leftarrow x+\operatorname{MoE}(\operatorname{RMSNorm}(x))`,
        steps: [
          { label: "x after attention residual", values: before },
          { label: "+ combined MoE output", values: target0.moe_output },
          { label: "= block 1 output", values: after },
        ],
      });
      noteEl(container, "This exact addition completes Transformer block 1.");
    },

    "final-norm"(container) {
      const T = TJ.store.data();
      const W = TJ.store.weights();
      let beforeFinal = T.target_embedding.slice();
      for (const layer of T.layers) beforeFinal = addVectors(beforeFinal, layer.target.attn_output, layer.target.moe_output);
      rmsNormPipeline(container, beforeFinal, W.final_norm_gamma, "The input is reconstructed by summing every captured residual across all blocks.");
    },

    "lm-head"(container) {
      const cfg = TJ.store.cfg();
      const T = TJ.store.data();
      const W = TJ.store.weights();
      const logits = T.logits[T.target_position];
      const top = logits.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, 6);
      const matrix = transpose(W.embedding_full);
      const reconstructed = matVec(T.target_final_norm_output, matrix);
      renderMatMul(container, {
        formula: String.raw`\text{logits}=x_{final}W_{LM}^{\top},\qquad W_{LM}=W_{embedding}`,
        aLabel: "final normalized token", a: T.target_final_norm_output,
        bLabel: "tied embedding^T (top columns)", b: matrix,
        outLabel: "vocabulary logits", colLabels: T.vocab.map(tokenName),
        colIndices: top.map((item) => item.index), colWindow: top.length,
      });
      noteEl(container, `These are the real six highest-logit columns, not the first six vocabulary entries. Max reconstruction difference ${maxAbsDiff(reconstructed, logits).toExponential(2)}.`);
    },

    output(container) {
      const cfg = TJ.store.cfg();
      const T = TJ.store.data();
      const logits = T.logits[T.target_position];
      const maximum = Math.max(...logits);
      const exponentials = logits.map((value) => Math.exp(value - maximum));
      const denominator = exponentials.reduce((sum, value) => sum + value, 0);
      const top = logits.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value).slice(0, 6);
      renderElementwise(container, {
        title: String.raw`\operatorname{softmax}(z)_i=\dfrac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}`,
        steps: [
          { label: "real top-6 logits", values: top.map((item) => item.value) },
          { label: "exp(logit - max)", values: top.map((item) => Math.exp(item.value - maximum)) },
          { label: `divide by all-${cfg.vocab_size} sum ${denominator.toFixed(3)}`, values: top.map((item) => Math.exp(item.value - maximum) / denominator) },
        ],
      });
      const probability = Math.exp(top[0].value - maximum) / denominator;
      noteEl(container, `Top prediction: ${tokenName(T.vocab[top[0].index])} at p=${probability.toFixed(4)}.`);
    },
  };
})();
