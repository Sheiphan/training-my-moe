/* site/token-journey/js/steps.js
 * The ordered "slide deck" -- one real token's journey through one real
 * forward pass. Narrative text is beginner-friendly (defines terms), each
 * formula is a KaTeX string, `focus` is a diagram.js data-focus id (or
 * array of ids) for camera.js, and `real` (when true) tells app.js to ask
 * render-token.js for that step's real-number panel.
 *
 * `TJ.buildSteps(variantData)` returns the step list for ONE (attention,
 * positional) variant -- the 5 attention-block steps (attn-qkv, attn-rope,
 * attn-group, attn-sdpa, attn-out) change their narrative/formula/deep-dive
 * meaning per variant (see `_attentionSteps` below); every other step is
 * attention-variant-agnostic and stays the same shape for all 4 variants.
 */
(function () {
  "use strict";
  const TJ = (window.TJ = window.TJ || {});

  /** "n choose k", formatted with thousands separators. */
  function combinations(n, k) {
    let r = 1;
    for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
    return Math.round(r).toLocaleString();
  }

  function _attentionSteps(cfg) {
    if (cfg.attention_variant === "mla") {
      const headDim = cfg.d_model / cfg.n_heads;
      const ropeHeadDim = Math.floor(headDim / 2);
      return [
        {
          id: "attn-qkv",
          title: "Compress to a latent + project a content query",
          focus: "attn-qkv1",
          narrative: `Instead of separate Key/Value projections, MLA first compresses the normalized vector down to one small shared "latent" -- this latent is the ONLY thing that needs to be cached during generation. A plain (uncompressed) Query is projected the usual way.`,
          formula: "c_{kv} = xW_{dkv} \\in \\mathbb{R}^{d_{latent}}, \\qquad q_c = xW_q",
          real: "qkv",
        },
        {
          id: "attn-rope",
          title: "Up-project the latent back to Key and Value",
          focus: "attn-rope1",
          narrative: `The small latent is decompressed back to full-size Key and Value, once per head -- this decompression happens fresh on every forward pass (it's cheap; only the small latent itself was ever cached).`,
          formula: "k_c = c_{kv}W_{uk},\\qquad v = c_{kv}W_{uv} \\in \\mathbb{R}^{n_{heads}\\times d_h}",
          real: "mla_uproj",
        },
        {
          id: "attn-group",
          title: "Decoupled RoPE: a small rotating slice",
          focus: "attn-group1",
          narrative: `MLA carries ALL of its positional information in one small extra slice, rotated by RoPE exactly like before -- and this slice is ALWAYS rotated, no matter what ModelConfig.positional_scheme says elsewhere (a documented MLA exception). The query gets its own per-head slice; the key's slice is shared across every head.`,
          formula: `\\theta_i = \\text{pos}\\cdot ${cfg.rope_theta}^{-2i/${ropeHeadDim}},\\qquad q_r, k_r = \\text{RoPE}(xW_{qr}, xW_{kr})`,
          real: "mla_rope",
        },
        {
          id: "attn-sdpa",
          title: "Rejoin content + rotary parts (per head), then attention",
          focus: "attn-sdpa1",
          narrative: `For EACH head separately, its content slice and its rotary slice are joined back into one vector for Query and one for Key (this is a per-head rejoin, NOT the across-heads concatenation that happens later) -- then attention proceeds exactly as usual: compare against every earlier token (never a future one), softmax, weighted sum of Values.`,
          formula: `q=[q_c;q_r],\\ k=[k_c;k_r],\\qquad \\text{Attn} = \\text{softmax}\\!\\left(\\dfrac{qk^\\top}{\\sqrt{${headDim}+${ropeHeadDim}}} + M\\right)v`,
          real: "attn",
        },
        {
          id: "attn-out",
          title: "Weighted sum + output projection",
          focus: "attn-out1",
          narrative: `Just like GQA: THIS is where the heads actually combine -- all ${cfg.n_heads} heads' results are concatenated side-by-side into one long vector, then passed through one more learned projection (W_O) back to the model's normal width.`,
          formula: "\\text{out} = \\text{concat}(\\text{head}_1,\\dots,\\text{head}_h)\\,W_O",
          real: "attnout",
        },
      ];
    }

    const nRep = cfg.n_heads / cfg.n_kv_heads;
    const positionalStep = {
      rope: {
        title: "Apply RoPE (positional encoding)",
        narrative: `Attention alone has no sense of word order -- so before comparing Query and Key, each is rotated by an angle that depends on the token's position in the sequence (Rotary Position Embedding). Nearby positions end up with similar rotation angles, distant ones very different -- that's how the model learns "how far apart" two tokens are.`,
        formula: "\\theta_i = \\text{pos} \\cdot 10000^{-2i/d_h}, \\qquad q'_i = q_i\\cos\\theta_i + \\text{rotate\\_half}(q)_i\\sin\\theta_i",
        real: "rope",
      },
      nope: {
        title: "No positional encoding (NoPE)",
        narrative: `This variant applies NO positional operation at all here -- Query and Key pass through completely unchanged. Token order is carried only by the causal mask (a token can never see the future), which turns out to still be enough signal for the model to learn from.`,
        formula: "q' = q, \\qquad k' = k \\qquad (\\text{no-op})",
        real: false,
      },
      alibi: {
        title: "ALiBi (deferred to the score step)",
        narrative: `ALiBi doesn't touch Query or Key here at all -- instead it adds a fixed, distance-based penalty directly to the raw attention SCORES in the next step: the further back a key token is, the more its score gets penalized, with a per-head slope fixed by a geometric sequence (no learned parameters).`,
        formula: "\\text{bias}_h[i,j] = -\\,\\text{slope}_h \\cdot (i-j), \\qquad \\text{applied at the SDPA step, not here}",
        real: "alibi_bias",
      },
    }[cfg.positional_scheme];

    // QK-norm (moe-modernization Requirement 7.3) sits between the QKV
    // projection and the positional transform, and only exists as a step
    // when the checkpoint actually has it -- so BASELINE and BASELINE_V2
    // differ by a visible extra stop on the walkthrough, not just by a
    // number.
    const qkNormStep = cfg.qk_norm
      ? [
          {
            id: "attn-qknorm",
            title: "QK-norm: pin the Query/Key scale",
            focus: "attn-qknorm1",
            narrative: `Before position is applied, Query and Key are each passed through their own RMSNorm, separately for every head. Why: the residual stream grows as the network gets deeper, so q and k grow with it, and the attention score is a dot product of the two — meaning the scores grow roughly with the SQUARE of that drift. Left alone, deep layers push softmax toward a one-hot spike, and a saturated softmax has almost no gradient left to learn from. Normalizing q and k first pins their length no matter how deep we are, so the scores stay in a range softmax can actually work with.`,
            formula:
              "q \\leftarrow \\dfrac{q}{\\sqrt{\\frac{1}{d_h}\\sum_i q_i^2+\\epsilon}}\\odot\\gamma_q, \\qquad k \\leftarrow \\dfrac{k}{\\sqrt{\\frac{1}{d_h}\\sum_i k_i^2+\\epsilon}}\\odot\\gamma_k",
            real: "qknorm",
          },
        ]
      : [];

    return [
      {
        id: "attn-qkv",
        title: "Project to Query, Key, Value",
        focus: "attn-qkv1",
        narrative: `The normalized vector is projected through three separate learned matrices into a Query, a Key, and a Value. Query is "what am I looking for", Key is "what do I contain" (every other token has one too), and Value is "what do I actually contribute" once a match is found. Because this model uses Grouped-Query Attention, there are ${cfg.n_heads} query heads but only ${cfg.n_kv_heads} key/value heads.`,
        formula: "Q = xW_Q,\\qquad K = xW_K,\\qquad V = xW_V",
        real: "qkv",
      },
      ...qkNormStep,
      { id: "attn-rope", focus: "attn-rope1", ...positionalStep },
      {
        id: "attn-group",
        title: "GQA head grouping",
        focus: "attn-group1",
        narrative:
          nRep > 1
            ? `Since there are only ${cfg.n_kv_heads} key/value heads but ${cfg.n_heads} query heads, each key/value head is duplicated (repeat_interleave) so every query head has a matching key/value to compare against. This is the whole point of Grouped-Query Attention: fewer K/V heads means a smaller cache to store during generation, at almost no quality cost.`
            : `n_kv_heads equals n_heads here, so every query head already has its own dedicated key/value head -- this is plain multi-head attention, no grouping needed.`,
        formula: nRep > 1 ? "K, V \\leftarrow \\text{repeat\\_interleave}(K, V,\\; n_{rep}),\\qquad n_{rep} = n_{heads}/n_{kv}" : null,
        real: false,
      },
      {
        id: "attn-sdpa",
        title: "Scaled dot-product attention",
        focus: "attn-sdpa1",
        narrative: `Now our token's Query is compared (dot product) against every token's Key up to and including itself -- a future token is never allowed to influence the past, which is what the causal mask enforces. Softmax turns those comparison scores into a proper probability distribution: "how much should I pay attention to each earlier token?"${cfg.positional_scheme === "alibi" ? " ALiBi's fixed distance penalty is added right here, before softmax." : ""}`,
        formula:
          cfg.positional_scheme === "alibi"
            ? "\\text{Attn} = \\text{softmax}\\!\\left(\\dfrac{QK^\\top}{\\sqrt{d_h}} + \\text{bias} + M\\right)V"
            : "\\text{Attn} = \\text{softmax}\\!\\left(\\dfrac{QK^\\top}{\\sqrt{d_h}} + M\\right)V",
        real: "attn",
      },
      {
        id: "attn-out",
        title: "Weighted sum + output projection",
        focus: "attn-out1",
        narrative: `Those attention probabilities are used to take a weighted sum of every token's Value -- so our token now carries a blend of information from the tokens it attended to. THIS is where the heads actually combine: all ${cfg.n_heads} heads' results are concatenated side-by-side into one long vector, then passed through one more learned projection (W_O) back to the model's normal width.`,
        formula: "\\text{head}_i = \\text{Attn}_i \\cdot V_i,\\qquad \\text{out} = \\text{concat}(\\text{head}_1,\\dots,\\text{head}_h)\\,W_O",
        real: "attnout",
      },
    ];
  }

  TJ.buildSteps = function buildSteps(variantData) {
    const cfg = variantData.config;
    const attnSteps = _attentionSteps(cfg).map((step) => ({ ...step, chapter: "Attention \u00b7 Block 1" }));

    return [
      {
        id: "intro",
        chapter: "Setup",
        title: "Meet the token",
        focus: "overview",
        narrative: `We're going to follow ONE real token through this model's actual forward pass: the prompt is "${variantData.prompt}", and we'll track its LAST character ("${variantData.target_token === "\n" ? "\\n" : variantData.target_token}"), because that token's output is what the model uses to predict what comes next. Every number you'll see from here on is real, captured from an actual trained checkpoint -- nothing here is illustrative.`,
        formula: null,
        real: false,
      },
      {
        id: "embed",
        chapter: "Setup",
        title: "Tokenize & Embed",
        focus: ["tokens", "embedding", "embeddings-vec"],
        narrative: `First the text is split into tokens (here, one token per character). Each token id is looked up in the embedding table -- a big learned matrix with one row per vocabulary entry -- producing a ${cfg.d_model}-number vector that represents that character.`,
        formula: "x_0 = \\text{Embedding}[\\text{token\\_id}] \\in \\mathbb{R}^{d_{model}}",
        real: "embed",
      },
      {
        id: "attn-norm",
        chapter: "Attention \u00b7 Block 1",
        title: "Pre-Attention RMSNorm",
        focus: "attn-norm1",
        narrative: `Before doing anything else, the block rescales the vector with RMSNorm: divide by the root-mean-square of its own values, then multiply by a learned per-dimension scale \u03b3. This keeps activations at a stable, consistent size no matter how deep the network is -- and it happens BEFORE attention ("pre-norm"), which trains more stably than normalizing after.`,
        formula: "\\text{RMSNorm}(x) = \\dfrac{x}{\\sqrt{\\frac{1}{d}\\sum_i x_i^2 + \\epsilon}} \\odot \\gamma",
        real: false,
      },
      ...attnSteps,
      {
        id: "attn-plus",
        chapter: "Attention \u00b7 Block 1",
        title: "Residual add",
        focus: "attn-plus1",
        narrative: `The attention output is added back to the vector from BEFORE this whole attention step (the "residual connection"). This shortcut is what makes it possible to stack many blocks without gradients vanishing on the way back during training -- the original signal always has a direct path through.`,
        formula: "x \\leftarrow x + \\text{Attention}(\\text{RMSNorm}(x))",
        real: "attnresid",
      },
      {
        id: "moe-norm",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: "Pre-MoE RMSNorm",
        focus: "moe-norm1",
        narrative: `Same idea as before -- the vector is rescaled with RMSNorm again, this time before the Mixture-of-Experts feed-forward step.`,
        formula: "\\text{RMSNorm}(x) = \\dfrac{x}{\\sqrt{\\frac{1}{d}\\sum_i x_i^2 + \\epsilon}} \\odot \\gamma",
        real: false,
      },
      {
        id: "moe-router",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: cfg.router_scoring === "sigmoid" ? "Router \u2192 sigmoid affinities" : "Router \u2192 softmax",
        focus: ["moe-router1", "moe-balance1"],
        narrative:
          cfg.router_scoring === "sigmoid"
            ? `Instead of running every token through one giant feed-forward network, this model has ${cfg.n_routed_experts} smaller "expert" networks and a router that decides which ones each token should use. The router is one more learned linear layer producing a score per expert — but unlike the original design, each score is turned into an affinity INDEPENDENTLY with a sigmoid, then the whole vector is renormalized. That independence is the point: with a softmax all experts share one denominator, so raising one expert's score necessarily pushes every other one down and a single expert can run away with the token. A sigmoid asks each expert "how well do you match this token?" on its own terms, which leaves the gate weights flatter and keeps more than one expert receiving real gradient.`
            : `Instead of running every token through one giant feed-forward network, this model has ${cfg.n_routed_experts} smaller "expert" networks and a router that decides which ones each token should use. The router is just one more learned linear layer producing a score per expert, turned into a probability distribution by softmax — one shared denominator, so the experts compete directly against each other.`,
        formula:
          cfg.router_scoring === "sigmoid"
            ? "s_i = \\sigma(xW_{router})_i,\\qquad p_i = s_i \\big/ \\textstyle\\sum_j s_j"
            : "p = \\text{softmax}(xW_{router}) \\in \\mathbb{R}^{n_{experts}}",
        real: "router",
      },
      {
        id: "moe-topk",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: "Top-k selection",
        focus: "moe-topk1",
        narrative:
          `Only the top ${cfg.top_k} highest-scoring experts are actually used for this token -- the rest are skipped entirely (no compute spent on them at all). The chosen experts' weights are renormalized so they sum to 1, then used to weight each expert's contribution.` +
          (cfg.n_routed_experts >= 16
            ? ` This model uses DeepSeekMoE fine-grained experts: ${cfg.n_routed_experts} narrow experts of width ${cfg.d_ff} instead of a handful of wide ones, picking ${cfg.top_k} of them. Both the total expert width and the width each token activates are unchanged versus the coarse version -- what changes is that there are now ${combinations(cfg.n_routed_experts, cfg.top_k)} distinct ways to combine experts, so routing can specialize far more finely for the same compute.`
            : ""),
        formula: "\\text{top-}k(p),\\qquad \\hat p_i = p_i \\big/ \\textstyle\\sum_{j\\,\\in\\,\\text{top-}k} p_j",
        real: "topk",
      },
      {
        id: "moe-experts",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: "Expert computation",
        focus: ["moe-experts1", "moe-shared1"],
        narrative: `Each chosen expert is a small SwiGLU feed-forward network: it expands the vector to a wider hidden size, applies a gated nonlinearity, then projects back down. Alongside the ${cfg.top_k} chosen routed experts, ${cfg.n_shared_experts} "shared" expert always runs too, for every token, no matter what the router decides -- a safety net of always-on general-purpose capacity.`,
        formula: "\\text{Expert}(x) = W_{down}\\big(\\text{SiLU}(W_{gate}x) \\odot W_{up}x\\big)",
        real: "experts",
      },
      {
        id: "moe-combine",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: "Combine",
        focus: "moe-combine1",
        narrative: `Every active expert's output is combined into one vector: the routed experts are weighted by their renormalized gate value, the shared expert(s) are added at full weight (unweighted), and it all sums together.`,
        formula: "y = \\sum_{i\\,\\in\\,\\text{top-}k} \\hat p_i \\cdot E_i(x) \\;+\\; \\sum_j S_j(x)",
        real: "combine",
      },
      {
        id: "moe-plus",
        chapter: "Mixture of Experts \u00b7 Block 1",
        title: "Residual add",
        focus: "moe-plus1",
        narrative: `Just like after attention, the MoE output is added back to the vector from before this step. That completes one full Transformer block.`,
        formula: "x \\leftarrow x + \\text{MoE}(\\text{RMSNorm}(x))",
        real: "moeresid",
      },
      {
        id: "repeat",
        chapter: "Repeat & Output",
        title: `This repeats \u00d7 ${cfg.n_layers} layers`,
        focus: "block1-container",
        narrative: `Everything you just saw -- RMSNorm, attention, residual, RMSNorm, MoE, residual -- is ONE block ("layer"). This model stacks ${cfg.n_layers} of them SEQUENTIALLY, each with its own independently learned weights: block 1's output becomes block 2's input, and so on -- layers are NEVER concatenated together (that's what happens to attention HEADS, back at the "Weighted sum + output projection" step; layers just hand the same-sized vector forward, one at a time, so the representation gets progressively richer before it ever reaches the output).`,
        formula: null,
        real: false,
      },
      {
        id: "final-norm",
        chapter: "Repeat & Output",
        title: "Final RMSNorm",
        focus: "final-norm",
        narrative: `After the very last block, one more RMSNorm is applied -- this is the one moment the model "settles" the representation before turning it into predictions.`,
        formula: "\\text{RMSNorm}(x) = \\dfrac{x}{\\sqrt{\\frac{1}{d}\\sum_i x_i^2 + \\epsilon}} \\odot \\gamma",
        real: "finalnorm",
      },
      {
        id: "lm-head",
        chapter: "Repeat & Output",
        title: "LM Head \u2192 logits",
        focus: "lm-head",
        narrative: `One last linear layer projects the ${cfg.d_model}-number vector back up to a score ("logit") for every one of the ${cfg.vocab_size} possible next characters.${cfg.tie_word_embeddings ? " This layer's weights are literally the SAME matrix as the embedding table from step 2 (\u201ctied embeddings\u201d), reused in reverse." : ""}`,
        formula: "\\text{logits} = xW_{LM}^\\top" + (cfg.tie_word_embeddings ? ",\\qquad W_{LM} = \\text{Embedding}" : ""),
        real: "logits",
      },
      {
        id: "output",
        chapter: "Repeat & Output",
        title: "Softmax \u2192 prediction",
        focus: ["softmax", "output-bars"],
        narrative: `Softmax turns those logits into a real probability distribution over every possible next character. Every bar in the diagram above is a genuine measured probability from this exact checkpoint -- not a guess.`,
        formula: "p = \\text{softmax}(\\text{logits})",
        real: "output",
      },
    ];
  };
})();

