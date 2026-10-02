/* site/shared/qwenmoe-model.js
 * The QwenMoE architecture as data: configuration slots and their variants, parameter
 * counting, the forward-pass step list, and the paper-style figure. Pure functions, no DOM,
 * so the same code runs in the browser (window.QM) and in Node (module.exports), which is
 * how scripts/render_model_figure.py exports the static SVG and how
 * scripts/render_tensor_walk.py verifies params() against the real PyTorch model.
 *
 * Variants marked repo:true exist in qwenmoe/; the others are drawn from the cited papers
 * and are labelled "not in this repo" wherever they appear.
 */
(function (root) {
  "use strict";

  const CITE = {
    layernorm: { label: "Ba et al. 2016, Layer Normalization", href: "https://arxiv.org/abs/1607.06450" },
    vaswani: { label: "Vaswani et al. 2017, Attention Is All You Need", href: "https://arxiv.org/abs/1706.03762" },
    olmo2: { label: "OLMo 2 (2025): norm on the sublayer output", href: "https://arxiv.org/abs/2501.00656" },
    gemma2: { label: "Gemma 2 (2024): pre- and post-norm", href: "https://arxiv.org/abs/2408.00118" },
    gpt2: { label: "Radford et al. 2019 (GPT-2): learned positions", href: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
    mistral: { label: "Mistral 7B (2023): sliding-window attention", href: "https://arxiv.org/abs/2310.06825" },
    gemma3: { label: "Gemma 3 (2025): 5 local : 1 global layers", href: "https://arxiv.org/abs/2503.19786" },
    primer: { label: "So et al. 2021 (Primer): squared ReLU", href: "https://arxiv.org/abs/2109.08668" },
    expertChoice: { label: "Zhou et al. 2022, Expert Choice Routing", href: "https://arxiv.org/abs/2202.09368" },
    switch: { label: "Fedus et al. 2021 (Switch): capacity factor", href: "https://arxiv.org/abs/2101.03961" },
    deepseek3: { label: "DeepSeek-V3 (2024): first layers dense", href: "https://arxiv.org/abs/2412.19437" },
    llama4: { label: "Llama 4 (2025): dense and MoE layers interleaved", href: "https://huggingface.co/blog/llama4-release" },
  };

  /* every slot the model has, and what can go in it */
  const SLOTS = [
    { id: "norm", label: "Normalization", group: "Norms", options: [
      { id: "rms", label: "RMSNorm", repo: true },
      { id: "layer", label: "LayerNorm", cite: CITE.layernorm } ] },
    { id: "place", label: "Norm placement", group: "Norms", options: [
      { id: "pre", label: "Pre-norm", repo: true },
      { id: "post", label: "Post-norm (after the add)", cite: CITE.vaswani },
      { id: "out", label: "Output norm (inside the residual)", cite: CITE.olmo2 },
      { id: "sandwich", label: "Pre + output (sandwich)", cite: CITE.gemma2 } ] },
    { id: "pos", label: "Positions", group: "Attention", options: [
      { id: "rope", label: "RoPE", repo: true },
      { id: "nope", label: "NoPE (none)", repo: true },
      { id: "alibi", label: "ALiBi", repo: true },
      { id: "learned", label: "Learned absolute", cite: CITE.gpt2 },
      { id: "sinusoidal", label: "Sinusoidal absolute", cite: CITE.vaswani } ] },
    { id: "attn", label: "Attention", group: "Attention", options: [
      { id: "gqa", label: "GQA (4 KV heads)", repo: true },
      { id: "mha", label: "MHA (12 KV heads)", repo: true },
      { id: "mqa", label: "MQA (1 KV head)", repo: true },
      { id: "mla", label: "MLA (latent 192)", repo: true } ] },
    { id: "qknorm", label: "QK-Norm", group: "Attention", options: [
      { id: "on", label: "On", repo: true }, { id: "off", label: "Off", repo: true } ] },
    { id: "window", label: "Attention span", group: "Attention", options: [
      { id: "global", label: "Global (causal)", repo: true },
      { id: "sliding", label: "Sliding window (256)", cite: CITE.mistral } ] },
    { id: "attnPattern", label: "Span across layers", group: "Layers", options: [
      { id: "uniform", label: "Same in every layer", repo: true },
      { id: "gemma3", label: "5 sliding : 1 global", cite: CITE.gemma3 },
      { id: "alternate", label: "Alternate sliding / global", cite: CITE.mistral } ] },
    { id: "ffn", label: "Feed-forward", group: "Feed-forward", options: [
      { id: "moe", label: "Mixture of Experts", repo: true },
      { id: "dense", label: "Dense FFN", cite: CITE.vaswani } ] },
    { id: "ffnPattern", label: "FFN across layers", group: "Layers", options: [
      { id: "uniform", label: "Same in every layer", repo: true },
      { id: "firstDense", label: "First layer dense, rest MoE", cite: CITE.deepseek3 },
      { id: "alternate", label: "Alternate dense / MoE", cite: CITE.llama4 } ] },
    { id: "experts", label: "Experts", group: "Feed-forward", options: [
      { id: "coarse", label: "8 × 1,536, top-2", repo: true },
      { id: "fine", label: "32 × 384, top-8", repo: true } ] },
    { id: "shared", label: "Shared expert", group: "Feed-forward", options: [
      { id: "1", label: "1 shared", repo: true }, { id: "0", label: "None", repo: true } ] },
    { id: "act", label: "Expert / FFN type", group: "Feed-forward", options: [
      { id: "swiglu", label: "SwiGLU", repo: true },
      { id: "geglu", label: "GeGLU", repo: true },
      { id: "gelu", label: "GELU (no gate)", repo: true },
      { id: "relu2", label: "Squared ReLU (no gate)", cite: CITE.primer } ] },
    { id: "router", label: "Routing", group: "Feed-forward", options: [
      { id: "sigmoid", label: "Sigmoid, token choice", repo: true },
      { id: "softmax", label: "Softmax, token choice", repo: true },
      { id: "expertChoice", label: "Expert choice", cite: CITE.expertChoice } ] },
    { id: "balance", label: "Load balancing", group: "Training", options: [
      { id: "aux", label: "Auxiliary loss", repo: true },
      { id: "lossfree", label: "Loss-free bias", repo: true },
      { id: "capacity", label: "Aux loss + capacity 1.25 (drops tokens)", cite: CITE.switch } ] },
    { id: "zloss", label: "Extra losses", group: "Training", options: [
      { id: "none", label: "None", repo: true },
      { id: "router", label: "Router z-loss", repo: true },
      { id: "output", label: "Output z-loss", repo: true },
      { id: "both", label: "Both z-losses", repo: true } ] },
    { id: "tie", label: "Output layer", group: "Output", options: [
      { id: "tied", label: "Tied to embedding", repo: true }, { id: "untied", label: "Separate weight", repo: true } ] },
    { id: "mtp", label: "Multi-token prediction", group: "Output", options: [
      { id: "0", label: "Off", repo: true }, { id: "1", label: "1 extra head", repo: true } ] },
  ];
  const SLOT = Object.fromEntries(SLOTS.map(s => [s.id, s]));
  const opt = (slot, id) => SLOT[slot].options.find(o => o.id === id);

  const DEFAULT = { norm: "rms", place: "pre", pos: "rope", attn: "gqa", qknorm: "on", window: "global",
    attnPattern: "uniform", ffn: "moe", ffnPattern: "uniform", experts: "coarse", shared: "1", act: "swiglu",
    router: "sigmoid", balance: "aux", zloss: "none", tie: "tied", mtp: "0" };

  /* slots that a choice makes irrelevant, with the reason shown in the UI */
  function inactive(cfg) {
    const off = {};
    if (cfg.attn === "mla") {
      off.pos = "MLA applies its own decoupled RoPE whatever this is set to (attention.py)";
      off.qknorm = "MLA in this repo has no QK-Norm";
    }
    if (cfg.attnPattern !== "uniform") off.window = "set per layer by the span pattern";
    const anyMoe = cfg.ffn === "moe" || cfg.ffnPattern !== "uniform";
    if (!anyMoe) for (const s of ["experts", "shared", "router", "balance"]) off[s] = "no MoE layers";
    if (cfg.router === "expertChoice" && anyMoe) off.balance = "expert choice balances by construction";
    return off;
  }

  function dims(base, cfg) {
    const d = base.d_model, H = base.n_heads, dk = d / H;
    const Hkv = { gqa: base.n_kv_heads, mha: H, mqa: 1, mla: H }[cfg.attn];
    const fine = cfg.experts === "fine";
    const N = fine ? base.n_routed_experts * 4 : base.n_routed_experts;
    const k = fine ? base.top_k * 4 : base.top_k;
    const F = fine ? base.d_ff / 4 : base.d_ff;
    const ns = +cfg.shared;
    return {
      d, H, dk, Hkv, N, k, F, ns, V: base.vocab_size, L: base.n_layers, ctx: base.max_seq_len,
      theta: base.rope_theta, eps: 1e-6, dc: d / 4, dr: dk / 2, window: 256,
      Fd: base.top_k * base.d_ff + +cfg.shared * base.d_ff,   // dense FFN width = the MoE's active width
      gated: cfg.act === "swiglu" || cfg.act === "geglu",
    };
  }

  /* per-layer types */
  function layers(base, cfg) {
    const L = base.n_layers, out = [];
    for (let l = 0; l < L; l++) {
      let local = cfg.window === "sliding";
      if (cfg.attnPattern === "gemma3") local = (l % 6) !== 5;
      if (cfg.attnPattern === "alternate") local = l % 2 === 0;
      let dense = cfg.ffn === "dense";
      if (cfg.ffnPattern === "firstDense") dense = l === 0;
      if (cfg.ffnPattern === "alternate") dense = l % 2 === 0;
      out.push({ local, dense });
    }
    return out;
  }

  function isRepo(cfg) {
    const off = inactive(cfg);
    return SLOTS.every(s => off[s.id] || opt(s.id, cfg[s.id]).repo);
  }

  /* ---------------- parameter counting ---------------- */
  function params(base, cfg) {
    const D = dims(base, cfg), { d, H, dk, Hkv, N, k, F, ns, V } = D;
    const normP = cfg.norm === "layer" ? 2 * d : d;
    const normsPerLayer = cfg.place === "sandwich" ? 4 : 2;
    let attn;
    if (cfg.attn === "mla") {
      attn = d * D.dc + 2 * D.dc * H * dk + d * H * dk + d * H * D.dr + d * D.dr + H * dk * d;
    } else {
      attn = d * H * dk + 2 * d * Hkv * dk + H * dk * d + (cfg.qknorm === "on" ? 2 * dk : 0);
    }
    const expert = (D.gated ? 3 : 2) * d * F;
    const dense = (D.gated ? 3 : 2) * d * D.Fd;
    const moeTotal = d * N + N * expert + ns * expert;
    const moeActive = d * N + k * expert + ns * expert;
    let attnTotal = 0, ffnTotal = 0, ffnActive = 0;
    for (const ly of layers(base, cfg)) {
      attnTotal += attn + normP * (normsPerLayer / 2);
      const norms = normP * (normsPerLayer / 2);
      ffnTotal += norms + (ly.dense ? dense : moeTotal);
      ffnActive += norms + (ly.dense ? dense : moeActive);
    }
    const posP = cfg.pos === "learned" && cfg.attn !== "mla" ? D.ctx * d : 0;
    const embedding = V * d + normP + posP + (cfg.tie === "untied" ? V * d : 0);
    const mtpOne = 2 * d + 2 * d * d + d + attn + d + 3 * d * F + d;   // MTPHead: norms, fuse, attn, dense SwiGLU(d_ff)
    return {
      embedding, attention: attnTotal, ffn: ffnTotal,
      total: embedding + attnTotal + ffnTotal,
      active: embedding + attnTotal + ffnActive,
      mtp: +cfg.mtp * mtpOne,
      exact: isRepo(cfg),
    };
  }

  /* ---------------- the forward pass as steps ---------------- */
  const T = (name, shape, role = "act") => ({ name, shape, role });

  function buildSteps(base, cfg, block) {
    const D = dims(base, cfg), { d, H, dk, Hkv, N, k, F, ns, V, L } = D;
    const ly = layers(base, cfg)[block];
    const off = inactive(cfg);
    const normName = cfg.norm === "layer" ? "LayerNorm" : "RMSNorm";
    const normParam = cfg.norm === "layer" ? T("gain g, bias b", [2, d], "weight") : T("gain g", [1, d], "weight");
    const normFormula = cfg.norm === "layer"
      ? String.raw`\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot g + b`
      : String.raw`\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{${d}}\sum_j x_j^2 + \epsilon}} \odot g`;
    const lit = (slot) => { const o = opt(slot, cfg[slot]); return o.repo ? null : o.cite; };
    const steps = [];
    const S = (o) => { steps.push(o); return o; };
    const blockName = `block ${block + 1}`;

    S({ key: "ids", group: "Input", title: "Text becomes token ids", zoom: "col", focus: ["f-in"], op: "ids", spec: {},
      formula: String.raw`\text{text} \to [t_0, \dots, t_{S-1}],\ t_i \in \{0, \dots, ${V - 1}\}`,
      text: `The byte-level BPE tokenizer turns text into S integer ids from a vocabulary of ${V.toLocaleString("en-US")}. Each id is one integer; nothing is a vector yet.` });
    S({ key: "embed", group: "Input", title: "Embedding lookup", zoom: "col", focus: ["f-emb"], op: "matmul",
      spec: { a: T("one-hot ids", ["S", V]), b: T("E (embedding)", [V, d], "weight"), c: T("X", ["S", d]) },
      formula: String.raw`X = \mathrm{onehot}(t)\,E,\quad E \in \mathbb{R}^{${V} \times ${d}}`,
      text: "Looking up row t of E is the same as multiplying a one-hot row by E, which is why the one-hot rows are so long and thin. In code it is an index, not a multiply.",
      params: V * d });
    if ((cfg.pos === "learned" || cfg.pos === "sinusoidal") && !off.pos) {
      const learned = cfg.pos === "learned";
      S({ key: "pos_add", group: "Input", title: learned ? "Add learned position embeddings" : "Add sinusoidal position encodings",
        zoom: "col", focus: ["f-posadd"], op: "add", cite: lit("pos"),
        spec: { a: T("X", ["S", d]), b: T(learned ? "P[0:S] (learned)" : "P[0:S] (fixed sin/cos)", ["S", d], learned ? "weight" : "act"), out: T("X + P", ["S", d]) },
        formula: learned ? String.raw`X \leftarrow X + P_{0:S},\quad P \in \mathbb{R}^{${D.ctx} \times ${d}}`
                         : String.raw`P_{p,2i} = \sin(p / 10000^{2i/${d}}),\ P_{p,2i+1} = \cos(\cdot)`,
        text: learned ? `A second table with one learned row per position (${D.ctx} rows) is added before the first block. Attention itself then never sees positions.`
                      : "A fixed pattern of sines and cosines is added before the first block, so each position has a distinct signature without any learned weights.",
        params: learned ? D.ctx * d : 0 });
    }

    const normStep = (key, title, focus, inName, outName, cite) => S({
      key, group: key.startsWith("norm_f") ? "Feed-forward" : "Attention", title, zoom: "block", focus: [focus], op: "elementwise", cite: cite || lit("norm"),
      spec: { rows: [{ inp: T(inName, ["S", d]), out: T(outName, ["S", d]), stack: 1 }], param: normParam, label: normName },
      formula: normFormula,
      text: cfg.norm === "layer"
        ? `LayerNorm subtracts each token's mean before dividing by its standard deviation, then applies a gain and a bias: ${2 * d} parameters, twice RMSNorm's ${d}.`
        : "Each token's numbers are rescaled so their root-mean-square is 1, then multiplied by a learned gain. Rows are independent; the shape does not change.",
      params: cfg.norm === "layer" ? 2 * d : d });
    const addStep = (key, title, focus, a, b, out, group, formula) => S({
      key, group, title, zoom: "block", focus: [focus], op: "add",
      spec: { a: T(a, ["S", d]), b: T(b, ["S", d]), out: T(out, ["S", d]) },
      formula,
      text: "The residual connection adds the sublayer's result to the stream it read from. Same shape in, same shape out." });

    /* --- attention sublayer --- */
    const placeCite = lit("place");
    const pre = cfg.place === "pre" || cfg.place === "sandwich";
    if (pre) normStep("norm_a", `${normName} before attention`, "f-n1", "X", "X̂");
    if (cfg.attn === "mla") mlaSteps(S, D, lit);
    else gqaSteps(S, D, cfg, ly, lit, off);
    if (cfg.place === "out" || cfg.place === "sandwich")
      normStep("norm_a_out", `${normName} on the attention output`, "f-n1b", "attention out", "normed out", placeCite || lit("norm"));
    addStep("res1", "Residual add", "f-r1", "X", "attention out", "X₁", "Attention",
      cfg.place === "out" || cfg.place === "sandwich" ? String.raw`X_1 = X + \mathrm{Norm}(\mathrm{Attn}(\cdot))` : String.raw`X_1 = X + \mathrm{Attn}(\cdot)`);
    if (cfg.place === "post") normStep("norm_a_post", `${normName} after the add (post-norm)`, "f-n1", "X₁", "X̂₁", placeCite);

    /* --- feed-forward sublayer --- */
    if (pre) normStep("norm_f", `${normName} before the ${ly.dense ? "FFN" : "MoE"}`, "f-n2", "X₁", "X̂₁");
    if (ly.dense) denseSteps(S, D, cfg, lit);
    else moeSteps(S, D, cfg, lit);
    if (cfg.place === "out" || cfg.place === "sandwich")
      normStep("norm_f_out", `${normName} on the ${ly.dense ? "FFN" : "MoE"} output`, "f-n2b", "FFN out", "normed out", placeCite || lit("norm"));
    addStep("res2", "Residual add", "f-r2", "X₁", ly.dense ? "FFN out" : "MoE out", `block ${block + 1} out`, "Feed-forward",
      String.raw`X_2 = X_1 + \mathrm{${ly.dense ? "FFN" : "MoE"}}(\cdot)`);
    if (cfg.place === "post") normStep("norm_f_post", `${normName} after the add (post-norm)`, "f-n2", "X₂", "X̂₂", placeCite);

    const blockOps = steps.filter(s => s.group === "Attention" || s.group === "Feed-forward").length;
    const pat = layers(base, cfg);
    S({ key: "blocks", group: "Output", title: `All ${L} blocks`, zoom: "block", focus: ["f-block", "f-layers"], op: "blocks",
      spec: { n: L, layers: pat, walked: block }, cite: (cfg.attnPattern !== "uniform" ? lit("attnPattern") : null) || (cfg.ffnPattern !== "uniform" ? lit("ffnPattern") : null),
      formula: String.raw`X_{\ell+1} = \mathrm{Block}_\ell(X_\ell),\quad \ell = 1, \dots, ${L}`,
      text: `You walked ${blockName} (${blockOps} operations). Every block runs its own copy of them with its own weights` +
        (cfg.attnPattern !== "uniform" || cfg.ffnPattern !== "uniform" ? "; the strip shows which blocks use which attention span and feed-forward type, so some blocks run a different chain from the one you walked." : ".") +
        ` The tensor between blocks is always (S, ${d}).` });
    normStep("norm_final", `Final ${normName}`, "f-fnorm", "X_L", "X̂_L");
    steps[steps.length - 1].group = "Output"; steps[steps.length - 1].zoom = "col";
    S({ key: "head", group: "Output", title: cfg.tie === "tied" ? "Output layer (tied)" : "Output layer (separate weight)", zoom: "col", focus: ["f-head"], op: "matmul",
      spec: { a: T("X̂_L", ["S", d]), b: T(cfg.tie === "tied" ? "Eᵀ (tied)" : "W_out", [d, V], "weight"), c: T("logits", ["S", V]) },
      formula: cfg.tie === "tied" ? String.raw`\mathrm{logits} = \hat{X}_L E^{\top}` : String.raw`\mathrm{logits} = \hat{X}_L W_{\text{out}}`,
      text: `The widest operation in the model: ${d} → ${V.toLocaleString("en-US")}. ` + (cfg.tie === "tied" ? "It reuses the embedding matrix, transposed, so it adds no parameters." : `A separate matrix adds ${(V * d).toLocaleString("en-US")} parameters.`),
      params: cfg.tie === "tied" ? 0 : V * d });
    if (cfg.mtp === "1") {
      S({ key: "mtp", group: "Output", title: "Multi-token prediction head", zoom: "col", focus: ["f-mtp"], op: "matmul",
        spec: { a: T("[norm(h), norm(emb(t+1))]", ["S", 2 * d]), b: T("W_fuse", [2 * d, d], "weight"), c: T("fused", ["S", d]) },
        formula: String.raw`h' = W_{\text{fuse}}[\mathrm{norm}(h);\ \mathrm{norm}(\mathrm{emb}(t_{i+1}))]`,
        text: "During training only, one extra head joins the trunk's hidden state with the next token's embedding, runs one more attention + dense SwiGLU, and predicts the token after next through the shared output layer.",
        params: params(base, cfg).mtp });
    }
    S({ key: "next", group: "Output", title: "Next-token probabilities", zoom: "col", focus: ["f-out"], op: "next", spec: { v: V },
      formula: String.raw`p(t_S \mid t_{<S}) = \mathrm{softmax}(\mathrm{logits}_{S-1})`,
      text: `Only the last position's row is needed to generate: a softmax over all ${V.toLocaleString("en-US")} tokens gives the next-token distribution.` });
    const terms = lossTerms(D, cfg);
    for (const t of terms)
      S({ key: `loss_${t.key}`, group: "Training", title: t.kind === "update" ? `${t.name} (not a loss)` : t.name, zoom: "all",
        focus: [`f-loss-${t.key}`, `f-tap-${t.key}-0`, `f-tap-${t.key}-1`, `f-link-${t.key}`], op: "losses",
        spec: { terms, focus: t.key }, formula: t.latex, text: t.text, cite: t.cite || null });
    S({ key: "loss_total", group: "Training", title: "Total loss, then backward()", zoom: "loss", focus: ["f-loss-total"], op: "losses",
      spec: { terms, focus: "total" }, formula: totalLatex(terms),
      text: "Every loss term is added with its weight into one number, and backward() sends gradients from it to every weight in the model. Updates that are not losses (like the loss-free bias) happen separately after the step. Inference computes none of this: it stops at the logits." });
    return steps.map(s => Object.assign({ params: 0, cite: null }, s, { repo: !s.cite }));
  }

  function gqaSteps(S, D, cfg, ly, lit, off) {
    const { d, H, dk, Hkv } = D, kv = Hkv * dk, rep = H / Hkv;
    const kind = cfg.attn === "mha" ? "multi-head" : cfg.attn === "mqa" ? "multi-query" : "grouped-query";
    S({ key: "q", group: "Attention", title: "Project to queries", zoom: "attn", focus: ["f-qproj", "f-attn"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_Qᵀ", [d, H * dk], "weight"), c: T("Q", ["S", H * dk]) },
      formula: String.raw`Q = \hat{X} W_Q^{\top}`, text: `A square ${d}×${d} weight: each of the ${H} query heads gets ${dk} columns.`, params: d * H * dk });
    S({ key: "k", group: "Attention", title: "Project to keys", zoom: "attn", focus: ["f-kproj", "f-attn"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_Kᵀ", [d, kv], "weight"), c: T("K", ["S", kv]) },
      formula: String.raw`K = \hat{X} W_K^{\top}`,
      text: Hkv === H ? `Multi-head attention: as many key heads as query heads, so W_K is as wide as W_Q.`
        : `${Hkv} key head${Hkv > 1 ? "s" : ""}, so W_K is ${d}×${kv}: ${rep} times narrower than W_Q. This is where ${kind} attention saves parameters and KV-cache memory.`,
      params: d * kv });
    S({ key: "v", group: "Attention", title: "Project to values", zoom: "attn", focus: ["f-vproj", "f-attn"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_Vᵀ", [d, kv], "weight"), c: T("V", ["S", kv]) },
      formula: String.raw`V = \hat{X} W_V^{\top}`, text: `Same shape as the keys: ${Hkv} value head${Hkv > 1 ? "s" : ""} of ${dk}.`, params: d * kv });
    S({ key: "split", group: "Attention", title: "Split into heads", zoom: "attn", focus: ["f-qproj", "f-kproj", "f-vproj"], op: "reshape",
      spec: { rows: [{ inp: T("Q", ["S", H * dk]), heads: H, out: T("q", ["S", dk]) }, { inp: T("K", ["S", kv]), heads: Hkv, out: T("k", ["S", dk]) }, { inp: T("V", ["S", kv]), heads: Hkv, out: T("v", ["S", dk]) }] },
      formula: String.raw`(S, ${H * dk}) \to (${H}, S, ${dk}),\quad (S, ${kv}) \to (${Hkv}, S, ${dk})`,
      text: `No arithmetic: columns are cut into ${dk}-wide slices and stacked as heads.` });
    if (cfg.qknorm === "on")
      S({ key: "qknorm", group: "Attention", title: "QK-Norm", zoom: "attn", focus: ["f-qkn-q", "f-qkn-k"], op: "elementwise",
        spec: { rows: [{ inp: T("q", ["S", dk]), out: T("q̂", ["S", dk]), stack: H }, { inp: T("k", ["S", dk]), out: T("k̂", ["S", dk]), stack: Hkv }], param: T("gains", [1, dk], "weight"), label: "RMSNorm" },
        formula: String.raw`\hat{q} = \mathrm{RMSNorm}_{${dk}}(q),\ \hat{k} = \mathrm{RMSNorm}_{${dk}}(k)`,
        text: `Each ${dk}-wide head vector of q and k is normalized on its own, which bounds the attention logits.`, params: 2 * dk });
    if (cfg.pos === "rope")
      S({ key: "rope", group: "Attention", title: "Rotary position embedding", zoom: "attn", focus: ["f-rope-q", "f-rope-k", "f-rope"], op: "elementwise",
        spec: { rows: [{ inp: T("q", ["S", dk]), out: T("q′", ["S", dk]), stack: H }, { inp: T("k", ["S", dk]), out: T("k′", ["S", dk]), stack: Hkv }], label: "rotate pairs" },
        formula: String.raw`(x_{2i}, x_{2i+1}) \mapsto R(p\,\theta^{-2i/${dk}})(x_{2i}, x_{2i+1}),\ \theta = ${D.theta}`,
        text: "Each pair of numbers is rotated by an angle that grows with the token's position p. Shapes do not change; position now lives in the angles." });
    if (rep > 1)
      S({ key: "gqa", group: "Attention", title: Hkv === 1 ? "Share the single key/value head" : "Share each key/value head", zoom: "attn", focus: ["f-core"], op: "repeat",
        spec: { rows: [{ inp: T("k", ["S", dk]), heads: Hkv }, { inp: T("v", ["S", dk]), heads: Hkv }], factor: rep, heads: H },
        formula: String.raw`(${Hkv}, S, ${dk}) \to (${H}, S, ${dk})`,
        text: `Each of the ${Hkv} key and value head${Hkv > 1 ? "s" : ""} serves ${rep} query heads. The explicit path copies them; the fused path broadcasts without copying.` });
    S({ key: "scores", group: "Attention", title: "Attention scores", zoom: "attn", focus: ["f-core"], op: "matmul",
      spec: { a: T("q (per head)", ["S", dk]), b: T("kᵀ (per head)", [dk, "S"]), c: T("scores", ["S", "S"]), batch: H },
      formula: String.raw`\mathrm{scores}_h = q_h k_h^{\top} / \sqrt{${dk}}`,
      text: `For every head, every position scores every position: an S×S matrix, ${H} times. This is the part that grows with the square of the sequence length.` });
    if (cfg.pos === "alibi")
      S({ key: "alibi", group: "Attention", title: "Add the ALiBi distance bias", zoom: "attn", focus: ["f-core", "f-rope"], op: "mask",
        spec: { shape: ["S", "S"], stack: H, variant: "alibi" },
        formula: String.raw`\mathrm{scores}_{h,ij} \mathrel{+}= -m_h\,(i - j)`,
        text: "Instead of rotating q and k, ALiBi subtracts a penalty proportional to distance, with a different slope per head. Nearby tokens score higher." });
    S({ key: "mask", group: "Attention", title: ly.local ? `Sliding-window mask (${D.window})` : "Causal mask", zoom: "attn", focus: ["f-core"], op: "mask",
      spec: { shape: ["S", "S"], stack: H, window: ly.local ? D.window : 0 }, cite: ly.local ? (lit("window") || lit("attnPattern") || opt("window", "sliding").cite) : null,
      formula: ly.local ? String.raw`\mathrm{scores}_{ij} = -\infty\ \text{for } j > i \text{ or } i - j \ge ${D.window}` : String.raw`\mathrm{scores}_{ij} = -\infty \quad \text{for } j > i`,
      text: ly.local ? `In this block each position may look back at most ${D.window} positions, so the allowed region is a band along the diagonal. At S ≤ ${D.window} it is identical to the causal mask; try S = 1,024.`
        : "A position may not look at later positions, so everything above the diagonal is set to −∞ and becomes exactly 0 after the softmax." });
    S({ key: "softmax", group: "Attention", title: "Softmax over each row", zoom: "attn", focus: ["f-core"], op: "softmax",
      spec: { shape: ["S", "S"], stack: H, window: ly.local ? D.window : 0 },
      formula: String.raw`P_{ij} = e^{s_{ij}} / \textstyle\sum_{j'} e^{s_{ij'}}`,
      text: "Each row becomes weights that sum to 1: how much position i takes from each position it may see." });
    S({ key: "weighted", group: "Attention", title: "Weighted sum of values", zoom: "attn", focus: ["f-core"], op: "matmul",
      spec: { a: T("P (per head)", ["S", "S"]), b: T("v (per head)", ["S", dk]), c: T("o (per head)", ["S", dk]), batch: H },
      formula: String.raw`o_h = P_h v_h`, text: `The inner dimension is now the sequence: each output row mixes value rows. Back to ${dk} per head.` });
    S({ key: "concat", group: "Attention", title: "Concatenate heads", zoom: "attn", focus: ["f-wo"], op: "reshape",
      spec: { rows: [{ inp: T("O", ["S", H * dk]), heads: H, out: T("o", ["S", dk]) }], reverse: true },
      formula: String.raw`(${H}, S, ${dk}) \to (S, ${H * dk})`, text: `The reverse of the split: ${H} heads side by side make ${H * dk} columns again.` });
    S({ key: "wo", group: "Attention", title: "Output projection", zoom: "attn", focus: ["f-wo"], op: "matmul",
      spec: { a: T("O", ["S", H * dk]), b: T("W_Oᵀ", [H * dk, d], "weight"), c: T("attention out", ["S", d]) },
      formula: String.raw`\mathrm{Attn}(X) = O W_O^{\top}`, text: "A square weight mixes information across heads and returns to the model width.", params: H * dk * d });
  }

  function mlaSteps(S, D) {
    const { d, H, dk, dc, dr } = D;
    S({ key: "mla_dkv", group: "Attention", title: "Compress to the latent", zoom: "attn", focus: ["f-dkv", "f-attn"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_DKVᵀ", [d, dc], "weight"), c: T("c_KV", ["S", dc]) },
      formula: String.raw`c^{KV} = \hat{X} W_{DKV}^{\top}`,
      text: `One shared ${dc}-wide latent per token replaces every head's keys and values. This is all that has to be cached for generation (plus a small RoPE key).`, params: d * dc });
    S({ key: "mla_uk", group: "Attention", title: "Up-project keys from the latent", zoom: "attn", focus: ["f-uk"], op: "matmul",
      spec: { a: T("c_KV", ["S", dc]), b: T("W_UKᵀ", [dc, H * dk], "weight"), c: T("K content", ["S", H * dk]) },
      formula: String.raw`K^{C} = c^{KV} W_{UK}^{\top}`, text: `The narrow latent is widened back to ${H} full key heads on the fly.`, params: dc * H * dk });
    S({ key: "mla_uv", group: "Attention", title: "Up-project values from the latent", zoom: "attn", focus: ["f-uv"], op: "matmul",
      spec: { a: T("c_KV", ["S", dc]), b: T("W_UVᵀ", [dc, H * dk], "weight"), c: T("V", ["S", H * dk]) },
      formula: String.raw`V = c^{KV} W_{UV}^{\top}`, text: "Values come from the same latent.", params: dc * H * dk });
    S({ key: "q", group: "Attention", title: "Project to queries", zoom: "attn", focus: ["f-qproj"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_Qᵀ", [d, H * dk], "weight"), c: T("Q content", ["S", H * dk]) },
      formula: String.raw`Q^{C} = \hat{X} W_Q^{\top}`, text: "Queries are not compressed in this simplified MLA.", params: d * H * dk });
    S({ key: "mla_qr", group: "Attention", title: "Query RoPE slices", zoom: "attn", focus: ["f-qr"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_QRᵀ", [d, H * dr], "weight"), c: T("q rope", ["S", H * dr]) },
      formula: String.raw`Q^{R} = \hat{X} W_{QR}^{\top}`, text: `A small ${dr}-wide slice per head carries position, because a rotated key could not be rebuilt from the latent.`, params: d * H * dr });
    S({ key: "mla_kr", group: "Attention", title: "Shared key RoPE slice", zoom: "attn", focus: ["f-kr"], op: "matmul",
      spec: { a: T("X̂", ["S", d]), b: T("W_KRᵀ", [d, dr], "weight"), c: T("k rope", ["S", dr]) },
      formula: String.raw`k^{R} = \hat{X} W_{KR}^{\top}`, text: `One ${dr}-wide positional key shared by every head: the only other thing that gets cached.`, params: d * dr });
    S({ key: "mla_rope", group: "Attention", title: "Rotate the RoPE slices", zoom: "attn", focus: ["f-rope-q", "f-rope-k"], op: "elementwise",
      spec: { rows: [{ inp: T("q rope", ["S", dr]), out: T("q rope′", ["S", dr]), stack: H }, { inp: T("k rope", ["S", dr]), out: T("k rope′", ["S", dr]), stack: 1 }], label: "rotate pairs" },
      formula: String.raw`(q^R, k^R) \leftarrow \mathrm{RoPE}(q^R, k^R)`, text: "Only the small slices are rotated; the content parts stay position-free." });
    S({ key: "mla_cat", group: "Attention", title: "Join content and RoPE parts", zoom: "attn", focus: ["f-core"], op: "reshape",
      spec: { rows: [{ inp: T("q = [content; rope]", ["S", H * (dk + dr)]), heads: H, out: T("q", ["S", dk + dr]) }] },
      formula: String.raw`q_h = [q^C_h;\ q^R_h],\ k_h = [k^C_h;\ k^R] \in \mathbb{R}^{${dk + dr}}`,
      text: `Every head's query and key become ${dk} + ${dr} = ${dk + dr} wide; the one RoPE key is appended to all ${H} heads.` });
    S({ key: "scores", group: "Attention", title: "Attention scores", zoom: "attn", focus: ["f-core"], op: "matmul",
      spec: { a: T("q (per head)", ["S", dk + dr]), b: T("kᵀ (per head)", [dk + dr, "S"]), c: T("scores", ["S", "S"]), batch: H },
      formula: String.raw`\mathrm{scores}_h = q_h k_h^{\top} / \sqrt{${dk + dr}}`, text: `The inner dimension is ${dk + dr}, wider than GQA's ${dk}, because of the RoPE slices.` });
    S({ key: "mask", group: "Attention", title: "Causal mask", zoom: "attn", focus: ["f-core"], op: "mask", spec: { shape: ["S", "S"], stack: H },
      formula: String.raw`\mathrm{scores}_{ij} = -\infty \quad \text{for } j > i`, text: "Everything above the diagonal is set to −∞." });
    S({ key: "softmax", group: "Attention", title: "Softmax over each row", zoom: "attn", focus: ["f-core"], op: "softmax", spec: { shape: ["S", "S"], stack: H },
      formula: String.raw`P_{ij} = e^{s_{ij}} / \textstyle\sum_{j'} e^{s_{ij'}}`, text: "Each row becomes weights that sum to 1." });
    S({ key: "weighted", group: "Attention", title: "Weighted sum of values", zoom: "attn", focus: ["f-core"], op: "matmul",
      spec: { a: T("P (per head)", ["S", "S"]), b: T("v (per head)", ["S", dk]), c: T("o (per head)", ["S", dk]), batch: H },
      formula: String.raw`o_h = P_h v_h`, text: `Values carry no RoPE part, so outputs are back to ${dk} per head.` });
    S({ key: "concat", group: "Attention", title: "Concatenate heads", zoom: "attn", focus: ["f-wo"], op: "reshape",
      spec: { rows: [{ inp: T("O", ["S", H * dk]), heads: H, out: T("o", ["S", dk]) }], reverse: true },
      formula: String.raw`(${H}, S, ${dk}) \to (S, ${H * dk})`, text: "Heads side by side again." });
    S({ key: "wo", group: "Attention", title: "Output projection", zoom: "attn", focus: ["f-wo"], op: "matmul",
      spec: { a: T("O", ["S", H * dk]), b: T("W_Oᵀ", [H * dk, d], "weight"), c: T("attention out", ["S", d]) },
      formula: String.raw`\mathrm{Attn}(X) = O W_O^{\top}`, text: "Back to the model width.", params: H * dk * d });
  }

  function expertMats(S, D, cfg, lit, who, rows, keyPrefix, focusUp, focusDown, title) {
    const { d } = D, F = keyPrefix === "dense" ? D.Fd : D.F;
    const actName = { swiglu: "SiLU", geglu: "GELU", gelu: "GELU", relu2: "ReLU²" }[cfg.act];
    const cite = lit("act");
    if (D.gated)
      S({ key: `${keyPrefix}_up`, group: "Feed-forward", title: `${title}: gate and up projections`, zoom: "swiglu", focus: focusUp, op: "matmul", cite,
        spec: { a: T(who, [rows, d]), b: T("W_gateᵀ, W_upᵀ", [d, F], "weight"), c: T("h", [rows, F]), batch: 2, batchLabel: "gate and up" },
        formula: String.raw`h = \mathrm{${actName}}(x W_{\text{gate}}^{\top}) \odot (x W_{\text{up}}^{\top})`,
        text: `Widen from ${d} to ${F.toLocaleString("en-US")} twice, apply ${actName} to the gate branch, and multiply the two elementwise.`, params: 2 * d * F });
    else
      S({ key: `${keyPrefix}_up`, group: "Feed-forward", title: `${title}: up projection`, zoom: "swiglu", focus: focusUp, op: "matmul", cite,
        spec: { a: T(who, [rows, d]), b: T("W_1ᵀ", [d, F], "weight"), c: T("h", [rows, F]) },
        formula: cfg.act === "relu2" ? String.raw`h = \max(0, x W_1^{\top})^2` : String.raw`h = \mathrm{GELU}(x W_1^{\top})`,
        text: `No gate: one matrix widens ${d} → ${F.toLocaleString("en-US")}, then ${actName}. One matrix fewer than SwiGLU.`, params: d * F });
    S({ key: `${keyPrefix}_down`, group: "Feed-forward", title: `${title}: down projection`, zoom: "swiglu", focus: focusDown, op: "matmul", cite,
      spec: { a: T("h", [rows, F]), b: T(D.gated ? "W_downᵀ" : "W_2ᵀ", [F, d], "weight"), c: T("out", [rows, d]) },
      formula: String.raw`y = h W_{\text{down}}^{\top}`, text: `Back from ${F.toLocaleString("en-US")} to ${d}.`, params: F * d });
  }

  function denseSteps(S, D, cfg, lit) {
    expertMats(S, D, cfg, lit, "X̂₁", "S", "dense", ["f-moe", "f-gate", "f-up"], ["f-moe", "f-down"], "Dense FFN");
  }

  function moeSteps(S, D, cfg, lit) {
    const { d, N, k, ns } = D;
    const ec = cfg.router === "expertChoice";
    const fn = cfg.router === "softmax" ? "softmax" : "sigmoid";
    S({ key: "router", group: "Feed-forward", title: "Router scores", zoom: "moe", focus: ["f-router", "f-moe"], op: "matmul", cite: lit("router"),
      spec: { a: T("X̂₁", ["S", d]), b: T("W_routerᵀ", [d, N], "weight"), c: T("affinities", ["S", N]) },
      formula: ec ? String.raw`A = \mathrm{softmax}(\hat{X}_1 W_r^{\top})` : fn === "sigmoid" ? String.raw`a = \sigma(\hat{X}_1 W_r^{\top})` : String.raw`a = \mathrm{softmax}(\hat{X}_1 W_r^{\top})`,
      text: `A very narrow weight: ${N} columns, one per routed expert. ` + (fn === "sigmoid" && !ec ? "Each token gets independent sigmoid scores." : "Each token's scores are a softmax over the experts."),
      params: d * N });
    if (ec)
      S({ key: "ec_select", group: "Feed-forward", title: "Each expert picks its tokens", zoom: "moe", focus: ["f-router"], op: "topk", cite: lit("router"),
        spec: { experts: N, k, expertChoice: true },
        formula: String.raw`\text{expert } e \text{ takes its top } C = S k / N \text{ tokens}`,
        text: `The selection is transposed: instead of each token choosing ${k} experts, each expert chooses the C tokens it scores highest. Every expert gets exactly C tokens; a token may get any number of experts, including none. Routing shown is an example.` });
    else
      S({ key: "topk", group: "Feed-forward", title: `Pick the top ${k} experts`, zoom: "moe", focus: ["f-router", "f-e3"], op: "topk",
        spec: { experts: N, k },
        formula: String.raw`\mathcal{T}_i = \mathrm{top}_{${k}}(a_i),\ g_{ie} = a_{ie} / \textstyle\sum_{e' \in \mathcal{T}_i} a_{ie'}`,
        text: `Each token keeps its ${k} highest-scoring experts and renormalizes those scores to sum to 1.` + (cfg.balance === "lossfree" ? " With loss-free balancing, a per-expert bias is added to the scores for this choice only." : "") + " Routing shown is an example pattern." });
    S({ key: "dispatch", group: "Feed-forward", title: cfg.balance === "capacity" && !ec ? "Dispatch, dropping overflow" : "Send tokens to their experts", zoom: "moe",
      focus: ["f-e3", ...Array.from({ length: N }, (_, i) => `f-e${i + 1}`).filter(id => id !== "f-e3")], op: "dispatch",
      cite: cfg.balance === "capacity" && !ec ? lit("balance") : (ec ? lit("router") : null),
      spec: { experts: N, k, d, capacity: cfg.balance === "capacity" && !ec ? 1.25 : 0, expertChoice: ec },
      formula: cfg.balance === "capacity" && !ec ? String.raw`C = 1.25 \cdot ${k} S / ${N}` : String.raw`\textstyle\sum_e n_e = ${k}\,S`,
      text: ec ? "Every expert receives the same number of tokens, so no balancing loss is needed."
        : cfg.balance === "capacity" ? "Each expert accepts at most C tokens. Tokens past an expert's capacity skip it and pass through the residual unchanged."
        : `Every token goes to ${k} experts, so the experts process ${k}·S rows in total, unevenly.` });
    expertMats(S, D, cfg, lit, "tokens for E3", "n_e", "expert", ["f-gate", "f-up", "f-e3"], ["f-down", "f-e3"], "One expert");
    if (ns > 0)
      S({ key: "shared", group: "Feed-forward", title: "Shared expert", zoom: "moe", focus: ["f-shared"], op: "matmul",
        spec: D.gated ? { a: T("X̂₁", ["S", d]), b: T("W_gateᵀ, W_upᵀ", [d, D.F], "weight"), c: T("h (shared)", ["S", D.F]), batch: 2, batchLabel: "gate and up" }
                      : { a: T("X̂₁", ["S", d]), b: T("W_1ᵀ", [d, D.F], "weight"), c: T("h (shared)", ["S", D.F]) },
        formula: String.raw`s = E_{\text{shared}}(\hat{X}_1)`,
        text: "The shared expert has the same shape as a routed one but runs on every token, with no routing.", params: (D.gated ? 3 : 2) * d * D.F });
    S({ key: "combine", group: "Feed-forward", title: "Weighted sum", zoom: "moe", focus: ["f-sum"], op: "add",
      spec: { a: T("Σ g·expert outputs", ["S", d]), b: T(ns ? "shared output" : "(no shared expert)", ["S", d]), out: T("MoE out", ["S", d]) },
      formula: ns ? String.raw`\mathrm{MoE}(x_i) = \textstyle\sum_{e} g_{ie} E_e(x_i) + s(x_i)` : String.raw`\mathrm{MoE}(x_i) = \textstyle\sum_{e} g_{ie} E_e(x_i)`,
      text: "Each token's expert outputs are weighted by its gates and added" + (ns ? " to the shared expert's output." : " together.") });
  }

  /* every training-time term, in the order scripts/train.py's trainer adds them, and where it reads from */
  function lossTerms(D, cfg) {
    const anyMoe = cfg.ffn === "moe" || cfg.ffnPattern !== "uniform";
    const t = [];
    t.push({ key: "ce", name: "Cross-entropy", tap: "logits", reads: ["S", D.V], plain: "−log p(next token)", weight: "× 1",
      latex: String.raw`\mathcal{L}_{\text{CE}} = -\tfrac{1}{S}\textstyle\sum_i \log p(t_{i+1} \mid t_{\le i})`,
      text: "The main objective: at every position, the log-probability the model gave to the token that actually came next. It reads the full (S, vocabulary) logits." });
    if (anyMoe && cfg.router === "expertChoice")
      t.push({ key: "ec", kind: "note", name: "No balancing term", tap: "router", reads: [D.N], plain: "each expert takes exactly C tokens", weight: "nothing to add", cite: CITE.expertChoice,
        latex: String.raw`n_e = C \text{ for every expert } e`,
        text: "Expert-choice routing balances load by construction, so no balancing loss or bias is needed." });
    if (anyMoe && cfg.router !== "expertChoice" && (cfg.balance === "aux" || cfg.balance === "capacity"))
      t.push({ key: "aux", name: "Load-balancing loss", tap: "router", reads: ["S", D.N], plain: "α·N·Σₑ fₑ·Pₑ per MoE layer", weight: "α = 0.01, mean over MoE layers",
        latex: String.raw`\mathcal{L}_{\text{aux}} = \alpha N \textstyle\sum_e f_e P_e,\quad \alpha = 0.01`,
        text: `Per MoE layer: f is the fraction of tokens dispatched to each expert, P the mean router probability. Their product is smallest when load is spread evenly over the ${D.N} experts. The layers' values are averaged and added to the loss.` });
    if (anyMoe && cfg.router !== "expertChoice" && cfg.balance === "lossfree")
      t.push({ key: "bias", kind: "update", name: "Loss-free bias update", tap: "router", reads: [D.N], plain: "bₑ nudged toward equal load", weight: "after each step, no gradient",
        latex: String.raw`b_e \leftarrow b_e + \gamma\,\mathrm{sign}(\bar{n} - n_e)`,
        text: "Not a loss: after each step, each expert's selection bias goes up if it was under-loaded and down if over-loaded. It changes which experts get picked, without adding a gradient." });
    if (anyMoe && (cfg.zloss === "router" || cfg.zloss === "both"))
      t.push({ key: "rz", name: "Router z-loss", tap: "router", reads: ["S", D.N], plain: "λ·mean(logsumexp(router logits)²)", weight: "added to each layer's term",
        latex: String.raw`\mathcal{L}_{rz} = \lambda_{rz}\,\overline{\big(\log \textstyle\sum_e e^{r_e}\big)^2}`,
        text: "Keeps the router's logits from growing large, which would make routing brittle. In this code it is added inside each MoE layer, alongside the balancing term." });
    if (cfg.zloss === "output" || cfg.zloss === "both")
      t.push({ key: "oz", name: "Output z-loss", tap: "logits", reads: ["S", D.V], plain: "λ·mean(logsumexp(logits)²)", weight: "× z_loss_weight",
        latex: String.raw`\mathcal{L}_{z} = \lambda_z\,\overline{\big(\log \textstyle\sum_v e^{\ell_v}\big)^2}`,
        text: "Pulls the softmax normalizer toward 1 without changing the probabilities, which keeps bf16 logits in a safe range." });
    if (cfg.mtp === "1")
      t.push({ key: "mtp", name: "MTP cross-entropy", tap: "mtp", reads: ["S", D.V], plain: "CE of the extra head", weight: "× λ = 0.3",
        latex: String.raw`\lambda_{\text{MTP}}\,\mathcal{L}_{\text{CE}}^{(t+2)},\quad \lambda = 0.3`,
        text: "The extra head's prediction of the token after next, scored the same way as the main loss and down-weighted." });
    t.forEach((x, i) => { x.n = i + 1; x.kind = x.kind || "loss"; });
    return t;
  }

  function totalLatex(terms) {
    const L = ["\\mathcal{L}_{\\text{CE}}"];
    const layer = terms.filter(t => t.key === "aux" || t.key === "rz").map(t => t.key === "aux" ? "\\mathcal{L}_{\\text{aux}}" : "\\mathcal{L}_{rz}");
    if (layer.length) L.push(`\\overline{(${layer.join(" + ")})}_{\\text{layers}}`);
    if (terms.some(t => t.key === "oz")) L.push("\\mathcal{L}_{z}");
    if (terms.some(t => t.key === "mtp")) L.push("0.3\\,\\mathcal{L}_{\\text{MTP}}");
    return "\\mathcal{L} = " + L.join(" + ");
  }

  /* ---------------- the figure ---------------- */
  const P = { ink: "#1E2430", ink2: "#555B66", rule: "#CFC6AF", paper: "#F5F1E6", panel: "#ECE6D6",
    norm: ["#FCE7C8", "#B9781C"], attn: ["#D7E6F8", "#2A5DB0"], moe: ["#D3EFDF", "#1F7A4A"], emb: ["#E7DEFA", "#6B3FB8"],
    router: ["#FFF1B8", "#9A6200"], lit: "#B4421E" };
  const FONT = "'IBM Plex Sans','Helvetica Neue',Helvetica,Arial,sans-serif";
  const escx = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const f = n => n.toLocaleString("en-US");
  const mill = x => x >= 10e6 ? (x / 1e6).toFixed(1) + "M" : (x / 1e6).toFixed(2) + "M";

  function Svg() {
    const parts = [];
    const s = {
      add: x => parts.push(x),
      open: id => parts.push(`<g id="${id}" class="fig-node">`),
      close: () => parts.push("</g>"),
      text(x, y, t, size = 13, weight = 400, fill = P.ink, anchor = "start", italic = false) {
        parts.push(`<text x="${x}" y="${y}" font-family="${FONT}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}"${italic ? ' font-style="italic"' : ""}>${escx(t)}</text>`);
      },
      box(x, y, w, h, col, label, sub, o = {}) {
        if (o.id) s.open(o.id);
        const dash = o.lit ? ' stroke-dasharray="6 3"' : "";
        parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.rx ?? 10}" fill="${col[0]}" stroke="${o.lit ? P.lit : col[1]}" stroke-width="${o.sw ?? 1.6}"${dash}/>`);
        const size = o.size ?? 15, cx = x + w / 2;
        if (sub) { s.text(cx, y + h / 2 - 3, label, size, o.weight ?? 600, P.ink, "middle"); s.text(cx, y + h / 2 + 14, sub, o.subSize ?? 12.5, 400, P.ink2, "middle"); }
        else s.text(cx, y + h / 2 + size * 0.35, label, size, o.weight ?? 600, P.ink, "middle");
        if (o.id) s.close();
      },
      line(pts, o = {}) {
        parts.push(`<polyline points="${pts.map(p => p.join(",")).join(" ")}" fill="none" stroke="${o.color || P.ink}" stroke-width="${o.width || 1.5}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}${o.arrow === false ? "" : ' marker-end="url(#arr)"'} stroke-linejoin="round"/>`);
      },
      plus(cx, cy, id) {
        if (id) s.open(id);
        parts.push(`<circle cx="${cx}" cy="${cy}" r="13" fill="#FFFFFF" stroke="${P.ink}" stroke-width="1.5"/><path d="M${cx - 6},${cy} H${cx + 6} M${cx},${cy - 6} V${cy + 6}" stroke="${P.ink}" stroke-width="1.8"/>`);
        if (id) s.close();
      },
      callout(xt, y, lines, target) {
        lines.forEach((t, i) => s.text(xt, y + i * 17, t, 13, i ? 400 : 500, i ? P.ink2 : P.ink, "end"));
        s.line([[xt + 8, y - 4], target], { arrow: false, dash: "3 4", color: P.ink2, width: 1.1 });
      },
      render(w, h, title, desc) {
        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-labelledby="fig-title fig-desc">` +
          `<title id="fig-title">${escx(title)}</title><desc id="fig-desc">${escx(desc)}</desc>` +
          `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${P.ink}"/></marker></defs>` +
          `<rect width="${w}" height="${h}" fill="${P.paper}"/>` + parts.join("") + "</svg>";
      },
    };
    return s;
  }

  function drawFigure(base, cfg, block = 0, title = "QwenMoE") {
    const D = dims(base, cfg), { d, H, dk, Hkv, N, k, F, ns, V, L } = D;
    const pr = params(base, cfg), ly = layers(base, cfg), cur = ly[block];
    const off = inactive(cfg), s = Svg();
    const isLit = slot => !off[slot] && !opt(slot, cfg[slot]).repo;
    const normName = cfg.norm === "layer" ? "LayerNorm" : "RMSNorm";
    const normLit = isLit("norm") || isLit("place");
    const kind = cfg.attn === "mla" ? "Multi-head latent attention" : cfg.attn === "mha" ? "Masked multi-head attention" : cfg.attn === "mqa" ? "Masked multi-query attention" : "Masked grouped-query attention";
    const attnSub = (cfg.attn === "mla" ? `${H} heads · latent ${D.dc} · decoupled RoPE` : `${H} query heads · ${Hkv} KV head${Hkv > 1 ? "s" : ""}` + (cfg.qknorm === "on" ? " · QK-Norm" : "")) + (cur.local ? ` · window ${D.window}` : "");
    const ffnLabel = cur.dense ? "Dense feed-forward" : "Mixture of Experts";
    const actName = { swiglu: "SwiGLU", geglu: "GeGLU", gelu: "GELU", relu2: "ReLU²" }[cfg.act];
    const ffnSub = cur.dense ? `${d} → ${f(D.Fd)} → ${d} · ${actName}` : `${N} routed (top-${k})${ns ? " + 1 shared" : ""} · ${actName}`;

    s.text(40, 58, title, 30, 700);
    s.text(40, 86, `${mill(pr.total)} total parameters · ${mill(pr.active)} active per token${pr.exact ? "" : " (formula: includes variants not in this repo)"} · decoder-only transformer`, 14.5, 400, P.ink2);

    const cx = 520, bx = 380, bw = 280;
    s.open("f-out"); s.text(cx, 132, `Next-token logits (${f(V)})`, 13.5, 500, P.ink, "middle"); s.close();
    s.box(bx, 150, bw, 46, P.emb, cfg.tie === "tied" ? "Linear output layer" : "Linear output layer (untied)", `${d} → ${f(V)}`, { id: "f-head" });
    s.line([[cx, 150], [cx, 140]]);
    if (cfg.mtp === "1") {
      s.box(680, 150, 110, 46, ["#FFFFFF", P.emb[1]], "MTP head", "training only", { id: "f-mtp", size: 13, subSize: 11 });
      s.line([[680, 173], [662, 173]]);
    }
    s.box(bx, 236, bw, 40, P.norm, `Final ${normName}`, null, { id: "f-fnorm", lit: isLit("norm") });
    s.line([[cx, 236], [cx, 196]]);

    // block contents, bottom to top: [items before the add] + plus + [items after the add], twice
    const sub = [];
    const normBox = (id, label) => ({ id, kind: "norm", label, h: 36 });
    const a1 = [], a2 = [], f1 = [], f2 = [];
    if (cfg.place === "pre" || cfg.place === "sandwich") a1.push(normBox("f-n1", `${normName} 1`));
    a1.push({ id: "f-attn", kind: "attn", h: 62 });
    if (cfg.place === "out" || cfg.place === "sandwich") a1.push(normBox("f-n1b", `${normName} (output)`));
    if (cfg.place === "post") a2.push(normBox("f-n1", `${normName} 1 (post)`));
    if (cfg.place === "pre" || cfg.place === "sandwich") f1.push(normBox("f-n2", `${normName} 2`));
    f1.push({ id: "f-moe", kind: "ffn", h: 58 });
    if (cfg.place === "out" || cfg.place === "sandwich") f1.push(normBox("f-n2b", `${normName} (output)`));
    if (cfg.place === "post") f2.push(normBox("f-n2", `${normName} 2 (post)`));
    const gap = 22, plusH = 26;
    const seq = [...a1, { id: "f-r1", kind: "plus", h: plusH }, ...a2, ...f1, { id: "f-r2", kind: "plus", h: plusH }, ...f2];
    const contentH = seq.reduce((a, it) => a + it.h, 0) + gap * (seq.length + 1);
    const panelTop = 296, chipsH = 34, panelH = contentH + 30 + chipsH, panelBot = panelTop + panelH;
    s.add(`<rect id="f-block" class="fig-panel" x="300" y="${panelTop}" width="440" height="${panelH}" rx="18" fill="${P.panel}" stroke="${P.rule}" stroke-width="1.5"/>`);
    s.text(318, panelTop + 28, "Transformer block", 15, 700);
    s.open("f-layers"); s.text(458, panelTop + 28, `×${L}`, 22, 700, P.router[1]);
    // per-layer chips: attention span (top half) and FFN type (bottom half), walked block outlined
    ly.forEach((t, l) => {
      const x = 510 + l * 21, y = panelTop + 12;
      s.add(`<rect x="${x}" y="${y}" width="17" height="10" rx="2" fill="${t.local ? "#FFFFFF" : P.attn[0]}" stroke="${P.attn[1]}" stroke-width="1"/>`);
      s.add(`<rect x="${x}" y="${y + 11}" width="17" height="10" rx="2" fill="${t.dense ? "#FFFFFF" : P.moe[0]}" stroke="${P.moe[1]}" stroke-width="1"/>`);
      if (l === block) s.add(`<rect x="${x - 2}" y="${y - 2}" width="21" height="25" rx="3" fill="none" stroke="${P.router[1]}" stroke-width="2"/>`);
    });
    s.close();
    s.text(318, panelTop + 46, `walking block ${block + 1} · chips: attention span (blue = global) over FFN (green = MoE)`, 10.5, 400, P.ink2);

    let y = panelBot - gap;            // fill from the bottom
    const pos = {};
    for (const it of seq) {
      const top = y - it.h; pos[it.id] = { top, bot: y, mid: top + it.h / 2 };
      if (it.kind === "norm") s.box(bx, top, bw, it.h, P.norm, it.label, null, { id: it.id, lit: normLit, size: 14 });
      else if (it.kind === "attn") s.box(bx, top, bw, it.h, P.attn, kind, attnSub, { id: it.id, lit: isLit("window") || cur.local, size: 14.5 });
      else if (it.kind === "ffn") s.box(bx, top, bw, it.h, P.moe, ffnLabel, ffnSub, { id: it.id, lit: cur.dense || isLit("act") || isLit("router") || isLit("balance"), size: 14.5 });
      else s.plus(cx, top + it.h / 2, it.id);
      y = top - gap;
    }
    // vertical arrows between consecutive items
    s.line([[cx, panelBot], [cx, pos[seq[0].id].bot]]);
    for (let i = 1; i < seq.length; i++) s.line([[cx, pos[seq[i - 1].id].top], [cx, pos[seq[i].id].bot]]);
    s.line([[cx, pos[seq[seq.length - 1].id].top], [cx, 276]]);
    // residuals: from the stream point below each sublayer to its plus
    const r1 = pos["f-r1"], r2 = pos["f-r2"];
    const start1 = pos[seq[0].id].bot + gap / 2, start2 = pos[(a2.length ? a2[a2.length - 1] : { id: "f-r1" }).id].top - gap / 2;
    s.line([[cx, start1], [712, start1], [712, r1.mid], [533, r1.mid]]);
    s.line([[cx, start2], [712, start2], [712, r2.mid], [533, r2.mid]]);
    s.text(720, (start1 + r1.mid) / 2, "residual", 11.5, 400, P.ink2);
    s.text(720, (start2 + r2.mid) / 2, "residual", 11.5, 400, P.ink2);
    // positional pill beside attention
    const am = pos["f-attn"].mid;
    const pill = cfg.attn === "mla" ? "dec. RoPE" : { rope: "RoPE", alibi: "ALiBi", nope: null, learned: null, sinusoidal: null }[cfg.pos];
    if (pill) {
      s.open("f-rope");
      s.add(`<rect x="304" y="${am - 15}" width="66" height="30" rx="15" fill="${P.emb[0]}" stroke="${P.emb[1]}" stroke-width="1.4"/>`);
      s.text(337, am + 5, pill, 12, 600, P.ink, "middle"); s.close();
      s.line([[370, am], [bx, am]]);
    }

    // input side
    const posAdd = (cfg.pos === "learned" || cfg.pos === "sinusoidal") && !off.pos;
    const embTop = panelBot + (posAdd ? 70 : 38);
    if (posAdd) {
      s.plus(cx, panelBot + 32, "f-posadd");
      s.box(560, panelBot + 19, 170, 26, ["#FFFFFF", P.emb[1]], cfg.pos === "learned" ? "learned positions" : "sinusoidal positions", null, { size: 12, lit: true, rx: 13 });
      s.line([[560, panelBot + 32], [533, panelBot + 32]]);
      s.line([[cx, embTop], [cx, panelBot + 45]]);
    }
    s.box(bx, embTop, bw, 46, P.emb, "Token embedding layer", `${f(V)} × ${d}`, { id: "f-emb" });
    s.line([[cx, embTop + 84], [cx, embTop + 46]]);
    s.open("f-in"); s.text(cx, embTop + 104, "Tokenized text (byte-level BPE)", 13.5, 500, P.ink, "middle"); s.close();
    const embY = embTop - 50;

    // callouts
    const xt = 262;
    s.callout(xt, 168, [cfg.tie === "tied" ? "Output layer reuses the" : "Separate output matrix", cfg.tie === "tied" ? "embedding matrix (tied)" : `${f(V * d)} more parameters`], [bx, 173]);
    s.callout(xt, 252, [`${normName} ${{ pre: "before each sublayer", post: "after each add", out: "on each output", sandwich: "before and after" }[cfg.place]}`, `and once at the end, ε = 10⁻⁶`], [bx, 256]);
    s.callout(xt, pos["f-moe"].mid - 6, cur.dense ? [`Dense FFN, width ${f(D.Fd)}`, "every token, every weight"] : [`${N} routed experts, ${k} active`, `per token${ns ? ", plus 1 shared" : ""}`], [bx, pos["f-moe"].mid]);
    s.callout(xt, am + 40, cfg.attn === "mla" ? [`${H} heads from one ${D.dc}-wide latent`, `head dim ${dk} + ${D.dr} RoPE`] : [`${H} query heads share`, `${Hkv} key/value head${Hkv > 1 ? "s" : ""}, dim ${dk}`], [bx, pos["f-attn"].bot - 6]);
    s.callout(xt, embTop + 18, [`Vocabulary of ${f(V)} tokens`, `Embedding dimension ${d}`], [bx, embTop + 23]);

    // zoom connectors
    const mo = pos["f-moe"], at = pos["f-attn"];
    for (const [a, b] of [[[660, mo.top], [800, 110]], [[660, mo.bot], [800, 664]], [[660, at.top], [800, 692]], [[660, at.bot], [800, 1150]]])
      s.line([a, b], { arrow: false, dash: "2 5", color: "#9C9381", width: 1.1 });

    drawFfnInset(s, D, cfg, cur, isLit);
    if (cfg.attn === "mla") drawMlaInset(s, D); else drawAttnInset(s, D, cfg, cur, isLit);
    const panelTop2 = Math.max(embTop + 144, 990);
    drawExpertPanel(s, D, cfg, cur, isLit, panelTop2);

    const lossY = Math.max(panelTop2 + 180, 1166);
    const routerGeo = cur.dense ? null : { x: 830, w: ns ? 400 : 540, y: 432 };
    drawLosses(s, D, cfg, lossY, { mo: pos["f-moe"], router: routerGeo });
    const footY = lossY + 186;
    const lits = SLOTS.filter(sl => isLit(sl.id)).map(sl => `${sl.label}: ${opt(sl.id, cfg[sl.id]).label}`);
    s.text(40, footY, `Base: configs/presets.py SCALE_300M, vocab ${f(V)}, context ${f(D.ctx)}. Dashed red outlines mark parts that are not in this repo's code` + (lits.length ? ":" : " (none in this configuration)."), 12, 400, P.ink2);
    if (lits.length) s.text(40, footY + 18, lits.join(" · "), 12, 500, P.lit);
    const desc = `Decoder-only transformer, ${L} blocks: ${kind.toLowerCase()} and ${ffnLabel.toLowerCase()}; ${f(pr.total)} parameters, ${f(pr.active)} active per token.`;
    return s.render(1440, footY + 40, `${title} architecture`, desc);
  }

  function drawLosses(s, D, cfg, y0, geo) {
    const terms = lossTerms(D, cfg);
    s.add(`<rect id="f-losses" class="fig-panel" x="40" y="${y0}" width="1370" height="160" rx="16" fill="#FFFFFF" fill-opacity=".55" stroke="${P.router[1]}" stroke-width="1.4"/>`);
    s.text(60, y0 + 28, "Training losses", 14.5, 700);
    s.text(196, y0 + 28, "computed only while training; inference stops at the logits. Numbered dots on the figure mark where each term reads from.", 12, 400, P.ink2);
    const totalX = 1220, n = terms.length, gapB = 14, bw = Math.min(250, (totalX - 70 - (n - 1) * gapB) / n), by = y0 + 44, bh = 74, bus = y0 + 140;
    const counters = { logits: 0, router: 0 }, nRouter = terms.filter(t => t.tap === "router").length;
    terms.forEach((t, i) => {
      const x = 60 + i * (bw + gapB), dashed = t.kind !== "loss";
      s.open(`f-loss-${t.key}`);
      s.add(`<rect x="${x}" y="${by}" width="${bw}" height="${bh}" rx="10" fill="${dashed ? "#FFFFFF" : P.router[0]}" stroke="${t.cite ? P.lit : P.router[1]}" stroke-width="1.5"${dashed ? ' stroke-dasharray="5 3"' : ""}/>`);
      s.add(`<circle cx="${x + 16}" cy="${by + 16}" r="10" fill="${P.ink}"/>`); s.text(x + 16, by + 20.5, String(t.n), 11.5, 700, "#FFFFFF", "middle");
      s.text(x + 32, by + 21, t.name, 13, 700);
      s.text(x + 12, by + 42, t.plain, 11.5, 500, P.ink);
      s.text(x + 12, by + 60, t.weight, 11, 400, P.ink2);
      s.close();
      if (!dashed) s.line([[x + bw / 2, by + bh], [x + bw / 2, bus]], { arrow: false });
      // where this term reads from
      const where = [];
      if (t.tap === "logits") where.push([628 + counters.logits++ * 24, 128]);
      const slot = nRouter - 1 - counters.router;   // left-to-right in term order
      if (t.tap === "router" && geo.mo) where.push([646 - slot * 24, geo.mo.top + 1]);
      if (t.tap === "router" && geo.router) where.push([geo.router.x + geo.router.w - 14 - slot * 24, geo.router.y + 1]);
      if (t.tap === "router") counters.router++;
      if (t.tap === "mtp") where.push([786, 152]);
      where.forEach(([tx, ty], j) => {
        s.open(`f-tap-${t.key}-${j}`);
        s.add(`<circle cx="${tx}" cy="${ty}" r="10" fill="${P.ink}" stroke="#FFFFFF" stroke-width="1.5"/>`);
        s.text(tx, ty + 4.5, String(t.n), 11.5, 700, "#FFFFFF", "middle");
        s.close();
      });
      if (where.length) {
        const [tx, ty] = where[where.length - 1];
        s.add(`<g id="f-link-${t.key}" class="fig-link" visibility="hidden"><polyline points="${tx},${ty + 10} ${x + bw / 2},${by - 4}" fill="none" stroke="${P.router[1]}" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arr)"/></g>`);
      }
    });
    const lossCount = terms.filter(t => t.kind === "loss").length;
    const firstX = 60 + bw / 2, lastLoss = terms.map((t, i) => [t, i]).filter(([t]) => t.kind === "loss").pop()[1];
    s.line([[firstX, bus], [Math.max(60 + lastLoss * (bw + gapB) + bw / 2, firstX), bus], [totalX - 6, bus], [totalX - 6, by + bh / 2], [totalX, by + bh / 2]], { arrow: true });
    s.open("f-loss-total");
    s.add(`<rect x="${totalX}" y="${by}" width="170" height="${bh}" rx="12" fill="${P.ink}" stroke="${P.ink}"/>`);
    s.text(totalX + 85, by + 30, `Σ total loss (${lossCount} term${lossCount > 1 ? "s" : ""})`, 13.5, 700, "#FFFFFF", "middle");
    s.text(totalX + 85, by + 52, "→ backward()", 12.5, 500, "#F6D77A", "middle");
    s.close();
  }

  function drawFfnInset(s, D, cfg, cur, isLit) {
    const { d, N, k, ns } = D;
    s.add(`<rect id="f-moe-inset" class="fig-panel" x="800" y="110" width="610" height="554" rx="16" fill="#FFFFFF" fill-opacity=".55" stroke="${P.moe[1]}" stroke-width="1.4"/>`);
    if (cur.dense) {
      s.text(822, 140, "Dense feed-forward, zoomed in", 17, 700);
      s.text(822, 162, `this block has no routing: every token uses all ${D.gated ? 3 : 2} weight matrices`, 12.5, 400, P.ink2);
      s.box(880, 320, 300, 70, ["#FFFFFF", P.lit], `${d} → ${f(D.Fd)} → ${d}`, `width = the MoE's active width`, { id: "f-sum", lit: true, size: 15 });
      s.line([[1030, 560], [1030, 390]]); s.line([[1030, 320], [1030, 200]]);
      s.text(1030, 192, `FFN output (${d}) → residual add`, 12.5, 500, P.ink, "middle");
      s.text(1030, 580, `x: the normed token (${d})`, 12.5, 500, P.ink, "middle");
      return;
    }
    const ec = cfg.router === "expertChoice";
    s.text(822, 140, "Mixture of Experts, zoomed in", 17, 700);
    s.text(822, 162, ec ? `expert choice: each of ${N} experts takes a fixed number of tokens` : `${N} routed experts; this token uses ${k} (green, an example)${ns ? " plus the shared expert" : ""}`, 12.5, 400, P.ink2);
    s.text(1030, 192, `MoE output (${d}) → residual add`, 12.5, 500, P.ink, "middle");
    s.open("f-sum");
    s.add(`<rect x="880" y="222" width="300" height="42" rx="21" fill="#FFFFFF" stroke="${P.ink}" stroke-width="1.4"/>`);
    s.text(1030, 248, ns ? "weighted sum + shared expert" : "weighted sum", 13.5, 600, P.ink, "middle");
    s.close();
    s.line([[1030, 222], [1030, 200]]);
    const rowW = ns ? 400 : 540, gapE = N > 16 ? 2 : 8, ew = (rowW - (N - 1) * gapE) / N, ex0 = 830, ey = 330;
    const active = N > 16 ? [1, 5, 9, 13, 17, 21, 25, 29] : [2, 5];
    for (let i = 0; i < N; i++) {
      const x = ex0 + i * (ew + gapE), on = active.includes(i) && !ec;
      s.open(`f-e${i + 1}`);
      s.add(`<rect x="${x}" y="${ey}" width="${ew}" height="64" rx="${N > 16 ? 3 : 8}" fill="${on ? P.moe[0] : "#FFFFFF"}" stroke="${on ? P.moe[1] : "#B9AF97"}" stroke-width="${on ? 2.2 : 1.2}"/>`);
      if (N <= 16) s.text(x + ew / 2, ey + 37, `E${i + 1}`, 13, on ? 700 : 500, on ? P.ink : "#8C8574", "middle");
      s.close();
      if (on) s.line([[x + ew / 2, ey], [x + ew / 2, 264]]);
    }
    if (ns) {
      s.box(1260, ey, 130, 64, P.moe, "Shared expert", "always on", { id: "f-shared", size: 13.5, sw: 2.2, subSize: 12 });
      s.line([[1325, ey], [1325, 243], [1182, 243]]);
      s.line([[1030, 560], [1325, 560], [1325, 394]]);
    }
    const rl = ec ? `Router: Linear ${d} → ${N}, softmax` : `Router: Linear ${d} → ${N}, ${cfg.router === "softmax" ? "softmax" : "sigmoid score each"}`;
    const rs = ec ? "each expert keeps its top C = S·k/N tokens" : cfg.balance === "capacity" ? `top-${k}; experts drop tokens past capacity 1.25` : cfg.balance === "lossfree" ? `top-${k} on scores + bias, renormalize` : `keep top-${k}, renormalize to sum to 1`;
    s.box(830, 432, rowW, 54, P.router, rl, rs, { id: "f-router", size: 13.5, subSize: 12, lit: isLit("router") || isLit("balance") });
    if (!ec) for (const i of active) { const x = ex0 + i * (ew + gapE) + ew / 2; s.line([[x, 432], [x, 394]]); }
    s.line([[1030, 610], [1030, 486]]);
    s.text(1030, 630, `x: the normed token (${d})`, 12.5, 500, P.ink, "middle");
    if (!ec) s.text(1240, 655, "training-time terms: see Training losses below", 11, 400, P.ink2, "middle", true);
  }

  function drawAttnInset(s, D, cfg, cur, isLit) {
    const { d, H, dk, Hkv } = D, kv = Hkv * dk, group = H / Hkv;
    s.add(`<rect id="f-attn-inset" class="fig-panel" x="800" y="692" width="610" height="458" rx="16" fill="#FFFFFF" fill-opacity=".55" stroke="${P.attn[1]}" stroke-width="1.4"/>`);
    const name = cfg.attn === "mha" ? "Multi-head attention" : cfg.attn === "mqa" ? "Multi-query attention" : "Grouped-query attention";
    s.text(822, 722, `${name}, zoomed in`, 17, 700);
    s.text(822, 744, Hkv === H ? `${H} heads, each with its own key and value` : `${H} query heads${Hkv > 1 ? ` in ${Hkv} groups of ${group}` : ""}; ${Hkv > 1 ? "each group shares one key/value head" : "all share one key/value head"}`, 12.5, 400, P.ink2);
    const cols = [920, 1105, 1290];
    s.box(960, 760, 290, 34, P.emb, `W_O · concat ${H} heads → ${d}`, null, { id: "f-wo", size: 13, rx: 8 });
    s.line([[1250, 777], [1300, 777]]); s.text(1306, 781, "residual add", 12, 500);
    const maskTxt = cur.local ? `causal sliding window (${D.window})` : "causal mask";
    s.box(840, 808, 530, 54, P.attn, `softmax(QKᵀ / √${dk}${cfg.pos === "alibi" ? " − ALiBi" : ""} + ${cur.local ? "window" : "causal"} mask) · V`, cur.local ? `position i attends to i − ${D.window - 1} … i only` : "position i attends to positions 0 … i only", { id: "f-core", size: 14, subSize: 12, lit: cur.local });
    s.line([[1105, 808], [1105, 794]]);
    if (group > 1) { s.text(1197, 884, "each K/V head serves", 11.5, 400, P.ink2, "middle", true); s.text(1197, 898, `${group} query heads`, 11.5, 400, P.ink2, "middle", true); }
    [[cols[0], "W_Q", `${d} → ${d}  (${H} × ${dk})`, "f-qproj"], [cols[1], "W_K", `${d} → ${kv}  (${Hkv} × ${dk})`, "f-kproj"], [cols[2], "W_V", `${d} → ${kv}  (${Hkv} × ${dk})`, "f-vproj"]]
      .forEach(([c, lab, sub, id]) => s.box(c - 85, 1036, 170, 44, P.attn, lab, sub, { id, size: 13.5, rx: 8, subSize: 11.5 }));
    ["q", "k"].forEach((w, i) => {
      const c = cols[i]; let top = 1036;
      if (cfg.qknorm === "on") { s.box(c - 85, 978, 170, 34, P.norm, "QK-Norm (RMSNorm/head)", null, { id: `f-qkn-${w}`, size: 12, rx: 8 }); s.line([[c, top], [c, 1012]]); top = 978; }
      if (cfg.pos === "rope") { s.box(c - 85, 922, 170, 34, P.emb, "RoPE", null, { id: `f-rope-${w}`, size: 12.5, rx: 8 }); s.line([[c, top], [c, 956]]); top = 922; }
      s.line([[c, top], [c, 862]]);
    });
    if (cfg.pos === "alibi") s.box(1180, 940, 200, 30, ["#FFFFFF", P.emb[1]], "ALiBi bias −m·(i − j)", null, { size: 12, rx: 8 });
    s.line([[cols[2], 1036], [cols[2], 862]]);
    s.line([[cols[0], 1100], [cols[2], 1100]], { arrow: false });
    cols.forEach(c => s.line([[c, 1100], [c, 1080]]));
    s.line([[1105, 1122], [1105, 1100]], { arrow: false });
    s.text(1105, 1138, `x: the normed token (${d})`, 12.5, 500, P.ink, "middle");
  }

  function drawMlaInset(s, D) {
    const { d, H, dk, dc, dr } = D;
    s.add(`<rect id="f-attn-inset" class="fig-panel" x="800" y="692" width="610" height="458" rx="16" fill="#FFFFFF" fill-opacity=".55" stroke="${P.attn[1]}" stroke-width="1.4"/>`);
    s.text(822, 722, "Multi-head latent attention, zoomed in", 17, 700);
    s.text(822, 744, `keys and values rebuilt from one ${dc}-wide latent; a small shared RoPE key carries position`, 12.5, 400, P.ink2);
    s.box(960, 760, 290, 34, P.emb, `W_O · concat ${H} heads → ${d}`, null, { id: "f-wo", size: 13, rx: 8 });
    s.line([[1250, 777], [1300, 777]]); s.text(1306, 781, "residual add", 12, 500);
    s.box(840, 808, 530, 50, P.attn, `softmax([q_C; q_R][k_C; k_R]ᵀ / √${dk + dr} + causal mask) · V`, `head dim ${dk} content + ${dr} RoPE`, { id: "f-core", size: 13.5, subSize: 12 });
    s.line([[1105, 808], [1105, 794]]);
    // columns: q content, q rope, latent -> k/v, k rope
    s.box(820, 1040, 120, 44, P.attn, "W_Q", `${d} → ${H * dk}`, { id: "f-qproj", size: 13, rx: 8, subSize: 11 });
    s.box(950, 1040, 120, 44, P.attn, "W_QR", `${d} → ${H * dr}`, { id: "f-qr", size: 13, rx: 8, subSize: 11 });
    s.box(1080, 1040, 160, 44, P.router, "W_DKV → latent", `${d} → ${dc}`, { id: "f-dkv", size: 13, rx: 8, subSize: 11 });
    s.box(1250, 1040, 140, 44, P.attn, "W_KR (shared)", `${d} → ${dr}`, { id: "f-kr", size: 13, rx: 8, subSize: 11 });
    s.box(1080, 966, 76, 40, P.attn, "W_UK", `→ ${H * dk}`, { id: "f-uk", size: 12.5, rx: 8, subSize: 10.5 });
    s.box(1164, 966, 76, 40, P.attn, "W_UV", `→ ${H * dk}`, { id: "f-uv", size: 12.5, rx: 8, subSize: 10.5 });
    s.box(950, 980, 120, 30, P.emb, "RoPE", null, { id: "f-rope-q", size: 12, rx: 8 });
    s.box(1250, 980, 140, 30, P.emb, "RoPE", null, { id: "f-rope-k", size: 12, rx: 8 });
    s.line([[880, 1040], [880, 858]]); s.line([[1010, 1040], [1010, 1010]]); s.line([[1010, 980], [1010, 858]]);
    s.line([[1118, 1040], [1118, 1006]]); s.line([[1202, 1040], [1202, 1006]]); s.line([[1118, 966], [1118, 858]]); s.line([[1202, 966], [1202, 858]]);
    s.line([[1320, 1040], [1320, 1010]]); s.line([[1320, 980], [1320, 858]]);
    s.text(1160, 1100, `cached per token: ${dc} + ${dr} = ${dc + dr} numbers`, 11.5, 500, P.router[1], "middle", true);
    s.line([[880, 1112], [1320, 1112]], { arrow: false });
    s.text(1105, 1138, `x: the normed token (${d})`, 12.5, 500, P.ink, "middle");
  }

  function drawExpertPanel(s, D, cfg, cur, isLit, top) {
    const { d } = D, F = cur.dense ? D.Fd : D.F;
    const actName = { swiglu: "SiLU", geglu: "GELU", gelu: "GELU", relu2: "ReLU²" }[cfg.act];
    const lit = isLit("act");
    s.add(`<rect id="f-swiglu" class="fig-panel" x="40" y="${top}" width="700" height="160" rx="16" fill="#FFFFFF" fill-opacity=".55" stroke="${lit ? P.lit : P.moe[1]}" stroke-width="1.4"${lit ? ' stroke-dasharray="6 3"' : ""}/>`);
    const name = { swiglu: "SwiGLU", geglu: "GeGLU", gelu: "GELU", relu2: "squared-ReLU" }[cfg.act];
    s.text(60, top + 28, `Inside ${cur.dense ? "the dense FFN" : "every expert (routed and shared)"}: ${name} feed-forward`, 14.5, 700);
    const my = top + 95;
    s.text(84, my + 5, `x (${d})`, 12.5, 500, P.ink, "middle");
    if (D.gated) {
      s.line([[112, my], [130, my], [130, my - 27], [150, my - 27]]); s.line([[130, my], [130, my + 27], [150, my + 27]]);
      s.box(150, my - 45, 180, 36, P.moe, `Linear ${d} → ${f(F)}`, null, { id: "f-gate", size: 12.5, rx: 8 });
      s.box(350, my - 45, 60, 36, P.norm, actName, null, { id: "f-silu", size: 12.5, rx: 8 });
      s.line([[330, my - 27], [350, my - 27]]);
      s.box(150, my + 9, 180, 36, P.moe, `Linear ${d} → ${f(F)}`, null, { id: "f-up", size: 12.5, rx: 8 });
      s.open("f-mul"); s.add(`<circle cx="452" cy="${my}" r="14" fill="#FFFFFF" stroke="${P.ink}" stroke-width="1.5"/>`); s.text(452, my + 5, "×", 16, 600, P.ink, "middle"); s.close();
      s.line([[410, my - 27], [440, my - 9]]); s.line([[330, my + 27], [440, my + 9]]);
      s.box(490, my - 18, 180, 36, P.moe, `Linear ${f(F)} → ${d}`, null, { id: "f-down", size: 12.5, rx: 8 });
      s.line([[466, my], [490, my]]);
    } else {
      s.line([[112, my], [150, my]]);
      s.box(150, my - 18, 180, 36, P.moe, `Linear ${d} → ${f(F)}`, null, { id: "f-up", size: 12.5, rx: 8 });
      s.box(350, my - 18, 90, 36, P.norm, actName, null, { id: "f-silu", size: 12.5, rx: 8, lit });
      s.line([[330, my], [350, my]]); s.line([[440, my], [490, my]]);
      s.box(490, my - 18, 180, 36, P.moe, `Linear ${f(F)} → ${d}`, null, { id: "f-down", size: 12.5, rx: 8 });
    }
    s.line([[670, my], [700, my]]); s.text(704, my + 5, `${d}`, 12.5, 500);
    const per = (D.gated ? 3 : 2) * d * F;
    s.text(60, top + 150, `${D.gated ? 3 : 2} × ${d} × ${f(F)} = ${f(per)} parameters per ${cur.dense ? "dense FFN" : "expert"}`, 12, 400, P.ink2);
  }

  /* map a step from one chain onto another when the configuration changes */
  function mapStep(oldSteps, newSteps, i) {
    const keys = newSteps.map(s => s.key);
    for (let j = i; j < oldSteps.length; j++) { const m = keys.indexOf(oldSteps[j].key); if (m >= 0) return m; }
    for (let j = i; j >= 0; j--) { const m = keys.indexOf(oldSteps[j].key); if (m >= 0) return m; }
    return Math.min(i, newSteps.length - 1);
  }

  const QM = { CITE, SLOTS, DEFAULT, inactive, dims, layers, isRepo, params, buildSteps, drawFigure, mapStep, opt, lossTerms };
  if (typeof module !== "undefined" && module.exports) module.exports = QM;
  else root.QM = QM;
})(typeof window !== "undefined" ? window : globalThis);
