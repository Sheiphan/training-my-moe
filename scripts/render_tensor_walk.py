"""Generate site/tensor-walk/: an animated walk of one forward pass, every tensor drawn to scale,
for SCALE_300M or any variant of it built from the configurator on the page.

    uv run python scripts/render_tensor_walk.py

The architecture itself (slots, variants, steps, parameter formula, figure) lives in
site/shared/qwenmoe-model.js so the page can rebuild it live. This script supplies what the
browser cannot know and checks what it must not get wrong:

  * the base config and the doc/code link for every step that exists in this repo, each
    resolved against the real headings and source lines (a moved heading fails the build);
  * a verification that the JS parameter formula equals qwenmoe.train.scaling.count_params
    on the real PyTorch model, for a spread of repo-supported configurations. Needs `node`.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import render_model_figure as fig  # noqa: E402
from qwenmoe.model.transformer import QwenMoEModel  # noqa: E402
from qwenmoe.train.scaling import count_params  # noqa: E402

OUT = ROOT / "site" / "tensor-walk" / "index.html"
ENGINE = ROOT / "site" / "shared" / "qwenmoe-model.js"
TEMPLATE = Path(__file__).resolve().parent / "tensor_walk_template.html"
MATH = ROOT / "docs" / "math"
BASE_FIELDS = ("d_model", "n_heads", "n_kv_heads", "n_routed_experts", "top_k", "d_ff",
               "vocab_size", "n_layers", "max_seq_len", "rope_theta")


def _gh_slug(heading: str) -> str:
    t = heading.lstrip("#").strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def doc_link(n: int, pattern: str) -> dict:
    path = sorted(MATH.glob(f"{n:03d}-*.md"))[0]
    seen: dict[str, int] = {}
    fence = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fence = not fence
        if fence or not line.startswith("#"):
            continue
        slug = _gh_slug(line)
        k = seen.get(slug, 0)
        seen[slug] = k + 1
        if re.search(pattern, line):
            anchor = slug if k == 0 else f"{slug}-{k}"
            label = line.lstrip("#").strip()
            label = re.sub(r"\^\{?\\top\}?", "ᵀ", label)
            label = re.sub(r"\\sqrt\{([^}]*)\}", r"√\1", label)
            label = re.sub(r"[`*$\\]", "", label)
            label = re.sub(r"\{([^}]*)\}", r"\1", label)
            return {"href": f"../../docs/math/{path.name}#{anchor}", "label": f"{n:03d} · {label}"}
    raise AssertionError(f"doc {n:03d}: no heading matches {pattern!r}")


def code_link(path: str, pattern: str, nth: int = 0) -> dict:
    hits = [i for i, line in enumerate((ROOT / path).read_text().splitlines(), 1) if re.search(pattern, line)]
    assert len(hits) > nth, f"{path}: {pattern!r} has {len(hits)} match(es), wanted index {nth}"
    return {"href": f"../../{path}#L{hits[nth]}", "label": f"{path}:{hits[nth]}"}


ATT, MOE, BLK, TR = "qwenmoe/model/attention.py", "qwenmoe/model/moe.py", "qwenmoe/model/block.py", "qwenmoe/model/transformer.py"

# step key -> (derivation, code) for every step that exists in this repo
LINKS = {
    "ids": ((7, r"^## Part 3 — `encode\(\)`"), ("qwenmoe/tokenizer/bpe.py", r"def encode")),
    "embed": ((6, r"^### Step 1 — Embedding lookup"), (TR, r"x = self\.embedding\(input_ids\)")),
    "norm_a": ((6, r"^### Step 4 — Block 0: `attn_norm"), (BLK, r"x = x \+ self\.attn\(")),
    "norm_f": ((6, r"^### Step 7 — Block 0: `moe_norm"), (BLK, r"self\.moe_norm\(x\)")),
    "norm_final": ((6, r"^### Step 16"), (TR, r"normed = self\.final_norm\(x\)")),
    "q": ((4, r"^#### 1a\. Queries"), (ATT, r"q = self\.w_q\(x\)", 1)),
    "k": ((4, r"^#### 1b\. Keys"), (ATT, r"k = self\.w_k\(x\)", 1)),
    "v": ((4, r"^#### 1c\. Values"), (ATT, r"v = self\.w_v\(x\)", 1)),
    "split": ((4, r"^### Step 2 — Split"), (ATT, r"\.view\(batch, seq_len, self\.n_heads", 1)),
    "qknorm": ((22, r"^### Step 3 — QK-norm"), (ATT, r"q = self\.q_norm\(q\)", 1)),
    "rope": ((4, r"^#### 4·RoPE"), ("qwenmoe/model/positional.py", r"^class RotaryEmbedding")),
    "alibi": ((4, r"^### Step 8 — Add the positional score bias"), ("qwenmoe/model/positional.py", r"^class AlibiBias")),
    "gqa": ((4, r"^### Step 5 — GQA"), (ATT, r"k = k\.repeat_interleave")),
    "scores": ((4, r"^### Step 6 — Raw attention scores"), (ATT, r"scores = \(q @ k\.transpose")),
    "mask": ((4, r"^### Step 9 — The causal mask"), (ATT, r"^def _causal_position_mask")),
    "softmax": ((4, r"^### Step 10 — Softmax"), (ATT, r"probs = F\.softmax\(scores")),
    "weighted": ((4, r"^### Step 11 — Weighted sum"), (ATT, r"out = probs @ v")),
    "concat": ((4, r"^#### 12a\."), (ATT, r"out = out\.transpose\(1, 2\)\.contiguous\(\)", 1)),
    "wo": ((4, r"^#### 12b\. Output projection"), (ATT, r"return self\.w_o\(out\)")),
    "mla_dkv": ((16, r"^### Step 1 — Compress once"), (ATT, r"self\.w_dkv = nn\.Linear")),
    "mla_uk": ((16, r"^### Step 2 — Up-project the key"), (ATT, r"self\.w_uk = nn\.Linear")),
    "mla_uv": ((16, r"^### Step 3 — Up-project the value"), (ATT, r"self\.w_uv = nn\.Linear")),
    "mla_qr": ((16, r"^### Step 5 — The decoupled-RoPE slices"), (ATT, r"self\.w_qr = nn\.Linear")),
    "mla_kr": ((16, r"^### Step 5 — The decoupled-RoPE slices"), (ATT, r"self\.w_kr = nn\.Linear")),
    "mla_rope": ((16, r"^### Step 6 — Rotate the slices"), (ATT, r"q_rope, k_rope_new = self\.rope\.apply_to_qk")),
    "mla_cat": ((16, r"^### Step 8 — Concatenate content"), (ATT, r"q = torch\.cat\(\[q_content, q_rope\]")),
    "res1": ((6, r"^### Step 6 — Block 0: residual add"), (BLK, r"x = x \+ self\.attn\(")),
    "res2": ((6, r"^### Step 9 — Block 0: residual add"), (BLK, r"x = x \+ moe_out")),
    "router": ((25, r"^## Part 2 — `score_and_select`"), (MOE, r"^def score_and_select")),
    "topk": ((5, r"^### Step 4 — Selection scores"), (MOE, r"_, topk_indices = selection_scores\.topk")),
    "dispatch": ((5, r"^#### 6a\. Finding an expert's tokens"), (MOE, r"^def dispatch_batched")),
    "expert_up": ((5, r"^#### 6c\. What one SwiGLU expert"), (MOE, r"^def build_expert")),
    "expert_down": ((5, r"^#### 6c\. What one SwiGLU expert"), (MOE, r"^def build_expert")),
    "shared": ((5, r"^### Step 7 — The shared expert"), (MOE, r"for expert in self\.shared_experts")),
    "combine": ((5, r"^#### 6g\. Gate weights and the scatter-add"), (MOE, r"routed_output = dispatch_batched\(")),
    "blocks": ((6, r"^### Steps 10–15 — Block 1"), (TR, r"for i, block in enumerate\(self\.blocks\)")),
    "head": ((6, r"^### Step 17"), (TR, r"logits = self\.lm_head\(normed\)")),
    "mtp": ((28, r"^## Part 2 — MTP head, depth 1"), ("qwenmoe/model/mtp.py", r"^class MTPHead")),
    "next": ((11, r"^### Step 8 — Keep only the last position"), ("qwenmoe/model/generate.py", r"^def _sample_next")),
    "loss_ce": ((10, r"^### 4\.3 `F\.cross_entropy`"), ("qwenmoe/train/trainer.py", r"loss = F\.cross_entropy")),
    "loss_aux": ((5, r"^### Step 9 — The load-balancing"), (MOE, r"def aux_loss\(self, router_probs")),
    "loss_rz": ((25, r"^## Part 5 — The router z-loss"), (MOE, r"def _add_router_z_loss")),
    "loss_oz": ((24, r"^## Part 5 — Cross-entropy's shift-invariance"), ("qwenmoe/train/trainer.py", r"log_z = torch\.logsumexp")),
    "loss_mtp": ((28, r"^## Part 5 — Training: `mtp_lambda`"), ("qwenmoe/train/trainer.py", r"total_loss = total_loss \+ model\.config\.mtp_lambda")),
    "loss_bias": ((38, r"^## Part 5 — `LossFreeBalancer`"), (MOE, r"^class LossFreeBalancer")),
    "loss_total": ((10, r"^### 4\.4 Adding the auxiliary loss"), ("qwenmoe/train/trainer.py", r"\(total_loss \* loss_scale\)\.backward\(\)")),
}


def resolved_links() -> dict:
    return {key: {"doc": doc_link(n, pat), "code": code_link(*code)} for key, ((n, pat), code) in LINKS.items()}


# ---------------------------------------------------------------------------
# verification: the JS parameter formula vs the real model, on repo-supported configs
# ---------------------------------------------------------------------------
JS_DEFAULT = dict(norm="rms", place="pre", pos="rope", attn="gqa", qknorm="on", window="global", attnPattern="uniform",
                  ffn="moe", ffnPattern="uniform", experts="coarse", shared="1", act="swiglu", router="sigmoid",
                  balance="aux", zloss="none", tie="tied", mtp="0")
CHECKS = [
    {}, {"attn": "mha"}, {"attn": "mqa"}, {"attn": "mla"}, {"qknorm": "off"}, {"pos": "nope"}, {"pos": "alibi"},
    {"experts": "fine"}, {"shared": "0"}, {"act": "geglu"}, {"act": "gelu"}, {"router": "softmax"}, {"balance": "lossfree"},
    {"zloss": "both"}, {"tie": "untied"}, {"mtp": "1"},
    {"attn": "mla", "experts": "fine", "mtp": "1"}, {"attn": "mla", "act": "gelu", "tie": "untied"},
    {"attn": "mha", "qknorm": "off", "mtp": "1"}, {"attn": "mqa", "tie": "untied", "shared": "0"},
    {"experts": "fine", "act": "gelu", "shared": "0"}, {"experts": "fine", "act": "geglu", "mtp": "1", "tie": "untied"},
]


def to_model_config(base, js: dict):
    kw: dict = {"n_kv_heads": {"gqa": base.n_kv_heads, "mha": base.n_heads, "mqa": 1, "mla": base.n_kv_heads}[js["attn"]]}
    if js["attn"] == "mla":
        kw.update(attention_variant="mla", mla_latent_dim=base.d_model // 4)
    kw["qk_norm"] = js["qknorm"] == "on"
    kw["positional_scheme"] = js["pos"]
    if js["experts"] == "fine":
        kw.update(n_routed_experts=base.n_routed_experts * 4, top_k=base.top_k * 4, d_ff=base.d_ff // 4)
    kw["n_shared_experts"] = int(js["shared"])
    kw["activation"] = js["act"]
    kw["router_scoring"] = js["router"]
    kw["moe_balancing"] = {"aux": "aux_loss", "lossfree": "loss_free"}[js["balance"]]
    kw["router_z_loss_weight"] = 0.001 if js["zloss"] in ("router", "both") else 0.0
    kw["tie_word_embeddings"] = js["tie"] == "tied"
    kw["mtp_depth"] = int(js["mtp"])
    return dataclasses.replace(base, **kw)


run_engine = fig.run_engine


def verify(base) -> int:
    base_json = {k: getattr(base, k) for k in BASE_FIELDS}
    cfgs = [{**JS_DEFAULT, **c} for c in CHECKS]
    results = run_engine("A[1].map(c=>[QM.isRepo(c),QM.params(A[0],c)])", [base_json, cfgs])
    for c, (repo, p) in zip(cfgs, results):
        assert repo, f"check config is not repo-only: {c}"
        with torch.device("meta"):
            m = QwenMoEModel(to_model_config(base, c))
        cp = count_params(m)
        mtp = sum(t.numel() for t in m.mtp_heads.parameters())
        got, want = (p["total"], p["active"], p["mtp"]), (cp.total, cp.active, mtp)
        assert got == want, f"JS params {got} != PyTorch {want} for {c}"
    return len(cfgs)


def main() -> None:
    cfg, _, _ = fig.build()
    n_checked = verify(cfg)
    data = {"base": {k: getattr(cfg, k) for k in BASE_FIELDS}, "links": resolved_links(),
            "preset": fig.PRESET, "verified": n_checked}
    page = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__PRESET__", fig.PRESET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT}; JS parameter formula matches PyTorch count_params on {n_checked} configurations")


if __name__ == "__main__":
    main()
