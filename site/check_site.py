"""Check that every file the pages in site/ depend on exists and is tracked by git.

    uv run python site/check_site.py           # summary + any problems, exit 1 if any
    uv run python site/check_site.py --list    # also print every referenced file, per page

What it follows, per page:
  * static references: every local `src="..."` / `href="..."`
  * fonts inside linked stylesheets: `url(...)` in each local .css (KaTeX lists .woff2 first
    with .woff/.ttf fallbacks; only the .woff2 files are vendored, and browsers stop at the first
    format they support, so a missing fallback is fine when its .woff2 sibling exists)
  * runtime data: token-journey's `store.js` loads `data/<file>` for each entry in `data/manifest.js`
  * the architecture map's embedded links to math docs, journal entries, code lines and verify scripts
  * the tensor walk's per-step links to derivation sections and code lines

"Tracked" means committed or staged (`git add`), so a fresh clone gets the file.
External URLs (fonts, the Mermaid CDN) are listed as external, not checked.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


TRACKED = set(_git("ls-files", "--cached").splitlines())  # committed + staged


def _rel(base: Path, url: str) -> str | None:
    url = url.split("#")[0].split("?")[0]
    if not url or re.match(r"^(https?:|data:|mailto:|javascript:|\$\{)", url):
        return None
    return os.path.relpath((base / url).resolve(), ROOT)


def references(page: Path) -> tuple[dict[str, set[str]], set[str]]:
    base, text = page.parent, page.read_text()
    refs: dict[str, set[str]] = {}
    external = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', text))
    external |= set(re.findall(r'import\("(https?://[^"]+)"\)', text))

    def add(kind: str, rel: str | None) -> None:
        if rel:
            refs.setdefault(kind, set()).add(rel)

    for url in re.findall(r'(?:src|href)="([^"]+)"', text):
        add("static", _rel(base, url))
    for css in re.findall(r'href="([^"]+\.css)"', text):
        css_path = (base / css).resolve()
        if css_path.exists():
            for u in re.findall(r"url\(([^)]+)\)", css_path.read_text()):
                rel = _rel(css_path.parent, u.strip("'\""))
                woff2 = rel and re.sub(r"\.(woff|ttf)$", ".woff2", rel)
                if rel and rel != woff2 and (ROOT / woff2).exists():
                    continue  # format fallback; the .woff2 sibling is what browsers load
                add("css url()", rel)
    if page == SITE / "token-journey" / "index.html":
        manifest = (base / "data" / "manifest.js").read_text()
        for f in re.findall(r'"?file"?\s*:\s*"([^"]+)"', manifest):
            add("runtime data", _rel(base, "data/" + f))
    m = re.search(r"const (?:D|DATA) = (\{.*?\});\n", text, re.S)
    if m:
        data = json.loads(m.group(1))
        for n in data.get("nodes", []):  # architecture map
            for k in ("math", "journal", "code"):
                if n.get(k):
                    add("map link", _rel(base, n[k]))
        for d in data.get("docs", []):
            for k in ("math", "journal", "verify"):
                add("map link", _rel(base, d[k]))
        for st in data.get("steps", []):  # tensor walk
            for k in ("doc", "code"):
                add("step link", _rel(base, st[k]["href"]))
        for lk in data.get("links", {}).values():  # tensor walk (links keyed by step)
            for k in ("doc", "code"):
                add("step link", _rel(base, lk[k]["href"]))
    return refs, external


def status(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return "MISSING"
    if p.is_dir():
        return "ok"
    return "ok" if rel in TRACKED else "UNTRACKED"


def main() -> int:
    show_all = "--list" in sys.argv
    problems = 0
    for page in sorted(SITE.rglob("*.html")):
        prel = os.path.relpath(page, ROOT)
        refs, external = references(page)
        own = status(prel)
        problems += own != "ok"
        total = sum(len(v) for v in refs.values())
        print(f"{prel}  [{own}]  {total} local refs, {len(external)} external")
        for kind, rels in sorted(refs.items()):
            for rel in sorted(rels):
                st = status(rel)
                if st != "ok":
                    problems += 1
                    print(f"    {st:9s} {kind:12s} {rel}")
                elif show_all:
                    print(f"    ok        {kind:12s} {rel}")
        if show_all:
            for url in sorted(external):
                print(f"    external  {url}")
    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
