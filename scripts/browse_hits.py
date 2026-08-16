#!/usr/bin/env python
"""Interactively browse classified WildChat hits from a batch results file.

Reads a results JSONL (default runs/english_full/results/strong_hits.jsonl),
reconstructs each conversation from its shard (the exact rendered prompt we
scored -- no re-download), and pretty-prints it with the scores and the model's
reasoning. Navigate with the keyboard.

Usage:
    .venv/bin/python scripts/browse_hits.py
    .venv/bin/python scripts/browse_hits.py --file runs/english_full/results/conceptual_ge2.jsonl
    .venv/bin/python scripts/browse_hits.py --min-novelty 3          # tighten further
    .venv/bin/python scripts/browse_hits.py --sort novelty          # order by an axis
    .venv/bin/python scripts/browse_hits.py --start 40

Navigation (Enter after each):
    [Enter]  next        p  previous        <number>  jump to that #
    q        quit        (any results file with custom_id + shard_id works)
"""

import argparse
import gzip
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "runs" / "english_full" / "results" / "strong_hits.jsonl"
TURN_RE = re.compile(r'<turn index="(\d+)" role="(\w+)">')

USE_COLOR = sys.stdout.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def wrap(text, width, indent="  "):
    out = []
    for line in text.split("\n"):
        if line.strip():
            out.append(textwrap.fill(line, width=width, initial_indent=indent,
                                     subsequent_indent=indent))
        else:
            out.append("")
    return "\n".join(out)


def parse_turns(prompt):
    """Reconstruct [(role, content), ...] from a rendered classifier prompt."""
    a = prompt.find("<conversation>")
    b = prompt.find("</conversation>")
    if a == -1 or b == -1:
        return [], ""
    block = prompt[a + len("<conversation>"):b]
    turns, cur, note = [], None, ""
    for line in block.splitlines():
        s = line.strip()
        m = TURN_RE.match(s)
        if m:
            cur = [m.group(2), []]
            turns.append(cur)
        elif s == "</turn>":
            cur = None
        elif cur is None and s.startswith("[") and "later turn" in s:
            note = s
        elif cur is not None:
            cur[1].append(line)
    return [(role, "\n".join(lines).strip("\n")) for role, lines in turns], note


def load_prompts(hits, shards_dir):
    """One pass per shard: pull the rendered prompt for each hit's custom_id."""
    by_shard = {}
    for h in hits:
        by_shard.setdefault(h["shard_id"], set()).add(h["custom_id"])
    prompts = {}
    shards = sorted(by_shard)
    for i, sid in enumerate(shards, 1):
        want = by_shard[sid]
        path = shards_dir / f"shard_{sid}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as gz:
            for line in gz:
                rec = json.loads(line)
                if rec["custom_id"] in want:
                    prompts[rec["custom_id"]] = rec["params"]["messages"][0]["content"]
                    want = want - {rec["custom_id"]}
                    if not want:
                        break
        print(f"\r  loading conversations… {i}/{len(shards)} shards", end="", flush=True)
    print("\r" + " " * 50 + "\r", end="")
    return prompts


def show(hit, prompt, pos, total):
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    bar, thin = "=" * width, "-" * width
    print("\n" * 2)
    print(c("1", bar))
    print(c("1;32", f"[{pos}/{total}]  conceptual {hit['conceptual']}  "
                    f"novelty {hit['novelty']}  well-formed {hit['well_formed']}"))
    print(c("1", bar))
    print(f"Hash:      {hit.get('conversation_hash')}   (custom_id {hit['custom_id']})")
    print(f"Model:     {hit.get('src_model')}    Timestamp: {hit.get('timestamp')}")
    print(f"Language:  {hit.get('language')}    Country: {hit.get('country')}    "
          f"Turns: {hit.get('n_turns')}")
    print()
    print(c("1", "Classifier reasoning:"))
    print(wrap(hit.get("reasoning", ""), width))
    if prompt is None:
        print(c("33", "\n(conversation text unavailable — shard not found)"))
        return
    turns, note = parse_turns(prompt)
    for i, (role, content) in enumerate(turns, 1):
        color = "1;36" if role == "user" else "1;35"
        print(c("2", thin))
        print(c(color, f"[{i}] {role.upper()}"))
        print(c("2", thin))
        print(wrap(content, width))
    if note:
        print(c("2", f"\n  {note}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--min-conceptual", type=int, default=0)
    ap.add_argument("--min-novelty", type=int, default=0)
    ap.add_argument("--min-wellformed", type=int, default=0)
    ap.add_argument("--language", default=None, help="filter to one language (e.g. English)")
    ap.add_argument("--sort", choices=["conceptual", "novelty", "well_formed", "none"],
                    default="conceptual", help="primary sort axis (desc); default conceptual")
    ap.add_argument("--start", type=int, default=1, help="1-based index to start at")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"no such file: {path}")
    hits = [json.loads(l) for l in open(path, encoding="utf-8")]
    hits = [h for h in hits if h.get("status", "ok") == "ok"
            and h["conceptual"] >= args.min_conceptual
            and h["novelty"] >= args.min_novelty
            and h["well_formed"] >= args.min_wellformed
            and (args.language is None or (h.get("language") or "") == args.language)]
    if args.sort != "none":
        keys = {"conceptual": ("conceptual", "novelty", "well_formed"),
                "novelty": ("novelty", "conceptual", "well_formed"),
                "well_formed": ("well_formed", "conceptual", "novelty")}[args.sort]
        hits.sort(key=lambda h: tuple(-h[k] for k in keys))
    if not hits:
        sys.exit("no hits match the filters.")

    print(f"{len(hits)} hit(s) from {path.name}"
          + (f"  (filters: c>={args.min_conceptual} n>={args.min_novelty} "
             f"w>={args.min_wellformed}"
             + (f" lang={args.language}" if args.language else "") + ")"
             if (args.min_conceptual or args.min_novelty or args.min_wellformed or args.language)
             else ""))
    shards_dir = path.resolve().parent.parent / "shards"
    prompts = load_prompts(hits, shards_dir)

    interactive = sys.stdin.isatty()
    i = max(0, min(args.start - 1, len(hits) - 1))
    while True:
        h = hits[i]
        show(h, prompts.get(h["custom_id"]), i + 1, len(hits))
        if not interactive:
            i += 1
            if i >= len(hits):
                break
            continue
        try:
            ans = input(c("1", f"\n[{i+1}/{len(hits)}]  [Enter] next   [p] prev   "
                               f"[#] jump   [q] quit > ")).strip().lower()
        except EOFError:
            break
        if ans in ("q", "quit"):
            break
        elif ans == "p":
            i = max(0, i - 1)
        elif ans.isdigit():
            i = max(0, min(int(ans) - 1, len(hits) - 1))
        else:
            i += 1
            if i >= len(hits):
                print(c("1;32", "\n— end of hits —"))
                break


if __name__ == "__main__":
    main()
