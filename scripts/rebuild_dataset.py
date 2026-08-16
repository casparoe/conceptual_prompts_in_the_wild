#!/usr/bin/env python
"""Rebuild a filtered prompt dataset from committed scores -- WITHOUT re-running Haiku.

The repo commits only scores + join keys (scores/*.parquet), never the source text.
This script filters the scores by thresholds, re-fetches the public source data,
joins on `key`, and writes the selected prompts (with their conversation text).

Usage:
    .venv/bin/python scripts/rebuild_dataset.py --source lesswrong \
        --min-conceptual 2 --min-novelty 2 --min-wellformed 2

    .venv/bin/python scripts/rebuild_dataset.py --source philosophy_se --min-novelty 3

Sources: wildchat, lesswrong, eaforum, philosophy_se
Note: `wildchat` re-downloads ~15 GB of WildChat parquet (its text isn't stored
here); the forum/SE sources download small dumps (<600 MB total, cached in
runs/_sources/).
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_classify as bc          # noqa: E402  (WildChat parquet streaming)
import classify_sources as cs        # noqa: E402  (forum/SE adapters)

REPO_ROOT = bc.REPO_ROOT
SOURCES = {
    "wildchat":      {"scores": "scores/english_full.parquet", "kind": "wildchat"},
    "lesswrong":     {"scores": "scores/lesswrong_questions.parquet", "kind": "adapter",
                      "adapter": "lesswrong_questions"},
    "eaforum":       {"scores": "scores/eaforum_questions.parquet", "kind": "adapter",
                      "adapter": "eaforum_questions"},
    "philosophy_se": {"scores": "scores/philosophy_se.parquet", "kind": "adapter",
                      "adapter": "philosophy_se"},
    "sharegpt":      {"scores": "scores/sharegpt.parquet", "kind": "adapter",
                      "adapter": "sharegpt"},
    "prism":         {"scores": "scores/prism.parquet", "kind": "adapter",
                      "adapter": "prism"},
}


def load_selected(scores_path: Path, mc: int, mn: int, mw: int) -> dict:
    t = pq.read_table(scores_path)
    mask = pc.and_(pc.equal(t["status"], "ok"),
                   pc.and_(pc.greater_equal(t["conceptual"], mc),
                           pc.and_(pc.greater_equal(t["novelty"], mn),
                                   pc.greater_equal(t["well_formed"], mw))))
    t = t.filter(mask)
    sel = {}
    for k, c, n, w in zip(t["key"].to_pylist(), t["conceptual"].to_pylist(),
                          t["novelty"].to_pylist(), t["well_formed"].to_pylist()):
        sel[k] = {"conceptual": c, "novelty": n, "well_formed": w}
    return sel


def rebuild_adapter(adapter_name: str, sel: dict, source: str) -> list:
    out = []
    for rec in cs.ADAPTERS[adapter_name]():
        k = str(rec["id"])
        if k in sel:
            out.append({"key": k, "source": source, **sel[k],
                        "url": rec["manifest"].get("url"),
                        "title": rec["manifest"].get("title"),
                        "conversation": rec["conversation"]})
    return out


def rebuild_wildchat(sel: dict) -> list:
    remaining = set(sel)
    out = []
    tmp = REPO_ROOT / "runs" / "_rebuild_tmp.parquet"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    for i in range(bc.NUM_FILES):
        if not remaining:
            break
        print(f"  wildchat file {i:04d}/{bc.NUM_FILES} ({len(remaining):,} keys left)…", flush=True)
        bc.download_parquet(i, tmp)
        for row in bc.iter_rows(tmp):
            h = row.get("conversation_hash")
            if h in remaining:
                out.append({"key": h, "source": "wildchat", **sel[h],
                            "conversation": [{"role": t.get("role"), "content": t.get("content")}
                                             for t in row["conversation"]]})
                remaining.discard(h)
                if not remaining:
                    break
    if tmp.exists():
        tmp.unlink()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--min-conceptual", type=int, default=2)
    ap.add_argument("--min-novelty", type=int, default=2)
    ap.add_argument("--min-wellformed", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = SOURCES[args.source]
    scores_path = REPO_ROOT / spec["scores"]
    if not scores_path.exists():
        sys.exit(f"missing {scores_path} (run from the repo root)")
    sel = load_selected(scores_path, args.min_conceptual, args.min_novelty, args.min_wellformed)
    print(f"{args.source}: {len(sel):,} rows match "
          f"c>={args.min_conceptual} n>={args.min_novelty} w>={args.min_wellformed}")
    if not sel:
        return

    if spec["kind"] == "wildchat":
        out = rebuild_wildchat(sel)
    else:
        out = rebuild_adapter(spec["adapter"], sel, args.source)

    outpath = Path(args.out or (REPO_ROOT / "runs" / "rebuilt" /
                   f"{args.source}_c{args.min_conceptual}n{args.min_novelty}w{args.min_wellformed}.jsonl"))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out):,} prompts -> {outpath}")
    if len(out) < len(sel):
        print(f"  note: {len(sel) - len(out):,} selected keys not found in the current "
              f"source (dump drift / recency); scores still valid for the rest.")


if __name__ == "__main__":
    main()
