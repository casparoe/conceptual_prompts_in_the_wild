#!/usr/bin/env python
"""Prepare non-WildChat sources into batch_classify-compatible run directories.

Each adapter maps a source dataset to the conversation shape the classifier
expects, renders prompts with the SAME template, and writes shards + state that
scripts/batch_classify.py {submit,poll} consume unchanged. So every source is
scored by the identical Batch-API pipeline and is browsable with
scripts/browse_hits.py.

Sources:
  lesswrong_questions  LessWrong posts carrying the `question` flag (dump)
  eaforum_questions    EA Forum posts with an interrogative ('?') title
                       (dump has no question flag; the classifier's well_formed
                       axis filters out any non-questions)
  philosophy_se        Philosophy StackExchange Q&A (already conversation-shaped)

Usage:
  .venv/bin/python scripts/classify_sources.py prepare --source lesswrong_questions
  .venv/bin/python scripts/batch_classify.py --run-dir runs/lesswrong_questions submit --yes
  .venv/bin/python scripts/batch_classify.py --run-dir runs/lesswrong_questions poll --watch
  .venv/bin/python scripts/browse_hits.py --file runs/lesswrong_questions/results/strong_hits.jsonl
"""

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_classify as bc  # noqa: E402  (reuse schema, caps, helpers)

REPO_ROOT = bc.REPO_ROOT
SOURCES_DIR = REPO_ROOT / "runs" / "_sources"
CUSTOM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def strip_html(h: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h or ""))).strip()


def download_to(url: str, path: Path) -> None:
    """Stream a URL to path, sending the HF token if set (needed for gated datasets)."""
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers=bc.hf_headers())
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)


def ensure(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {path.name} …", flush=True)
    download_to(url, path)
    return path


# ---------------------------------------------------------------- adapters
# each yields {"id", "conversation": [{role, content}], "manifest": {...}}

def src_lesswrong_questions():
    files = [ensure(SOURCES_DIR / f"lw_{i}.parquet",
             "https://huggingface.co/datasets/x65617379/lesswrong_260509/"
             f"resolve/main/data/raw/train-0000{i}-of-00003.parquet") for i in range(3)]
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=["post"]):
            for r in b.to_pylist():
                p = r["post"]
                if not (isinstance(p, dict) and p.get("question")):
                    continue
                title = (p.get("title") or "").strip()
                text = (title + "\n\n" + strip_html(p.get("htmlBody"))).strip()
                if not text:
                    continue
                yield {"id": p.get("_id"),
                       "conversation": [{"role": "user", "content": text}],
                       "manifest": {"source": "lesswrong", "title": title,
                                    "url": p.get("pageUrl"), "base_score": p.get("baseScore"),
                                    "timestamp": str(p.get("postedAt")), "language": "English",
                                    "country": None, "src_model": "human"}}


def src_eaforum_questions():
    f = ensure(SOURCES_DIR / "eaforum.parquet",
               "https://huggingface.co/datasets/x65617379/eaforum_260506/resolve/main/raw.parquet")
    cols = ["post_id", "title", "htmlBody", "pageUrl", "baseScore", "postedAt"]
    for b in pq.ParquetFile(f).iter_batches(batch_size=300, columns=cols):
        for r in b.to_pylist():
            title = (r.get("title") or "").strip()
            if not title.endswith("?"):
                continue
            text = (title + "\n\n" + strip_html(r.get("htmlBody"))).strip()
            yield {"id": r.get("post_id"),
                   "conversation": [{"role": "user", "content": text}],
                   "manifest": {"source": "eaforum", "title": title, "url": r.get("pageUrl"),
                                "base_score": r.get("baseScore"), "timestamp": str(r.get("postedAt")),
                                "language": "English", "country": None, "src_model": "human"}}


def src_philosophy_se():
    f = ensure(SOURCES_DIR / "philosophy_se.parquet",
               "https://huggingface.co/datasets/mlfoundations-dev/stackexchange_philosophy/"
               "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")
    for b in pq.ParquetFile(f).iter_batches(batch_size=300,
                                            columns=["instruction", "completion", "conversations"]):
        for r in b.to_pylist():
            conv = []
            for t in (r.get("conversations") or []):
                role = "user" if t.get("from") in ("human", "user", "system") else "assistant"
                conv.append({"role": role, "content": t.get("value") or ""})
            if not conv and r.get("instruction"):
                conv = [{"role": "user", "content": r["instruction"]},
                        {"role": "assistant", "content": r.get("completion") or ""}]
            if not conv:
                continue
            q = r.get("instruction") or conv[0]["content"]
            yield {"id": hashlib.blake2b(q.encode("utf-8", "replace"), digest_size=12).hexdigest(),
                   "conversation": conv,
                   "manifest": {"source": "philosophy_se", "title": q[:100], "url": None,
                                "base_score": None, "timestamp": None, "language": "English",
                                "country": None, "src_model": "human"}}


def src_prism():
    f = ensure(SOURCES_DIR / "prism_conversations.parquet",
               "https://huggingface.co/datasets/HannahRoseKirk/prism-alignment/"
               "resolve/refs%2Fconvert%2Fparquet/conversations/train/0000.parquet")
    cols = ["conversation_id", "conversation_type", "conversation_history",
            "generated_datetime"]
    for b in pq.ParquetFile(f).iter_batches(batch_size=200, columns=cols):
        for r in b.to_pylist():
            # score the human side only (PRISM pairs multiple model candidates per turn)
            turns = [{"role": "user", "content": t.get("content") or ""}
                     for t in (r.get("conversation_history") or []) if t.get("role") == "user"]
            turns = [t for t in turns if t["content"].strip()]
            if not turns:
                continue
            yield {"id": r.get("conversation_id"),
                   "conversation": turns,
                   "manifest": {"source": "prism", "title": None, "url": None, "base_score": None,
                                "timestamp": str(r.get("generated_datetime")), "language": "English",
                                "country": None, "src_model": "human",
                                "conversation_type": r.get("conversation_type")}}


def src_sharegpt():
    f = ensure(SOURCES_DIR / "sharegpt_v3.json",
               "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
               "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json")
    data = json.load(open(f, encoding="utf-8"))
    for obj in data:
        conv = []
        for t in (obj.get("conversations") or []):
            role = "user" if t.get("from") in ("human", "user", "system") else "assistant"
            conv.append({"role": role, "content": t.get("value") or ""})
        conv = [t for t in conv if t["content"].strip()]
        if not conv:
            continue
        yield {"id": str(obj.get("id")),
               "conversation": conv,
               "manifest": {"source": "sharegpt", "title": None, "url": None, "base_score": None,
                            "timestamp": None, "language": None, "country": None,
                            "src_model": "unknown"}}


def src_lmsys():
    # LMSYS-Chat-1M is gated: needs HF_TOKEN (accept terms on the dataset page first).
    listing = "https://datasets-server.huggingface.co/parquet?dataset=lmsys%2Flmsys-chat-1m"
    req = urllib.request.Request(listing, headers=bc.hf_headers())
    files = [f for f in json.load(urllib.request.urlopen(req, timeout=120))["parquet_files"]
             if f["split"] == "train"]
    tmp = SOURCES_DIR / "_lmsys_tmp.parquet"
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    for i, fi in enumerate(files):
        print(f"  lmsys parquet {i + 1}/{len(files)} …", flush=True)
        download_to(fi["url"], tmp)
        for b in pq.ParquetFile(tmp).iter_batches(
                batch_size=200, columns=["conversation_id", "model", "conversation", "language"]):
            for r in b.to_pylist():
                if (r.get("language") or "").lower() != "english":
                    continue
                conv = [{"role": t.get("role"), "content": t.get("content") or ""}
                        for t in (r.get("conversation") or [])]
                conv = [t for t in conv if t["content"].strip()]
                if not conv:
                    continue
                yield {"id": r.get("conversation_id"),
                       "conversation": conv,
                       "manifest": {"source": "lmsys", "title": None, "url": None,
                                    "base_score": None, "timestamp": None, "language": "English",
                                    "country": None, "src_model": r.get("model")}}
    if tmp.exists():
        tmp.unlink()


def _hf_train_parquet_urls(repo):
    listing = ("https://datasets-server.huggingface.co/parquet?dataset="
               + urllib.parse.quote(repo, safe=""))
    req = urllib.request.Request(listing, headers=bc.hf_headers())
    data = json.load(urllib.request.urlopen(req, timeout=120))
    return [f["url"] for f in data["parquet_files"] if f["split"] == "train"]


def _stream_parquet(repo, columns, tmp_name):
    """Yield row dicts across all of a dataset's train parquet files (token-aware)."""
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SOURCES_DIR / tmp_name
    urls = _hf_train_parquet_urls(repo)
    for i, url in enumerate(urls):
        print(f"    {repo} parquet {i + 1}/{len(urls)} …", flush=True)
        download_to(url, tmp)
        for b in pq.ParquetFile(tmp).iter_batches(batch_size=300, columns=columns):
            yield from b.to_pylist()
    if tmp.exists():
        tmp.unlink()


def _blocks_text(content):
    """Flatten an Arena content field (string, or list of {type,text} blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(b.get("text") or "" for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return ""


_ARENA_ROLE_RE = re.compile(r"'role':\s*'(user|assistant)'")
_ARENA_ESC = {"n": "\n", "t": "\t", "r": "\r"}


def _read_quoted(s, i):
    q = s[i]
    i += 1
    out = []
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            out.append(_ARENA_ESC.get(s[i + 1], s[i + 1]))
            i += 2
        elif c == q:
            return "".join(out)
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _parse_arena_str(s):
    """Extract [(role, text)] from Arena's stringified-numpy conversation field
    (numpy-array repr: elements newline-separated, mixed quoting, so not literal-parseable)."""
    matches = list(_ARENA_ROLE_RE.finditer(s))
    starts = [m.start() for m in matches]
    turns = []
    for i, m in enumerate(matches):
        nxt = starts[i + 1] if i + 1 < len(starts) else len(s)
        ti = s.find("'text':", m.end(), nxt)
        if ti == -1:
            continue
        j = ti + 7
        while j < nxt and s[j] in " \t\n":
            j += 1
        if j < nxt and s[j] in ("'", '"'):
            turns.append((m.group(1), _read_quoted(s, j)))
    return turns


def _arena(repo, label):
    for r in _stream_parquet(repo, ["id", "conversation_a", "language"], "_arena_tmp.parquet"):
        if (r.get("language") or "").lower() not in ("en", "english"):
            continue
        ca = r.get("conversation_a")
        if isinstance(ca, str):  # arena-expert stores conversations as stringified numpy
            conv = [{"role": "user", "content": txt}
                    for role, txt in _parse_arena_str(ca) if role == "user" and txt.strip()]
        else:
            conv = [{"role": "user", "content": _blocks_text(t.get("content"))}
                    for t in (ca or []) if isinstance(t, dict) and t.get("role") == "user"]
            conv = [t for t in conv if t["content"].strip()]
        if not conv:
            continue
        yield {"id": r.get("id"), "conversation": conv,
               "manifest": {"source": label, "title": None, "url": None, "base_score": None,
                            "timestamp": None, "language": "English", "country": None,
                            "src_model": "human"}}


def src_arena_140k():
    yield from _arena("lmarena-ai/arena-human-preference-140k", "arena_140k")


def src_arena_expert():
    yield from _arena("lmarena-ai/arena-expert-5k", "arena_expert")


def src_oasst2():
    cols = ["message_id", "parent_id", "text", "role", "lang", "deleted"]
    for r in _stream_parquet("OpenAssistant/oasst2", cols, "_oasst_tmp.parquet"):
        # English root prompts only (the initial human ask of each tree)
        if r.get("role") != "prompter" or r.get("parent_id") is not None or r.get("deleted"):
            continue
        if (r.get("lang") or "") != "en":
            continue
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        yield {"id": r.get("message_id"), "conversation": [{"role": "user", "content": txt}],
               "manifest": {"source": "oasst2", "title": None, "url": None, "base_score": None,
                            "timestamp": None, "language": "English", "country": None,
                            "src_model": "human"}}


ADAPTERS = {"lesswrong_questions": src_lesswrong_questions,
            "eaforum_questions": src_eaforum_questions,
            "philosophy_se": src_philosophy_se,
            "prism": src_prism,
            "sharegpt": src_sharegpt,
            "lmsys": src_lmsys,
            "arena_140k": src_arena_140k,
            "arena_expert": src_arena_expert,
            "oasst2": src_oasst2}


def first_user(conv):
    for t in conv:
        if t["role"] == "user" and t.get("content"):
            return t["content"]
    return ""


def _stackexchange(site):
    """Generic adapter for any mlfoundations-dev/stackexchange_<site> (conversation-formatted)."""
    repo = f"mlfoundations-dev/stackexchange_{site}"
    for r in _stream_parquet(repo, ["conversations"], "_se_tmp.parquet"):
        conv = []
        for t in (r.get("conversations") or []):
            role = "user" if t.get("from") in ("human", "user", "system") else "assistant"
            conv.append({"role": role, "content": t.get("value") or ""})
        conv = [t for t in conv if t["content"].strip()]
        if not conv:
            continue
        q = conv[0]["content"]
        yield {"id": hashlib.blake2b(q.encode("utf-8", "replace"), digest_size=12).hexdigest(),
               "conversation": conv,
               "manifest": {"source": f"se_{site}", "title": q[:100], "url": None,
                            "base_score": None, "timestamp": None, "language": "English",
                            "country": None, "src_model": "human"}}


def cmd_prepare(args):
    if args.source == "stackexchange":
        if not getattr(args, "se_site", None):
            sys.exit("--se-site is required when --source stackexchange")
        gen = lambda: _stackexchange(args.se_site)
        source_name = f"se_{args.se_site}"
    else:
        gen = ADAPTERS[args.source]
        source_name = args.source
    run_dir = Path(args.run_dir or (REPO_ROOT / "runs" / source_name))
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(REPO_ROOT / "prompts"), undefined=StrictUndefined)
    template = env.get_template("conceptual_classifier.jinja")
    tmpl_sha = hashlib.sha256(
        (REPO_ROOT / "prompts" / "conceptual_classifier.jinja").read_bytes()).hexdigest()[:16]

    bc.write_json_atomic(run_dir / "config.json", {
        "dataset": source_name, "language": "n/a", "dedup_prefix": bc.DEDUP_PREFIX,
        "model": args.model, "max_tokens": bc.MAX_TOKENS, "schema_sha": bc.sha_json(bc.SCORE_SCHEMA),
        "template_sha": tmpl_sha, "shard_max_requests": bc.SHARD_MAX_REQUESTS,
        "shard_max_bytes": bc.SHARD_MAX_BYTES, "created_at": bc.now_utc_iso()})

    seen_ids, seen_txt, used = set(), set(), set()
    shards, survivors, total_chars = [], 0, 0
    cur, cur_bytes, cur_cnt, sub = [], 0, 0, 0

    def flush():
        nonlocal cur, cur_bytes, cur_cnt, sub
        if not cur_cnt:
            return
        path = shards_dir / f"shard_{sub:04d}.jsonl.gz"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as gz:
            gz.write("".join(cur))
        os.replace(tmp, path)
        shards.append({"shard_id": f"{sub:04d}", "path": str(path.relative_to(run_dir)),
                       "n_requests": cur_cnt, "n_bytes": cur_bytes})
        sub += 1
        cur, cur_bytes, cur_cnt = [], 0, 0

    for rec in gen():
        if rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        fu = first_user(rec["conversation"])
        tkey = " ".join(fu[:bc.DEDUP_PREFIX].lower().split())
        if not tkey or tkey in seen_txt:
            continue
        seen_txt.add(tkey)
        cid = str(rec["id"]) if (rec["id"] and CUSTOM_ID_RE.fullmatch(str(rec["id"]))
                                 and str(rec["id"]) not in used) else f"r{survivors:07d}"
        used.add(cid)
        prompt = template.render(conversation=rec["conversation"])
        params = {"model": args.model, "max_tokens": bc.MAX_TOKENS,
                  "output_config": {"format": {"type": "json_schema", "schema": bc.SCORE_SCHEMA}},
                  "messages": [{"role": "user", "content": prompt}]}
        man = dict(rec["manifest"])
        man.update({"conversation_hash": str(rec["id"]), "n_turns": len(rec["conversation"]),
                    "first_user_preview": " ".join(fu.split())[:90], "prompt_chars": len(prompt)})
        line = json.dumps({"custom_id": cid, "params": params, "manifest": man}) + "\n"
        api_bytes = len(json.dumps({"custom_id": cid, "params": params}))
        if cur_cnt and (cur_cnt + 1 > bc.SHARD_MAX_REQUESTS or cur_bytes + api_bytes > bc.SHARD_MAX_BYTES):
            flush()
        cur.append(line)
        cur_bytes += api_bytes
        cur_cnt += 1
        survivors += 1
        total_chars += len(prompt)
    flush()

    bc.write_json_atomic(run_dir / "prepare_state.json", {
        "files_done": [0], "shards": shards, "survivors": survivors, "english_seen": survivors,
        "total_prompt_chars": total_chars, "complete": True})
    print(f"\nprepared {survivors:,} records -> {len(shards)} shard(s) at {run_dir}")
    bc._print_cost_estimate({"survivors": survivors, "total_prompt_chars": total_chars,
                             "shards": shards})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", required=True, choices=list(ADAPTERS) + ["stackexchange"])
    p.add_argument("--se-site", default=None,
                   help="StackExchange site name when --source stackexchange (e.g. hermeneutics)")
    p.add_argument("--model", default=bc.MODEL_DEFAULT)
    p.set_defaults(func=cmd_prepare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
