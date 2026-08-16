#!/usr/bin/env python
"""Batch-classify the deduplicated English subset of WildChat-4.8M.

Scores every deduplicated English conversation on three 0-3 axes (conceptual,
novelty, well_formed) with prompts/conceptual_classifier.jinja on Claude Haiku
4.5, via the Anthropic **Batch API** (50% cheaper than the sync API). Designed to
survive the laptop being closed: every stage checkpoints to disk under a run
directory and is safe to re-run.

Pipeline (run in order):

    prepare   Download the 86 parquet files one at a time, keep English rows,
              drop first-message duplicates, render the classifier prompt for
              each survivor, and write batch-input shards (one per parquet file).
              No API calls, no cost. Resumable at file granularity. Prints the
              exact survivor count and an estimated cost when done.

    submit    Create one Batch API job per shard. Prints the cost and asks for
              confirmation (unless --yes). Strictly sequential with write-ahead
              state + orphan reconciliation, so a crash/close never double-charges
              a shard. Re-run to continue where it left off.

    poll      Check every submitted batch; as each finishes, download its results
              and store them. Idempotent; --watch loops until everything is done.
              Batches run on Anthropic's servers, so closing the laptop here loses
              nothing -- just re-run poll later.

    status    Print a dashboard of prepare/submit/poll progress.

    merge     Concatenate per-batch results into results/all_results.jsonl.

Typical use:

    .venv/bin/python scripts/batch_classify.py prepare
    .venv/bin/python scripts/batch_classify.py submit          # asks to confirm cost
    .venv/bin/python scripts/batch_classify.py poll --watch

Requires ANTHROPIC_API_KEY. HF_TOKEN is optional (only raises HF download rate
limits; the parquet files are public).
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = "allenai/WildChat-4.8M"
PARQUET_BASE = (
    "https://huggingface.co/datasets/allenai/WildChat-4.8M/"
    "resolve/refs%2Fconvert%2Fparquet/default/train"
)
NUM_FILES = 86  # 0000.parquet .. 0085.parquet

# Classification parameters (frozen into config.json on first prepare).
MODEL_DEFAULT = "claude-haiku-4-5"
MAX_TOKENS = 300
LANGUAGE = "english"          # top-level `language` field value, lowercased
DEDUP_PREFIX = 300            # first N chars of first user message, normalized

# Shard caps. A batch allows <=100k requests and <=256MB; we stay well under the
# byte limit (each request carries the full ~4k-token prompt) and align shards to
# parquet files, splitting a file only if it overflows one of these.
SHARD_MAX_REQUESTS = 10_000
SHARD_MAX_BYTES = 150 * 1024 * 1024

# Pricing ($/M tokens), Haiku 4.5, for the cost preview. Batch = 50% off.
PRICE_IN, PRICE_OUT = 1.0, 5.0
SUBMIT_MAX_ATTEMPTS = 5  # per-shard create attempts (with reconcile between)
BATCH_DISCOUNT = 0.5
# Rough estimate only; the real bill is computed from result usage after the fact.
CHARS_PER_TOKEN = 4.0
EST_OUTPUT_TOKENS = 116  # empirical mean output length with this prompt

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "conceptual": {"type": "integer", "enum": [0, 1, 2, 3]},
        "novelty": {"type": "integer", "enum": [0, 1, 2, 3]},
        "well_formed": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["reasoning", "conceptual", "novelty", "well_formed"],
    "additionalProperties": False,
}

CUSTOM_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


# ---------------------------------------------------------------- small helpers

def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default


def hf_headers() -> dict:
    headers = {"User-Agent": "conceptual-prompts-in-the-wild-batch/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:  # fall back to the huggingface-cli cached token file
        hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        try:
            with open(os.path.join(hf_home, "token")) as f:
                token = f.read().strip()
        except OSError:
            token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def first_user_content(conv) -> str:
    for turn in conv:
        if turn.get("role") == "user" and turn.get("content"):
            return turn["content"]
    return ""


def dedup_key(conv) -> str:
    return " ".join(first_user_content(conv)[:DEDUP_PREFIX].lower().split())


def key_digest(key: str) -> bytes:
    return hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=16).digest()


def preview(conv, width=90) -> str:
    oneline = " ".join(first_user_content(conv).split())
    return oneline[:width]


# --------------------------------------------------------------- parquet source

def download_parquet(file_idx: int, dest: Path, tries: int = 8) -> None:
    """Download one parquet file to dest, with patient retries. Verifies size."""
    url = f"{PARQUET_BASE}/{file_idx:04d}.parquet"
    part = dest.with_suffix(".part")
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=hf_headers())
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
            if total and got != total:
                raise IOError(f"short read: {got}/{total} bytes")
            os.replace(part, dest)
            return
        except (urllib.error.URLError, OSError) as e:
            if attempt == tries - 1:
                raise
            wait = min(2 ** attempt, 60)
            print(f"    (download {file_idx:04d} failed: {e}; retry in {wait}s)", flush=True)
            time.sleep(wait)


PARQUET_COLUMNS = ["conversation_hash", "model", "timestamp", "conversation", "turn",
                   "language", "country"]


def iter_rows(path: Path, batch_size: int = 256):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=PARQUET_COLUMNS):
        yield from batch.to_pylist()


# ---------------------------------------------------------------- render a request

def build_request(template, row, used_ids: set):
    """Return (custom_id, api_request_dict, manifest_dict) for one conversation."""
    conv = row["conversation"]
    prompt = template.render(conversation=conv)
    chash = row.get("conversation_hash") or ""
    cid = chash if (CUSTOM_ID_RE.fullmatch(chash) and chash not in used_ids) else None
    if cid is None:
        # synthetic fallback: unique within the shard
        base = f"row_{len(used_ids):06d}"
        cid = base
    used_ids.add(cid)
    ts = row.get("timestamp")
    api_request = {
        "custom_id": cid,
        "params": {
            "model": None,  # filled in by caller from config
            "max_tokens": MAX_TOKENS,
            "output_config": {"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
            "messages": [{"role": "user", "content": prompt}],
        },
    }
    manifest = {
        "conversation_hash": chash,
        "language": row.get("language"),
        "country": row.get("country"),
        "src_model": row.get("model"),
        "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else ts,
        "src_turn": row.get("turn"),
        "n_turns": len(conv),
        "first_user_preview": preview(conv),
        "prompt_chars": len(prompt),
    }
    return cid, api_request, manifest


# ------------------------------------------------------------------------ config

def load_or_create_config(run_dir: Path, model: str, force: bool) -> dict:
    template_bytes = (REPO_ROOT / "prompts" / "conceptual_classifier.jinja").read_bytes()
    cfg = {
        "dataset": DATASET,
        "language": LANGUAGE,
        "dedup_prefix": DEDUP_PREFIX,
        "model": model,
        "max_tokens": MAX_TOKENS,
        "schema_sha": sha_json(SCORE_SCHEMA),
        "template_sha": hashlib.sha256(template_bytes).hexdigest()[:16],
        "shard_max_requests": SHARD_MAX_REQUESTS,
        "shard_max_bytes": SHARD_MAX_BYTES,
    }
    path = run_dir / "config.json"
    existing = load_json(path)
    if existing:
        drift = {k: (existing.get(k), cfg[k]) for k in
                 ("template_sha", "schema_sha", "model", "dedup_prefix", "language")
                 if existing.get(k) != cfg[k]}
        if drift and not force:
            sys.exit(f"config.json drift vs current code/prompt: {drift}\n"
                     f"A run's prompt/schema must stay fixed. Use a fresh --run-dir, "
                     f"or --force to override (results will be inconsistent).")
        return existing
    cfg["created_at"] = now_utc_iso()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, cfg)
    return cfg


# ----------------------------------------------------------------------- prepare

def cmd_prepare(args):
    run_dir = Path(args.run_dir)
    cfg = load_or_create_config(run_dir, args.model, args.force)
    shards_dir = run_dir / "shards"
    seen_dir = run_dir / "seen"
    shards_dir.mkdir(parents=True, exist_ok=True)
    seen_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(REPO_ROOT / "prompts"), undefined=StrictUndefined)
    template = env.get_template("conceptual_classifier.jinja")

    pstate = load_json(run_dir / "prepare_state.json", {
        "files_done": [], "shards": [], "survivors": 0, "english_seen": 0,
        "total_prompt_chars": 0, "complete": False,
    })
    files_done = set(pstate["files_done"])

    # Rebuild the global dedup set from committed files only.
    seen = set()
    for idx in files_done:
        p = seen_dir / f"{idx:03d}.bin"
        if p.exists():
            blob = p.read_bytes()
            for i in range(0, len(blob), 16):
                seen.add(blob[i:i + 16])

    n_files = min(args.limit_files, NUM_FILES) if args.limit_files else NUM_FILES
    limit_requests = getattr(args, "limit_requests", 0)
    tmp_parquet = run_dir / "_current.parquet"

    print(f"prepare: run_dir={run_dir}  files_done={len(files_done)}/{n_files}  "
          f"survivors so far={pstate['survivors']:,}", flush=True)

    for file_idx in range(n_files):
        if file_idx in files_done:
            continue
        t0 = time.time()
        print(f"  file {file_idx:04d}: downloading...", flush=True)
        download_parquet(file_idx, tmp_parquet)

        # Process the whole file into memory, then commit transactionally.
        used_ids: set = set()
        cur_lines: list = []
        cur_bytes = 0
        cur_count = 0
        file_shards: list = []          # (path, n_requests, n_bytes) written this file
        new_digests = bytearray()
        eng_seen = 0
        survivors = 0
        prompt_chars = 0
        sub = 0

        def flush_shard():
            nonlocal cur_lines, cur_bytes, cur_count, sub
            if not cur_count:
                return
            name = f"shard_{file_idx:04d}_{sub:02d}.jsonl.gz"
            path = shards_dir / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as gz:
                gz.write("".join(cur_lines))
            os.replace(tmp, path)
            file_shards.append({"shard_id": f"{file_idx:04d}_{sub:02d}", "path": str(path.relative_to(run_dir)),
                                "n_requests": cur_count, "n_bytes": cur_bytes})
            sub += 1
            cur_lines, cur_bytes, cur_count = [], 0, 0

        for row in iter_rows(tmp_parquet):
            if (row.get("language") or "").lower() != cfg["language"]:
                continue
            eng_seen += 1
            dig = key_digest(dedup_key(row["conversation"]))
            if dig in seen:
                continue
            seen.add(dig)
            new_digests += dig
            survivors += 1
            cid, api_request, manifest = build_request(template, row, used_ids)
            api_request["params"]["model"] = cfg["model"]
            prompt_chars += manifest["prompt_chars"]
            line = json.dumps({"custom_id": cid, "params": api_request["params"],
                               "manifest": manifest}) + "\n"
            # byte cost that counts against the 256MB API limit (params only)
            api_bytes = len(json.dumps({"custom_id": cid, "params": api_request["params"]}))
            if cur_count and (cur_count + 1 > SHARD_MAX_REQUESTS or
                              cur_bytes + api_bytes > SHARD_MAX_BYTES):
                flush_shard()
            cur_lines.append(line)
            cur_bytes += api_bytes
            cur_count += 1
            if limit_requests and pstate["survivors"] + survivors >= limit_requests:
                break  # testing cap
        flush_shard()

        # Commit this file: seen digests, then state (atomic each).
        (seen_dir / f"{file_idx:03d}.bin").write_bytes(bytes(new_digests))
        pstate["files_done"].append(file_idx)
        pstate["shards"].extend(file_shards)
        pstate["survivors"] += survivors
        pstate["english_seen"] += eng_seen
        pstate["total_prompt_chars"] += prompt_chars
        write_json_atomic(run_dir / "prepare_state.json", pstate)
        dt = time.time() - t0
        print(f"  file {file_idx:04d}: english={eng_seen:,} survivors={survivors:,} "
              f"shards={len(file_shards)} ({dt:.0f}s)  cumulative survivors={pstate['survivors']:,}",
              flush=True)
        if limit_requests and pstate["survivors"] >= limit_requests:
            print(f"  (stopping: --limit-requests {limit_requests} reached)")
            break

    if tmp_parquet.exists():
        tmp_parquet.unlink()

    if len(pstate["files_done"]) >= n_files:
        pstate["complete"] = True
        write_json_atomic(run_dir / "prepare_state.json", pstate)

    _print_cost_estimate(pstate)
    print("\nprepare complete." if pstate["complete"] else
          "\nprepare paused (not all files done); re-run to continue.")


def _print_cost_estimate(pstate) -> None:
    survivors = pstate["survivors"]
    est_in = pstate["total_prompt_chars"] / CHARS_PER_TOKEN
    est_out = survivors * EST_OUTPUT_TOKENS
    batch_cost = (est_in * PRICE_IN + est_out * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
    sync_cost = (est_in * PRICE_IN + est_out * PRICE_OUT) / 1e6
    n_shards = len(pstate["shards"])
    print(f"\n  survivors (deduped English): {survivors:,}")
    if survivors:
        print(f"  mean prompt tokens (est):    {est_in / survivors:,.0f}")
    print(f"  shards / batches:            {n_shards}")
    print(f"  est. input tokens:           {est_in/1e6:,.1f}M")
    print(f"  est. output tokens:          {est_out/1e6:,.1f}M")
    print(f"  ESTIMATED BATCH COST:        ~${batch_cost:,.0f}  (sync would be ~${sync_cost:,.0f})")
    print("  (estimate only; actual billed from result usage)")


# ------------------------------------------------------------------------ submit

def _iter_shard_requests(run_dir: Path, shard_path: str):
    """Yield {custom_id, params} for the API (manifest stripped)."""
    with gzip.open(run_dir / shard_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            rec = json.loads(line)
            yield {"custom_id": rec["custom_id"], "params": rec["params"]}


def _reconcile_orphan(client, expected_count: int, known_ids: set, run_started: datetime):
    """After a create() whose outcome is unknown, look for the batch it may have
    created. Return its id if exactly one plausible orphan exists, None if none,
    or raise if ambiguous (favor stopping over double-charging)."""
    unknown = []
    for b in client.messages.batches.list(limit=100):
        if b.id in known_ids:
            continue
        created = b.created_at if isinstance(b.created_at, datetime) else None
        if created is not None and created < run_started:
            continue
        rc = b.request_counts
        total = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
        unknown.append((b.id, total))
    match = [bid for bid, total in unknown if total == expected_count]
    if len(match) == 1:
        return match[0]
    if not unknown:
        return None
    raise RuntimeError(
        f"orphan reconciliation is ambiguous: {len(unknown)} recent unknown batch(es) "
        f"{unknown}, {len(match)} match expected count {expected_count}. "
        f"Resolve manually (inspect the account's batches) before resubmitting.")


def cmd_submit(args):
    run_dir = Path(args.run_dir)
    cfg = load_json(run_dir / "config.json")
    if not cfg:
        sys.exit("no config.json; run `prepare` first.")
    pstate = load_json(run_dir / "prepare_state.json")
    if not pstate or not pstate.get("complete"):
        if not args.allow_incomplete:
            sys.exit("prepare is not complete; run `prepare` to finish, or pass "
                     "--allow-incomplete to submit only the shards prepared so far.")

    shards = pstate["shards"]
    sstate = load_json(run_dir / "submit_state.json", {
        "run_started_at": now_utc_iso(), "shards": {},
    })
    run_started = datetime.fromisoformat(sstate["run_started_at"])
    client = anthropic.Anthropic()
    # Disable SDK auto-retry on create: a silently-retried POST could double-create.
    create_client = client.with_options(max_retries=0)

    todo = [s for s in shards if sstate["shards"].get(s["shard_id"], {}).get("status") != "submitted"]
    done = len(shards) - len(todo)
    total_reqs = sum(s["n_requests"] for s in todo)
    _print_cost_estimate(pstate)
    print(f"\n  already submitted: {done}/{len(shards)} shards")
    print(f"  to submit now:     {len(todo)} shards, {total_reqs:,} requests")
    if not todo:
        print("nothing to submit."); return

    limit = getattr(args, "limit_shards", 0)
    if limit:
        print(f"  (--limit-shards {limit}: will submit at most {limit} new shard(s) this run)")
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("refusing to submit non-interactively without --yes.")
        ans = input(f"\nCreate {min(limit, len(todo)) if limit else len(todo)} batch job(s) now? "
                    f"This will incur cost. Type 'yes': ").strip()
        if ans.lower() != "yes":
            print("aborted."); return

    def save():
        write_json_atomic(run_dir / "submit_state.json", sstate)

    submitted_now = 0
    failed = 0
    for s in shards:
        if limit and submitted_now >= limit:
            print(f"  reached --limit-shards {limit}; {len(todo) - submitted_now} shard(s) left. "
                  f"Re-run submit to continue."); break
        sid = s["shard_id"]
        st = sstate["shards"].get(sid, {})
        known_ids = {v["batch_id"] for v in sstate["shards"].values() if v.get("batch_id")}

        if st.get("status") == "submitted":
            continue
        if st.get("status") == "submitting":
            # A prior run may have created this batch. Reconcile before retrying.
            adopted = _reconcile_orphan(create_client, s["n_requests"], known_ids, run_started)
            if adopted:
                sstate["shards"][sid] = {"status": "submitted", "batch_id": adopted,
                                         "n_requests": s["n_requests"], "created_at": now_utc_iso(),
                                         "adopted": True}
                save()
                print(f"  {sid}: adopted orphan batch {adopted}")
                continue

        sstate["shards"][sid] = {"status": "submitting", "n_requests": s["n_requests"]}
        save()
        requests = list(_iter_shard_requests(run_dir, s["path"]))
        # Retry with reconcile between attempts. max_retries=0 on the client means
        # the SDK never silently re-POSTs; if a create's outcome is unknown we look
        # for the batch it may have made (reconcile) before ever trying again, so a
        # transient blip can't double-charge.
        batch_id = None
        last_error = None
        for attempt in range(SUBMIT_MAX_ATTEMPTS):
            try:
                batch = create_client.messages.batches.create(requests=requests)
                batch_id = batch.id
                break
            except Exception as e:  # noqa: BLE001 - reconcile then decide
                last_error = e
                time.sleep(3)  # let batches.list() reflect a possibly-created batch
                adopted = _reconcile_orphan(create_client, s["n_requests"], known_ids, run_started)
                if adopted:
                    batch_id = adopted
                    break
                if attempt < SUBMIT_MAX_ATTEMPTS - 1:
                    wait = min(5 * 2 ** attempt, 120)
                    print(f"  {sid}: create failed ({e}); retry {attempt + 2}/"
                          f"{SUBMIT_MAX_ATTEMPTS} in {wait}s", flush=True)
                    time.sleep(wait)
        if batch_id is None:
            sstate["shards"][sid] = {"status": "pending", "n_requests": s["n_requests"],
                                     "last_error": str(last_error)}
            save()
            failed += 1
            print(f"  {sid}: FAILED after {SUBMIT_MAX_ATTEMPTS} attempts ({last_error}); "
                  f"left pending, continuing.", flush=True)
            continue
        sstate["shards"][sid] = {"status": "submitted", "batch_id": batch_id,
                                 "n_requests": s["n_requests"], "created_at": now_utc_iso()}
        save()
        submitted_now += 1
        print(f"  {sid}: submitted batch {batch_id} ({s['n_requests']:,} requests)", flush=True)

    if failed:
        print(f"\n{failed} shard(s) failed to submit and were left pending; re-run `submit` to retry them.")
    remaining = sum(1 for s in shards
                    if sstate["shards"].get(s["shard_id"], {}).get("status") != "submitted")
    if remaining:
        print(f"\n{submitted_now} submitted this run; {remaining} shard(s) still pending. "
              f"Re-run `submit` to continue.")
    else:
        print("\nall shards submitted. Run `poll --watch` to collect results.")


# -------------------------------------------------------------------------- poll

def _parse_result_entry(entry):
    """-> (status, conceptual, novelty, well_formed, reasoning, in_tok, out_tok)."""
    res = entry.result
    rtype = getattr(res, "type", None)
    if rtype != "succeeded":
        return (rtype or "unknown", None, None, None, "", 0, 0)
    msg = res.message
    usage = getattr(msg, "usage", None)
    itok = getattr(usage, "input_tokens", 0) or 0
    otok = getattr(usage, "output_tokens", 0) or 0
    if getattr(msg, "stop_reason", None) == "refusal":
        return ("refusal", None, None, None, "", itok, otok)
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
    try:
        p = json.loads(text)
        return ("ok", int(p["conceptual"]), int(p["novelty"]), int(p["well_formed"]),
                p.get("reasoning", ""), itok, otok)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return (f"parse_error: {e}", None, None, None, text[:200], itok, otok)


def _collect_batch(client, run_dir: Path, sid: str, batch_id: str, shard_path: str) -> dict:
    """Download one ended batch's results, join to manifest, write atomically."""
    manifest = {}
    with gzip.open(run_dir / shard_path, "rt", encoding="utf-8") as gz:
        for line in gz:
            rec = json.loads(line)
            manifest[rec["custom_id"]] = rec["manifest"]
    out_dir = run_dir / "results" / "by_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    counts = Counter()
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in client.messages.batches.results(batch_id):
            cid = entry.custom_id
            status, c, n, w, reasoning, itok, otok = _parse_result_entry(entry)
            counts[status if status in ("ok", "refusal") else "error"] += 1
            m = manifest.get(cid, {})
            # spread the full manifest so source-specific metadata (title, url,
            # base_score, source, …) survives into the results, then set the
            # standard result fields.
            row = {**m,
                   "custom_id": cid, "conversation_hash": m.get("conversation_hash"),
                   "status": status, "conceptual": c, "novelty": n, "well_formed": w,
                   "reasoning": reasoning, "input_tokens": itok, "output_tokens": otok,
                   "shard_id": sid, "batch_id": batch_id}
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, out_path)
    return dict(counts)


def cmd_poll(args):
    run_dir = Path(args.run_dir)
    pstate = load_json(run_dir / "prepare_state.json")
    sstate = load_json(run_dir / "submit_state.json")
    if not sstate:
        sys.exit("nothing submitted yet; run `submit` first.")
    shard_path = {s["shard_id"]: s["path"] for s in pstate["shards"]}
    client = anthropic.Anthropic()
    rstate = load_json(run_dir / "results_state.json", {"collected": {}})

    while True:
        submitted = {sid: v for sid, v in sstate["shards"].items() if v.get("status") == "submitted"}
        pending = {sid: v for sid, v in submitted.items() if sid not in rstate["collected"]}
        ended_now = 0
        still = 0
        for sid, v in pending.items():
            bid = v["batch_id"]
            try:
                b = client.messages.batches.retrieve(bid)
            except Exception as e:  # noqa: BLE001
                print(f"  {sid}: retrieve failed ({e}); will retry next cycle."); still += 1; continue
            if b.processing_status != "ended":
                still += 1
                continue
            try:
                counts = _collect_batch(client, run_dir, sid, bid, shard_path[sid])
            except Exception as e:  # noqa: BLE001
                print(f"  {sid}: collect failed ({e}); will retry next cycle."); still += 1; continue
            rstate["collected"][sid] = {"batch_id": bid, "counts": counts, "at": now_utc_iso()}
            write_json_atomic(run_dir / "results_state.json", rstate)
            ended_now += 1
            print(f"  {sid}: collected {sum(counts.values()):,} results {counts}", flush=True)

        total = len(submitted)
        coll = len(rstate["collected"])
        agg = Counter()
        for v in rstate["collected"].values():
            agg.update(v["counts"])
        print(f"[{now_utc_iso()}] collected {coll}/{total} batches; "
              f"remaining processing: {still}; results so far: {dict(agg)}", flush=True)

        if coll >= total:
            print("\nall submitted batches collected.")
            _merge(run_dir)
            return
        if not args.watch:
            return
        time.sleep(args.interval)


# ----------------------------------------------------------------- status / merge

def _merge(run_dir: Path) -> Path:
    by_batch = run_dir / "results" / "by_batch"
    out = run_dir / "results" / "all_results.jsonl"
    n = 0
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fout:
        for p in sorted(by_batch.glob("*.jsonl")):
            with open(p, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line); n += 1
    os.replace(tmp, out)
    print(f"merged {n:,} results -> {out}")
    return out


def cmd_merge(args):
    _merge(Path(args.run_dir))


def cmd_retry(args):
    """Re-classify the handful of non-ok results (usually max_tokens truncation)
    via the sync API with a higher token cap, patch their by_batch files, re-merge."""
    run_dir = Path(args.run_dir)
    pstate = load_json(run_dir / "prepare_state.json")
    shard_path = {s["shard_id"]: s["path"] for s in pstate["shards"]}
    client = anthropic.Anthropic()
    by_batch = run_dir / "results" / "by_batch"
    fixed = still = 0
    for f in sorted(by_batch.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        if all(r["status"] == "ok" for r in rows):
            continue
        sid = f.stem
        params_by_cid = {}
        with gzip.open(run_dir / shard_path[sid], "rt", encoding="utf-8") as gz:
            for line in gz:
                rec = json.loads(line)
                params_by_cid[rec["custom_id"]] = rec["params"]
        changed = False
        for r in rows:
            if r["status"] == "ok":
                continue
            params = dict(params_by_cid[r["custom_id"]])
            params["max_tokens"] = max(params.get("max_tokens", 300), args.max_tokens)
            try:
                resp = client.messages.create(**params)
                text = "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", None) == "text")
                pj = json.loads(text)
                r.update(status="ok", conceptual=int(pj["conceptual"]), novelty=int(pj["novelty"]),
                         well_formed=int(pj["well_formed"]), reasoning=pj.get("reasoning", ""),
                         input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)
                fixed += 1
                changed = True
            except Exception as e:  # noqa: BLE001
                print(f"  {sid}/{r['custom_id']}: still failing ({e})")
                still += 1
        if changed:
            tmp = f.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as out:
                for r in rows:
                    out.write(json.dumps(r) + "\n")
            os.replace(tmp, f)
    print(f"retry: fixed {fixed}, still-failing {still}")
    _merge(run_dir)


def cmd_status(args):
    run_dir = Path(args.run_dir)
    cfg = load_json(run_dir / "config.json")
    pstate = load_json(run_dir / "prepare_state.json")
    sstate = load_json(run_dir / "submit_state.json")
    rstate = load_json(run_dir / "results_state.json", {"collected": {}})
    print(f"run_dir: {run_dir}")
    if not cfg:
        print("  (not initialized; run `prepare`)"); return
    print(f"  model={cfg['model']} language={cfg['language']} template_sha={cfg['template_sha']}")
    if pstate:
        print(f"  prepare: files_done={len(pstate['files_done'])}/{NUM_FILES} "
              f"survivors={pstate['survivors']:,} shards={len(pstate['shards'])} "
              f"complete={pstate.get('complete')}")
        _print_cost_estimate(pstate)
    if sstate:
        byst = Counter(v.get("status") for v in sstate["shards"].values())
        print(f"  submit: {dict(byst)}")
    if rstate["collected"]:
        agg = Counter()
        for v in rstate["collected"].values():
            agg.update(v["counts"])
        print(f"  results: collected {len(rstate['collected'])} batches, "
              f"{sum(agg.values()):,} rows {dict(agg)}")


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(REPO_ROOT / "runs" / "english_full"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download+filter+dedup+render into shards")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--limit-files", type=int, default=0, help="process only the first N parquet files (testing)")
    p.add_argument("--limit-requests", type=int, default=0, help="stop after N survivors (testing)")
    p.add_argument("--force", action="store_true", help="override config drift check")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("submit", help="create batch jobs from prepared shards")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    p.add_argument("--limit-shards", type=int, default=0, help="submit at most N new shards this run (partial/test)")
    p.add_argument("--allow-incomplete", action="store_true", help="submit shards even if prepare isn't done")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("poll", help="collect results from submitted batches")
    p.add_argument("--watch", action="store_true", help="loop until all batches are collected")
    p.add_argument("--interval", type=int, default=120, help="seconds between poll cycles (--watch)")
    p.set_defaults(func=cmd_poll)

    p = sub.add_parser("status", help="print progress dashboard")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("merge", help="concatenate per-batch results into all_results.jsonl")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("retry", help="re-classify non-ok results via the sync API, then re-merge")
    p.add_argument("--max-tokens", type=int, default=600, help="token cap for retries (default 600)")
    p.set_defaults(func=cmd_retry)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
