#!/usr/bin/env python
"""Scan WildChat-4.8M for conversations the classifier prompt rates as conceptual.

Streams rows from the HuggingFace datasets-server (no local download), classifies
each conversation with prompts/conceptual_classifier.jinja on Claude Haiku 4.5
(three 0-3 scores: "conceptual", "novelty"/in-the-weeds, and "well_formed"), and
pretty-prints each conversation clearing the thresholds (conceptual >= --min-score,
and optionally novelty >= --min-novelty and well-formed >= --min-wellformed).
After each find (or after an empty round of --max-rows classifications), it pauses:
press Enter to keep searching, q+Enter to quit. When stdin is not a terminal, it
exits after the first find instead of prompting.

By default, rows are sampled from random pages spread across the whole dataset
(WildChat contains long runs of near-identical bot traffic, so sequential scanning
from one offset is a poor sample), and rows whose first user message duplicates an
already-classified one are skipped for free. Pass --offset for a sequential scan
from a fixed row (e.g. to re-display a known row).

Usage:
    .venv/bin/python scripts/find_conceptual.py                    # random sampling
    .venv/bin/python scripts/find_conceptual.py --seed 42          # reproducible sampling
    .venv/bin/python scripts/find_conceptual.py --offset 123 --max-rows 1   # re-display row 123
    .venv/bin/python scripts/find_conceptual.py --language all --min-score 3
    .venv/bin/python scripts/find_conceptual.py --language English,German,French

Requires ANTHROPIC_API_KEY (or another credential source the SDK resolves).
Optional: set HF_TOKEN (HuggingFace access token) for higher datasets-server rate
limits; anonymous access can hit HTTP 429 during longer sessions.
"""

import argparse
import itertools
import json
import os
import random
import shutil
import sys
import textwrap
import time
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = "allenai/WildChat-4.8M"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100  # max allowed by the /rows endpoint
FETCH_HARD_CAP = 50_000  # safety bound on rows fetched in sequential mode
FETCH_PAUSE = 0.5  # seconds between page fetches, so filter-skip bursts don't hammer the API

# Haiku 4.5 pricing, $/M tokens, for the running cost note
PRICE_IN, PRICE_OUT = 1.0, 5.0

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


@dataclass
class Verdict:
    conceptual: int | None  # None = classification failed (refusal / parse error / API error)
    novelty: int | None
    well_formed: int | None
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0


def _hf_headers() -> dict:
    headers = {"User-Agent": "conceptual-prompts-in-the-wild-scanner/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get_json(url: str, tries: int = 6) -> dict:
    """GET with patient retries: honors Retry-After on 429, backs off on 5xx/network errors."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=_hf_headers())
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if attempt == tries - 1 or e.code not in (429, 500, 502, 503, 504):
                raise
            if e.code == 429:
                try:
                    wait = float(e.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    wait = 5 * 2**attempt
                wait = min(max(wait, 5.0), 180.0)
                print(f"  (datasets-server rate limit hit; waiting {wait:.0f}s before retrying...)",
                      flush=True)
            else:
                wait = 2**attempt
            time.sleep(wait)
        except urllib.error.URLError as e:
            if attempt == tries - 1:
                raise
            wait = 2**attempt
            print(f"  (network error: {e.reason}; retrying in {wait:.0f}s...)", flush=True)
            time.sleep(wait)
        except OSError as e:
            # Raw socket errors — TimeoutError on a slow read, ConnectionResetError,
            # etc. — are OSError subclasses but NOT urllib.error.URLError, so they
            # would otherwise escape the retry loop and crash the whole scan.
            if attempt == tries - 1:
                raise
            wait = 2**attempt
            print(f"  (network {type(e).__name__}; retrying in {wait:.0f}s...)", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_page(offset: int, length: int) -> list[tuple[int, dict]]:
    url = (
        f"{DATASETS_SERVER}/rows?dataset={urllib.parse.quote(DATASET, safe='')}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )
    payload = http_get_json(url)
    return [(item["row_idx"], item["row"]) for item in payload.get("rows", [])]


def dataset_num_rows() -> int:
    """Row count of the served train split, taken from the /rows endpoint itself.

    Note: despite the dataset name, the hosted split has ~3.2M rows, not 4.8M —
    offsets past num_rows_total return empty pages, so this must be exact.
    """
    payload = http_get_json(
        f"{DATASETS_SERVER}/rows?dataset={urllib.parse.quote(DATASET, safe='')}"
        f"&config=default&split=train&offset=0&length=1"
    )
    return payload["num_rows_total"]


def classify(client: anthropic.Anthropic, template, model: str, row: dict) -> Verdict:
    prompt = template.render(conversation=row["conversation"])
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        return Verdict(None, None, None, f"API error: {e}")
    usage = resp.usage
    if resp.stop_reason == "refusal":
        return Verdict(None, None, None, "refusal", usage.input_tokens, usage.output_tokens)
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        parsed = json.loads(text)
        return Verdict(
            int(parsed["conceptual"]), int(parsed["novelty"]), int(parsed["well_formed"]),
            parsed["reasoning"], usage.input_tokens, usage.output_tokens,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return Verdict(None, None, None, f"parse error: {e}: {text[:200]}", usage.input_tokens, usage.output_tokens)


def first_user_content(row: dict) -> str:
    for turn in row["conversation"]:
        if turn["role"] == "user" and turn.get("content"):
            return turn["content"]
    return ""


def first_user_preview(row: dict, width: int = 70) -> str:
    oneline = " ".join(first_user_content(row).split())
    if not oneline:
        return "(no user content)"
    return oneline[:width] + ("…" if len(oneline) > width else "")


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ---------- pretty printing ----------

USE_COLOR = sys.stdout.isatty()


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def wrap_block(text: str, width: int, indent: str = "  ") -> str:
    out = []
    for line in text.split("\n"):
        if line.strip():
            out.append(
                textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
            )
        else:
            out.append("")
    return "\n".join(out)


def print_hit(row_idx: int, row: dict, verdict: Verdict, display_cap: int | None):
    width = min(shutil.get_terminal_size((100, 20)).columns, 100)
    bar = "=" * width
    thin = "-" * width
    conv = row["conversation"]
    n_user = sum(1 for t in conv if t["role"] == "user")

    print()
    print(c("1", bar))
    print(c("1;32", f"CONCEPTUAL CONVERSATION FOUND — conceptual {verdict.conceptual}, "
                    f"novelty {verdict.novelty}, well-formed {verdict.well_formed}"))
    print(c("1", bar))
    print(f"Row index:  {row_idx}")
    print(f"Hash:       {row.get('conversation_hash')}")
    print(f"Model:      {row.get('model')}    Timestamp: {row.get('timestamp')}")
    print(f"Language:   {row.get('language')}    Country: {row.get('country')}")
    print(f"Turns:      {len(conv)} ({n_user} user)")
    print()
    print(c("1", "Classifier reasoning:"))
    print(wrap_block(verdict.reasoning, width))
    print()
    for i, turn in enumerate(conv, 1):
        role = turn["role"].upper()
        color = "1;36" if turn["role"] == "user" else "1;35"
        print(c("2", thin))
        print(c(color, f"[{i}] {role}"))
        print(c("2", thin))
        content = turn.get("content") or ""
        if display_cap and len(content) > display_cap:
            content = content[:display_cap] + f"\n[... {len(content) - display_cap} chars truncated for display; use --full]"
        print(wrap_block(content, width))
    print(c("1", bar))
    print(f"Re-display exactly this conversation: --offset {row_idx} --max-rows 1")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offset", type=int, default=None,
                    help="scan sequentially from this row instead of random sampling "
                         "(use with --max-rows 1 to re-display a specific row)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for random-page sampling (default: random; printed for reproducibility)")
    ap.add_argument("--per-page", type=int, default=20,
                    help="in sampling mode, max rows classified per random page of 100 (default 20; "
                         "lower = better spread per dollar)")
    ap.add_argument("--min-score", type=int, default=2,
                    help="show conversations with conceptual score >= this (default 2)")
    ap.add_argument("--min-novelty", type=int, default=0,
                    help="additionally require novelty score >= this (default 0 = no constraint)")
    ap.add_argument("--min-wellformed", type=int, default=0,
                    help="additionally require well-formedness score >= this (default 0 = no constraint)")
    ap.add_argument("--max-rows", type=int, default=400,
                    help="max conversations to classify per search round, i.e. between pauses (default 400)")
    ap.add_argument("--language", default="English,German",
                    help="comma-separated list of top-level languages to classify (default: English,German); "
                         "'all' disables the filter. Skipped rows cost no API calls.")
    ap.add_argument("--concurrency", type=int, default=8, help="parallel classification requests (default 8)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--full", action="store_true", help="print full message contents (default: truncate very long messages for display)")
    args = ap.parse_args()

    env = Environment(loader=FileSystemLoader(REPO_ROOT / "prompts"), undefined=StrictUndefined)
    template = env.get_template("conceptual_classifier.jinja")
    client = anthropic.Anthropic()

    langs = {l.strip().lower() for l in args.language.split(",") if l.strip()}
    if "all" in langs:
        langs = None

    num_rows = dataset_num_rows()
    if args.offset is not None and not 0 <= args.offset < num_rows:
        sys.exit(f"--offset {args.offset} is out of range: the served train split has {num_rows:,} rows.")

    print(f"Dataset: {DATASET} ({num_rows:,} rows in served train split)")
    if args.offset is not None:
        print(f"Mode: sequential scan from offset {args.offset}")
    else:
        seed = args.seed if args.seed is not None else random.randrange(2**32)
        print(f"Mode: random-page sampling, seed {seed} (reproduce with --seed {seed})")
        rng = random.Random(seed)
    if langs:
        print(f"Language filter: {', '.join(sorted(l.capitalize() for l in langs))}")
    threshold = f"conceptual >= {args.min_score}" \
        + (f" and novelty >= {args.min_novelty}" if args.min_novelty > 0 else "") \
        + (f" and well-formed >= {args.min_wellformed}" if args.min_wellformed > 0 else "")
    print(f"Model: {args.model} | showing {threshold} | round size: {args.max_rows} classified rows\n")

    def raw_candidates():
        if args.offset is not None:
            fetched = 0
            offset = args.offset
            while offset < num_rows and fetched < FETCH_HARD_CAP:
                page = fetch_page(offset, min(PAGE_SIZE, num_rows - offset))
                if not page:
                    return
                fetched += len(page)
                offset += len(page)
                time.sleep(FETCH_PAUSE)
                yield from page
        else:
            while True:
                page_offset = rng.randrange(0, max(1, num_rows - PAGE_SIZE))
                page = fetch_page(page_offset, PAGE_SIZE)
                rng.shuffle(page)
                time.sleep(FETCH_PAUSE)
                yield from page[: args.per_page]

    stats = {"dup": 0, "lang_skipped": 0}
    seen_keys: set = set()
    seen_idx: set = set()

    def candidates():
        for row_idx, row in raw_candidates():
            if row_idx in seen_idx:
                continue
            seen_idx.add(row_idx)
            if langs and (row.get("language") or "").lower() not in langs:
                stats["lang_skipped"] += 1
                continue
            key = " ".join(first_user_content(row)[:300].lower().split())
            if key in seen_keys:
                stats["dup"] += 1
                continue
            seen_keys.add(key)
            yield row_idx, row

    dist_c: Counter = Counter()
    dist_n: Counter = Counter()
    dist_w: Counter = Counter()
    totals = {"scanned": 0, "failures": 0, "in": 0, "out": 0, "hits": 0}
    stream = candidates()

    def fmt_dist(d):
        return ", ".join(f"{s}: {d[s]}" for s in sorted(d)) or "n/a"

    def print_stats():
        cost = totals["in"] / 1e6 * PRICE_IN + totals["out"] / 1e6 * PRICE_OUT
        print(f"\nSession: {totals['scanned']} classified ({totals['failures']} failures; "
              f"{stats['dup']} duplicates skipped"
              + (f", {stats['lang_skipped']} other-language rows skipped" if langs else "") + ")")
        print(f"Conceptual: {fmt_dist(dist_c)} | Novelty: {fmt_dist(dist_n)} | Well-formed: {fmt_dist(dist_w)}")
        print(f"Tokens {totals['in']:,} in / {totals['out']:,} out (~${cost:.3f})")

    def scan_round(ex):
        """Classify up to --max-rows candidates. Returns (hits_in_round, stream_exhausted)."""
        n = 0
        for chunk in chunked(itertools.islice(stream, args.max_rows), args.concurrency):
            verdicts = list(ex.map(lambda item: classify(client, template, args.model, item[1]), chunk))
            round_hits = []
            for (row_idx, row), v in zip(chunk, verdicts):
                n += 1
                totals["scanned"] += 1
                totals["in"] += v.input_tokens
                totals["out"] += v.output_tokens
                if v.conceptual is None:
                    is_hit = False
                    totals["failures"] += 1
                    cell = c("33", f"{'ERR':<12}")
                else:
                    dist_c[v.conceptual] += 1
                    dist_n[v.novelty] += 1
                    dist_w[v.well_formed] += 1
                    is_hit = (v.conceptual >= args.min_score
                              and v.novelty >= args.min_novelty
                              and v.well_formed >= args.min_wellformed)
                    cell = f"{f'c={v.conceptual} n={v.novelty} w={v.well_formed}':<12}"
                    if is_hit:
                        cell = c("1;32", cell)
                lang = (row.get("language") or "?")[:10]
                print(f"  row {row_idx:>9}  {cell} {lang:<11} {first_user_preview(row)}")
                if is_hit:
                    round_hits.append((row_idx, row, v))
            if round_hits:
                return round_hits, False
        return [], n < args.max_rows

    interactive = sys.stdin.isatty()

    def wants_more(had_hit: bool) -> bool:
        if not interactive:
            return False
        what = "search for the next one" if had_hit else f"scan another {args.max_rows} rows"
        try:
            ans = input(c("1", f"\n[Enter] {what}   [q+Enter] quit > ")).strip().lower()
        except EOFError:
            return False
        return ans not in ("q", "quit", "n", "no")

    pending: deque = deque()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            while True:
                exhausted = False
                if pending:
                    hits = [pending.popleft()]
                else:
                    hits, exhausted = scan_round(ex)
                    if hits:
                        pending.extend(hits[1:])  # hits from the same chunk, shown on later rounds
                        hits = hits[:1]
                if hits:
                    row_idx, row, verdict = hits[0]
                    totals["hits"] += 1
                    print_hit(row_idx, row, verdict, display_cap=None if args.full else 5000)
                elif exhausted:
                    print("\nReached the end of the scannable rows for this mode.")
                    print_stats()
                    break
                else:
                    print(f"\nNo conversation scored >= {args.min_score} in this round.")
                print_stats()
                if not wants_more(had_hit=bool(hits)):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
        print_stats()
    except urllib.error.URLError as e:
        print(f"\nGiving up: datasets-server requests kept failing even with retries ({e}).")
        print("The HF datasets-server rate-limits anonymous clients. Wait a few minutes and rerun,")
        print("or set HF_TOKEN (a HuggingFace access token) in your environment for higher limits.")
        print_stats()
        sys.exit(2)

    if totals["hits"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
