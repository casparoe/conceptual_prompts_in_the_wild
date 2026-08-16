# Conceptual-reasoning prompt scores

A classifier and **precomputed scores** that identify *conceptual, in-the-weeds*
prompts across 22 public conversation and Q&A datasets — for building evaluation
or training sets that probe **conceptual reasoning** rather than knowledge recall.

Every prompt is scored **0–3 on three independent axes** by Claude Haiku 4.5:

| axis | question it answers |
|---|---|
| **conceptual** | Is progress driven by informal argument / a priori reasoning, with no straightforward empirical or mathematical verification? (prototype: philosophy) |
| **novelty** | How *in-the-weeds* / unprecedented is the ask? (0 = a stock question asked countless times; 3 = probably never publicly asked-and-answered) |
| **well_formed** | How clearly is a question actually being *asked*? |

The definition of "conceptual" follows *"Why and how to differentially accelerate
conceptual reasoning capabilities"* (Cooper, Oesterheld, Nguyen).

## What's here — and what isn't

This repo publishes **scores only, never the underlying conversation text.** Each
score row carries a join **key** + the three scores + coarse metadata (source
model, timestamp, language, country, turn count, URL where public). So you can
**rebuild filtered prompt sets** by re-fetching the public sources and joining on
the key — *without* re-running the classifier (which cost roughly $3,000 of
Batch-API calls to produce).

```
prompts/conceptual_classifier.jinja   the classifier prompt (three-axis rubric)
scores/*.parquet                      22 per-source score tables  (see scores/README.md)
scores/README.md                      score schema, per-source join recipe, yields
scripts/                              the pipeline (below)
```

Deliberately **not** included (gitignored): conversation/prompt text, the model's
reasoning traces, raw dataset dumps, and the private source document defining
"conceptual reasoning."

## Results

**16,385 strong hits** (conceptual ≥2 **and** novelty ≥2 **and** well_formed ≥2)
across 22 sources; **49,600 at conceptual ≥2**; 834 at novelty = 3.

Density tracks *source format*, not size:

| kind | examples | strong-hit rate |
|---|---|---|
| Question / Q&A corpora | Philosophy SE (29%), hermeneutics (18%), LessWrong/EA questions (22–25%) | **highest** |
| Difficulty-curated chat | Chatbot Arena expert (2.3%) | mid |
| Organic chat logs | WildChat (0.08%), ShareGPT (0.10%), LMSYS (0.05%) | lowest |

Full 22-source table and per-source yields are in [`scores/README.md`](scores/README.md).

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # only needed to (re)classify, not to read scores
```

**Rebuild a prompt set from the committed scores (no classifier calls):**

```bash
python scripts/rebuild_dataset.py --source philosophy_se \
    --min-conceptual 2 --min-novelty 2 --min-wellformed 2
# → re-fetches the public source, joins on key, writes prompts to runs/rebuilt/
```

`--source` accepts any of the 22 (e.g. `wildchat`, `lesswrong`, `se_hermeneutics`,
`arena_140k`, …). See `scores/README.md` for the list and per-source join keys.

**What rebuilding downloads.** 20 of the 22 sources are public and fetched
automatically (cached in `runs/_sources/`). The exceptions:

- `wildchat` — public, but the join streams the full **~15 GB** of WildChat
  parquet (one file at a time; only the file being scanned is kept on disk).
- `lmsys` — **gated**: accept the terms at
  [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), then
  either put a HF read token in `$HF_TOKEN` (or `~/.cache/huggingface/token`), or
  download the repo's `data/*.parquet` in a browser into `runs/_sources/lmsys/`.

Note on source availability: the LessWrong/EA Forum dumps live on a personal HF
account (`x65617379/*`, 2026-05 snapshots). If those repos disappear, equivalent
data can be re-exported from the forums' GraphQL API (the dumps' own origin), but
post `_id`s must match for the keys to join.

**Or just filter the scores directly:**

```python
import pyarrow.parquet as pq
t = pq.read_table("scores/philosophy_se.parquet").to_pandas()
hits = t[(t.status=="ok") & (t.conceptual>=2) & (t.novelty>=2) & (t.well_formed>=2)]
```

## Scripts

| script | purpose |
|---|---|
| `batch_classify.py` | Resumable Batch-API pipeline: `prepare` → `submit` → `poll` → `merge` (+ `status`, `retry`). Crash-safe on-disk state; orphan reconciliation so an interrupted submit never double-charges. Built for WildChat; the run dir format is shared by all sources. |
| `classify_sources.py` | Source adapters (forums, StackExchange, chat datasets) that render with the same prompt and emit `batch_classify`-compatible shards. `prepare --source <name>`. |
| `rebuild_dataset.py` | Reconstruct prompt sets from committed scores by re-fetching + joining. |
| `browse_hits.py` | Interactive terminal reviewer for a results file. |
| `find_conceptual.py` | Original single-conversation explorer (interactive; samples WildChat live). |

Re-scoring from scratch: `batch_classify.py prepare && … submit && … poll` for
WildChat; for other sources, `classify_sources.py prepare --source <name>` then
`batch_classify.py --run-dir runs/<name> submit|poll`.

## Data provenance & licensing

The score tables are **derived metadata — no source text.** When you re-fetch the
underlying data, respect each source's license; some are restrictive.

| score table(s) | source (HF) | terms |
|---|---|---|
| `english_full` | `allenai/WildChat-4.8M` | ODC-BY / AI2 ImpACT |
| `lmsys` | `lmsys/lmsys-chat-1m` | **gated**, LMSYS-Chat-1M license — see dataset page |
| `sharegpt` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | no stated license (shared-ChatGPT scrape) |
| `arena_140k`, `arena_expert` | `lmarena-ai/*` | see dataset pages |
| `prism` | `HannahRoseKirk/prism-alignment` | see dataset page |
| `oasst2` | `OpenAssistant/oasst2` | Apache-2.0 |
| `lesswrong_questions`, `eaforum_questions` | `x65617379/lesswrong_260509`, `…/eaforum_260506` | community content (LessWrong / EA Forum terms) |
| `philosophy_se`, `se_*` (13 sites) | `mlfoundations-dev/stackexchange_*` | StackExchange user content, CC-BY-SA 4.0 |

**Code and prompt** (`scripts/`, `prompts/`) are MIT-licensed (see `LICENSE`). The
**scores** are provided as-is as derived metadata; the underlying conversations
are governed by the sources above.

# Note on the use of LLMs

Everything you're seeing here (except for this section) is written by Opus/Fable. This includes the prompt for scoring the prompts in the source datasets. I have only given high-level instructions on this projects and done some limited validation on the classification prompt.