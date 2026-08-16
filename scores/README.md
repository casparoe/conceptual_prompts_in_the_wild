# Precomputed conceptual-reasoning scores

Claude Haiku 4.5 scores for conversations/questions from several public sources,
on three independent 0–3 axes:

- **conceptual** — is progress driven by informal argument / a priori reasoning (no
  straightforward empirical or mathematical verification)? (see the classifier prompt)
- **novelty** — how "in-the-weeds" / unprecedented is the ask (vs. well-trodden)?
- **well_formed** — how clearly is a question actually being *asked*?

These are committed so you can **rebuild filtered prompt datasets without re-running
the model**. We store only scores + a join key + metadata — **no conversation/prompt
text**, and **no model reasoning** (that's kept locally only). To get the actual
prompts, re-fetch the public source and join on `key` (see
`../scripts/rebuild_dataset.py`).

## Files

| file | source | rows | `key` joins to |
|---|---|---:|---|
| `english_full.parquet` | WildChat (English, deduped) | 605,842 | `conversation_hash` in `allenai/WildChat-4.8M` |
| `lesswrong_questions.parquet` | LessWrong "Question" posts | 2,689 | post `_id` in `x65617379/lesswrong_260509` (config `raw`) |
| `eaforum_questions.parquet` | EA Forum `?`-title posts | 1,060 | `post_id` in `x65617379/eaforum_260506` |
| `philosophy_se.parquet` | Philosophy StackExchange | 17,998 | `blake2b(instruction)` in `mlfoundations-dev/stackexchange_philosophy` (see recipe below) |
| `sharegpt.parquet` | ShareGPT (shared ChatGPT convos) | 68,582 | `id` in `anon8231489123/ShareGPT_Vicuna_unfiltered` (`ShareGPT_V3_unfiltered_cleaned_split.json`) |
| `prism.parquet` | PRISM (values-laden assistant prompts) | 7,764 | `conversation_id` in `HannahRoseKirk/prism-alignment` (config `conversations`) |
| `arena_140k.parquet` | Chatbot Arena (organic) | 66,198 | `id` in `lmarena-ai/arena-human-preference-140k` |
| `arena_expert.parquet` | Chatbot Arena (expert-curated) | 2,846 | `id` in `lmarena-ai/arena-expert-5k` |
| `oasst2.parquet` | OpenAssistant (English root prompts) | 5,076 | `message_id` in `OpenAssistant/oasst2` |

## Columns

- `source`, `key` — corpus and its join key
- `conceptual`, `novelty`, `well_formed` — 0–3 (filter on these)
- `status` — `"ok"`, or an error string; **filter to `status == "ok"`**
- `src_model`, `timestamp`, `country`, `language`, `n_turns`, `base_score`, `url` — metadata

## Rebuild a prompt dataset

```bash
# e.g. all "strong" LessWrong questions (conceptual/novelty/well_formed all >= 2)
.venv/bin/python scripts/rebuild_dataset.py --source lesswrong \
    --min-conceptual 2 --min-novelty 2 --min-wellformed 2
```

This filters the scores, re-fetches the source, joins on `key`, and writes the
selected prompts to `runs/rebuilt/`. WildChat re-downloads ~15 GB of parquet (its
text isn't stored here); the forum/SE dumps are small (<600 MB total).

Or filter the scores directly, e.g. in Python:

```python
import pyarrow.parquet as pq
t = pq.read_table("scores/philosophy_se.parquet").to_pandas()
hits = t[(t.status=="ok") & (t.conceptual>=2) & (t.novelty>=2) & (t.well_formed>=2)]
```

## Yields (strong = conceptual≥2, novelty≥2, well_formed≥2)

Philosophy SE 5,223 · LessWrong 685 · WildChat 481 · arena_140k 302 · EA Forum 230 ·
ShareGPT 72 · arena_expert 66 · oasst2 13 · PRISM 10 = **7,082 total** (19,566 at
conceptual≥2; 449 at novelty=3). Density gaps worth knowing:
- forum/Q&A **question** corpora (Philosophy SE, LessWrong/EA questions): 22–29% strong;
- **general** chat logs (WildChat, ShareGPT, PRISM): ~0.1% strong;
- **Chatbot Arena** chat is in between — `arena_140k` 0.46% (AI-savvy population) and the
  difficulty-curated `arena_expert` 2.32% — the best *organic* sources found.

## Philosophy SE key recipe

The source dataset has no native id, so the key is derived from the question text:

```python
import hashlib
q = row["instruction"] or row["conversations"][0]["value"]
key = hashlib.blake2b(q.encode("utf-8", "replace"), digest_size=12).hexdigest()
```

## Provenance

Scores produced by `scripts/batch_classify.py` (WildChat) and
`scripts/classify_sources.py` (forum/SE sources), both using
`prompts/conceptual_classifier.jinja` on `claude-haiku-4-5`. Dumps are 2026
snapshots; LessWrong/EA GraphQL APIs are behind bot-mitigation, so the HF dumps
are the practical source.
