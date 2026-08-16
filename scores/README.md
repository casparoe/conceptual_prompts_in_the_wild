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
| `lmsys.parquet` | LMSYS-Chat-1M (English, deduped) | 421,058 | `conversation_id` in `lmsys/lmsys-chat-1m` (gated — needs token/local files) |
| `se_hermeneutics.parquet` | StackExchange: hermeneutics | 14,750 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_hermeneutics` |
| `se_christianity.parquet` | StackExchange: christianity | 16,698 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_christianity` |
| `se_law.parquet` | StackExchange: law | 22,830 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_law` |
| `se_politics.parquet` | StackExchange: politics | 17,153 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_politics` |
| `se_linguistics.parquet` | StackExchange: linguistics | 11,119 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_linguistics` |
| `se_hsm.parquet` | StackExchange: history of sci/math | 4,564 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_hsm` |
| `se_opensource.parquet` | StackExchange: open source | 4,758 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_opensource` |
| `se_parenting.parquet` | StackExchange: parenting | 6,871 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_parenting` |
| `se_mythology.parquet` | StackExchange: mythology | 2,038 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_mythology` |
| `se_vegetarianism.parquet` | StackExchange: vegetarianism | 760 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_vegetarianism` |
| `se_matheducators.parquet` | StackExchange: math educators | 3,763 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_matheducators` |
| `se_cseducators.parquet` | StackExchange: CS educators | 1,208 | `blake2b(first question)` in `mlfoundations-dev/stackexchange_cseducators` |

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

**16,385 strong across 22 sources** (49,600 at conceptual≥2; 834 at novelty=3). Top:
Philosophy SE 5,223 · SE-hermeneutics 2,594 · SE-christianity 1,967 · SE-law 1,789 ·
SE-politics 1,017 · SE-linguistics 904 · LessWrong 685 · WildChat 481 · arena_140k 302 ·
SE-hsm 238 · EA Forum 230 · ShareGPT 72 · arena_expert 66 · oasst2 13 · PRISM 10.

Density gaps worth knowing:
- **question/Q&A corpora** are richest: Philosophy SE 29%, hermeneutics 18%, christianity
  12%, LessWrong/EA questions 22–25%, linguistics/law/hsm/politics 5–8%;
- **general** chat logs are poorest: WildChat 0.08%, ShareGPT 0.10%, PRISM 0.13%,
  LMSYS 0.05% (LMSYS's 2023 Vicuna/Arena traffic is coding/roleplay-heavy);
- **Chatbot Arena** chat is in between — arena_140k 0.46%, difficulty-curated arena_expert
  2.32% (best *organic* sources). Density tracks the population/format, not raw size.

## StackExchange key recipe

The `mlfoundations-dev/stackexchange_*` datasets have no native id, so the key is
blake2b of the first question text:

```python
import hashlib
q = row.get("instruction") or row["conversations"][0]["value"]
key = hashlib.blake2b(q.encode("utf-8", "replace"), digest_size=12).hexdigest()
```

(`philosophy_se` used `instruction`; the other SE sites use `conversations[0]["value"]` —
the line above handles both. `scripts/rebuild_dataset.py --source se_<site>` does it for you.)

## Provenance

Scores produced by `scripts/batch_classify.py` (WildChat) and
`scripts/classify_sources.py` (forum/SE sources), both using
`prompts/conceptual_classifier.jinja` on `claude-haiku-4-5`. Dumps are 2026
snapshots; LessWrong/EA GraphQL APIs are behind bot-mitigation, so the HF dumps
are the practical source.
