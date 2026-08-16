# Pipeline run log — by course

This is the **canonical** pipeline run log, grouped by course first and model
second (see project convention). It started as a reorganized, derived view of
`testing_cmds.txt`, but as of 2026-07-11 this is the file to update going
forward — log new runs here directly, including `(kept)` stats at the time
of the run.

`testing_cmds.txt` is kept around for fast, no-formatting notes jotted while
a run is in progress (see the note at its top) — fold anything worth keeping
from it into this file afterward. It is no longer the append point for
structured run records.

## How to read "(kept)"

Since `bcb76fc`/filter step, the pipeline logs a line like:

```
[filter] concepts kept=190/2225 (min_unique_chunks=3); mentions kept=1006/3364
```

`kept` is the count surviving the `min_unique_chunks` filter before
clustering/pairing. Recent entries in `testing_cmds.txt` (me320, me270, me310,
tam251) already record this as `mentions (kept N)` / `concepts (kept N)`. For
older entries that predate that convention, values below were **backfilled
from surviving `out/**/run.log` files** — never guessed. Where no log
survived, the cell is left blank.

**Important caveat found while backfilling:** `out/<course>/run.log` gets
overwritten every time a run reuses the same `--out-dir`, so most of the
early Qwen3B/14B/32B runs against the shared `out/sql`, `out/me200`,
`out/me400` paths have no surviving log — except where an archived
per-model subdirectory (e.g. `out/me200/Qwen14B_ffp/`) happened to preserve
a copy. That's the only reason backfill was possible for some legacy rows
and not others.

**Discrepancies found (flagged inline, not silently corrected — `testing_cmds.txt` is untouched):**
- In the legacy (pre-"kept") section, the plain `mentions` number is
  inconsistently either the **raw** mentions count or the **already-kept**
  count depending on the row (e.g. sql/Qwen14B's "6 mentions" = kept 6/45,
  but sql/Qwen32B's "59 mentions" = raw, kept was actually 6/59). Treat the
  backfilled kept column as authoritative where the two conflict.
- `me320`: recorded raw mentions = 2729, but the surviving log shows raw
  mentions = 1719 (kept 832/1719 does match the recorded "kept 832").
- `me400_ffp`: recorded clusters = 171, but the surviving log shows
  clusters = 177.
- `me340`: run was **still in progress** at the time of writing (started
  2026-07-10 23:43 UTC, PID 1572, had only just finished loading the model
  weights as of the last check). Stats were backfilled from `run.log` after
  completion — see its section.

## Course configs at a glance

The PDF filename tag drives OCR in `ingest.py`: `_scan` → `force_full_page_ocr=True`
(native text discarded, full-page OCR), `_text` or no tag → native text extraction.
One combined-notes PDF per course. All single-config runs used
Qwen/Qwen2.5-14B-Instruct end to end (pipeline and judge).

| Course | Input PDF | OCR | `VLLM_MAX_MODEL_LEN` |
|---|---|---|---|
| tam210 | `TAM210_CombinedNotes_text.pdf` | text | 16384 |
| tam212 | `TAM212_CombinedNotes_text.pdf` | text | 16384 |
| tam251 | `TAM251_CombinedNotes_text.pdf` | text | 16384 |
| me270 | `ME270_CombinedNotes_text.pdf` | text | 8192 (default) |
| me310 | `ME310&TAM335_CombinedNotes_scan.pdf` | ffp | 8192 (default) |
| me320 | `ME320_CombinedNotes_scan.pdf` | ffp | 16384 |
| me340 | `ME340_CombinedNotes_text.pdf` | text | 8192 (default) |
| cs401_403 | `cs401_403_Combined_text.pdf` | text | 16384 |

sql/me200/me400 ran multiple model × OCR variants — see their sections. me310's
PDF combines ME 310 and TAM 335 material; it is treated as me310 throughout.

---

## sql

### Qwen2.5-3B-Instruct (A100)
`LLM_PROVIDER=vllm python main.py --data-dir data/sql --steps ingest llm clustering pairpackets --out-dir out/sql --llm-model Qwen/Qwen2.5-3B-Instruct`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 13 | 4 / — | 57 / — | 2 | 0 | — (relations step not run) |

*No surviving `run.log` for this run — not backfillable.*

### Qwen2.5-14B-Instruct (H200)
`# [JR] switched to nohup + log so the run survives Jupyter connection drops`
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/sql --out-dir out/sql --llm-model Qwen/Qwen2.5-14B-Instruct > out/sql/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 13 | 45 / 6 | 35 / 2 | 3 | 1 | 1 |

### Qwen2.5-32B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/sql --out-dir out/sql --llm-model Qwen/Qwen2.5-32B-Instruct > out/sql/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 13 | 59 / 6 | 47 / 2 | 3 | 1 | 1 |

### Other sql commands logged (no stats recorded)
- `LLM_PROVIDER=hf HF_BATCH_SIZE=8 LLM_CHUNK_BATCH=4 python main.py --data-dir data/sql --steps ingest llm clustering pairpackets --out-dir out/sql --llm-model meta-llama/Llama-3.2-3B-Instruct`
- `python relation_judger.py --in out/sql/pairpackets.jsonl --out out/sql/relations_fixed.jsonl --model "meta-llama/Llama-3.2-3B-Instruct" --batch-size 32 --concurrency 12`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output`
- `python main.py --steps clustering pairpackets relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output`
- `python main.py --steps relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output_llama8b/`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output_llama3b/`

---

## me200

### Qwen2.5-3B-Instruct (A100)

`LLM_PROVIDER=vllm python main.py --data-dir data/me200 --steps ingest llm clustering pairpackets --out-dir out/me200 --llm-model Qwen/Qwen2.5-3B-Instruct`

| Variant | Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|---|
| Ch1 (handwritten scans) | 2 | 2 / — | 2 / — | — | — | ERROR: chunks must have more than one neighbor (I think) |
| CombinedNotes (handwritten scans, weak concept extraction) | 24 | 29 / — | 67 / — | 6 | 11 | — |
| CombinedNotes, same data + relations step (`--out-dir out/me200`, no `--steps` filter) | 24 | 29 / — | 67 / — | 6 | 11 | 11 |

`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me200 --out-dir out/me200 --llm-model Qwen/Qwen2.5-3B-Instruct > out/me200/run.log 2>&1 &`

| Variant | Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|---|
| CombinedNotes, force_full_page_ocr | 442 (labeled `chunks_ffp` in source) | 2413 / 1139 | 1301 / 176† | 116 | 3109 | 3109 |

† log's filter line shows denominator 1300, not 1301 (concept_cards count logged one write ahead of the filter check — not fixed here, reported as-is).

*Ch1 and the two default-OCR CombinedNotes runs above have no surviving `run.log` — not backfillable.*

### Qwen2.5-14B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me200 --out-dir out/me200 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me200/run.log 2>&1 &`
(add `VLLM_MAX_MODEL_LEN=16384` for the enriched variant)

| Variant | Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|---|
| default | 24 | 53 / 7 | 44 / 2 | 2 | 1 | 1 |
| force_full_page_ocr (ffp) | 442 | 1760 / 882 | 890 / 130 | 121 | 1711 | 1711 |
| ffp + enriched (`VLLM_MAX_MODEL_LEN=16384`) | 443 | 1833 / 1004 | 867 / 142 | 120 | 2195 | 2195 |

#### Evaluation — Qwen2.5-14B-Instruct as judge, 2×2 over variant × course string
`python evaluation/eval.py --input_file out/me200/Qwen14B_ffp/relations.jsonl --course_name me200 --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b > out/me200/Qwen14B_ffp/eval_me200_qwen14b_ffp_baretitle.log 2>&1`

Swap `--course_name me200_catalog` for the catalog condition and
`Qwen14B_ffp_enriched` for the enriched variant; one log per cell, named
`eval_me200_qwen14b_<variant>_<baretitle|catalog>.log` in the variant's dir.

| Variant | Course string | Nodes | Triplets | node_significance | triplet_accuracy |
|---|---|---|---|---|---|
| ffp | bare title | 129 | 1711 | 0.9845 ± 0.087 | 0.6137 ± 0.228 |
| ffp | catalog | 129 | 1711 | 0.9612 ± 0.134 | 0.6146 ± 0.220 |
| ffp + enriched | bare title | 140 | 2195 | 0.9857 ± 0.083 | 0.5827 ± 0.203 |
| ffp + enriched | catalog | 140 | 2195 | 0.9821 ± 0.093 | 0.5841 ± 0.203 |

~6 min per cell on the A100; no JSON parse failures. Relation mix: ffp 905
`depends_on` / 128 `part_of` / 678 `None`, enriched 1029 / 93 / 1073 — null
share 39.6% → 48.9%. Single-file mode ignores `--output_json`; the logs are the
only record. Analysis in findings.md 2026-08-14.

### Qwen2.5-32B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me200 --out-dir out/me200 --llm-model Qwen/Qwen2.5-32B-Instruct > out/me200/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 442 | 2328 / 1225 | 1120 / 182 | 121 | 3468 | 3468 |

Recorded simply as "me200" in the original log, but the chunk count (442)
and the archived dir name (`out/me200/Qwen32B_ffp/`) confirm this was the ffp
config, not default OCR.

---

## me400

### Qwen2.5-3B-Instruct (A100)
`LLM_PROVIDER=vllm python main.py --data-dir data/me400 --out-dir out/me400 --llm-model Qwen/Qwen2.5-3B-Instruct`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 725 | 2300 / — | 1423 / — | 180 | 4777 | 4777 |

*No surviving `run.log` — not backfillable.*

### Qwen2.5-14B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me400 --out-dir out/me400 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me400/run.log 2>&1 &`
(add `VLLM_MAX_MODEL_LEN=16384` for the enriched variant)

| Variant | Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|---|
| default | 725 | 2793 / 1799 | 1118 / 255 | 174 | 3158 | 3158 |
| force_full_page_ocr (ffp) | 749 | 2848 / 1839 | 1131 / 258 | 171 (recorded) / **177 (log — mismatch)** | 3168 | 3168 |
| enriched (`VLLM_MAX_MODEL_LEN=16384`) | 727 | 2991 / 1989 | 1136 / 274 | 178 | 3472 | 3472 |

A superseded "buggy_run" attempt for the enriched config also survives on
disk (2997/1138/172/3123) — not used here since the final run in the parent
dir replaced it.

The enriched outputs sit one level deeper than the other two variants:
`out/me400/Qwen14B_enriched/8191_chunks/`. A sibling `6000_chunks/` (120
relations) is a superseded chunk-budget attempt.

#### Evaluation — Qwen2.5-14B-Instruct as judge, 2×3 over variant × course string
`python evaluation/eval.py --input_file out/me400/Qwen14B/relations.jsonl --course_name me400 --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b > out/me400/Qwen14B/eval_me400_qwen14b_default_baretitle.log 2>&1`

Swap `--course_name me400_catalog` for the catalog condition; variants are
`Qwen14B` (default), `Qwen14B_ffp`, and `Qwen14B_enriched/8191_chunks`.

| Variant | Course string | Nodes | Triplets | node_significance | triplet_accuracy |
|---|---|---|---|---|---|
| default | bare title | 249 | 3158 | 0.9940 ± 0.055 | 0.6811 ± 0.257 |
| default | catalog | 249 | 3158 | 0.9920 ± 0.063 | 0.6884 ± 0.254 |
| ffp | bare title | 251 | 3168 | 0.9920 ± 0.063 | 0.6761 ± 0.255 |
| ffp | catalog | 251 | 3168 | 0.9920 ± 0.063 | 0.6853 ± 0.248 |
| enriched | bare title | 270 | 3472 | 0.9870 ± 0.079 | 0.6753 ± 0.261 |
| enriched | catalog | 270 | 3472 | 0.9833 ± 0.090 | 0.6822 ± 0.259 |

`6000_chunks` and `buggy_run` excluded as superseded. No JSON parse failures.
Null share 40.8% (default) → 39.0% (ffp) → 32.9% (enriched). Analysis in
findings.md 2026-08-14.

### Qwen2.5-32B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me400 --out-dir out/me400 --llm-model Qwen/Qwen2.5-32B-Instruct > out/me400/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 725 | 3600 / 2494 | 1254 / 320 | 174 | 5108 | 5108 |

---

## me320 — Qwen2.5-14B-Instruct (H200), ffp + enriched
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me320 --out-dir out/me320 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me320/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 424 | 2729 (recorded) / **1719 (log — mismatch)**, kept 832 | 889 / 118 | 122 | 1295 | 1295 |

Kept counts (832 mentions, 118 concepts) match the log exactly; only the raw
mentions total disagrees with what's recorded in `testing_cmds.txt`.

Eval 2026-08-15 (Qwen2.5-14B-Instruct as judge, A100-80GB, ~9 min),
`out/me320/Qwen14B_ffp_enriched/eval_me320_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 1295 | 629 | 48 | 618 | 117 | 0.9530 ± 0.146 | 0.6189 ± 0.244 |

## me270 — Qwen2.5-14B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me270 --out-dir out/me270 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me270/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 716 | 3364 / 1006 | 2225 / 190 | 206 | 819 | 819 |

Fully verified against `run.log` — no discrepancies.

Eval 2026-08-15 (Qwen2.5-14B-Instruct as judge, A100-80GB, ~7 min),
`out/me270/eval_me270_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 819 | 145 | 106 | 568 | 147 | 0.9354 ± 0.168 | 0.6630 ± 0.266 |

Highest null share on record (69.4%) and highest `part_of` share of real edges
(42.2%) — both traced to the checklist-structured corpus, findings.md 2026-08-15.

## me310 — Qwen2.5-14B-Instruct (H200), ffp
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me310 --out-dir out/me310 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me310/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 235 | 1017 / 501 | 550 / 94 | 64 | 928 | 928 |

Fully verified against `run.log` — no discrepancies.

Eval 2026-08-15 (Qwen2.5-14B-Instruct as judge, A100-80GB, ~8 min),
`out/me310/eval_me310_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 928 | 506 | 56 | 366 | 92 | 0.9402 ± 0.162 | 0.6207 ± 0.254 |

## me340 — Qwen2.5-14B-Instruct (H200)
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/me340 --out-dir out/me340 --llm-model Qwen/Qwen2.5-14B-Instruct > out/me340/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 306 | 1294 / 707 | 603 / 90 | 79 | 656 | 656 |

Eval 2026-08-15 (Qwen2.5-14B-Instruct as judge, A100-80GB, ~7 min),
`out/me340/eval_me340_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 656 | 458 | 27 | 171 | 89 | 0.9888 ± 0.074 | 0.7995 ± 0.248 |

Highest triplet accuracy in the set by 0.11, and lowest null share (26.1%).
Largely compositional — 94.4% of real edges are `depends_on` in a linearly
ordered domain, so the score approaches pure direction accuracy; see
findings.md 2026-08-15 before citing it against mixed-relation courses.

## tam251 — Qwen2.5-14B-Instruct (H200), enriched
`nohup env LLM_PROVIDER=vllm python main.py --data-dir data/tam251 --out-dir out/tam251 --llm-model Qwen/Qwen2.5-14B-Instruct > out/tam251/run.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 73 | 383 / 94 | 289 / 21 | 21 | 95 | 95 |

Fully verified against `run.log` (`out/tam251/Qwen14B_enriched/run.log`) — no
discrepancies.

Eval 2026-08-15 (Qwen2.5-14B-Instruct as judge, A100-80GB, ~5 min),
`out/tam251/Qwen14B_enriched/eval_tam251_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 95 | 59 | 7 | 29 | 17 | 0.9118 ± 0.191 | 0.6737 ± 0.238 |

## tam210 — Qwen2.5-14B-Instruct (A100-80GB), enriched
Run 2026-08-15 via the combined pipeline → graph → eval loop below
(`for C in tam210 tam212`), `VLLM_MAX_MODEL_LEN=16384`; pipeline ~22 min.

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 87 | 448 / 77 | 350 / 19 | 22 | 68 | 68 |

Thinnest corpus in the set, and the only course where clusters (22) exceed
kept concepts (19).

Eval (~5 min), `out/tam210/eval_tam210_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 68 | 46 | 4 | 18 | 19 | 1.0000 ± 0.000 | 0.6250 ± 0.289 |

The perfect node-significance score is a ceiling artifact at n=19, not a
result — see findings.md 2026-08-15.

## tam212 — Qwen2.5-14B-Instruct (A100-80GB), enriched
Run 2026-08-15 via the same loop as tam210; pipeline ~43 min.

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 237 | 759 / 321 | 444 / 72 | 62 | 331 | 331 |

Eval (~6 min), `out/tam212/eval_tam212_qwen14b.log`:

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 331 | 181 | 47 | 103 | 68 | 0.9706 ± 0.118 | 0.6571 ± 0.245 |

14.2% `part_of` of all records — the highest share measured under the 14B judge.

---

## cs401_403 — Qwen2.5-14B-Instruct (A100-80GB), combined 379-page text PDF
`nohup bash -c 'set -e; LLM_PROVIDER=vllm VLLM_MAX_MODEL_LEN=16384 PYTHONUNBUFFERED=1 python main.py --data-dir data/cs401_403 --out-dir out/cs401_403 --llm-model Qwen/Qwen2.5-14B-Instruct; test -s out/cs401_403/relations.jsonl; PYTHONUNBUFFERED=1 python evaluation/eval.py --input_file out/cs401_403/relations.jsonl --course_name cs401_403 --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b' > out/cs401_403/overnight.log 2>&1 &`

| Chunks | Mentions (raw/kept) | Concepts (raw/kept) | Clusters | Pairs | Relations |
|---|---|---|---|---|---|
| 279 | 950 / 323 | 605 / 71 | 74 | 180 | 180 |

Pipeline 05:06:26 → 05:53:15 (~47 min), eval → 05:59:45 (~6.5 min). Relations
break down as 91 `depends_on`, 2 `part_of`, 87 `None` — 93 real edges over 61
nodes. `VLLM_MAX_MODEL_LEN=16384` was set for the whole run (not just relations)
as overflow insurance on an unattended run; no overflow occurred.

Eval, same judge model: node_significance 0.951 ± 0.149 (61 nodes),
triplet_accuracy 0.683 ± 0.278 (180 triplets) — within noise of cs401's 0.943 /
0.679 at a third the corpus size. See findings.md 2026-08-14.

Graph rendered via `knowledge_graph_visualization.ipynb` (93 edges, 61 nodes, 0
isolated) → `out/cs401_403/kg_visualization.html`.

---

## cs401 — evaluation run (Qwen2.5-14B-Instruct as judge, A100-80GB)
`nohup python evaluation/eval.py --input_file out/cs401/relations.jsonl --course_name cs401 --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b > out/cs401/eval_cs401_qwen14b.log 2>&1 &`

| Records | depends_on | part_of | None | Nodes scored | Node significance | Triplet accuracy |
|---|---|---|---|---|---|---|
| 126 | 54 | 2 | 70 | 35 | 0.943 ± 0.159 | 0.679 ± 0.263 |

Scores are raw 0–2 normalized by ÷2. Node significance covers only the 35
concepts appearing in non-null edges (`eval.py:103` skips `relation: null`);
triplet accuracy covers all 126 records, including the 70 nulls, which the
rubric scores as a valid "None" judgment. Judging took ~38s for 161 prompts;
model load dominated (~5 min).

Needs the `--course_name cs401` map entry and the `--max_model_len` fix — see
findings.md 2026-08-14. Single-file mode prints results to stdout only, so the
log is the sole record.

---

## Graph rendering (any course)

`knowledge_graph_visualization.ipynb` reads `KG_RELATIONS` and writes
`kg_visualization.html` beside that file.

`KG_RELATIONS=out/<course>/relations.jsonl jupyter nbconvert --to notebook --execute --stdout knowledge_graph_visualization.ipynb > /dev/null`

~15s, no GPU. `--stdout` keeps the tracked notebook free of outputs. All
courses already run:

`for f in out/*/relations.jsonl; do KG_RELATIONS=$f jupyter nbconvert --to notebook --execute --stdout knowledge_graph_visualization.ipynb > /dev/null && echo "ok $f"; done`

### Combined pipeline → graph → eval

```bash
for C in tam210 tam212; do
  mkdir -p out/$C
  nohup bash -c "set -e
LLM_PROVIDER=vllm VLLM_MAX_MODEL_LEN=16384 PYTHONUNBUFFERED=1 python main.py \
  --data-dir data/$C --out-dir out/$C --llm-model Qwen/Qwen2.5-14B-Instruct
test -s out/$C/relations.jsonl
KG_RELATIONS=out/$C/relations.jsonl jupyter nbconvert --to notebook --execute \
  --stdout knowledge_graph_visualization.ipynb > /dev/null
PYTHONUNBUFFERED=1 python evaluation/eval.py --input_file out/$C/relations.jsonl \
  --course_name $C --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b \
  --output_json out/$C/eval_${C}_qwen14b.json \
  > out/$C/eval_${C}_qwen14b.log 2>&1" > out/$C/run.log 2>&1
done &
```

Courses run sequentially, one GPU at a time. Pipeline and graph output to
`out/<course>/run.log`, eval to `out/<course>/eval_<course>_<model>.log`.
Requires a `course_map` entry in `evaluation/eval.py` for `$C`. As of
2026-08-15 single-file mode writes `--output_json` too; earlier runs' results
live only in their log tails.

---

## anlp / algo (predecessor's commands — no stats recorded)

These predate the stats-tracking convention entirely; only commands were
logged, so there's nothing to backfill.

- `LLM_PROVIDER=hf python main.py --steps ingest llm clustering pairpackets --out-dir out --llm-model meta-llama/Llama-3.2-3B-Instruct`
- `LLM_PROVIDER=hf HF_BATCH_SIZE=8 LLM_CHUNK_BATCH=4 python main.py --data-dir data/algo --steps ingest llm --out-dir out/algo --llm-model meta-llama/Llama-3.2-3B-Instruct`
- `LLM_PROVIDER=hf HF_BATCH_SIZE=8 LLM_CHUNK_BATCH=4 python main.py --data-dir data/anlp --steps ingest llm clustering pairpackets --out-dir out/anlp --llm-model meta-llama/Llama-3.2-3B-Instruct`
- `LLM_PROVIDER=hf HF_BATCH_SIZE=8 LLM_CHUNK_BATCH=4 python relation_judger.py --data-dir data/anlp --out-dir out/anlp --llm-model meta-llama/Llama-3.2-3B-Instruct`
- `python relation_judger.py --in out/anlp/pairpackets.jsonl --out out/anlp/relations_fixed.jsonl --model "meta-llama/Llama-3.2-3B-Instruct" --batch-size 8`
- `python relation_judger.py --in out/algo/pairpackets.jsonl --out out/algo/relations_fixed.jsonl --model "meta-llama/Llama-3.2-3B-Instruct" --batch-size 32 --concurrency 12`
- `python main.py --data-dir data/anlp --out-dir out/anlp --llm-model "meta-llama/Llama-3.1-8B-Instruct"`
- `LLM_PROVIDER=hf RELATION_DEBUG=1 RELATION_DEBUG_N=3 python main.py --steps ingest llm clustering pairpackets relations --data-dir data/sql --out-dir out/sql --llm-model "meta-llama/Llama-3.1-8B-Instruct" --batch-size 32 --concurrency 12`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir testing`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output`
- `python main.py --steps clustering pairpackets relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output`
- `python main.py --steps clustering pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir algo_output`
- `python main.py --steps clustering pairpackets relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir algo_output`
- `python main.py --steps relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output_llama3b`
- `python main.py --steps relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir sql_output_llama8b`
- `python main.py --steps relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output`
- `python main.py --steps relations --llm-model Qwen/Qwen2.5-14B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output_qwen14b`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output_llama8b/`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.1-8B-Instruct --concurrency 1 --batch-size 32 --out-dir algo_output_llama8b/`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir anlp_output_llama3b/`
- `python main.py --steps pairpackets relations --llm-model meta-llama/Llama-3.2-3B-Instruct --concurrency 1 --batch-size 32 --out-dir algo_output_llama3b/`
- `python eval.py --input_file "/home/alrabah2/graphika/graphika/evaluation/relations_algo_llama8b (1).jsonl" --course_name algo`
