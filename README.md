<div align="center">

<img src="banner.png" alt="InstructKG" width="50%">

### Automated Knowledge Graph Construction from Educational Content

*Building concept dependency graphs from lecture materials using multi-LLM architectures*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)


</div>

---

> **Fork note.** This is a fork of [aalrabah/InstructKG](https://github.com/aalrabah/InstructKG),
> extended for a study over UIUC course materials. The Overview and Architecture below are the
> original author's. This fork adds local-model support via vLLM, an evaluation harness, graph
> visualization, and the operational documentation needed to reproduce a run end to end.
>
> Two files carry most of what was learned running it. **`findings.md`** is a dated,
> append-only investigation log — root causes, benchmarks, and things ruled out; check it
> before re-investigating anything that may already be diagnosed. **`testing_cmds.md`** is the
> per-course run log, organized by course then model, with the stage-by-stage output counts a
> healthy run should produce.

---

## Overview

**InstructKG** is an automated framework that extracts structured knowledge graphs from educational materials. It analyzes lecture slides, textbooks, and course materials to identify concepts and their pedagogical relationships — helping students understand learning paths and prerequisite dependencies.

### What It Does

Given educational PDFs, InstructKG:
1. **Extracts concepts** mentioned across lectures using LLMs
2. **Identifies roles** (Definition, Example, Assumption, NA) for each concept
3. **Clusters contexts** to find where concepts appear together
4. **Judges relationships** between concept pairs to build dependency graphs

### Why It Matters

- 📚 **For Students**: Understand which concepts to learn first and how topics connect
- 👨‍🏫 **For Educators**: Automatically generate course roadmaps and learning paths
- 🔬 **For Instructors**: Course content insights and knowledge tracing

---

## Architecture

InstructKG uses a **five-stage pipeline**:

```
PDFs → Chunking → Concept Extraction → Clustering → Pair Generation → Relation Judgment → Knowledge Graph
```

1. **Ingestion**: Converts PDFs to semantically meaningful chunks
2. **LLM Extraction**: Identifies concepts and classifies each mention's role
3. **Clustering**: Groups similar contexts using UMAP + HDBSCAN
4. **Pair Packets**: Aggregates evidence for concept pairs from co-occurrences
5. **Relation Judgment**: LLM classifies each pair as `depends_on`, `part_of`, or no edge

---

## Quick Start

### Installation

```bash
git clone https://github.com/aalrabah/InstructKG.git
cd InstructKG

pip install -r requirements.txt
```

On shared or container environments where the Python prefix is not writable, install with
`--user` so packages persist across sessions:

```bash
pip install --user -r requirements.txt
```

`neo4j-viz` and `json_repair` in particular fail or silently disappear without this on
`/opt/conda`-style installs.

On headless machines, follow with:

```bash
pip install --force-reinstall --no-deps opencv-python-headless
```

(rapidocr pulls the full `opencv-python`, which can overwrite the headless `cv2` and then
fail to import without `libGL.so.1`.)

### Choosing an LLM backend

`LLM_PROVIDER` selects the backend. It defaults to `openai`.

| `LLM_PROVIDER` | Backend | Notes |
|---|---|---|
| `openai` (default) | OpenAI API | needs `OPENAI_API_KEY` |
| `anthropic` / `claude` | Anthropic API | needs `ANTHROPIC_API_KEY` |
| `vllm` | local vLLM engine | GPU; used for all local runs |
| `hf` / `huggingface` / `local` | local vLLM engine | alias for `vllm` |
| `hf_legacy` | transformers pipeline | slower fallback path |

Local runs use `vllm`:

```bash
LLM_PROVIDER=vllm python main.py \
  --data-dir data/me400 --out-dir out/me400 \
  --llm-model Qwen/Qwen2.5-14B-Instruct
```

### Environment

Configuration is read from a `.env` file in the repo root (see `config.py`).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | backend selection (table above) |
| `OPENAI_API_KEY` | — | OpenAI credentials |
| `ANTHROPIC_API_KEY` | — | Anthropic credentials |
| `OUT_DIR` | `out` | default output directory |
| `MAX_TOKENS` | `8191` | chunker token budget |
| `CONCURRENCY` | `5` | request concurrency |
| `VLLM_MAX_MODEL_LEN` | `8192` | vLLM context window |
| `VLLM_MAX_NEW_TOKENS` | `512` | generation cap |
| `VLLM_TEMPERATURE` | `0.1` | sampling temperature |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | vLLM memory fraction |

Long-context stages can overflow at the default 8192. Set `VLLM_MAX_MODEL_LEN=16384` for
relation judging on large corpora.

### Input file naming — this controls OCR

`ingest.py` picks its OCR mode from a tag in the PDF filename. This is the highest-impact
operational detail in the repo:

| Filename tag | OCR mode | Use for |
|---|---|---|
| `*_scan.pdf` | `force_full_page_ocr=True` | scanned or handwritten notes with no usable text layer |
| `*_text.pdf` | native text extraction | PDFs with a clean embedded text layer |

Getting this wrong is not subtle. The same scanned document produced 24 chunks with native
extraction and 442 with full-page OCR.

---

## Running the full pipeline in one command

Pipeline → graph → evaluation, detached, one course at a time:

```bash
C=me400
mkdir -p out/$C
nohup bash -c "set -e
LLM_PROVIDER=vllm VLLM_MAX_MODEL_LEN=16384 PYTHONUNBUFFERED=1 python main.py \
  --data-dir data/$C --out-dir out/$C --llm-model Qwen/Qwen2.5-14B-Instruct
test -s out/$C/relations.jsonl
KG_RELATIONS=out/$C/relations.jsonl jupyter nbconvert --to notebook --execute \
  --stdout knowledge_graph_visualization.ipynb > /dev/null
PYTHONUNBUFFERED=1 python evaluation/eval.py --input_file out/$C/relations.jsonl \
  --course_name $C --model_name Qwen/Qwen2.5-14B-Instruct --method_name instructkg_qwen14b \
  > out/$C/eval_${C}_qwen14b.log 2>&1" > out/$C/run.log 2>&1 &
```

- Pipeline and graph output land in `out/<course>/run.log`; evaluation in
  `out/<course>/eval_<course>_<model>.log`.
- The graph step runs before evaluation deliberately — it is free, and under `set -e` an
  evaluation error would otherwise cost you the graph too.
- Evaluation requires a `course_map` entry for `$C` in `evaluation/eval.py`.

### Running a subset of stages

Stages read and write the same `--out-dir`, so partial reruns are normal:

```bash
python main.py --data-dir data/me400 --out-dir out/me400 \
  --steps ingest llm clustering pairpackets relations
```

`main.py` skips `ingest` automatically when a `chunks*.jsonl` already exists in the out-dir.
Delete it to force re-ingestion.

---

## Pipeline Stages

### 1. Ingestion (`ingest`)
Converts PDF lectures into chunks with metadata, via docling + `HybridChunker`.

**Output**: `chunks.jsonl`

### 2. Concept Extraction (`llm`)
One call per chunk extracts candidate concepts; one call per (chunk, concept) classifies the
mention's role as Definition, Example, Assumption or NA. Mentions are then aggregated per
concept across all lectures.

**Output**: `mentions.jsonl`, `concept_cards.jsonl`

### 3. Clustering (`clustering`)
Groups similar contexts where concepts appear.

**Techniques**: Sentence embeddings → UMAP → HDBSCAN

**Output**: `context_clusters.jsonl`

### 4. Pair Packet Generation (`pairpackets`)
Aggregates evidence for concept pairs that co-occur in the same chunk or the same cluster —
temporal order, role-grounded snippets, co-occurrence counts.

**Output**: `pairpackets.jsonl`

### 5. Relation Judgment (`relations`)
An LLM judges each pair packet, preferring chunk-level evidence over cluster-level evidence.
The output vocabulary is `depends_on`, `part_of`, or no edge (`null`).

**Output**: `relations.jsonl` (final knowledge graph edges)

---

## Evaluation

`evaluation/eval.py` scores a `relations.jsonl` with an LLM judge on two metrics:
**node significance** (is this concept meaningful for the course?) and **triplet accuracy**
(is this relation judgment correct?). Both are ordinal 0–2, reported normalized to 0–1.

```bash
python evaluation/eval.py \
  --input_file out/me400/relations.jsonl \
  --course_name me400 \
  --model_name Qwen/Qwen2.5-14B-Instruct \
  --method_name instructkg_qwen14b
```

`--course_name` must be a key in the `course_map` inside `eval.py`. The mapped value is injected
into the judge prompt as the course title and scope, and node significance is judged
course-relative — so the string matters to the score, not just as a label.

`--max_model_len` defaults to the model's own configured maximum. Judging is precision-oriented:
it scores the edges the pipeline produced and has no recall term.

---

## Visualization

`knowledge_graph_visualization.ipynb` renders a `relations.jsonl` to an interactive HTML graph.
It is parameterized by environment variable, so it runs headless without editing:

```bash
KG_RELATIONS=out/me400/relations.jsonl \
  jupyter nbconvert --to notebook --execute --stdout \
  knowledge_graph_visualization.ipynb > /dev/null
```

`kg_visualization.html` is written beside the input file. `--stdout` discards the executed
notebook so the tracked file never accumulates outputs.

| Variable | Default | Purpose |
|---|---|---|
| `KG_RELATIONS` | `out/cs401_403/relations.jsonl` | input relations file |
| `KG_MAX_EDGES` | `500` | edge render ceiling |

Above the ceiling the notebook samples rather than truncating, so raise `KG_MAX_EDGES` when a
course exceeds it. Output size is roughly 6.5 MB of fixed library payload plus ~4 KB per edge.

---

## Instructor review packet

`make_packet.py` turns a `relations.jsonl` into a Word document an instructor can mark up:
Part A (are these significant concepts?), Part B (are these relationships right?), and Part C
(three short questions). Set the course once at the top:

```bash
C=cs401_403
TITLE="iCAN Algorithms (CS 401/403)"
M=qwen14b

python make_packet.py \
  --relations out/$C/relations.jsonl \
  --course-title "$TITLE" \
  --run-label "$C, Qwen-14B, August 2026" \
  --eval-json out/$C/eval_${C}_${M}.json \
  --out out/$C/InstructKG_Review_Packet_${C}.docx
```

The two LLM-judge scores are required.
If no eval JSON, drop `--eval-json` and pass `--node-significance` and `--triplet-accuracy`
from the tables in `testing_cmds.md`.

Sampling is deterministic for a given `--seed`, so regenerating a packet reproduces the same
questions. Change the seed to draw a fresh sample.

| Argument | Default | Description |
|----------|---------|-------------|
| `--seed` | `597` | Controls Part A tail draws and Part B sampling |
| `--head-concepts` | `12` | Most-referenced concepts in Part A |
| `--tail-concepts` | `3` | Seeded tail draws added to Part A |
| `--glance-concepts` | `8` | Concepts listed in the summary paragraph |
| `--relationships` | `15` | Relationships sampled for Part B |
| `--justification-chars` | `420` | Truncation cap on each justification |
| `--name-overrides` | `None` | JSON file mapping raw names to display names |
| `--part-a-note` | `None` | Extra sentence in the Part A footnote |
| `--sample` | off | Render the banner label in red |
| `--return-by` / `--return-email` | `[date]` / `[email]` | Fill the return lines |

---

## Backing up outputs

`data/`, `out/`, `*.jsonl` and `*.log` are gitignored — input PDFs and every pipeline output are
local-only by design. Course PDFs run 20–80 MB each and every run adds a multi-megabyte HTML
graph, none of which belongs in git history. This study mirrored `out/` to a Google Drive folder
with [rclone](https://rclone.org/drive/) instead.

One-time setup: `rclone config`, choose Google Drive, name the remote `gdrive`. Then create a
Drive folder to hold the outputs and copy its ID out of the folder URL
(`drive.google.com/drive/folders/<ID>`).

```bash
rclone copy out/ "gdrive,root_folder_id=<PASTE_ID_HERE>:"
```

- Safe to rerun at any time — `copy` only uploads new or changed files, and never deletes on the
  remote. (`rclone sync` does delete; don't substitute it here.)

---

## Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Folder containing PDF lectures |
| `--out-dir` | `out` (`OUT_DIR`) | Output directory for results |
| `--llm-model` | `OPENAI_CONCEPTS_MODEL` | LLM model for extraction |
| `--batch-size` | `8` | Batch size for LLM calls |
| `--concurrency` | `1` | Relation judger concurrency (keep at 1 for vLLM) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Model for embeddings |
| `--min-cluster-size` | `2` | Minimum cluster size for HDBSCAN |
| `--min-cooc-chunks` | `0` | 0 allows cluster-only pairs |
| `--max-pairs` | `None` | Limit number of concept pairs (for testing) |
| `--steps` | all | `ingest llm clustering pairpackets relations` |

---

## Example

A 379-page combined-notes PDF (CS 401/403), Qwen2.5-14B-Instruct via vLLM, ~47 minutes on an
A100-80GB:

```
chunks:        279
mentions:      950  (323 after the min_unique_chunks filter)
concepts:      605  (71 after filtering)
clusters:       74
pairs:         180
relations:     180  →  93 edges over 61 nodes
```

**Sample extracted relationships:**
- `CONTRAPOSITIVE` → `NEGATION` (depends_on)
- `BIJECTION` → `ONE_TO_ONE` (depends_on)
- `CONCATENATION` → `REGULAR_EXPRESSIONS` (part_of)

---

## Project Structure

```
InstructKG/
├── main.py                            # Pipeline orchestrator
├── ingest.py                          # PDF → chunks (docling; OCR mode from filename tag)
├── llm.py                             # Concept extraction + role classification
├── clustering.py                      # Context clustering (UMAP + HDBSCAN)
├── pairpackets.py                     # Evidence aggregation for pairs
├── relation_judger.py                 # Relation classification
├── adapters.py                        # LLM provider abstraction (OpenAI/Anthropic/vLLM/HF)
├── prompts.py                         # Prompt templates for every LLM stage
├── response_parsing.py                # Tolerant JSON → Pydantic parsing
├── config.py                          # Configuration defaults (.env)
├── make_packet.py                     # relations.jsonl -> instructor review packet (.docx)
├── requirements.txt                   # Python dependencies
├── evaluation/
│   ├── eval.py                        # LLM-judge scorer
│   └── final_eval.json                # Reference evaluation set
├── knowledge_graph_visualization.ipynb # relations.jsonl → interactive HTML
├── students_mapping.ipynb             # Concept-card analysis
├── data/                              # Input PDFs (*_scan.pdf / *_text.pdf)
└── out/                               # Output files, per course
    ├── chunks.jsonl
    ├── mentions.jsonl
    ├── concept_cards.jsonl
    ├── context_clusters.jsonl
    ├── pairpackets.jsonl
    ├── relations.jsonl                # Final knowledge graph
    ├── kg_visualization.html
    └── InstructKG_Review_Packet_<course>.docx
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
