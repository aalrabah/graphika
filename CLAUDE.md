# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

InstructKG extracts a concept dependency knowledge graph from educational PDFs (lecture slides, scanned notes). It runs a five-stage pipeline per course dataset: PDF → chunks → concept mentions → context clusters → pair evidence → judged relations (`depends_on` / `part_of`).

## Commands

Run the full pipeline:
```bash
python main.py --data-dir data/<course> --out-dir out/<course> --llm-model <model>
```

Run a subset of stages (stages read/write the same `out-dir`, so partial reruns are normal):
```bash
python main.py --data-dir data/sql --out-dir out/sql --steps ingest llm clustering pairpackets relations
```
`main.py` skips `ingest` automatically if `chunks.jsonl` already exists in `out-dir` — delete it to force re-ingestion.

Select the LLM backend via `LLM_PROVIDER` env var (`openai`, `anthropic`, `vllm`, `hf` — `hf` now routes to the vLLM engine too; use `hf_legacy` for the old transformers pipeline):
```bash
LLM_PROVIDER=vllm python main.py --data-dir data/me400 --out-dir out/me400 --llm-model Qwen/Qwen2.5-3B-Instruct
```
Long vLLM runs are typically launched with `nohup ... > out/<course>/run.log 2>&1 &` since they run on a remote GPU box and need to survive a dropped connection (see `testing_cmds.txt`).

Individual stages also have their own standalone CLIs, e.g.:
```bash
python relation_judger.py --in out/sql/pairpackets.jsonl --out out/sql/relations.jsonl --model <model> --batch-size 8 --concurrency 12
python clustering.py --chunks out/chunks.jsonl --mentions out/mentions.jsonl --out out/context_clusters.jsonl --use-umap
```

There is no automated test suite and no lint/format config in this repo. `testing_cmds.txt` is a running log of ad hoc pipeline invocations across datasets/models used to manually verify behavior — check it for known-working command shapes before inventing new ones. `evaluation/eval.py` is a separate vLLM-based scorer for judging output quality against `evaluation/final_eval.json`.

## Architecture

**Pipeline stages** (each stage is a module, orchestrated by `main.py`):
1. `ingest.py` — docling converts PDFs to `Document` objects, then `HybridChunker` splits into token-bounded chunks → `chunks.jsonl`.
2. `llm.py` — for each chunk, one LLM call extracts candidate concepts, then one call per (chunk, concept) classifies its role (Definition/Example/Assumption/NA) → `mentions.jsonl`, and `build_concept_cards` deterministically aggregates mentions per concept across all lectures → `concept_cards.jsonl`.
3. `clustering.py` — embeds chunk text (sentence-transformers) → UMAP → HDBSCAN to group chunks discussing similar context → `context_clusters.jsonl`.
4. `pairpackets.py` — for concept pairs that co-occur in the same chunk or same cluster, aggregates evidence (temporal order, role-grounded snippets, co-occurrence counts) → `pairpackets.jsonl`.
5. `relation_judger.py` — LLM judges each pair packet into `depends_on`, `part_of`, or no edge, preferring chunk-level evidence over cluster-level evidence → `relations.jsonl` (final graph edges). `ALLOWED_RELATIONS = {"depends_on", "part_of"}` is the actual output vocabulary.

Downstream: `knowledge_graph_visualization.ipynb` and `students_mapping.ipynb` consume `relations.jsonl`/`concept_cards.jsonl`.

**LLM provider abstraction (`adapters.py`)** is the key indirection layer: every stage calls `client.responses.create(model=..., instructions=..., input=...)` against a common interface, and `get_llm_client()` returns one of `OpenAICompatClient`, `AnthropicCompatClient`, or `VLLMCompatClient`/`HFCompatClient` based on `LLM_PROVIDER`. The vLLM/HF engines lazily load a model on first use and cache it on the client instance; `HF_MODELS_MAP` maps short names (`qwen3b`, `llama8b`, ...) to full HF model IDs. Batched calls (list-of-list `input`) are supported for local engines to reduce per-call overhead; OpenAI/Anthropic paths fall back to one call per item via `asyncio.gather`.

**Structured LLM output parsing** (`response_parsing.py` + `prompts.py`): local models don't reliably emit strict JSON, so `parse_pydantic_from_llm_text` strips code fences, scans for the first balanced `{...}`/`[...]`, and validates against a Pydantic model (`ConceptExtractionOutput`, `RoleTaggerOutput`, `RelationJudgmentOutput`). Prefer extending this path over adding ad hoc JSON parsing when handling new LLM output shapes.

**Config** (`config.py`, loaded via `.env`): `OUT_DIR`, `MAX_TOKENS` (chunker token budget), `CONCURRENCY`, provider API keys/model defaults. Per-provider generation knobs (`VLLM_MAX_NEW_TOKENS`, `VLLM_TEMPERATURE`, `HF_BATCH_SIZE`, etc.) are read directly from env in `adapters.py` rather than `config.py`.

## Known gotchas

- **docling OCR config**: `force_full_page_ocr` must be set on the nested `ocr_options=OcrAutoOptions(...)` object passed to `PdfPipelineOptions`, not as a top-level `PdfPipelineOptions` kwarg — pydantic silently drops unknown top-level kwargs with no error. When `force_full_page_ocr=True`, docling discards native/programmatic text cells entirely and replaces them with a fresh OCR pass for the *whole* page (not just image regions) — good for scanned/handwritten PDFs with no usable text layer, but it reintroduces OCR noise into PDFs that already have a clean embedded text layer. See `findings.md` for the full investigation (ME200 vs ME400 chunk-count discrepancy) and candidate conditional-OCR fixes not yet implemented.
- **vLLM is pinned to `0.19.1`** in `requirements.txt` — newer vLLM pulls in a torch build that requires CUDA 13, which fails silently (surfaces as an opaque "EngineCore socket handshake" error) on this project's GPU driver (max CUDA 12.8). Don't bump past this pin without confirming the driver supports it.
- `data/`, `*.jsonl`, and `*.log` are gitignored — input PDFs and all pipeline outputs (`out/`) are local-only, not tracked in git.
- `findings.md` is an append-only, dated investigation log (root causes, benchmarks, things ruled out) — check it before re-investigating something that may already be diagnosed.
