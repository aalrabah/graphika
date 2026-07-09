# InstructKG — Running Findings Log

Append-only log of investigation findings, diagnostics, and root causes across this project. Not limited to one subject — add a new dated section per topic as things come up. Newest entries at the bottom.

**Entry format:**

```
## [YYYY-MM-DD] <short subject line>

<findings, evidence, root cause, open questions>
```

---

## [2026-07-04] ME200 vs ME400 chunking discrepancy

### Observed symptom

- ME200: 80MB PDF, ~250 pages of handwritten notes -> only 24 chunks, 44 concept cards.
- ME400: 20MB PDF, ~800 typed slides -> 725 chunks, 1423 concept cards.
- Question: why does the smaller/shorter-page-count file produce ~30x more chunks?

### Finding 1 — chunk size is roughly constant across both files

Average chunk text length (chars):

| Run | Count | Min | Max | Avg |
|---|---|---|---|---|
| ME200 (Qwen14B run) | 24 | 29 | 1811 | 473.1 |
| ME400 (Qwen3B run) | 725 | 1 | 9147 | 464.2 |

The chunker (docling HybridChunker, config `MAX_TOKENS` in `ingest.py`) targets a roughly fixed text size per chunk. Average chunk length is nearly identical between the two documents (**473 vs 464 chars**). This means chunk COUNT is driven by total extracted text volume, not by page count or file size.

### Finding 2 — most ME200 pages produced zero extractable text

Distinct pages referenced across all chunks:

| Run | Distinct pages w/ text | % of pages |
|---|---|---|
| ME200 (both Qwen14B and Qwen3B runs) | 56 / ~246 total | ~23% contributed any text at all; ~190 pages contributed nothing |
| ME400 (Qwen3B run) | 782 / ~805 total | ~97% of pages |

Total extracted characters across the whole document:

| Run | Total chars |
|---|---|
| ME200 Qwen14B | 11,354 |
| ME200 Qwen3B | 11,363 |
| ME400 Qwen3B | 336,521 (~30x more text than ME200, despite ME200's PDF being ~4x larger in file size) |

### Finding 3 — text quality on the pages ME200 DID extract looks like weak OCR

Sample chunk text from ME200:

```
Air in y<control :
Boundory surface)
(wdo)
```

```
Ke Tang. All Rights Reserved.
Ch1-3 Ke Tang. All Rights Reserved.
```

This reads like OCR output struggling with handwritten diagrams/text — garbled words, dropped content, mostly boilerplate ("All Rights Reserved") surviving because it's likely printed/typed text on an otherwise handwritten page.

Sample chunk text from ME400: clean, coherent typed sentences (e.g. faculty bio text, course topics).

### Root cause — CONFIRMED

`ingest.py:34` calls `DocumentConverter()` with no explicit pipeline/OCR configuration, so whatever docling defaults to is what ran on both files.

Checked installed docling (v2.109.0) defaults directly:

- `do_ocr = True`
- `ocr_options = OcrAutoOptions(force_full_page_ocr=False, bitmap_area_threshold=0.05)`

"Auto" mode probes the environment at pipeline init and picks the first available engine in this priority order:

1. `ocrmac` — macOS only — skipped, this is Linux
2. `nemotron` — not installed — skipped
3. `rapidocr+onnxruntime` — onnxruntime not installed — skipped
4. `easyocr` — not installed — skipped
5. `rapidocr+torch` — torch IS installed — **SELECTED**

So both ME200 and ME400 were processed with RapidOCR running on the torch backend. RapidOCR is a general printed-text/document OCR engine — it is not trained or tuned for handwriting recognition. This fully explains the observed pattern:

- ME400 (typed slides): no handwriting to struggle with -> real embedded/printed text, extracted cleanly on ~97% of pages.
- ME200 (handwritten notes): RapidOCR produces garbled/partial output on the minority of pages where it detects an OCR-able bitmap region, and produces nothing usable (zero doc_items, page silently dropped) on the ~77% of pages where confidence is too low or the handwritten content isn't cleanly classified as an OCR-able bitmap area. Defaults of `force_full_page_ocr=False` + `bitmap_area_threshold=0.05` mean OCR is only attempted on regions docling's layout model flags as bitmap images covering more than 5% of the page — not a full-page OCR pass.

### Bottom line

The low ME200 chunk/concept count is not a chunker bug — chunk size is consistent across both runs. It's an upstream text-extraction problem: docling defaulted to RapidOCR (torch backend) because onnxruntime/easyocr aren't installed, and RapidOCR — like all three candidate engines here — is built for printed text, not handwriting. **~77% of ME200's pages** are producing no usable text at all as a result.

### Possible next steps (not yet implemented)

- Install onnxruntime and/or easyocr to get a different default engine (may or may not help much — none of these are handwriting-specialized).
- Set `force_full_page_ocr=True` in `PdfPipelineOptions` to stop docling from skipping pages based on bitmap-region detection.
- Use an OCR/model actually suited for handwriting (e.g. a TrOCR-handwritten checkpoint, or a vision-LLM-based transcription pass) instead of relying on docling's default OCR stack for the ME200-style scanned/handwritten inputs.

---

## [2026-07-05] force_full_page_ocr fix — implementation + verification

### Action taken

Implemented "possible next step" #2 from 2026-07-04: force docling to OCR every page instead of only pages its layout model flags as a bitmap-image region >5% of page area (`bitmap_area_threshold=0.05` default).

Change is in `ingest.py`, `pdf_to_chunks()`, where `DocumentConverter()` is constructed. Now passes explicit `PdfPipelineOptions` with `ocr_options=OcrAutoOptions(force_full_page_ocr=True)`.

### Finding 1 — first attempt was a silent no-op

Initial edit passed the flag directly as `PdfPipelineOptions(force_full_page_ocr=True)`. This is WRONG: `force_full_page_ocr` belongs to the nested `ocr_options` object (`OcrAutoOptions`), not to `PdfPipelineOptions` itself. Because docling's pipeline options are pydantic models with default "ignore extra fields" behavior, this misplaced kwarg was silently dropped — no error, no warning. Confirmed directly:

```
PdfPipelineOptions(force_full_page_ocr=True).ocr_options
  -> OcrAutoOptions(force_full_page_ocr=False, ...)   # unchanged!
```

A full ME200 pipeline run was executed with this broken version and produced IDENTICAL output to the original bug (chunks=24, same garbled text), confirming the setting never took effect. The only anomaly visible in that run's log was a benign vLLM teardown-race message ("EngineCore died unexpectedly, shutting down client") appearing AFTER "Shutdown complete" — a red herring, unrelated to the OCR issue.

Corrected form:

```python
pdf_pipeline_options = PdfPipelineOptions(
    ocr_options=OcrAutoOptions(force_full_page_ocr=True)
)
```

### Finding 2 — corrected fix verified directly against ME200 PDF

Ran `pdf_to_chunks()` directly (ingest step only, isolated from LLM/clustering/relations steps) against `data/me200/ME200_CombinedNotes.pdf` with the corrected fix in place:

| Metric | Before (broken) | After (fixed) |
|---|---|---|
| Total chunks | 24 | 442 |
| Distinct pages w/ text | 56 / ~246 (~23%) | 247 / 248 (~99.6%) |
| Total extracted chars | 11,363 | 220,504 (~19x increase) |

Sample chunk text is also qualitatively better — coherent phrases and section headers are now coming through instead of mostly-garbled fragments, e.g.:

```
Ch1-1 Introduction and Defining Systems Ke Tang. All Rights Reserved.
Objectives: Reading: 1.1, 1.2 -Thermodynamics <Identify and Explain>...
```

Some OCR garbling remains in spots, apparently on rotated/upside-down text regions, e.g.:

```
hywon paxf Y :<sou jo4uoo) wo4sh paso10 of matter.
```

This is consistent with Finding 3 from 2026-07-04 (RapidOCR is a printed-text engine, not handwriting-specialized) — `force_full_page_ocr` fixes page COVERAGE (Finding 2 from 2026-07-04), it does not fix OCR ACCURACY on handwritten/rotated content (Finding 3 from 2026-07-04, still open).

### Status / open items

- `out/me200/chunks.jsonl` on disk still holds the OLD 24-chunk broken output from the last full pipeline run (predates the ocr_options fix). The 442-chunk verified result above was computed in-memory only, via a throwaway test script, and was never written to disk.
- Not yet done: rerun the full pipeline (ingest -> llm -> clustering -> pairpackets -> relations) on ME200 with the fix in place to get updated concept_cards/relations output.
- Handwriting-specific OCR accuracy (Finding 3, 2026-07-04) remains unaddressed — still a candidate follow-up (TrOCR-handwritten checkpoint, or vision-LLM transcription pass) if garbled text on scanned/handwritten pages is still a problem after this fix.

---

## [2026-07-05] force_full_page_ocr side effect on ME400 (typed doc) + fix options

### Observed symptom

Ran ME400 (typed slide deck, has a clean embedded/native text layer) with the `force_full_page_ocr=True` fix from the entry above, and compared against the earlier default-OCR run:

| Metric | Old (default OCR) | New (force_full_page_ocr) |
|---|---|---|
| Chunks | 725 | 749 |
| Total chars | 336,385 | 342,775 |

Small net increase (+24 chunks, +6.4K chars) — much smaller than ME200's ~18x jump, as expected since ME400 already had a good text layer. But a direct diff of chunk text (exact-match set difference) showed **499 old chunks** with no exact match in the new run and **517 new chunks** with no exact match in the old run — far more churn than the small net change suggests.

### Finding 1 — most of the "new" content is OCR-corrupted duplicates of already-clean text, not new coverage

Fuzzy-matched (difflib, cutoff=0.6) each of the 531 new-only chunks against the old chunk set:

- **465 / 531 (88%)** matched an old chunk at high similarity (0.94-1.00) — i.e. same underlying content, but the new version has OCR-introduced corruption: stray inserted characters/digits ("Thermal 1 energy", "crossplatform 1 integrated"), dropped words / case changes ("deals With" for "deals with", "Thef four laws" for "The four laws"), lost punctuation ("2023 -Date" -> "2023 Date").
- **66 / 531 (12%)** had no close old match at all. Sampled these directly — consistently formula/table/code-screenshot regions: thermodynamic property tables ("h = 3022.9 + 320 - 280"), a Python/CoolProp code screenshot, LaTeX-ish equation blocks that were previously rendered as `<!-- formula-not-decoded -->` placeholders in the old run. This portion is genuine new coverage, not corruption.

### Root cause — CONFIRMED, read docling 2.109.0 source directly

`docling/models/base_ocr_model.py`:

- **Lines 89-113 (rect selection):** with `force_full_page_ocr=False` (default), OCR is only applied to detected bitmap regions covering more than `bitmap_area_threshold` (default 0.05 = 5%) of the page. With `force_full_page_ocr=True`, this check is skipped and the ENTIRE page is always treated as one OCR region.
- **Lines 159-186 (`_combine_cells` / `post_process_cells`):** this is the actual bug for ME400. When `force_full_page_ocr=False`, OCR cells are filtered to drop any that overlap existing native/programmatic cells (`_filter_ocr_cells`, line 116), then APPENDED to the native cells — native text is preserved, OCR only fills gaps. When `force_full_page_ocr=True`, this is skipped entirely: `combined = ocr_cells` (line 178) means native cells are thrown away outright, and `post_process_cells` (line 163) goes further and filters `word_cells`/`char_cells` down to `from_ocr=True` only. So `force_full_page_ocr` doesn't just add OCR coverage — it replaces the ENTIRE page's text with a fresh OCR pass, discarding a perfectly good native text layer if one existed. This is exactly why already-clean ME400 text came back with new OCR-typo corruption: it was never using the native extraction anymore, even on pages that didn't need OCR at all.

### Bottom line

`force_full_page_ocr=True` is a clear win for ME200 (scanned/handwritten, ~77% of pages had ~nothing to lose). For ME400 (typed, has a real text layer) it's a mixed bag: recovers real content trapped in formula/code/table image regions, but also corrupts already-good native text on every other page by fully discarding it in favor of a fresh, lower-fidelity OCR pass. A single global `force_full_page_ocr=True` setting can't serve both document types well.

### Candidate fixes (not yet implemented — pipeline run in progress, action deferred until it completes)

1. **Lower `bitmap_area_threshold`** instead of/in addition to `force_full_page_ocr`
   - Docling default is 0.05 (5% of page area). Lowering it (e.g. to ~0.0-0.01) with `force_full_page_ocr` left False should pick up small embedded raster images (screenshots, small diagrams) that fall under the current 5% threshold, WITHOUT discarding native text elsewhere — because the append + overlap-filter path (`_filter_ocr_cells`) only runs when `force_full_page_ocr=False`. Cheapest change to try, single kwarg.
   - Caveat: only helps if the missed content is in a raster bitmap region docling's layout model actually detects (`page._backend.get_bitmap_rects()`). Vector-drawn formulas/diagrams (not embedded raster images) won't be picked up by this at all — see option 2.

2. **Enable `do_formula_enrichment=True` and `do_code_enrichment=True`** on `PdfPipelineOptions` (`pipeline_options.py:1682-1747`)
   - These are dedicated, non-OCR VLM-based recognition stages built into docling specifically for math formulas (-> LaTeX) and code blocks, backed by `CodeFormulaVlmOptions`. This looks like the correct, purpose-built fix for the 12% "genuinely new" ME400 content bucket found above (property-table formulas, the CoolProp code screenshot) — replacing "formula-not-decoded" placeholders properly instead of leaning on generic full-page OCR for that content. Doesn't touch/replace native text elsewhere on the page, so no corruption risk for already-clean pages. Needs a model download/runtime check (VLM-backed) before relying on it — not yet verified whether the required model is available in this environment.

3. **Document-level conditional two-pass OCR** (bigger change, addresses the ME200-vs-ME400 split directly)
   - Run a first pass with default settings (`force_full_page_ocr=False`, option-1 lowered threshold, option-2 enrichment flags on). Measure extracted text density (e.g. total chars / page count).
   - If density is above some threshold (ME400-like: dense native text layer already present), keep that result as-is.
   - If density is at/near zero (ME200-like: ~46 chars/page in the original default run), re-convert that document with `force_full_page_ocr=True`, since there's ~nothing to lose by discarding a native layer that barely existed.
   - This is a per-document heuristic, not per-page within one document; docling's `force_full_page_ocr` is a global pipeline option with no built-in per-page override, so true per-page conditional OCR would require subclassing/patching `base_ocr_model`'s rect-selection logic (bigger lift, not recommended as a first step).
   - `force_backend_text=True` (`pipeline_options.py:1700-1709`, "bypasses the layout model's text detection and uses the embedded text from the PDF file directly... useful for PDFs with reliable programmatic text layers") may be worth testing alongside this as a way to keep native text authoritative on the ME400-like branch specifically — not yet tested for compatibility with the OCR path.

Recommended order to try (lowest risk/effort first): 1 and 2 together first (cheap, no architecture change, directly targets the ME400 corruption + missed-formula issues) with ME200 re-tested to confirm 1 alone doesn't regress it back toward the original bug; fall back to 3 only if ME200 and ME400 still can't both be served well by a single set of global options.

---

## [2026-07-05] ME400 downstream (concept/relation) impact of force_full_page_ocr

### Context

Full pipeline (ingest -> llm -> clustering -> pairpackets -> relations) run to completion on ME400 with `force_full_page_ocr=True`, same Qwen14B model as the existing default-OCR baseline (`out/me400/Qwen14B/*`), so this isolates the OCR-setting effect from model choice. Follows directly from the chunk-level corruption finding in the entry above.

### Finding — downstream impact is much smaller than the chunk-level churn implied

| Metric | Old (default OCR) | New (force_full_page_ocr) | Delta |
|---|---|---|---|
| Chunks | 725 | 749 | +3.3% |
| Mentions | 1,799 | 1,839 | +2.2% |
| Concept cards | 1,118 | 1,131 | +1.2% |
| Clusters | 174 | 177 | +1.7% |
| Pairpackets | 3,158 | 3,168 | +0.3% |
| Relations — depends_on | 1,608 | 1,693 | +5.3% |
| Relations — part_of | 260 | 238 | -8.5% |
| Relations — none | 1,290 | 1,237 | -4.1% |

Despite 499 chunks disappearing and 517 new ones appearing at the text level (entry above), only **~13 net concepts** and **~10 net relations** changed overall. The LLM concept-extraction step appears robust to the OCR-introduced typos/stray characters — it doesn't seem to be manufacturing garbage concepts out of corrupted text.

### Quality check

Sampled concept_label values unique to each run (concept_ids present in one run's `concept_cards.jsonl` but not the other's) to check for OCR-garbage polluting the graph. None found: both sides read as clean, plausible thermo concepts (old-only: "vapor cycles", "Kelvin temperature scale", "EOS dry air"; new-only: "entropy calculation", "Butane", "Tab Twb Ta", "oxidation reaction" — consistent with the genuinely-new formula/property-table content identified in the entry above). No garbled/junk concept labels observed on either side.

**Caveat** — concept ID churn is larger than the net-count change suggests: 230 concept_ids only in the old run, 243 only in the new run, out of ~1,118-1,131 total (**~21% churn**). This is NOT a quality signal in either direction — it's chunk-boundary reshuffling (the same content lands in differently-sized/ordered chunks) changing what's visible to the LLM per extraction call, not better or worse extraction.

### Bottom line / recommendation

`force_full_page_ocr=True` is not badly damaging ME400 at the concept/relation level (no garbage-concept pollution observed), but it also isn't earning its keep: a ~1-5% volume bump doesn't justify the native-text-corruption risk already documented in the entry above, when cheaper/more targeted fixes are already identified and not yet implemented. See "Candidate fixes" in the [2026-07-05] force_full_page_ocr side effect entry (this same date, above) — still recommend trying options 1 (lower bitmap_area_threshold) + 2 (do_formula_enrichment / do_code_enrichment) first, in `ingest.py`'s `pdf_to_chunks()`, before falling back to option 3 (two-pass conditional). Not yet implemented — no `ingest.py` changes made as a result of this entry.

---

## [2026-07-06] Formula and diagram content is invisible to concept extraction

### Observed symptom

ME200/ME400 are equation- and diagram-heavy engineering course material. Suspected the chunker/LLM extraction step wasn't capturing that content well.

### Finding — quantified via placeholder markers in chunks.jsonl

Counted docling's HTML-comment-style placeholder markers across each course's `chunks.jsonl`:

| Course | Chunks | formula-not-decoded | image/diagram placeholders |
|---|---|---|---|
| ME200 | 442 | 689 | 0 |
| ME400 | 725 | 687 | 0 |
| SQL | 13 | 1 | 0 |

More than one unrendered formula per chunk on average in the ME courses. More strikingly: **0 image placeholders anywhere** — and confirmed via a full scan of every distinct `<!--...-->` marker in `me400/chunks.jsonl` that `<!-- formula-not-decoded -->` is the ONLY marker docling emits. Diagrams, charts, and photos don't get a placeholder at all — they contribute zero text to the chunk, so they don't just render poorly, they don't exist as far as concept extraction is concerned.

### Root cause — confirmed in docling 2.109.0 source

`pipeline_options.py:1257-1265`: `do_picture_description` (VLM-generated textual descriptions of pictures, "for accessibility and searchability") defaults to False. Combined with `do_formula_enrichment` also defaulting to False (see the [2026-07-05] force_full_page_ocr side effect entry above, candidate fix #2), two entire content categories — formulas and diagrams — are currently invisible to the LLM concept-extraction step, in every course, regardless of OCR/scan setting.

### Bottom line

This is a separate, likely higher-impact gap than the OCR tuning above: it affects every course (not just scanned ones) and strikes directly at concept-extraction completeness for STEM material. Not yet implemented — candidate fix is enabling `do_formula_enrichment` + `do_picture_description` (+ `do_code_enrichment` for code screenshots, e.g. the CoolProp example from the ME400 entry above) on one course and re-measuring concept/relation coverage before rolling out further.

---

## [2026-07-06] force_full_page_ocr is now auto-detected from filename tag

### Change

`ingest.py`'s `pdf_to_chunks()` no longer hardcodes `force_full_page_ocr`. A new helper, `_is_scanned_from_filename(pdf_path)`, splits the filename stem on `"_"` and checks whether `"scan"` is one of the resulting tokens:

- `"..._scan.pdf"` -> True (scanned/handwritten, no native text layer)
- `"..._text.pdf"` / no tag -> False (born-digital, has a native text layer)

Renamed the 5 active course PDFs to carry this tag, classification based on a pypdfium2 sampled-page text-density check (~10 pages/doc, same method used for the original ME200 vs ME400 comparison):

| File | Notes |
|---|---|
| `data/me200/ME200_CombinedNotes_scan.pdf` | ~4 chars/sampled page avg |
| `data/me320/ME320_CombinedNotes_scan.pdf` | ~4 chars/sampled page avg |
| `data/me400/ME400_CombinedNotes_text.pdf` | ~265 chars/sampled page avg |
| `data/sql/3-SQL2-JOINS_Nulls_text.pdf` | — |
| `data/tam251/TAM251_CombinedNotes_text.pdf` | ~265 chars/sampled page avg |

`data/me200/archive/ME200_Ch1.pdf` left untouched (unused leftover; not picked up anyway since `ingest_pdfs` globs `*.pdf` non-recursively).

### Why

Removes the failure mode from the [2026-07-05] entries above where the setting was a single hardcoded line someone had to remember to flip back and forth per course. Now the correct setting travels with the file.

---

## [2026-07-06] ME200/Qwen3B relations step: silent failure produced 100% fake null results, masked as a normally-running job

### Observed symptom

A `main.py --data-dir data/me200 --llm-model Qwen/Qwen2.5-3B-Instruct` run (on the H200) appeared to still be running after 3.5 hours (ps showed elapsed 03:39:08) — way past what a Qwen3B/ME200 run should take. Root cause was GPU contention: a separate ME400/Qwen32B enrichment smoke-test job was launched around the same time on the same GPU, and between the two, available GPU memory dropped to **~37 MiB free out of 143771 MiB total**.

### Root cause — confirmed by reading relation_judger.py and run.log

`relation_judger.py` (pre-fix) wrapped the per-batch LLM call in a bare try/except that caught ANY exception — including a failed vLLM engine initialization from GPU memory contention — and silently substituted `responses = ["{}"] * len(prompts)` for that batch, then continued to the next batch. Because the vLLM engine's `_loaded` flag never gets set when `load()` raises partway through, EVERY subsequent batch re-attempted engine init from scratch, hit the same GPU-memory error, and got the same fake substitution.

### Mechanics — NOT a hang on one batch, a full grind through fake work

`judge_pairpackets_file()` splits ~3044 pairs into ~380 batches and works through that list once, not in a retry loop on a single item. For each batch in sequence: attempt a fresh vLLM engine init -> fail (GPU OOM) -> catch, substitute fake responses for that batch's 8 pairs -> write them to disk -> move to the next batch, where the identical cycle repeats with a brand-new doomed engine-init attempt. Counted **240 such cycles** in run.log, **~52s apart**, each cycle producing **~100 log lines** (full vLLM startup sequence — config resolution, GPU memory checks, warnings — plus a full multi-process Python traceback for the failure). This is why the log was tens of thousands of lines long despite representing wasted, repetitive work rather than diverse progress. Directly inspected `relations.jsonl`: all **1976 records** written before the job was killed had `"relation": null` with the identical generic justification ("No clear relation is supported by the provided evidence."). That justification text alone isn't a reliable fingerprint (it's the normal fallback whenever parsed justification is empty, which a genuine model output could also produce) — what's actually diagnostic is the complete absence of any non-null relation across ~2000 diverse pairs, which no working run has produced all session.

The job was NOT stuck/hung in the sense of making zero progress — it would have kept grinding forward through the full ~380-batch backlog and eventually terminated NORMALLY ("Done. Wrote N records"), indistinguishable in exit behavior from a real successful run, unless GPU memory freed up partway through (in which case only pairs processed after that point would get real judgments, silently mixed with fake ones before it, with no recorded boundary).

### Compounding landmine — resume logic treats fake records as done

`judge_pairpackets_file()` has a resume mechanism (`_load_done_pairs(out_file)`) that skips any pair already present in the existing `relations.jsonl` on a rerun. Writes are incremental (`_append_jsonl` per batch), which normally makes resume safe — but it means the 1976 fake null records were fully indistinguishable from genuine "done" work to that resume logic. Resuming with `--steps relations` against the as-was `relations.jsonl` would have permanently skipped all 1976 corrupted pairs rather than reprocessing them. The corrupted file was deleted (manually, by the user, in a separate session) before any resume was attempted.

### Fix applied

`relation_judger.py`: removed the try/except around the batch LLM call (same "loud fail" pattern already used in `adapters.py`'s `_VLLMLocalEngine.generate_many` — that `adapters.py` change predates this session and was never separately documented here either). Old block commented out in a `'''...'''` string rather than deleted, matching the existing convention. Now an engine-init failure propagates and crashes the run immediately instead of silently continuing. Safe specifically because of the incremental-write + resume design above: crashing loses no genuinely-completed work, it just stops at the last real batch instead of quietly overwriting `relations.jsonl` with thousands of fake records with no error marker anywhere in the output data.

### Process note — original run.log no longer exists

`out/me200/run.log` was overwritten before this entry was written. A follow-up pipeline run (on the A100, same shared filesystem) was pointed at the same `--out-dir out/me200`, and the shell's `> out/me200/run.log` redirect truncated the old file the moment that run started — the original ~24,600+ line log with the 240 failure cycles is gone, replaced by the new run's (much shorter) log. All figures in this entry (240 cycles, 1976 fake records, ~52s/cycle, ~100 lines/cycle) were captured by direct inspection during the incident, before the overwrite happened, not reconstructed afterward. **Lesson:** a shared run.log path gets silently destroyed by the next job that reuses it — anything worth keeping from a log should be copied out under a distinct name before rerunning against the same out-dir.

### Bottom line

Not advisable to ever let this kind of failure "ride it out" hoping GPU memory frees up — even in the best case (contention resolves mid-run), you'd get a silently mixed fake/real file with an undocumented boundary. Killed the runaway process, fixed the silent-failure path in `relation_judger.py`, and the underlying GPU contention itself (separate root cause) is being avoided going forward by moving Qwen3B tests to the dedicated A100 rather than sharing the H200 with larger model runs.

---

## [2026-07-06] Qwen3B/14B/32B quality comparison, and formula/code enrichment vs. force_full_page_ocr independence

### Context

Full pipeline runs completed for sql, me200 (_ffp only — non-ffp me200 runs are the known-broken chunking from the 2026-07-04 entry, excluded), and me400 across Qwen2.5-3B/14B/32B-Instruct (9 runs total). Question: raw concept/pair/relation counts don't say which model is "best" — more concepts/pairs isn't automatically better. Needed quality signals derived from the jsonl outputs themselves, since there's no ground-truth eval set for this comparison.

### Finding 1 — role-tagging quality (mentions.jsonl "role" field)

% of role-tagged mentions classified "definition" vs. dumped into "assumption":

| Model | sql | me200_ffp | me400 |
|---|---|---|---|
| Qwen3B | 3.0% def / 75% assum | 4.7% def / 72% assum | 5.0% def / 80% assum |
| Qwen14B | 42.2% def / 100% assum | 29.3% def / 67% assum | 34.2% def / 58% assum |
| Qwen32B | 23.7% def / 83% assum | 18.1% def / 69% assum | 17.9% def / 60% assum |

Qwen3B almost never tags anything "definition" and collapses most mentions into "assumption" — consistent across all 3 courses despite each course being different source material. Since the same chunks feed all three models, this is a model-capability signal, not a corpus artifact.

### Finding 2 — relation-type discrimination (relations.jsonl depends_on vs part_of ratio)

| Model | Ratio | me200_ffp | me400 |
|---|---|---|---|
| Qwen3B | ~1:1 | 1130:1364 | 1830:1957 |
| Qwen14B | ~7:1 | 905:128 | 1608:260 |
| Qwen32B | ~7.5:1 | 2172:294 | 3177:428 |

14B and 32B agree with each other on this ratio; 3B is near coin-flip between the two relation types, suggesting it isn't reliably distinguishing "depends_on" from "part_of" as a judgment task.

### Finding 3 — justification grounding

% of non-null relation justifications that textually reference both concept names (loose token match, tokens >3 chars): Qwen14B highest (**98.4-99.6%**), Qwen32B middle (**97.3-99.2%**), Qwen3B lowest (**95.5-97.2%**). Small gaps but a consistent ordering across all 3 courses.

### Finding 4 — inter-model agreement on shared concept-pairs (strongest signal)

For concept-pairs that two models both happened to extract in the same course, % agreement on relation type:

| Comparison | me200_ffp | me400 |
|---|---|---|
| 14B vs 32B | 70.3% exact / 93.7% (both nonnull) | 75.4% / 92.8% |
| 3B vs 14B | 34.2% exact / 48.3% (both nonnull) | 44.2% / 58.5% |
| 3B vs 32B | 36.9% exact / 46.7% (both nonnull) | 45.6% / 58.1% |

(sql excluded — Qwen3B found 0 pairs there entirely, and 14B/32B each produced only 1 relation, too sparse to compare.)

14B and 32B converge with each other on **~93%** of pairs where both assert a relation exists — two independently-run models landing on the same judgment that often is strong evidence they're tracking real structure in the material. 3B's agreement with either larger model is barely above chance between two relation types.

No self-loops, no directly-contradictory relation pairs (A depends_on B AND B depends_on A simultaneously), and no duplicate canonical concept labels found in any of the 9 runs — ruling out those specific failure modes as a confound in the above.

### Bottom line — model choice

Qwen3B is unreliable across all four signals and not recommended for rollout. Qwen14B and Qwen32B are mutually consistent; between them, 14B edges out 32B on definition-tagging rate and justification-grounding, and 32B shows *more* "na"-tagged mentions than 14B despite being the larger model (more classification uncertainty, not less). Combined with 14B's lower compute cost, **Qwen14B is the recommended model going forward**. Caveat: none of this proves 14B/32B are "correct" in an absolute sense, only that they're self-consistent — a targeted spot-check of the cases where 14B and 32B *disagree* with each other (a small, high-signal set) would be the next level of verification if needed.

### Finding 5 — do_formula_enrichment/do_code_enrichment is independent of force_full_page_ocr

`ingest.py` was changed (uncommitted, in progress) to unconditionally set `do_formula_enrichment=True` and `do_code_enrichment=True` on `PdfPipelineOptions`, addressing the "formula/diagram content invisible to concept extraction" gap from the 2026-07-06 entry above. Question raised: does `force_full_page_ocr=True` (used for `_scan`-tagged courses) prevent this enrichment from working, since it discards native text cells wholesale?

Read docling 2.109.0 source directly to check (`legacy_standard_pdf_pipeline.py`, `layout_model.py`, `code_formula_model.py`):

- `force_full_page_ocr` only controls which TEXT CELLS get attached to a page (native vs. full-page OCR'd) — `base_ocr_model.py`, already documented in the 2026-07-05 entries above.
- The LAYOUT MODEL (which detects/labels regions including FORMULA and CODE clusters) runs on the raw rendered page image (`layout_model.py:172`, `page.get_image()`) — completely independent of OCR mode or text cell content. It runs after OCR in the pipe (`legacy_standard_pdf_pipeline.py:88-103`) but does not consume OCR's output.
- `CodeFormulaModel.is_processable()` (`code_formula_model.py:152`) fires on any element the layout model already labeled FORMULA or CODE, and `__call__` re-recognizes that region from a CROPPED IMAGE of it (`el.image`) using its own dedicated VLM (CodeFormulaV2) — it never reads the OCR/native text cells at all.

**Conclusion:** `force_full_page_ocr` and formula/code enrichment are orthogonal settings. ffp=True does NOT block formula/code enrichment from working — the layout model still detects and labels FORMULA/CODE regions from the page image regardless of OCR mode, and enrichment re-recognizes those regions independently via image crop. This means the per-course `_scan`/`_text` OCR decision (2026-07-06 entry above) does not need to change, and enrichment is a candidate improvement on BOTH branches, not just `_text`-tagged courses. (Whether a noisier scanned page image makes the layout model's box detection less accurate in practice is a separate, untested empirical question — not a hard incompatibility.)

### Status / next steps — not yet run

Planned test sequence, in order:

1. **Ingest-only validation** (no GPU/LLM): re-run `pdf_to_chunks()` in isolation on me400 (`_text`, 687 formula placeholders — primary test case), me200_ffp (`_scan`, 689 placeholders — secondary, to confirm enrichment also fires under ffp=True), and sql (`_text`, only 1 placeholder — quick sanity check, low signal expected). Compare old vs. new: formula-not-decoded placeholder count, chunk/char count, and critically a chunk-level diff to confirm already-clean/native chunks are byte-identical and not altered (the same regression check that caught the `force_full_page_ocr` corruption on me400 in the 2026-07-05 entry — the `ingest.py` comment claims this change is "purely additive" but that has not yet been empirically verified). Sample newly-recovered formula/code chunks directly to confirm real LaTeX/code output, not VLM hallucination.
2. If clean, full pipeline rerun (ingest -> relations) on me400 and me200_ffp with enrichment on, Qwen14B only (no need to redo the 3-model comparison — Finding 1-4 above already settled model choice). Compare concept_cards/relations against the existing Qwen14B baselines, looking specifically for new concepts traceable to formula/code content rather than just volume growth.
3. If validated, roll out to the untested courses (me320 — `_scan` tagged, tam251 — `_text` tagged) with enrichment on and Qwen14B as the model.

---

## [2026-07-07] Ingest-only validation of formula/code enrichment on ME400 — "purely additive" claim was FALSE, comment corrected

### Context

Executes step 1 of the planned test sequence from the entry above. Compared `out/me400/Qwen32B/chunks.jsonl` (725 chunks, pre-enrichment, same file referenced in the 2026-07-06 Qwen32B run) against `out/me400/chunks.jsonl` (725 chunks, produced 2026-07-06 22:22 with `do_formula_enrichment=True` and `do_code_enrichment=True`, per the `ingest.py` change). Same source PDF (`data/me400/ME400_CombinedNotes_text.pdf`), so `chunk_index` aligns 1:1 across both files (`chunk_id` prefixes differ only because the PDF was renamed with the `_text` tag between runs — see 2026-07-06 filename-tag entry).

### Finding 1 — formula placeholders fully eliminated, chunk count unchanged

| Metric | Old (pre-enrichment) | New (enrichment on) |
|---|---|---|
| Chunks | 725 | 725 |
| Total chars | 336,385 | 627,622 |
| "formula-not-decoded" markers | 687 | 0 |

All 687 formula placeholders identified in the 2026-07-06 entry are gone, replaced with real `$$...$$` LaTeX. Chunk count and page coverage are unaffected — this part is a clean win, as expected.

### Finding 2 — the enrichment is not purely additive: CODE/FORMULA-labeled regions are unconditionally overwritten, even when already clean

Of 725 chunks: **483 byte-identical, 242 changed**. Of the 242 changed:

- **193 chunks:** had a formula-not-decoded placeholder in the old text — exactly the intended, additive case.
- **49 chunks:** had NO placeholder in the old text (i.e. already had usable native/OCR text) but were changed anyway. 46 of these 49 contain a ` ``` ` code fence in the old text — these are CoolProp/EES-style numeric-output screenshots that the layout model labels as a CODE region regardless of whether that region already extracted cleanly. `CodeFormulaModel` re-recognizes the region from a fresh image crop (per the 2026-07-06 Finding 5 mechanics) and its output REPLACES the old text outright — there is no merge/fallback to the prior extraction.

This means `ingest.py`'s original comment describing the change as "purely additive... doesn't touch already-clean native text elsewhere on the page" was correct about the OCR text-cell path (enrichment models don't touch that) but wrong about CODE/FORMULA-labeled *regions* specifically — those get unconditionally overwritten by the enrichment models even when they already had good text. Comment corrected in `ingest.py` to describe this scope accurately (only text outside FORMULA/CODE-labeled regions is untouched).

### Finding 3 — re-recognition of code/output screenshots trades readability for numeric-value corruption in a meaningful minority of cases

Sampled the 46 code-fence chunks directly. Most show a genuine readability improvement (proper variable-name formatting: `T_1 = 300 [K]` vs. the old `T [1 = 300 [K]` linear dump with no assignment structure). But several also corrupt the numeric values themselves, not just formatting:

| idx | Old | New | Issue |
|---|---|---|---|
| 208 | P_1 8666 [kPa] | P_1 = 8000~[kPa] | value changed |
| 156 | T_3 101 [C] | T_3 = 10°C | value changed |
| 422 | eta 0.6315775314437779 | 0.631577531437779 | digit dropped |
| 218 | T_3 41.50876547963833 | 41.50867547963833 | digit swapped |
| 303 | P_1 372.92464275358316 | 372.92464275538316 | digits swapped |

Also one case of a hallucinated caption line with no clear source (idx 45): new text prepends "DataUcase IN 1 YTHON USING Spyder YDLE" (garbled, not present in the old extraction at all) before the actual CoolProp code.

Separately, 3 of the 49 non-placeholder-changed chunks (idx 114, 145, 381) were pre-existing `$$...$$` inline math (unicode subscripts like "𝑠2 -𝑠3 = 𝑄out 𝑚𝑇c") that got reformatted into proper LaTeX fractions (`\frac{\dot{Q}_{out}}{\dot{m}T_{c}}`) — a clear net improvement, no corruption observed in this subset.

Also worth flagging from the earlier idx 24-26 sample (non-code-fence formula region, not counted in the 46): unicode math-italic symbols for density (𝜌) collapsed to plain "p" in the new text, colliding with the already-present "Pressure: p" on the same line/list — a semantic ambiguity introduced by the enrichment pass, not present in the old extraction. "Isochoric specific heat" also came back mis-recognized as "Isochroni" in one chunk and "Isomorphic" in another (same term, two different wrong spellings) — consistent, low-confidence recognition noise on that specific label.

### Bottom line

`do_formula_enrichment`/`do_code_enrichment` successfully closes the "formula content invisible to extraction" gap from 2026-07-06 (0 placeholders left) without touching the 483 chunks outside any FORMULA/CODE-labeled region. But it is not the "purely additive" change the original `ingest.py` comment claimed: **~6.3% of all chunks (46/725)** are CODE-region screenshots that get unconditionally re-recognized and, in a meaningful fraction of sampled cases, come back with silently altered numeric values alongside genuine formatting gains — the same "looks like a clear win but corrupts a subset" pattern already documented for `force_full_page_ocr` on ME400 (2026-07-05 entries). Likely lower-stakes here than that OCR case, since these are worked-example numeric results (not concept definitions) and concept/relation extraction cares about textual concepts more than exact numbers — but not verified empirically.

### Not yet done

Downstream (concept_cards/relations) impact check for this change, analogous to the 2026-07-05 "ME400 downstream impact of force_full_page_ocr" entry — i.e. planned step 2 from the entry above, still outstanding.

---

## [2026-07-07] me400_enriched relations crash x2 — GPU contention, then a genuine token-overflow bug in relation_judger.py — MAX_TOKENS reverted, VLLM_MAX_MODEL_LEN raised via env var, main.py chunks-file bug fixed

### Context

This is the me400_enriched run logged in `testing_cmds.txt` (727 chunks, 3003 mentions, 1135 concepts, 179 clusters, 3066 pairpackets — `do_formula_enrichment`/`do_code_enrichment=True`, `MAX_TOKENS=6000` per the entry above). Ingest -> llm -> clustering -> pairpackets all completed cleanly; relations crashed twice, for two unrelated reasons.

### Crash 1 — GPU contention on H200

Relations step failed immediately on engine load: "ValueError: Free memory on device cuda:0 (116.87/139.81 GiB) on startup is less than desired GPU memory utilization (0.9, 125.83 GiB)". `nvidia-smi` confirmed another tenant's job using **120,429/143,771 MiB** at 100% util with "No running processes found" (separate container/namespace, not ours, not killable). **0 relations written** — this is the "loud fail" fix from the 2026-07-06 ME200/Qwen3B entry working correctly (crashed instead of silently writing fake nulls). Resolved by re-running the relations step on the dedicated A100 instead of waiting on the contended H200, per the same entry's precedent. GPU choice for a downstream stage does not affect run quality/determinism — same model/weights/vLLM version either way, `pairpackets.jsonl` is a deterministic on-disk artifact the relations step reads fresh regardless of which GPU produced it.

### Crash 2 — genuine token-length overflow in relation_judger.py (A100)

Relations step got through **120/3066 pairs** then crashed: "VLLMValidationError: ... your prompt contains at least 8193 input tokens". Directly tokenized all 3066 pairpackets with Qwen2.5-14B-Instruct's real tokenizer (offline/cached) through relation_judger's actual chat-template path: **23/3066 pairs** exceed 8192 tokens outright (worst: 9,897, MOIST_AIR/VAPOR_QUALITY), **121/3066** exceed 6000. The crash pair is confirmed as pairpackets index 126 (ENERGY_CONSERVATION/SPECIFIC_ENTHALPY, 9,204 tokens, batch 15 = indices 120-127) — exactly matches `relations.jsonl` having 120 records written before the crash.

Root cause: `relation_judger.py` has NO token-budget check anywhere in `_select_evidence_chunks` / `_format_evidence_block` / `build_prompt_from_pairpacket` before handing the assembled prompt (up to 3 aggregated evidence chunks) to vLLM. Unlike `ingest.py`'s chunker, nothing bounds this prompt's size.

Separately: `VLLM_MAX_MODEL_LEN=8192` (`adapters.py:222` default) turned out to be a self-imposed cap, not a model limit. Read Qwen2.5-14B-Instruct's config directly (offline, no download): `max_position_embeddings=32768`, `rope_type='default'` — native 32K support, no YaRN scaling needed.

### The MAX_TOKENS=6000 fix (entry above) does not actually solve either crash, and reverting it to the original 8191 is safe — explained below

Traced why `MAX_TOKENS=6000` was set: a reactive fix for an EARLIER, separate overflow ("chunk 549" during an ingest/llm-stage call, referenced in a `config.py` comment but never actually written up as its own findings.txt entry until now). Investigating this exposed a real, independent bug:

docling's `HybridChunker` (`ingest.py:89`) is constructed with no explicit tokenizer argument, so it silently uses `docling_core`'s default (`sentence-transformers/all-MiniLM-L6-v2`) — a completely different vocabulary than Qwen's. `MAX_TOKENS` therefore has never measured chunk size in the same currency as `VLLM_MAX_MODEL_LEN`; the two numbers were only ever coincidentally comparable. Confirmed directly: chunk 549's actual text, at `MAX_TOKENS=8191`, is **7,770 Qwen tokens**; plus role-tagging's fixed prompt overhead (measured directly: 636 tokens) = **8,406** — which does exceed 8192, reproducing the original bug. (Concept-extraction's fixed overhead is much smaller, 119 tokens.) So `MAX_TOKENS=6000` happened to dodge that one chunk, but for the wrong reason (shrinking a budget counted in the wrong tokenizer), while doing nothing for relation_judger's separate overflow (crash 2 above, which happened anyway, on 6000-token chunking) and diverging me400's chunking from every other course in the corpus for no real benefit.

### Fix applied

- `config.py`: `MAX_TOKENS` reverted **6000 -> 8191** (original default); comment corrected to describe the tokenizer-mismatch finding instead of the abandoned "leaves headroom" rationale.
- `adapters.py`: `VLLM_MAX_MODEL_LEN` left unchanged in code for now (still defaults to 8192). Raising it is being done as a command-line env var override (`VLLM_MAX_MODEL_LEN=16384`) rather than a code default change, deliberately scoped to the relations step — `llm.py`'s concept-extraction/role-tagging calls run per chunk x concept (thousands of calls/course) and don't need the bigger ceiling (worst case measured: 8,406), so there's no reason to pay vLLM's reduced-KV-cache-concurrency cost (~28x concurrency at 8192 vs ~7x at 32768, on a fixed 233,104-token pool) on every one of those calls when only relation_judger's evidence aggregation is actually at risk. Will fold a verified number into the code default (with a comment) once we see whether 16384 survives the run below — not yet done.
- `main.py`: fixed a latent bug found while staging a differently-named chunks file for this test (`chunks_enriched_8191.jsonl`, to reuse existing verified chunks and skip re-running the expensive OCR/enrichment ingest pass). Ingest's skip-check already correctly glob-matched any `chunks*.jsonl` file and printed "found existing, skipping ingest" — but `chunks_path` itself (what every downstream stage's `read_jsonl` call actually reads) stayed hardcoded to the literal `out_dir/"chunks.jsonl"`, so a differently-named staged file would pass the skip-check and then crash with a `FileNotFoundError` in the very next (llm) step. One-line fix:

  ```python
  chunks_path = str(next(out_dir.glob("chunks*.jsonl"), out_dir / "chunks.jsonl"))
  ```

### Status — in progress, not yet verified

Old `MAX_TOKENS=6000` downstream artifacts (mentions/concept_cards/context_clusters/pairpackets/relations.jsonl + both crash run logs) archived to `out/me400/archive_maxtok6000/` rather than deleted, so evidence from both crashes above is preserved. `chunks_enriched_8191.jsonl` (725 chunks, `MAX_TOKENS=8191`, enrichment on, 627,622 total chars — same file measured above, produced 2026-07-06 22:22) staged in `out/me400/` as the chunks file to reuse.

Relaunched (H200, uncontended at relaunch time):

```
nohup env LLM_PROVIDER=vllm VLLM_MAX_MODEL_LEN=16384 python main.py \
  --data-dir data/me400 --out-dir out/me400 \
  --steps ingest llm clustering pairpackets relations \
  --llm-model Qwen/Qwen2.5-14B-Instruct > out/me400/run.log 2>&1 &
```

Confirmed ingest step correctly skipped via the `main.py` fix above; llm stage running as of this entry.

**Open question, not yet resolved:** is `VLLM_MAX_MODEL_LEN=16384` actually sufficient for relation_judger's worst-case REAL (not theoretical-ceiling) evidence aggregation once mentions/pairpackets regenerate from the reverted 8191-token chunking? A crude ceiling using the two largest known chunks (549: 7,770 + 496: 6,271 = 19,615 combined, before template overhead) exceeds 16384, but that assumes those two chunks actually co-occur for the same concept pair, which was not true of the previous (6000-token) chunking's crash pair and is not yet confirmed either way for this one. If this run hits the same class of crash, next step is raising `VLLM_MAX_MODEL_LEN` for the relations step specifically (e.g. 32768, the model's real ceiling) rather than pipeline-wide.

**Addendum:** both original logs for the two crashes above (`out/me400/run.log` / `partial_run.log`, and `relations_run.log`) were lost — confirmed via `~/.bash_history` that `run.log` was silently overwritten by later reruns sharing the same `> out/me400/run.log` redirect target (the exact failure mode already documented in the 2026-07-06 ME200/Qwen3B entry's process note); `relations_run.log`'s disappearance is unexplained. Recreated a partial reconstruction from excerpts already quoted during this session at `out/me400/Qwen14B_enriched/6000_chunks/run_recreated.log` — clearly marked as a reconstruction, not a full/authentic capture.

---

## [2026-07-08] me400_enriched/8191_chunks completed — first successful formula/code-enrichment run through relations — but relations.jsonl has 3003 lines vs. run.log's reported 3123

The `VLLM_MAX_MODEL_LEN=16384` relaunch from the entry above finished cleanly, end to end: **727 chunks, 1135 concept cards, 179 clusters, 3123 pairpackets**. Output relocated to `out/me400/Qwen14B_enriched/8191_chunks/`. Next planned step: compare against the existing `out/me400/Qwen14B` baseline.

**Discrepancy:** run.log ends with "Done. Wrote 3123 new records to out/me400/relations.jsonl", no errors/tracebacks anywhere in the log — but the relocated `relations.jsonl` has only **3003 lines, 120 short**. Checked `relation_judger.py`'s write path (`judge_pairpackets_file`): the "wrote N" counter increments in lockstep with each `_append_jsonl` call under the same lock, so the script itself has no code path to over-report. The original `out/me400/relations.jsonl` no longer exists (moved before the gap was noticed), so old-vs-new can't be diffed and the missing 120 records can't be identified. Not investigated further — recorded here in case the pattern repeats on a future run. **Takeaway:** verify `wc -l` against the log's write count before trusting a run's output for analysis. Plan is to rerun the me400_enriched pipeline once more for a verified-clean file before doing the baseline comparison.

---

## [2026-07-08] me400_enriched/8191_chunks vs. Qwen14B baseline — formula/code enrichment downstream impact, verdict: worthwhile but modest

### Context

Rerun (per entry above) completed clean: `relations.jsonl` has **3472 lines**, matching run.log's "Wrote 3472 new records" exactly (no repeat of the 120-record gap). Compared `out/me400/Qwen14B_enriched/8191_chunks/` (enrichment on, `MAX_TOKENS=8191`, `VLLM_MAX_MODEL_LEN=16384`) against `out/me400/Qwen14B/` (pre-enrichment baseline, same Qwen2.5-14B-Instruct model) to answer whether `do_formula_enrichment`/`do_code_enrichment` is worth carrying forward to future courses.

### Finding 1 — volume

| Metric | OLD (baseline) | NEW (enriched) | Delta |
|---|---|---|---|
| Chunks | 725 | 727 | +0.3% |
| Mentions | 1,799 | 1,989 | +10.6% |
| Concept cards | 1,118 | 1,136 | +1.6% net |
| Clusters | 174 | 178 | +2.3% |
| Pairpackets | 3,158 | 3,472 | +9.9% |
| Relations | 3,158 | 3,472 | +9.9% |

Chunk count barely moved (as expected — enrichment replaces content within existing FORMULA/CODE-labeled regions, per the 2026-07-06 Finding 5 in the entry above, it doesn't change chunk boundaries). Mentions/pairpackets/relations all grew ~10%, concept cards net grew only 1.6% (see churn below).

### Finding 2 — role-tagging and relation-type discrimination are unchanged

| Signal | OLD | NEW |
|---|---|---|
| definition % | 34.2% | 32.7% |
| assumption % | 58.0% | 58.2% |
| na % | 4.6% | 5.2% |
| example % | 3.2% | 4.0% |
| depends_on:part_of ratio | 6.18:1 | 7.85:1 |
| justification grounding | 96.4% | 97.0% |

All four signals from the 2026-07-06 model-comparison entry stayed within noise of each other. Enrichment is not degrading (or improving) the LLM's judgment quality on the content it can already see — consistent with the expectation that enrichment only changes what text exists in a minority of chunks, not how the judging/tagging models behave.

### Finding 3 — stability check: relation judgments on concept pairs present in BOTH runs are highly consistent, no corruption signal

**1,861 concept pairs** were judged in both runs (same canonical A/B names). Of those:

- **83.9%** exact agreement including both-null (1,562/1,861)
- **96.9%** agreement among the 1,169 pairs where BOTH runs asserted a non-null relation (1,133/1,169)
- 36 pairs flipped depends_on <-> part_of
- 109 pairs: old asserted a relation, new said none
- 154 pairs: new asserted a relation, old said none

96.9% same-pair agreement is in the same range as the 14B-vs-32B cross-model agreement (92.8% on me400) from the 2026-07-06 entry — i.e. enrichment changes existing judgments about as little as swapping in a different (comparably-sized) model would, not a red flag. This directly tests the corruption concern raised in the 2026-07-07 entry (CODE/FORMULA regions get unconditionally overwritten, sometimes with corrupted numeric values) — the answer is that this corruption is not visibly propagating into relation-judgment instability at the pair level.

### Finding 4 — new-concept yield: majority of new-only concepts DO trace to formula/code content, but pairpacket/relation yield from them is small

**201 concept_ids** appear only in the new run. Checked each one's evidence chunk_text for enrichment markers (`$$` LaTeX blocks, ` ``` ` code fences):

- **107/201 (53%)** linked to `$$` formula evidence
- **28/201 (14%)** linked to code-fence evidence
- **66/201 (33%)** neither — plain-text evidence, i.e. chunk-boundary/extraction churn unrelated to enrichment (same background noise pattern documented for `force_full_page_ocr` in the 2026-07-05 entry)

So ~67% of new concepts are genuinely attributable to the enrichment feature working as intended (recovering formula/code content that was previously invisible per the 2026-07-06 "Formula and diagram content is invisible" entry). However, downstream structure gained from them is thin:

- Only **49/3,472 pairpackets (1.4%)** involve any new-only concept at all
- 0 pairpackets pair two new-only concepts with each other
- Of those 49, only 17 resolved to a non-null relation (all depends_on), the rest (32) judged no-relation
- Net: **17 new relations** directly attributable to enrichment, out of 3,472 total (0.5%)

Reason: `pairpackets.py` pairs concepts that co-occur in the same chunk or cluster. Formula/code-recovered concepts are concentrated in a small subset of chunks (~239 of 727, per the 2026-07-07 entry's 193 placeholder + 46 code-region counts) and mostly co-occur with pre-existing concepts already in the baseline graph, not with each other — so enrichment adds new nodes and a handful of new edges to the existing graph, it does not surface an independent sub-cluster of formula-derived structure.

### Finding 5 — old-only concept loss is comparable in size to new-only churn, not a regression

**183 concept_ids** appear only in the old run. Fuzzy-matched (difflib, cutoff=0.6) against the 201 new-only concepts:

- **134/183 (73%)** fuzzy-match a new-only concept (e.g. CARNOT_CYCLE_EFFICIENCY -> CARMOT_CYCLE, AHU_INLET_AIR -> AHU_INLET) — same-concept renaming/chunk-boundary churn, not lost content.
- **49/183 (27%)** have no close match — possibly genuinely dropped.

49 genuinely-dropped vs. 66 genuinely-new (Finding 4) is the same order of magnitude — consistent with ordinary extraction churn already documented elsewhere in this log (2026-07-05 ME400 OCR entry, 2026-07-06 model-comparison entry), not a new failure mode introduced by enrichment.

### Bottom line — recommendation

Formula/code enrichment is a small, low-risk, directionally-positive change: it closes a real completeness gap (formula/diagram content previously invisible to concept extraction, 2026-07-06 entry), the majority of new concepts trace directly to that recovered content, and it does not destabilize judgments on the ~1,861 concept pairs shared with the baseline (96.9% agreement, in the same range as normal model-to-model variance). The downstream yield is modest, though: **+18 net concepts, +17 relations out of 3,472 (0.5%)** directly attributable to the feature — most of the recovered formula/code concepts don't end up paired with each other or judged into new relations at all, because pairpacket formation depends on co-occurrence with concepts already in the graph. Numeric-value corruption within re-recognized CODE regions (2026-07-07 entry, e.g. digit swaps in worked-example results) remains a known, unresolved risk, but worked-example numeric values are a low-stakes category for a concept/relation graph specifically (as opposed to a fidelity-critical use case).

**Recommendation for future course runs:** keep `do_formula_enrichment`/`do_code_enrichment` ON by default for STEM-heavy courses (formula-dense material like ME400/ME200/TAM251) — the completeness win is real and the measured corruption risk doesn't propagate into graph-level instability. Not worth the extra ingest/relations compute+risk for courses with little formula/code content (e.g. SQL, which had only 1 formula-not-decoded marker total in the 2026-07-06 placeholder-count entry) since there's nothing to recover there. Still open: the CODE-region numeric-corruption risk (2026-07-07 entry) has no mitigation implemented — a candidate follow-up if a future use case ever needs exact numeric fidelity from worked examples rather than just concept/relation structure.

---

## [2026-07-08] me200 (_scan, force_full_page_ocr=True) + formula/code enrichment — recognition quality on handwritten pages is visibly worse than ME400's typed slides, driving a much higher no-relation rate

### Context

me200 is the SCANNED/HANDWRITTEN course (`data/me200/ME200_CombinedNotes_scan.pdf` — the same course from the original 2026-07-04 ME200-vs-ME400 chunking entry), not a text course — clarifying a mix-up from earlier in this session. Compared `out/me200/Qwen14B_ffp_enriched/` (`force_full_page_ocr=True` + `do_formula_enrichment`/`do_code_enrichment=True`, both settings are hardcoded on in `ingest.py`, not per-run flags) against `out/me200/Qwen14B_ffp/` (`force_full_page_ocr=True` only, pre-enrichment baseline, same Qwen2.5-14B-Instruct model) — same comparison methodology as the me400 entry above. `relations.jsonl` verified clean first (2195/2195 lines match run.log's "Wrote 2195 new records", no repeat of the 2026-07-08 me400 write-count gap).

### Finding 1 — volume: concept count is flat-to-down, but pairs/relations grew much more than me400's equivalent comparison

| Metric | OLD (ffp only) | NEW (ffp+enriched) | Delta |
|---|---|---|---|
| Chunks | 442 | 443 | +0.2% |
| Mentions | 882 | 1,004 | +13.8% |
| Concept cards | 890 | 867 | -2.6% (net) |
| Clusters | 121 | 120 | -0.8% |
| Pairpackets | 1,711 | 2,195 | +28.3% |
| Relations | 1,711 | 2,195 | +28.3% |

Compare to me400's +9.9% pairpacket/relation growth (entry above) — me200's growth is roughly **3x larger** in relative terms, despite concept_cards actually shrinking net (unlike me400's +1.6%). formula-not-decoded markers confirmed fully eliminated (0/443 chunks), same clean win as me400 Finding 1.

### Finding 2 — role-tagging, justification grounding, and pair-level stability all look as healthy as me400's

- definition/assumption/na/example %: OLD 27.2/66.9/1.5/4.4, NEW 26.9/65.1/3.2/4.8 — within noise, same as me400.
- justification grounding: OLD **98.7%**, NEW **99.6%** — both higher than me400's 96-97%, no regression.
- 969 concept-pairs judged in both runs; **98.1%** agreement among the 470 where both assert a non-null relation — comparable to me400's 96.9% and to the 2026-07-06 14B-vs-32B cross-model baseline (92.8%). No corruption signal at the pair-stability level, same conclusion as me400.

### Finding 3 — unlike me400, most of the new pairpacket volume resolves to "no relation," and part_of count actually shrinks

| Relation type | OLD | NEW | Delta |
|---|---|---|---|
| depends_on | 905 | 1,029 | +124 |
| part_of | 128 | 93 | -35 |
| none | 678 | 1,073 | +395 (82% of the total +484 pair growth) |

me400's equivalent breakdown (this same date, entry above) went the other way: none DECREASED (-146) while depends_on absorbed nearly all the growth (+457). For me200, **82%** of the new pairpacket volume is being judged "no relation" rather than becoming a new edge, and part_of shrinks in absolute terms despite more pairs overall being formed. Net new relation edges: +124 depends_on, -35 part_of = **+89 net**, out of +484 new pairs — a much lower conversion rate than me400's (+460 net relations out of +314 new pairs, i.e. more relations were created than net new pairs, because me400's none bucket also shrank).

### Finding 4 — root cause: formula/code enrichment's VLM recognition is visibly noisier on handwritten source pages than on me400's typed slides

Sampled the 157/443 chunks (35%) containing new `$$` LaTeX blocks directly (this is a far higher fraction of chunks than me400's, consistent with handwritten engineering notes being more equation-dense per page). Quality is a mixed bag, clearly worse on average than the me400 formula samples in the 2026-07-07 entry:

- Some formulas are clean and correct, e.g. chunk 0072: `\dot{Q} = -0.2[1-e^{(-0.05t)}]` — a legible first-order decay expression.
- Many are visibly garbled/non-parseable, e.g. chunk 0132: `\frac{1}{tan(20)}\sqrt{3}(state)(2), T_2 = \frac{P_2 v_2}{R} O` — text and math fragments interleaved with no coherent structure; chunk 0261's Carnot-efficiency-shaped formula has mismatched/duplicated subscripts; a MASS_BALANCE evidence sample showed `\frac{dE}{dt} = \underbrace{\dot{Q}}_{\widehat{A}_1-\dot{Q}_{A2}} - \dot{y} S e d y \underbrace{...}` — not a coherent physical equation.
- The surrounding plain-text OCR in these same chunks is also garbled ("cydles", "reservoixs", "α pey mole babis") — but that's the pre-existing handwriting-OCR problem from the 2026-07-04 entry, not new from enrichment. What IS new here is that the formula-recognition VLM inherits the same handwriting-vs-print quality gap: `CodeFormulaModel` re-recognizes a cropped image of the FORMULA-labeled region (per the 2026-07-06 Finding 5 mechanics, unchanged), and a handwritten equation crop is a much harder recognition target than me400's typed/printed equations were.

This plausibly explains Finding 3 above: if a large fraction of the newly recovered formula evidence is only partially legible, the relation-judging LLM is (correctly) declining to assert a confident relation from it more often than it does on me400's cleaner recovered formulas, rather than hallucinating structure from noise — a graceful-degradation read, not a corruption one, but it does mean the "more pairs" growth here is lower quality/yield than me400's equivalent growth.

### Finding 5 — concept-label quality remains clean despite noisier evidence

Sampled only-new concept_id labels directly (160 total, 108 formula-linked + 7 code-linked + 45 neither, same methodology as me400 Finding 4): labels read as legitimate thermo/heat-transfer concepts (CARNOT_CYCLES, CONVECTION_HEAT_TRANSFER, ADIABATIC_INDEX, COMPRESSIBLE_SYSTEMS, CONVERGING_DIVERGING_NOZZLE, 1_D_CONDUCTION, etc.) — no garbled/junk labels observed, consistent with the me400 finding and the 2026-07-05/06 OCR-typo-robustness findings: the concept-extraction step tolerates noisy underlying text/evidence without manufacturing garbage concept names, even when (per Finding 4) the formula content backing those concepts is itself partly illegible. **55/183** only-old concepts (30%) have no fuzzy match in the only-new set (vs. me400's 49/183, 27%) — comparable background churn, not a new regression.

### Bottom line

Formula/code enrichment closes the same completeness gap on me200 as it did on me400 (0 formula-not-decoded markers, plausible new concept labels, stable pair-level agreement, no corruption propagating into relation judgments) — so it is not actively harmful here either, and the earlier per-course `_scan`/`_text` tagging (2026-07-06 entry) does not need to change on account of this test. But the YIELD is meaningfully worse: ~3x the relative pairpacket growth of me400 converts mostly into "no relation" judgments (82% of new pairs) rather than new edges, because the underlying formula recognition is measurably noisier on handwritten source pages than on me400's typed slides — the same fundamental handwriting-vs-print gap already documented for OCR text in the 2026-07-04 entry now also shows up in the VLM-based formula/code recognition path. Practical implication for future scanned/handwritten courses (e.g. me320, also `_scan`-tagged): expect enrichment to add graph volume with a good chunk resolving to no-op judgments rather than me400-level relation yield — still worth leaving on (no evidence of harm, some genuine recovered concepts/relations), but temper yield expectations rather than assuming me400's ~+18-net-concepts/+17-relations profile will repeat on scanned material.

---

## [2026-07-08] VLLM_MAX_MODEL_LEN=16384 promoted from manual override to a relation_judger.py default; parameters now settled, next courses are production rollout, not comparisons

### Change made

`relation_judger.py` now sets `os.environ.setdefault("VLLM_MAX_MODEL_LEN", "16384")` right before constructing `_LLM_CLIENT`, rather than requiring `VLLM_MAX_MODEL_LEN=16384` to be passed manually on the command line every time (per the 2026-07-07 entry's "will fold into the code default once we see whether 16384 survives the run" — it has now survived two full clean runs, me400 and me200, with no token-overflow crash).

Deliberately NOT changed: `adapters.py`'s own default (still 8192). Bumping that global default would also raise `max_model_len` for `llm.py`'s per-chunk-x-concept calls (thousands per course), which the 2026-07-07 entry already established don't need the bigger ceiling (worst measured case 8,406 tokens) and would pay a real cost for no benefit — vLLM's KV-cache-backed concurrency drops roughly 28x -> 7x going from 8192 to 32768 on this box's fixed KV-cache pool. `relation_judger.py` is imported lazily inside `run_relation_judger()` (`main.py:273`), not at `main.py`'s top-level imports, so this `setdefault` only ever fires when the relations step actually runs — confirmed it can't leak into `llm.py`'s engine load even when both stages run in the same `main.py` process. `setdefault` (not a hard assignment) so an explicit env var still overrides it, e.g. if a future course's evidence aggregation needs more than 16384 (the 2026-07-07 entry flagged a theoretical 19,615-token ceiling from the two largest known me400 chunks co-occurring, not yet observed in practice).

### Where things stand — parameters now settled across both material types

Per-course auto-detected via filename tag (2026-07-06 entry):

- **force_full_page_ocr:** True for `_scan` (handwritten/no native text layer), False for `_text` (born-digital).
- **do_formula_enrichment / do_code_enrichment:** hardcoded True always (`ingest.py`) — validated as a net positive on both a `_text` course (me400, 2026-07-08 entry) and a `_scan` course (me200, 2026-07-08 entry): closes the formula/diagram invisibility gap on both, no corruption propagating into relation-judgment stability (96.9%/98.1% pair agreement respectively), though yield is lower on scanned/handwritten material since the formula-recognition VLM inherits the same handwriting-vs-print quality gap already known from OCR (2026-07-04 entry).
- **Model:** Qwen2.5-14B-Instruct (2026-07-06 4-signal comparison across sql/me200/me400) — not re-validated against enriched data specifically, but role-tagging %/relation-type ratio/justification-grounding all stayed stable across the enrichment on/off comparisons for both me400 and me200, i.e. the signals that drove the model pick aren't sensitive to enrichment being on. Judged not worth a fresh 3-model bake-off on enriched output given that stability.
- **VLLM_MAX_MODEL_LEN:** 16384 default for the relations step specifically (this entry), 8192 elsewhere (unchanged).

### Plan — remaining courses are rollout, not further comparison

Two courses have data on disk but zero pipeline output yet:

- `data/me320/ME320_CombinedNotes_scan.pdf` — `_scan`, same category as me200 (handwritten). No new settings question here — run with the settled parameters above. Expect a similar profile to me200's 2026-07-08 entry (formula-not-decoded fully eliminated, but a good chunk of the added pairpacket volume likely resolving to no-relation judgments rather than me400-level net-new-relation yield, since it's also handwritten source material). Worth a quick post-run sanity check (relations.jsonl line count vs. run.log's reported write count, formula-placeholder count = 0) but not a full repeat study.
- `data/tam251/TAM251_CombinedNotes_text.pdf` — `_text`, same category as me400 (typed/born-digital). Same reasoning — run with settled parameters, expect an me400-like profile, quick sanity check only.
- `data/sql/3-SQL2-JOINS_Nulls_text.pdf` has old output (`out/sql/`) from the pre-enrichment 2026-07-06 model comparison but hasn't been rerun with current settings. Low priority: only 1 formula-not-decoded marker existed in the whole course (2026-07-06 placeholder-count entry), so enrichment has ~nothing to recover there either way.

---

## [2026-07-08] me320 + tam251 rollout sanity check clean — but surfaced a pipeline-wide bug: concept_cards.jsonl is never regenerated against the min_unique_chunks filter, so 76-93% of every course's "concept cards" are orphaned (zero relations, below the pipeline's own evidence threshold)

### Context

Ran the me320/tam251 rollout planned in the entry above (settled parameters, no new comparison needed). Quick sanity check per that plan: `relations.jsonl` line count vs. run.log's reported write count, formula-not-decoded placeholder count.

### Finding 1 — both rollout runs are clean

- **me320** (`out/me320/Qwen14B_ffp_enriched/`): 1295/1295 pairpackets->relations match run.log's "Wrote 1295 new records"; 0/424 chunks have a formula-not-decoded placeholder.
- **tam251** (`out/tam251/Qwen14B_enriched/`): 95/95 pairpackets->relations match run.log's "Wrote 95 new records"; 0/73 chunks have a formula-not-decoded placeholder.

depends_on:part_of ratios (me320 629:48 ~13:1, tam251 59:7 ~8.4:1) land in the same range as the me200 (~11:1) and me400 (~7.85:1) baselines — no red flags on relation-type discrimination.

### Finding 2 — concept_cards.jsonl count (289) exceeds mentions.jsonl count (94) for tam251, which is structurally impossible if cards are built from mentions as documented

`llm.py`'s `build_concept_cards()` groups its output strictly from the `mentions` list passed in (`by_concept` = group by concept_id, one card per key) — card count can never exceed unique concept_ids in mentions. But tam251's `concept_cards.jsonl` has **289 unique concept_ids** while `mentions.jsonl` has only **94 records / 21 unique concept_ids** — 268 cards reference concepts with zero records in the mentions file sitting next to them.

### Root cause — main.py's llm step filters mentions.jsonl AFTER cards are built, and never rebuilds cards

`main.py`'s "llm" step (around line 423-436):

1. `run_llm_and_cards()` writes `mentions.jsonl` (unfiltered) and, using that same unfiltered list, builds and writes `concept_cards.jsonl`.
2. `main.py` then re-reads `mentions.jsonl`, calls `filter_mentions_by_min_unique_chunks(mentions, min_unique_chunks=3)` — drops any concept appearing in fewer than 3 distinct chunks — and overwrites `mentions.jsonl` with the filtered set.

`concept_cards.jsonl` is never regenerated against the filtered list. File mtimes confirm the ordering directly: tam251's `concept_cards.jsonl` finished writing at 03:24:09, `mentions.jsonl`'s filtered rewrite finished at 03:24:14, 5 seconds later.

Clustering and pairpackets both read `mentions_path` (the on-disk, filtered file) per `main.py`'s step wiring, so the actual graph (clusters/pairpackets/relations) is NOT corrupted by this — only `concept_cards.jsonl` is stale.

### Finding 3 — this is not tam251-specific, it's present in every course run to date, confirmed via each run.log's own "[filter] concepts kept=X/Y" line

| Course | concept_cards.jsonl | kept after filter | orphaned |
|---|---|---|---|
| me200_ffp_enriched | 867 | 142 | 83.6% |
| me400_enriched (8191_chunks) | 1,136 | 274 | 75.9% |
| me320 | 889 | 118 | 86.7% |
| tam251 | 289 | 21 | 92.7% |

**76-93%** of every course's `concept_cards.jsonl` is for concepts below the pipeline's own `min_unique_chunks=3` evidence bar and absent from `relations.jsonl` entirely (relations only ever reference concepts that survived the filter, since pairpackets is built from the filtered `mentions.jsonl`). This does NOT invalidate the 2026-07-08 me400/me200 enrichment A/B comparisons above — both sides of each comparison used the same unfiltered `concept_cards.jsonl` consistently, so the relative deltas still hold — but it does mean `concept_cards.jsonl` as a standalone artifact has never matched the filtered graph on any course run so far.

### Downstream impact — not yet measured

`knowledge_graph_visualization.ipynb` and `students_mapping.ipynb` consume `concept_cards.jsonl` directly (per CLAUDE.md). Any code path in those notebooks that builds a node list or joins against `concept_cards.jsonl` will include hundreds of orphaned, zero-relation concept nodes per course. Not yet checked whether either notebook already filters these out independently (e.g. by only rendering concepts that appear in `relations.jsonl`) or whether they currently display the inflated/orphaned set as-is.

### Status — logged, not fixed

**Decision:** log now, fix later, to keep momentum on bringing in remaining course content first. Candidate fix (not yet implemented): rebuild/rewrite `concept_cards.jsonl` from the filtered `mentions.jsonl` immediately after the `min_unique_chunks` filter runs in `main.py`'s llm step (~line 436), or emit a separate `concept_cards_filtered.jsonl` and point the notebooks at that instead. Affects every existing course's `concept_cards.jsonl` (me200, me400, me320, sql, tam251) — a fix should include regenerating those, not just changing behavior for future runs, if the notebooks are to be trusted.
