# Step 1 later: docling -> chunks with metadata -> write out/chunks.jsonl
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrAutoOptions
from docling.chunking import HybridChunker

from config import MAX_TOKENS, OUT_DIR


def _safe_pages(chunk) -> List[int]:
    """Try to extract page numbers from docling chunk provenance."""
    pages = set()
    try:
        for item in chunk.meta.doc_items:
            for prov in item.prov:
                if getattr(prov, "page_no", None) is not None:
                    pages.add(int(prov.page_no))
    except Exception:
        pass
    return sorted(pages) if pages else []


# [JR] new helper: infer scan vs. text from a "_scan"/"_text" filename tag,
# so switching between courses doesn't require hand-editing this file
def _is_scanned_from_filename(pdf_path: str) -> bool:
    tokens = Path(pdf_path).stem.lower().split("_")
    return "scan" in tokens


def pdf_to_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Convert one PDF to chunk records (text + metadata).
    Output is designed to be stable and extensible.
    """
    pdf_path = str(pdf_path)
    lecture_id = Path(pdf_path).stem

    # [JR] force_full_page_ocr: must be set on ocr_options, not PdfPipelineOptions
    # directly (pydantic silently drops unknown top-level kwargs, so this failed
    # silently before). Now auto-detected from the filename tag above instead
    # of hardcoded.
    #   False (no tag, or "_text") -- text-based material (typed slides, text
    #     PDFs). Docling keeps the native/embedded text layer and only OCRs
    #     bitmap regions its layout model flags (>5% of page area by default).
    #     Safe: doesn't touch already-clean text.
    #   True ("_scan") -- scanned/handwritten material with little to no usable
    #     native text layer. Forces OCR on the entire page, but this DISCARDS
    #     native text cells wherever they exist -- fine when there's ~nothing
    #     to lose (e.g. ME200), actively harmful on typed material (re-OCRs and
    #     corrupts already-good text; see ME400 case in findings.txt).
    is_scanned = _is_scanned_from_filename(pdf_path)
    print(f"   [ingest] {lecture_id}: force_full_page_ocr={is_scanned}")

    # [JR] enable docling's formula/code recognition (CodeFormulaV2 model) so
    # equations and code blocks get real text instead of a "formula-not-decoded"
    # placeholder -- see findings.txt 2026-07-06 entry (687-689 unrendered
    # formula placeholders per ME200/ME400 doc). NOT purely additive: any
    # region the layout model labels FORMULA or CODE gets unconditionally
    # re-recognized from an image crop and overwrites the existing text there,
    # even if that region already had good native/OCR text (e.g. a
    # CoolProp/EES numeric-output screenshot) -- see findings.txt 2026-07-07
    # entry for the ME400 chunk-diff that found this, including a handful of
    # cases where re-recognition silently altered numeric values. Only text
    # OUTSIDE FORMULA/CODE-labeled regions is untouched.
    # To restore docling's default (no enrichment), set both back to False.
    ENABLE_FORMULA_ENRICHMENT = True
    ENABLE_CODE_ENRICHMENT = True

    pdf_pipeline_options = PdfPipelineOptions(
        ocr_options=OcrAutoOptions(force_full_page_ocr=is_scanned),
        do_formula_enrichment=ENABLE_FORMULA_ENRICHMENT,
        do_code_enrichment=ENABLE_CODE_ENRICHMENT,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )
    # [JR] end force_full_page_ocr block

    result = converter.convert(pdf_path)
    dl_doc = result.document

    chunker = HybridChunker(max_tokens=MAX_TOKENS, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=dl_doc))

    records: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        pages = _safe_pages(ch)
        rec = {
            "lecture_id": lecture_id,
            "source_pdf": pdf_path,
            "chunk_id": f"{lecture_id}__{i:04d}",
            "chunk_index": i,
            "page_numbers": pages,
            "text": ch.text,
            # Neighbor pointers (helps later for cross-chunk context)
            "prev_chunk_id": f"{lecture_id}__{i-1:04d}" if i > 0 else None,
            "next_chunk_id": f"{lecture_id}__{i+1:04d}" if i < len(chunks) - 1 else None,
        }
        records.append(rec)

    return records


def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ingest_pdfs(pdf_paths: List[str], out_name: str = "chunks.jsonl", out_dir: str = OUT_DIR) -> str:

    import re
    from pathlib import Path

    def _pdf_sort_key(p: str):
        """
        Natural sort:
        1) prefer explicit lecture/chapter-like numbers (lec/lecture/ch/chapter/lesson/week/module/unit)
        2) else use first number anywhere
        3) else fallback to name
        """
        name = Path(p).stem.lower()

        # (1) explicit sequence markers
        # examples: "lec 3", "lecture-03", "ch.2", "chapter_10", "week5", "module 7", "unit02"
        m = re.search(r"(?:lec|lecture|ch|chapter|chap|lesson|week|wk|module|unit)\s*[\.\-_:]*\s*(\d+)", name)
        if m:
            return (0, int(m.group(1)), name)

        # (2) first number anywhere (handles "03_intro", "part2", etc.)
        m2 = re.search(r"(\d+)", name)
        if m2:
            return (1, int(m2.group(1)), name)

        # (3) no numbers => stable fallback
        return (2, 10**9, name)

    pdf_paths = sorted([str(p) for p in pdf_paths], key=_pdf_sort_key)

    all_records: List[Dict[str, Any]] = []
    for p in pdf_paths:
        print(f"📄 Ingesting: {p}")
        recs = pdf_to_chunks(p)
        print(f"   -> {len(recs)} chunks")
        all_records.extend(recs)

    out_path = str(Path(out_dir) / out_name)

    write_jsonl(all_records, out_path)
    print(f"✅ Wrote {len(all_records)} chunks to: {out_path}")
    return out_path



if __name__ == "__main__":
    # Example: put PDFs in a folder named "data/"
    data_dir = Path("data/algo")
    pdfs = sorted([str(p) for p in data_dir.glob("*.pdf")])

    if not pdfs:
        raise SystemExit("❌ No PDFs found. Put your PDFs in ./data/")

    ingest_pdfs(pdfs)

