# Step 1 later: docling -> chunks with metadata -> write out/chunks.jsonl
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter
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


def pdf_to_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Convert one PDF to chunk records (text + metadata).
    Output is designed to be stable and extensible.
    """
    pdf_path = str(pdf_path)
    lecture_id = Path(pdf_path).stem

    converter = DocumentConverter()
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


def ingest_pdfs(pdf_paths: List[str], out_name: str = "chunks.jsonl") -> str:
    all_records: List[Dict[str, Any]] = []
    for p in pdf_paths:
        print(f"📄 Ingesting: {p}")
        recs = pdf_to_chunks(p)
        print(f"   -> {len(recs)} chunks")
        all_records.extend(recs)

    out_path = str(Path(OUT_DIR) / out_name)
    write_jsonl(all_records, out_path)
    print(f"✅ Wrote {len(all_records)} chunks to: {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: put PDFs in a folder named "data/"
    data_dir = Path("data")
    pdfs = sorted([str(p) for p in data_dir.glob("*.pdf")])

    if not pdfs:
        raise SystemExit("❌ No PDFs found. Put your PDFs in ./data/")

    ingest_pdfs(pdfs)
