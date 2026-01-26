# clustering.py
"""
Minimal per-concept unsupervised context clustering using sentence-transformers + (optional) UMAP + HDBSCAN.

Reads:
  - out/chunks.jsonl   (chunk_id -> chunk text)
  - out/mentions.jsonl (from llm.py: concept_id + chunk_id, etc.)

Writes:
  - out/context_clusters.jsonl (one record per concept_id) with ONLY:

    {
      "concept_id": "LEFT_OUTER_JOIN",
      "context_clusters": [
        { "cluster_id": 4, "count_chunks": 6, "label_hint": "joins-and-null-semantics" },
        { "cluster_id": 1, "count_chunks": 3, "label_hint": "query-examples" }
      ]
    }

Notes:
- We deduplicate contexts by (concept_id, chunk_id) so each chunk counts once per concept.
- label_hint is a compact slug derived from c-TF-IDF top terms for that cluster.

Install deps:
  pip install sentence-transformers scikit-learn
  pip install umap-learn hdbscan   # recommended if you want real clustering beyond fallback

Run:
  python clustering.py \
    --chunks out/chunks.jsonl \
    --mentions out/mentions.jsonl \
    --out out/context_clusters.jsonl \
    --use-umap

If you don't pass --use-umap, the script will fallback to a single cluster per concept
(keeps behavior safe/deterministic).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("clustering")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ----------------------------
# JSONL I/O
# ----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    # supports JSON list or JSONL
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{path} is JSON but not a list.")
            return data
    return read_jsonl(path)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Text helpers
# ----------------------------

def slugify_tokens(tokens: List[str], max_tokens: int = 4) -> str:
    """
    Make a compact label like "joins-and-null-semantics" from term tokens.
    - uses first 2 tokens as "<a>-and-<b>" when possible
    - then appends remaining tokens with "-"
    """
    toks: List[str] = []
    for t in tokens:
        t = (t or "").lower()
        t = re.sub(r"[^a-z0-9]+", "", t)
        if not t:
            continue
        if t in toks:
            continue
        toks.append(t)
        if len(toks) >= max_tokens:
            break

    if not toks:
        return "misc"
    if len(toks) == 1:
        return toks[0]
    if len(toks) == 2:
        return f"{toks[0]}-and-{toks[1]}"
    return f"{toks[0]}-and-{toks[1]}-" + "-".join(toks[2:])


def terms_to_label_hint(terms: List[str], max_tokens: int = 4) -> str:
    """
    Convert c-TF-IDF terms (which may be multiword ngrams) into a stable-ish slug.
    Example terms: ["null semantics", "outer join", "query example"]
    -> tokens ["null","semantics","outer","join"] -> "null-and-semantics-outer-join"
    """
    flat_tokens: List[str] = []
    for term in terms or []:
        for w in (term or "").split():
            if not w:
                continue
            flat_tokens.append(w)
            if len(flat_tokens) >= 12:  # cap before slugify
                break
        if len(flat_tokens) >= 12:
            break
    return slugify_tokens(flat_tokens, max_tokens=max_tokens)


# ----------------------------
# Context building from llm.py outputs
# ----------------------------

def build_chunks_index(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        cid = ch.get("chunk_id")
        if cid is None:
            continue
        by_id[str(cid)] = ch
    return by_id


def build_concept_to_chunk_ids(mentions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Deduplicate by (concept_id, chunk_id). We only need chunk IDs per concept for clustering.
    """
    seen = set()
    out: Dict[str, List[str]] = defaultdict(list)
    for m in mentions:
        concept_id = m.get("concept_id")
        chunk_id = m.get("chunk_id")
        if not concept_id or not chunk_id:
            continue
        key = (str(concept_id), str(chunk_id))
        if key in seen:
            continue
        seen.add(key)
        out[str(concept_id)].append(str(chunk_id))
    return out


def concept_context_texts(
    concept_id: str,
    chunk_ids: List[str],
    chunks_by_id: Dict[str, Dict[str, Any]],
    *,
    include_concept_in_text: bool = False,
    concept_label: Optional[str] = None,
) -> List[str]:
    texts: List[str] = []
    for cid in chunk_ids:
        ch = chunks_by_id.get(cid, {})
        t = (ch.get("text") or "").strip()
        if not t:
            continue
        if include_concept_in_text:
            label = concept_label or concept_id
            t = f"{t}\n\nCONCEPT: {label}"
        texts.append(t)
    return texts


# ----------------------------
# ML bits (embedding + optional UMAP + HDBSCAN + c-TF-IDF)
# ----------------------------

def _import_or_die():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.preprocessing import normalize
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies. Install:\n"
            "  pip install sentence-transformers scikit-learn\n"
            "Optional but recommended for clustering:\n"
            "  pip install umap-learn hdbscan\n"
        ) from e

    try:
        import umap  # type: ignore
    except Exception:
        umap = None

    try:
        import hdbscan  # type: ignore
    except Exception:
        hdbscan = None

    return SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan


def embed_texts(texts: List[str], model_name: str, batch_size: int, normalize_embeddings: bool):
    SentenceTransformer, np, *_ = _import_or_die()
    model = SentenceTransformer(model_name)
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32", copy=False)
    return X


def reduce_umap(X, *, n_neighbors: int, n_components: int):
    *_ , umap, _ = _import_or_die()[5:]  # just to keep lint quiet
    # Actually fetch proper tuple:
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    if umap is None:
        raise RuntimeError("umap-learn not installed. Install: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(X)


def cluster_hdbscan(Xr, *, min_cluster_size: int, min_samples: Optional[int]):
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    if hdbscan is None:
        raise RuntimeError("hdbscan not installed. Install: pip install hdbscan")
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    return clusterer.fit_predict(Xr)


def ctfidf_terms_per_cluster(cluster_docs: List[str], top_terms: int) -> List[List[str]]:
    SentenceTransformer, np, CountVectorizer, TfidfTransformer, normalize, umap, hdbscan = _import_or_die()
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    counts = vec.fit_transform(cluster_docs)
    tfidf = TfidfTransformer(norm=None).fit_transform(counts)
    terms = vec.get_feature_names_out()

    out: List[List[str]] = []
    for i in range(tfidf.shape[0]):
        row = tfidf[i].toarray().ravel()
        if row.size == 0:
            out.append([])
            continue
        idx = row.argsort()[-top_terms:][::-1]
        out.append([str(terms[j]) for j in idx if row[j] > 0])
    return out


# ----------------------------
# Per-concept clustering (minimal output)
# ----------------------------

def cluster_concept(
    concept_id: str,
    texts: List[str],
    *,
    embedding_model: str,
    batch_size: int,
    normalize_embeddings: bool,
    use_umap: bool,
    umap_neighbors: int,
    umap_components: int,
    min_contexts_to_cluster: int,
    min_cluster_size: Optional[int],
    min_samples: Optional[int],
    top_terms: int,
) -> Dict[str, Any]:
    """
    Returns ONLY:
      { "concept_id": ..., "context_clusters": [ {cluster_id, count_chunks, label_hint}, ... ] }
    """
    n = len(texts)
    if n == 0:
        return {"concept_id": concept_id, "context_clusters": []}

    # Always embed (even if we later fallback) so behavior is consistent if you enable UMAP/HDBSCAN.
    X = embed_texts(texts, model_name=embedding_model, batch_size=batch_size, normalize_embeddings=normalize_embeddings)

    # Fallback: too few contexts => single cluster
    if n < min_contexts_to_cluster or not use_umap:
        return {
            "concept_id": concept_id,
            "context_clusters": [
                {"cluster_id": 0, "count_chunks": n, "label_hint": "misc"}
            ],
        }

    # UMAP reduction
        # PCA reduction (UMAP replacement to avoid llvmlite/numba)
    n_components = min(umap_components, n, X.shape[1])
    Xr = PCA(n_components=n_components, random_state=42).fit_transform(X)


    # HDBSCAN clustering
    mcs = min_cluster_size if min_cluster_size is not None else max(3, int(round(0.05 * n)))
    labels = cluster_hdbscan(Xr, min_cluster_size=mcs, min_samples=min_samples)

    # Group contexts by label
    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[int(lab)].append(i)

    # Ignore noise for labeling; if everything is noise, treat as one cluster
    non_noise = sorted([lab for lab in by_label.keys() if lab != -1])
    if not non_noise:
        return {
            "concept_id": concept_id,
            "context_clusters": [
                {"cluster_id": 0, "count_chunks": n, "label_hint": "misc"}
            ],
        }

    # c-TF-IDF per cluster for label_hint
    cluster_docs = [" ".join(texts[i] for i in by_label[lab]) for lab in non_noise]
    term_lists = ctfidf_terms_per_cluster(cluster_docs, top_terms=top_terms)

    context_clusters: List[Dict[str, Any]] = []
    for lab, terms in zip(non_noise, term_lists):
        context_clusters.append(
            {
                "cluster_id": int(lab),
                "count_chunks": len(by_label[lab]),
                "label_hint": terms_to_label_hint(terms),
            }
        )

    # Sort biggest first (useful downstream)
    context_clusters.sort(key=lambda x: -x["count_chunks"])

    return {"concept_id": concept_id, "context_clusters": context_clusters}