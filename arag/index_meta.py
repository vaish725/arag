"""index_meta.json: records how the vector indexes were built.

Written by the index builders, read at load time so an index built with one
embedding model is never searched with query vectors from another.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Optional

META_FILENAME = "index_meta.json"


class IndexMismatchError(RuntimeError):
    """The configured embedding model does not match the built indexes."""


def meta_path(data_dir: str) -> str:
    return os.path.join(data_dir or ".", META_FILENAME)


def read_meta(data_dir: str) -> Optional[Dict[str, Any]]:
    path = meta_path(data_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(data_dir: str, meta: Dict[str, Any]) -> None:
    os.makedirs(data_dir or ".", exist_ok=True)
    with open(meta_path(data_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


def record_build(
    data_dir: str,
    info: Dict[str, Any],
    dim: int,
    fingerprint: Optional[Dict[str, Any]] = None,
    **counts: int,
) -> None:
    """Record a finished index build (``num_chunks=`` and/or ``num_sentences=``).

    If the embedding model differs from the one already recorded, the counts of
    the index that was NOT rebuilt are cleared: that index is stale.
    """
    meta = read_meta(data_dir) or {}
    same_model = (
        meta.get("provider") == info["provider"]
        and meta.get("model") == info["model"]
        and meta.get("dimension") == dim
    )
    if not same_model:
        meta["num_chunks"] = None
        meta["num_sentences"] = None
    meta.update(
        provider=info["provider"],
        model=info["model"],
        dimension=dim,
        build_date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    if fingerprint is not None:
        meta["model_revision"] = fingerprint["revision"]
        meta["model_files"] = fingerprint["files"]
    meta.update(counts)
    _write(data_dir, meta)


def record_wikipedia_revisions(data_dir: str, revisions: Dict[str, int]) -> None:
    meta = read_meta(data_dir) or {}
    meta.setdefault("wikipedia_revisions", {}).update(revisions)
    _write(data_dir, meta)


def check_compat(
    data_dir: str,
    info: Dict[str, Any],
    chunk_index: Any = None,
    sentence_index: Any = None,
    fingerprint: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise IndexMismatchError if the loaded indexes don't fit ``info``.

    ``info`` is the configured embedding setup ({provider, model, dim}).
    Does nothing when no vector index is loaded.
    """
    if chunk_index is None and sentence_index is None:
        return
    configured = f"{info['provider']}/{info['model']} (dim {info['dim']})"
    meta = read_meta(data_dir)
    if meta is None:
        raise IndexMismatchError(
            f"No {META_FILENAME} next to the vector indexes in {data_dir!r}, so their embedding "
            f"model is unknown. Configured: {configured}. Rebuild the indexes."
        )
    built = f"{meta.get('provider')}/{meta.get('model')} (dim {meta.get('dimension')})"
    if (
        meta.get("provider") != info["provider"]
        or meta.get("model") != info["model"]
        or (info["dim"] is not None and meta.get("dimension") != info["dim"])
    ):
        raise IndexMismatchError(
            f"Embedding mismatch: indexes were built with {built} but the configured embedding "
            f"is {configured}. Set ARAG_EMBEDDING_PROVIDER/ARAG_EMBEDDING_MODEL to match, or rebuild."
        )
    if fingerprint is not None and meta.get("model_files"):
        if meta.get("model_revision") != fingerprint["revision"]:
            raise IndexMismatchError(
                f"The embedding model revision differs: indexes were built with "
                f"{meta.get('model_revision')} but the loaded model is {fingerprint['revision']}. Rebuild."
            )
        built_files, loaded_files = meta["model_files"], fingerprint["files"]
        changed = sorted(n for n in set(built_files) | set(loaded_files) if built_files.get(n) != loaded_files.get(n))
        if changed:
            raise IndexMismatchError(
                f"The embedding model files differ from the ones the indexes were built with "
                f"(sha256 mismatch: {', '.join(changed)}). Re-download the model or rebuild the indexes."
            )
    for label, index, count_key in (
        ("chunk", chunk_index, "num_chunks"),
        ("sentence", sentence_index, "num_sentences"),
    ):
        if index is None:
            continue
        if index.d != meta.get("dimension"):
            raise IndexMismatchError(
                f"The {label} FAISS index has dimension {index.d} but {META_FILENAME} says "
                f"{meta.get('dimension')} ({built}). It was built with a different model; rebuild."
            )
        if meta.get(count_key) != index.ntotal:
            raise IndexMismatchError(
                f"The {label} FAISS index has {index.ntotal} vectors but {META_FILENAME} records "
                f"{meta.get(count_key)}. It is stale or was built with a different model; rebuild."
            )
