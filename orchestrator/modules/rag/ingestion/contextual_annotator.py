"""Contextual chunk annotations at ingestion (PRD-188 S2).

Anthropic's contextual-retrieval pattern (the dossier's "Adopt 2"): a cheap
model reads the parent document and writes a ~50-100-token situating preface
per chunk, prepended BEFORE embedding — so the vector knows what the chunk is
*about* beyond its ~500 characters. Because ``document_chunks.content`` stores
the annotated text, the tsvector maintenance trigger indexes the preface too,
which strengthens the S3 BM25 leg with the same one-time spend.

Contract:

* **Immutable** — returns NEW :class:`DocumentChunk` objects
  (``dataclasses.replace``); the caller's chunks are never mutated.
* **Idempotent** — a chunk whose metadata already carries
  ``contextual_annotation`` passes through untouched (no double-prepend on a
  re-run; the corpus reprocess path rebuilds chunks from raw text anyway).
* **Loud on failure** — an annotation-LLM error logs a WARNING and the chunk
  falls back to its raw text with ``contextual_annotation_error`` recorded in
  metadata. Never a silent ``except`` swallow, never a failed ingest: the
  annotation is an enhancement, the document always lands.
* The annotation itself is stored in chunk metadata
  (``contextual_annotation``) so it is inspectable and re-annotation is
  detectable.

Flag-gated by the caller (``config.RAG_CONTEXTUAL_ANNOTATIONS_ENABLED``,
default OFF until the existing background reprocess has re-annotated the
corpus — live retrieval must never see a half-annotated store).
"""
import asyncio
import logging
from dataclasses import replace
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ANNOTATION_METADATA_KEY = "contextual_annotation"
ANNOTATION_ERROR_KEY = "contextual_annotation_error"

# The parent document is prompt-cached across a document's chunks in a
# provisioned run (~$1.02/M doc tokens one-time, dossier §H); the hard cap
# below only guards degenerate inputs from blowing the annotation context.
MAX_PARENT_CHARS = 24_000

# Bounded concurrency: annotation is a per-chunk LLM call at ingestion time —
# parallel enough to not serialise a large document, bounded enough to not
# stampede the provider.
MAX_CONCURRENT_ANNOTATIONS = 5

# Anthropic's contextual-retrieval prompt, near-verbatim
# (anthropic.com/engineering/contextual-retrieval, via rag-retrieval §D).
_ANNOTATION_PROMPT = """<document>
{parent_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def _default_llm() -> Optional[Any]:
    """The same LLM seam the RAG module already uses (query_enhancer idiom),
    pinned to the cheap annotation model from canonical config."""
    try:
        from config import config as app_config
        from core.llm import create_llm_manager

        return create_llm_manager(
            service_name="rag", model=app_config.RAG_CONTEXTUAL_ANNOTATION_MODEL
        )
    except Exception as e:
        logger.warning(f"Contextual annotator: LLM manager unavailable ({e})")
        return None


async def _annotate_one(chunk, parent_text: str, llm, semaphore) -> Any:
    """Return a NEW chunk carrying its situating context (or the raw chunk with
    the failure recorded — loud, never silent, never fatal)."""
    metadata = dict(chunk.metadata or {})
    if metadata.get(ANNOTATION_METADATA_KEY):
        return chunk  # already annotated — idempotent passthrough

    prompt = _ANNOTATION_PROMPT.format(
        parent_text=parent_text[:MAX_PARENT_CHARS], chunk_text=chunk.content
    )
    try:
        async with semaphore:
            response = await llm.generate_response([{"role": "user", "content": prompt}])
        annotation = (getattr(response, "content", "") or "").strip()
    except Exception as e:
        logger.warning(
            f"Contextual annotation failed for doc {chunk.document_id} "
            f"chunk {chunk.chunk_index} — falling back to raw text: {e}"
        )
        return replace(chunk, metadata={**metadata, ANNOTATION_ERROR_KEY: str(e)})

    if not annotation:
        logger.warning(
            f"Contextual annotation empty for doc {chunk.document_id} "
            f"chunk {chunk.chunk_index} — falling back to raw text"
        )
        return replace(chunk, metadata={**metadata, ANNOTATION_ERROR_KEY: "empty annotation"})

    return replace(
        chunk,
        content=f"{annotation}\n\n{chunk.content}",
        metadata={**metadata, ANNOTATION_METADATA_KEY: annotation},
    )


async def annotate_chunks(chunks: List[Any], parent_text: str, llm_provider: Any = None) -> List[Any]:
    """Situate every chunk within its parent document (new objects, same order).

    ``llm_provider`` is injectable for tests and provisioned runs; the default
    resolves the rag-service LLM manager with the configured annotation model.
    With no LLM available the chunks pass through unchanged — loudly.
    """
    if not chunks:
        return chunks

    llm = llm_provider if llm_provider is not None else _default_llm()
    if llm is None:
        logger.warning(
            f"Contextual annotations enabled but no LLM available — "
            f"{len(chunks)} chunks embed raw"
        )
        return list(chunks)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANNOTATIONS)
    annotated = await asyncio.gather(
        *[_annotate_one(chunk, parent_text or "", llm, semaphore) for chunk in chunks]
    )

    ok = sum(1 for c in annotated if (c.metadata or {}).get(ANNOTATION_METADATA_KEY))
    logger.info(f"Contextual annotations: {ok}/{len(annotated)} chunks situated")
    return list(annotated)
