"""RAG-CS: Retrieval-Augmented Consistency Scoring for hallucination detection.

Heavy dependencies (torch, transformers, faiss) are imported lazily so that the
lightweight modules (features, classifier, segmentation) can be used and tested
without a GPU stack.
"""

__version__ = "0.1.0"
__all__ = ["RAGCS", "RAGCSConfig", "__version__"]


def __getattr__(name):
    if name == "RAGCS":
        from .pipeline import RAGCS

        return RAGCS
    if name == "RAGCSConfig":
        from .config import RAGCSConfig

        return RAGCSConfig
    raise AttributeError(f"module 'ragcs' has no attribute {name!r}")
