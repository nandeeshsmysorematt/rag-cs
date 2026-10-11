"""Central configuration. Defaults follow Section 4.3 of the paper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGCSConfig:
    # --- Models (Sec. 4.3). Pin the *_revision fields to commit hashes for
    # --- exact reproducibility; None means "latest on the Hub".
    dpr_question_model: str = "facebook/dpr-question_encoder-single-nq-base"
    dpr_question_revision: Optional[str] = None
    dpr_context_model: str = "facebook/dpr-ctx_encoder-single-nq-base"
    dpr_context_revision: Optional[str] = None
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    nli_revision: Optional[str] = None
    spacy_model: Optional[str] = "en_core_web_sm"

    # --- Retrieval
    k: int = 5                      # retrieval depth
    embedding_dim: int = 768
    max_query_tokens: int = 256     # query = [q ; s_i]; q is truncated, s_i kept
    max_passage_encode_tokens: int = 256
    prepend_title_for_nli: bool = True

    # --- NLI scoring
    max_nli_tokens: int = 512
    max_passage_tokens: int = 450   # passages are right-truncated to this
    nli_batch_size: int = 32
    # "sentence_first": input is [CLS] s_i [SEP] p_j [SEP]   (as written in the paper)
    # "premise_first": input is [CLS] p_j [SEP] s_i [SEP]    (standard NLI direction)
    nli_input_order: str = "sentence_first"

    # --- Segmentation / aggregation
    min_sentence_tokens: int = 5
    tau: float = 0.0                # neutrality threshold

    # --- Runtime
    device: Optional[str] = None    # None -> cuda if available else cpu
    fp16: bool = True               # half precision on CUDA only
    seed: int = 42
