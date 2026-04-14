"""
Hallucination detection using vectara/hallucination_evaluation_model.

The model scores (premise, hypothesis) pairs in [0, 1] where higher means
more consistent (less likely to be a hallucination). We split the answer
into sentences, score each sentence against the full context, and report
a per-sentence breakdown plus a conservative overall score (minimum).
"""

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.api.schemas import HallucinationResult, SentenceScore
from app.utils.text import split_sentences

logger = logging.getLogger(__name__)

_MODEL_ID = "vectara/hallucination_evaluation_model"


class HallucinationChecker:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading hallucination model '%s' on %s …", _MODEL_ID, device)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )
        self._model.to(device).eval()
        self._device = device
        logger.info("Hallucination model ready.")

    def check(self, answer: str, context_chunks: list[str]) -> HallucinationResult:
        """
        Score each sentence in *answer* against the joined *context_chunks*.

        Args:
            answer: The generated answer text.
            context_chunks: List of raw chunk texts used as the grounding context.

        Returns:
            HallucinationResult with per-sentence scores and conservative overall.

        Raises:
            ValueError: If context_chunks is empty (cannot check ungrounded answers).
        """
        if not context_chunks:
            raise ValueError("context_chunks must not be empty")

        sentences = split_sentences(answer)
        if not sentences:
            return HallucinationResult(overall_score=1.0, consistent=True, sentences=[])

        premise = "\n\n".join(context_chunks)
        pairs = [[premise, sentence] for sentence in sentences]

        with torch.no_grad():
            scores = self._model.predict(pairs)

        # scores may be a tensor or list — normalise to plain floats
        if isinstance(scores, torch.Tensor):
            float_scores = scores.tolist()
        else:
            float_scores = list(scores)

        # Clamp to [0, 1] in case model output drifts slightly outside range
        float_scores = [max(0.0, min(1.0, float(s))) for s in float_scores]

        sentence_scores = [
            SentenceScore(
                sentence=sent,
                score=round(score, 4),
                flagged=score < 0.5,
            )
            for sent, score in zip(sentences, float_scores)
        ]

        overall = min(s.score for s in sentence_scores)
        return HallucinationResult(
            overall_score=round(overall, 4),
            consistent=overall >= 0.5,
            sentences=sentence_scores,
        )
