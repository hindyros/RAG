"""
Intent detection — classifies user queries before any retrieval happens.

Classification drives two decisions:
1. Whether to trigger the knowledge base at all (conversational → skip).
2. Which answer template to use (factual / list / comparison / table).

We use Mistral's JSON mode (response_format={"type": "json_object"}) to get
structured output.  The response is validated against our IntentResult schema
and falls back to sensible defaults on parse failures so that one bad LLM
response never crashes the whole pipeline.
"""

import json
import logging

from app.api.schemas import IntentResult, PrimaryIntent, SubIntent
from app.config import Settings
from app.llm.client import MistralClient
from app.llm.prompts.intent import build_intent_messages

logger = logging.getLogger(__name__)


class IntentDetector:
    def __init__(self, client: MistralClient, settings: Settings) -> None:
        self._client = client
        self._confidence_min = settings.intent_confidence_min

    async def detect(self, query: str) -> IntentResult:
        """
        Classify the query's intent.

        On any failure (JSON parse error, missing fields, low confidence),
        returns a safe default: knowledge_seeking / factual.  This errs on the
        side of doing retrieval rather than skipping it — a false negative
        (doing retrieval for a greeting) costs one extra API call; a false
        positive (skipping retrieval for a real question) returns no answer.
        """
        messages = build_intent_messages(query)
        try:
            raw = await self._client.chat(
                messages=messages,
                temperature=0.0,   # deterministic classification
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            result = self._parse(raw)
        except Exception as exc:
            logger.warning("Intent detection failed: %s — using default.", exc)
            result = self._default_intent()

        # If confidence is below the threshold, collapse to the safer default
        if result.confidence < self._confidence_min:
            logger.info(
                "Intent confidence %.2f < %.2f — defaulting to factual.",
                result.confidence, self._confidence_min,
            )
            result = IntentResult(
                primary=result.primary,
                sub=SubIntent.FACTUAL,
                confidence=result.confidence,
                reasoning=result.reasoning + " (sub-intent defaulted due to low confidence)",
            )

        logger.info(
            "Intent: primary=%s sub=%s confidence=%.2f",
            result.primary, result.sub, result.confidence,
        )
        return result

    def _parse(self, raw: str) -> IntentResult:
        """Parse the JSON response from the LLM into an IntentResult."""
        data = json.loads(raw)

        primary_str = data.get("primary", "knowledge_seeking")
        sub_str = data.get("sub", "factual")

        # Safely map strings to enums
        try:
            primary = PrimaryIntent(primary_str)
        except ValueError:
            primary = PrimaryIntent.KNOWLEDGE_SEEKING

        try:
            sub = SubIntent(sub_str)
        except ValueError:
            sub = SubIntent.FACTUAL

        return IntentResult(
            primary=primary,
            sub=sub,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    @staticmethod
    def _default_intent() -> IntentResult:
        return IntentResult(
            primary=PrimaryIntent.KNOWLEDGE_SEEKING,
            sub=SubIntent.FACTUAL,
            confidence=0.5,
            reasoning="Default intent (detection failed).",
        )
