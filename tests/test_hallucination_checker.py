"""
Tests for HallucinationChecker.

All tests mock out AutoModelForSequenceClassification and AutoTokenizer before
importing the checker so no model weights are downloaded (~300 MB).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


# ── Shared mock fixture ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_transformers(monkeypatch):
    """Prevent real model download for all tests in this module."""
    mock_model_instance = MagicMock()

    with patch(
        "app.hallucination.checker.AutoModelForSequenceClassification"
    ) as mock_model_cls:
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        # Make .to() and .eval() chainable
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance

        yield mock_model_instance


# ── Tests ─────────────────────────────────────────────────────────────────

def test_empty_answer_returns_consistent(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    checker = HallucinationChecker()
    result = checker.check("", ["Some context."])

    assert result.overall_score == 1.0
    assert result.consistent is True
    assert result.sentences == []


def test_single_sentence_high_score(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    mock_transformers.predict.return_value = torch.tensor([0.92])

    checker = HallucinationChecker()
    result = checker.check("The sky is blue.", ["The sky appears blue due to Rayleigh scattering."])

    assert result.consistent is True
    assert len(result.sentences) == 1
    assert result.sentences[0].flagged is False
    assert abs(result.sentences[0].score - 0.92) < 0.01


def test_single_sentence_low_score(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    mock_transformers.predict.return_value = torch.tensor([0.3])

    checker = HallucinationChecker()
    result = checker.check("The sky is green.", ["The sky appears blue due to Rayleigh scattering."])

    assert result.consistent is False
    assert len(result.sentences) == 1
    assert result.sentences[0].flagged is True
    assert abs(result.sentences[0].score - 0.3) < 0.01


def test_overall_score_is_minimum(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    mock_transformers.predict.return_value = torch.tensor([0.9, 0.4, 0.8])

    checker = HallucinationChecker()
    answer = "First sentence is fine. Second sentence is sketchy. Third sentence is okay."
    result = checker.check(answer, ["Context about the topic."])

    assert abs(result.overall_score - 0.4) < 0.01
    assert result.consistent is False  # 0.4 < 0.5


def test_empty_context_raises_value_error(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    checker = HallucinationChecker()

    with pytest.raises(ValueError):
        checker.check("Some answer.", [])


def test_pairs_count_matches_sentence_count(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    scores = [0.8, 0.7, 0.9]
    mock_transformers.predict.return_value = torch.tensor(scores)

    checker = HallucinationChecker()
    answer = "Alpha. Beta. Gamma."
    result = checker.check(answer, ["Context here."])

    assert len(result.sentences) == len(scores)
    mock_transformers.predict.assert_called_once()
    pairs = mock_transformers.predict.call_args[0][0]
    assert len(pairs) == len(scores)


def test_context_joined_with_double_newline(mock_transformers):
    from app.hallucination.checker import HallucinationChecker

    mock_transformers.predict.return_value = torch.tensor([0.85])

    checker = HallucinationChecker()
    chunks = ["Chunk one.", "Chunk two.", "Chunk three."]
    checker.check("Single sentence answer.", chunks)

    mock_transformers.predict.assert_called_once()
    pairs = mock_transformers.predict.call_args[0][0]
    # premise is the first element of the first pair
    premise = pairs[0][0]
    assert "\n\n" in premise
    assert "Chunk one." in premise
    assert "Chunk three." in premise
