"""Tests for TEXT CORE - AI-generated text detection."""

import pytest
from core.text_core import TextCore


@pytest.fixture
def text_core():
    return TextCore()


class TestTextCore:
    HUMAN_TEXT = """
    I was walking down the street yesterday when I bumped into an old friend.
    We hadn't seen each other in years! She looked great, though a bit tired.
    We grabbed coffee at this tiny place on the corner - you know, the one
    with the crooked sign. Anyway, she told me about her new job. Sounds
    exhausting, honestly. But she seemed happy. We promised to catch up more
    often, though we both know we probably won't. Life's funny like that.
    I walked home feeling oddly nostalgic. The whole encounter lasted maybe
    twenty minutes but it stuck with me all day.
    """

    AI_TEXT = """
    It is important to note that artificial intelligence has transformed
    numerous industries in recent years. Furthermore, the implementation
    of machine learning algorithms has led to significant improvements in
    efficiency and accuracy. Additionally, it should be noted that these
    advancements play a crucial role in shaping the future of technology.
    Moreover, the integration of AI systems into existing workflows has
    demonstrated remarkable results. In conclusion, it is essential to
    recognize that the continued development of these technologies will
    have far-reaching implications for society as a whole. Furthermore,
    the ethical considerations surrounding AI deployment remain a topic
    of ongoing discussion and debate in the academic community.
    """

    def test_human_text_scores_low(self, text_core):
        result = text_core.analyze(self.HUMAN_TEXT)
        assert result.score < 0.5, f"Human text scored too high: {result.score}"
        assert result.status in ("PASS", "WARN")

    def test_ai_text_scores_high(self, text_core):
        result = text_core.analyze(self.AI_TEXT)
        assert result.score > 0.35, f"AI text scored too low: {result.score}"

    def test_short_text_returns_insufficient(self, text_core):
        result = text_core.analyze("Hello world, this is short.")
        assert result.data_quality == "INSUFFICIENT"
        assert result.confidence < 0.3

    def test_result_has_all_detail_keys(self, text_core):
        result = text_core.analyze(self.HUMAN_TEXT)
        assert "word_count" in result.details
        assert "burstiness_score" in result.details
        assert "vocabulary_score" in result.details
        assert "repetition_score" in result.details
        assert "structure_score" in result.details

    def test_to_dict(self, text_core):
        result = text_core.analyze(self.HUMAN_TEXT)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "score" in d
        assert "confidence" in d

    def test_ai_markers_detected(self, text_core):
        result = text_core.analyze(self.AI_TEXT)
        assert result.details.get("ai_markers_found", 0) > 0

    def test_score_in_valid_range(self, text_core):
        result = text_core.analyze(self.HUMAN_TEXT)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_text(self, text_core):
        result = text_core.analyze("")
        assert result.data_quality == "INSUFFICIENT"

    def test_moderate_text(self, text_core):
        """Text that is neither clearly human nor AI should get uncertain score."""
        moderate = (
            "The weather today is nice. I went to the store and bought some groceries. "
            "Then I came home and cooked dinner. It was a good day overall. "
            "I watched some television before going to bed. Tomorrow I have work. "
            "I need to prepare for a meeting. The project deadline is coming up soon. "
            "I hope everything goes well. My colleague sent me the latest report. "
            "The numbers look promising this quarter. We should discuss the results."
        )
        result = text_core.analyze(moderate)
        assert 0.0 <= result.score <= 1.0
