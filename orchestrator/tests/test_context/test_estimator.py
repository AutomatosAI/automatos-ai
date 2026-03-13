"""Unit tests for TokenEstimator."""

import pytest

from modules.context.estimator import TokenEstimator


class TestTokenEstimator:
    """Tests for the fast character-based token estimator."""

    def setup_method(self):
        self.estimator = TokenEstimator()

    # -- estimate() --

    def test_estimate_empty_string(self):
        assert self.estimator.estimate("") == 0

    def test_estimate_none_like_empty(self):
        """Empty string returns 0; callers should never pass None."""
        assert self.estimator.estimate("") == 0

    def test_estimate_short_text(self):
        # 12 chars -> 3 tokens
        assert self.estimator.estimate("hello world!") == 3

    def test_estimate_exact_multiple(self):
        text = "a" * 400  # exactly 100 tokens
        assert self.estimator.estimate(text) == 100

    def test_estimate_non_multiple(self):
        text = "a" * 401  # 401 // 4 = 100
        assert self.estimator.estimate(text) == 100

    def test_estimate_single_char(self):
        assert self.estimator.estimate("x") == 0  # 1 // 4 = 0

    def test_estimate_four_chars(self):
        assert self.estimator.estimate("abcd") == 1

    def test_estimate_realistic_prompt(self):
        """A realistic system prompt should produce a reasonable estimate."""
        prompt = (
            "You are Test Agent, an AI agent on the Automatos platform.\n"
            "Your role: assistant\n"
            "Workspace: TestWorkspace\n"
        ) * 10  # ~1000 chars
        estimate = self.estimator.estimate(prompt)
        assert 200 <= estimate <= 300

    # -- precise() --

    def test_precise_empty_string(self):
        assert self.estimator.precise("") == 0

    def test_precise_falls_back_gracefully(self):
        """precise() should return a positive number even without tiktoken."""
        result = self.estimator.precise("hello world this is a test")
        assert result > 0

    def test_precise_matches_estimate_order_of_magnitude(self):
        """precise() and estimate() should be in the same ballpark."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        fast = self.estimator.estimate(text)
        precise = self.estimator.precise(text)
        # Within 50% of each other (generous tolerance)
        assert abs(fast - precise) / max(fast, precise, 1) < 0.5
