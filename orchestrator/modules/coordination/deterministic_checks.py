"""
Deterministic Checker — PRD-82A Sequential Mission Coordinator
================================================================

Validates task output against deterministic criteria before burning an LLM
verification call. If any must_pass check fails, short-circuits with an
immediate FAIL verdict.

8 check types:
  format_regex, min_length, max_length, required_sections,
  json_schema, url_valid, contains_keywords, word_count_range

Source: PRD-103 Section 4 (deterministic checks)
        PRD-82A Section 11 (verification guardrails)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckFailure:
    """A single deterministic check that failed."""

    check_type: str
    description: str
    must_pass: bool


@dataclass(frozen=True)
class DeterministicResult:
    """Immutable result of all deterministic checks on a task output."""

    passed: bool
    failures: List[CheckFailure] = field(default_factory=list)
    short_circuited: bool = False  # True if a must_pass check failed


# ---------------------------------------------------------------------------
# Check handler type
# ---------------------------------------------------------------------------

# handler(output, value, criterion) -> Optional[str]
# Returns None on success, or a failure description string.
CheckHandler = Callable[[str, Any, Dict[str, Any]], Optional[str]]


# ---------------------------------------------------------------------------
# DeterministicChecker
# ---------------------------------------------------------------------------


class DeterministicChecker:
    """
    Runs deterministic checks on task output against verification criteria.

    Each criterion dict: {'type': '<check_type>', 'value': <check_value>, 'must_pass': bool}
    If must_pass is True and the check fails, short-circuits immediately.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, CheckHandler] = {
            "format_regex": self._check_format_regex,
            "min_length": self._check_min_length,
            "max_length": self._check_max_length,
            "required_sections": self._check_required_sections,
            "json_schema": self._check_json_schema,
            "url_valid": self._check_url_valid,
            "contains_keywords": self._check_contains_keywords,
            "word_count_range": self._check_word_count_range,
        }

    def check(
        self,
        output: str,
        criteria: Optional[List[Dict[str, Any]]],
    ) -> DeterministicResult:
        """
        Run all deterministic checks on the output.

        Args:
            output: The task output text to validate.
            criteria: List of criterion dicts, each with 'type', 'value',
                      and optional 'must_pass' (defaults to False).

        Returns:
            DeterministicResult with pass/fail and failure details.
        """
        if not criteria:
            return DeterministicResult(passed=True)

        failures: List[CheckFailure] = []

        for criterion in criteria:
            check_type = criterion.get("type", "")
            check_value = criterion.get("value")
            must_pass = criterion.get("must_pass", False)

            handler = self._handlers.get(check_type)
            if handler is None:
                logger.warning(
                    "Unknown deterministic check type: %s (skipping)",
                    check_type,
                )
                continue

            try:
                failure_desc = handler(output, check_value, criterion)
            except Exception:
                logger.error(
                    "Error running deterministic check %s",
                    check_type,
                    exc_info=True,
                )
                failure_desc = f"Check '{check_type}' raised an exception"

            if failure_desc is not None:
                check_failure = CheckFailure(
                    check_type=check_type,
                    description=failure_desc,
                    must_pass=must_pass,
                )
                failures.append(check_failure)

                # Short-circuit on must_pass failure
                if must_pass:
                    logger.info(
                        "must_pass check failed (short-circuit): %s — %s",
                        check_type,
                        failure_desc,
                    )
                    return DeterministicResult(
                        passed=False,
                        failures=failures,
                        short_circuited=True,
                    )

        passed = len(failures) == 0
        return DeterministicResult(passed=passed, failures=failures)

    # -----------------------------------------------------------------------
    # Individual check handlers
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_format_regex(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output matches the provided regex pattern."""
        if not isinstance(value, str):
            return "format_regex value must be a string pattern"
        try:
            if not re.search(value, output, re.DOTALL):
                return f"Output does not match regex pattern: {value}"
        except re.error as e:
            return f"Invalid regex pattern '{value}': {e}"
        return None

    @staticmethod
    def _check_min_length(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output meets minimum character count."""
        try:
            min_len = int(value)
        except (TypeError, ValueError):
            return f"min_length value must be an integer, got: {value}"
        actual = len(output)
        if actual < min_len:
            return (
                f"Output length {actual} is below minimum {min_len} characters"
            )
        return None

    @staticmethod
    def _check_max_length(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output does not exceed maximum character count."""
        try:
            max_len = int(value)
        except (TypeError, ValueError):
            return f"max_length value must be an integer, got: {value}"
        actual = len(output)
        if actual > max_len:
            return (
                f"Output length {actual} exceeds maximum {max_len} characters"
            )
        return None

    @staticmethod
    def _check_required_sections(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output contains all required markdown section headers."""
        if not isinstance(value, list):
            return "required_sections value must be a list of section headers"
        missing = []
        for section in value:
            # Match markdown headers: # Section, ## Section, ### Section
            # Also match the section text without the # prefix
            pattern = rf"^#{1,6}\s+{re.escape(section)}\s*$"
            if not re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
                # Also try matching the exact header string (e.g., "## Summary")
                if section not in output:
                    missing.append(section)
        if missing:
            return f"Missing required sections: {', '.join(missing)}"
        return None

    @staticmethod
    def _check_json_schema(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output is valid JSON conforming to provided schema."""
        if not isinstance(value, dict):
            return "json_schema value must be a JSON schema dict"

        # Try to parse output as JSON
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            return f"Output is not valid JSON: {e}"

        # Basic schema validation (type checking without jsonschema dependency)
        schema_type = value.get("type")
        if schema_type == "object" and not isinstance(parsed, dict):
            return f"Expected JSON object, got {type(parsed).__name__}"
        if schema_type == "array" and not isinstance(parsed, list):
            return f"Expected JSON array, got {type(parsed).__name__}"

        # Check required properties
        required = value.get("required", [])
        if isinstance(parsed, dict) and required:
            missing = [k for k in required if k not in parsed]
            if missing:
                return f"Missing required JSON properties: {', '.join(missing)}"

        # Check property types
        properties = value.get("properties", {})
        if isinstance(parsed, dict) and properties:
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            for prop_name, prop_schema in properties.items():
                if prop_name in parsed:
                    expected_type = prop_schema.get("type")
                    if expected_type and expected_type in type_map:
                        python_type = type_map[expected_type]
                        if not isinstance(parsed[prop_name], python_type):
                            return (
                                f"Property '{prop_name}' expected type "
                                f"'{expected_type}', got "
                                f"'{type(parsed[prop_name]).__name__}'"
                            )

        return None

    @staticmethod
    def _check_url_valid(
        output: str,
        _value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that all URLs in output are well-formed."""
        # Extract URLs using a simple pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]*'
        urls = re.findall(url_pattern, output)

        if not urls:
            # No URLs to validate — pass
            return None

        invalid = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    invalid.append(url)
            except Exception:
                invalid.append(url)

        if invalid:
            truncated = invalid[:5]
            return (
                f"Invalid URLs found: {', '.join(truncated)}"
                + (f" (and {len(invalid) - 5} more)" if len(invalid) > 5 else "")
            )
        return None

    @staticmethod
    def _check_contains_keywords(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that output contains all listed keywords."""
        if not isinstance(value, list):
            return "contains_keywords value must be a list of keywords"
        output_lower = output.lower()
        missing = [kw for kw in value if kw.lower() not in output_lower]
        if missing:
            return f"Missing required keywords: {', '.join(missing)}"
        return None

    @staticmethod
    def _check_word_count_range(
        output: str,
        value: Any,
        _criterion: Dict[str, Any],
    ) -> Optional[str]:
        """Check that word count is within [min, max] bounds."""
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return "word_count_range value must be [min, max]"
        try:
            min_words = int(value[0])
            max_words = int(value[1])
        except (TypeError, ValueError):
            return f"word_count_range bounds must be integers, got: {value}"

        word_count = len(output.split())
        if word_count < min_words:
            return (
                f"Word count {word_count} is below minimum {min_words}"
            )
        if word_count > max_words:
            return (
                f"Word count {word_count} exceeds maximum {max_words}"
            )
        return None
