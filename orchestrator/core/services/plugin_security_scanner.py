"""
Plugin Security Scanner
=======================

Two-stage security scanning for marketplace plugins:
  Stage 1: Static pattern-based analysis (fast, free)
  Stage 2: LLM-based deep analysis via Claude Haiku (US-006)
"""

import re
import logging
from typing import Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blocked pattern definitions
# ---------------------------------------------------------------------------

BLOCKED_CODE_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"\bimport\s+subprocess\b", "type": "dangerous_import", "severity": "critical", "description": "Imports subprocess module for shell command execution"},
    {"pattern": r"\bfrom\s+subprocess\s+import\b", "type": "dangerous_import", "severity": "critical", "description": "Imports from subprocess module"},
    {"pattern": r"\bsubprocess\.(run|call|Popen|check_output|check_call|getoutput|getstatusoutput)\b", "type": "dangerous_call", "severity": "critical", "description": "Calls subprocess for shell execution"},
    {"pattern": r"\bexec\s*\(", "type": "dangerous_call", "severity": "critical", "description": "Uses exec() to execute arbitrary code"},
    {"pattern": r"\beval\s*\(", "type": "dangerous_call", "severity": "critical", "description": "Uses eval() to evaluate arbitrary expressions"},
    {"pattern": r"\b__import__\s*\(", "type": "dangerous_call", "severity": "critical", "description": "Uses __import__() for dynamic module loading"},
    {"pattern": r"\bcompile\s*\(", "type": "dangerous_call", "severity": "high", "description": "Uses compile() which can prepare code for exec/eval"},
    {"pattern": r"\bos\.system\s*\(", "type": "dangerous_call", "severity": "critical", "description": "Uses os.system() for shell command execution"},
    {"pattern": r"\bos\.popen\s*\(", "type": "dangerous_call", "severity": "critical", "description": "Uses os.popen() for shell command execution"},
    {"pattern": r"\bos\.exec", "type": "dangerous_call", "severity": "critical", "description": "Uses os.exec* for process replacement"},
    {"pattern": r"\bctypes\b", "type": "dangerous_import", "severity": "high", "description": "Uses ctypes for low-level memory access"},
    {"pattern": r"\bimport\s+pickle\b", "type": "dangerous_import", "severity": "high", "description": "Imports pickle which can execute arbitrary code on deserialization"},
    {"pattern": r"\bfrom\s+pickle\s+import\b", "type": "dangerous_import", "severity": "high", "description": "Imports from pickle module"},
]

BLOCKED_NETWORK_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"\bimport\s+requests\b", "type": "network_access", "severity": "high", "description": "Imports requests library for HTTP access"},
    {"pattern": r"\bfrom\s+requests\s+import\b", "type": "network_access", "severity": "high", "description": "Imports from requests library"},
    {"pattern": r"\bimport\s+urllib\b", "type": "network_access", "severity": "high", "description": "Imports urllib for HTTP access"},
    {"pattern": r"\bfrom\s+urllib\b", "type": "network_access", "severity": "high", "description": "Imports from urllib"},
    {"pattern": r"\bimport\s+socket\b", "type": "network_access", "severity": "high", "description": "Imports socket for raw network access"},
    {"pattern": r"\bfrom\s+socket\s+import\b", "type": "network_access", "severity": "high", "description": "Imports from socket module"},
    {"pattern": r"\bimport\s+http\.client\b", "type": "network_access", "severity": "high", "description": "Imports http.client for HTTP access"},
    {"pattern": r"\bimport\s+aiohttp\b", "type": "network_access", "severity": "high", "description": "Imports aiohttp for async HTTP access"},
    {"pattern": r"\bimport\s+httpx\b", "type": "network_access", "severity": "high", "description": "Imports httpx for HTTP access"},
]

BLOCKED_FS_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"\bopen\s*\(.+['\"]w['\"]", "type": "file_write", "severity": "high", "description": "Opens file in write mode"},
    {"pattern": r"\bopen\s*\(.+['\"]a['\"]", "type": "file_write", "severity": "medium", "description": "Opens file in append mode"},
    {"pattern": r"\bos\.remove\s*\(", "type": "file_delete", "severity": "high", "description": "Deletes a file via os.remove()"},
    {"pattern": r"\bos\.unlink\s*\(", "type": "file_delete", "severity": "high", "description": "Deletes a file via os.unlink()"},
    {"pattern": r"\bshutil\.rmtree\s*\(", "type": "file_delete", "severity": "critical", "description": "Recursively deletes directory tree"},
    {"pattern": r"\bos\.rmdir\s*\(", "type": "file_delete", "severity": "high", "description": "Removes a directory"},
    {"pattern": r"\bos\.makedirs\s*\(", "type": "file_delete", "severity": "medium", "description": "Creates directories on the filesystem"},
    {"pattern": r"\bpathlib\.Path\(.+\.write_", "type": "file_write", "severity": "high", "description": "Writes to file via pathlib"},
]

PROMPT_INJECTION_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "type": "prompt_injection", "severity": "critical", "description": "Attempts to override system instructions"},
    {"pattern": r"ignore\s+your\s+instructions", "type": "prompt_injection", "severity": "critical", "description": "Attempts to override system instructions"},
    {"pattern": r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", "type": "prompt_injection", "severity": "critical", "description": "Attempts to override system instructions"},
    {"pattern": r"jailbreak", "type": "prompt_injection", "severity": "critical", "description": "Contains jailbreak reference"},
    {"pattern": r"system\s+prompt\s+override", "type": "prompt_injection", "severity": "critical", "description": "Attempts to override system prompt"},
    {"pattern": r"override\s+(the\s+)?system\s+prompt", "type": "prompt_injection", "severity": "critical", "description": "Attempts to override system prompt"},
    {"pattern": r"exfiltrate", "type": "prompt_injection", "severity": "critical", "description": "References data exfiltration"},
    {"pattern": r"hidden\s+instruction", "type": "prompt_injection", "severity": "critical", "description": "Contains hidden instruction reference"},
    {"pattern": r"you\s+are\s+now\s+(a|an|in)", "type": "prompt_injection", "severity": "high", "description": "Attempts to redefine AI identity"},
    {"pattern": r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", "type": "prompt_injection", "severity": "critical", "description": "Attempts to remove safety restrictions"},
    {"pattern": r"pretend\s+(you\s+are|to\s+be)\s+.*(unrestricted|unfiltered)", "type": "prompt_injection", "severity": "critical", "description": "Attempts to bypass filters via role play"},
    {"pattern": r"do\s+not\s+follow\s+your\s+(guidelines|rules|policy)", "type": "prompt_injection", "severity": "critical", "description": "Attempts to make AI ignore guidelines"},
]

# All pattern groups combined for iteration
ALL_PATTERN_GROUPS = [
    BLOCKED_CODE_PATTERNS,
    BLOCKED_NETWORK_PATTERNS,
    BLOCKED_FS_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
]


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class StaticFinding(BaseModel):
    """A single finding from static analysis."""
    type: str = Field(..., description="Category of the finding")
    severity: str = Field(..., description="critical, high, medium, or low")
    file: str = Field(..., description="File path within plugin")
    line: int = Field(..., description="Line number where pattern was found")
    pattern: str = Field(..., description="Regex pattern that matched")
    matched_text: str = Field(..., description="The text that matched the pattern")
    description: str = Field(..., description="Human-readable description of the issue")


class StaticScanResult(BaseModel):
    """Result of the static security scan."""
    status: str = Field(..., description="passed or flagged")
    findings: List[StaticFinding] = Field(default_factory=list, description="List of findings")


# ---------------------------------------------------------------------------
# Static scan implementation
# ---------------------------------------------------------------------------

async def static_scan(plugin_files: Dict[str, str]) -> StaticScanResult:
    """Run static pattern-based security analysis on plugin files.

    Args:
        plugin_files: Mapping of file path -> file content (text files only).

    Returns:
        StaticScanResult with status='passed' if clean, 'flagged' if issues found.
    """
    findings: List[StaticFinding] = []

    for file_path, content in plugin_files.items():
        lines = content.split("\n")
        for line_num, line_text in enumerate(lines, start=1):
            for pattern_group in ALL_PATTERN_GROUPS:
                for pat_def in pattern_group:
                    match = re.search(pat_def["pattern"], line_text, re.IGNORECASE)
                    if match:
                        findings.append(
                            StaticFinding(
                                type=pat_def["type"],
                                severity=pat_def["severity"],
                                file=file_path,
                                line=line_num,
                                pattern=pat_def["pattern"],
                                matched_text=match.group(0),
                                description=pat_def["description"],
                            )
                        )

    status = "passed" if not findings else "flagged"
    logger.info(
        "Static scan complete: status=%s, findings=%d", status, len(findings)
    )
    return StaticScanResult(status=status, findings=findings)
