"""
Jira Bug Triage Recipe (PRD-50: US-012)
=======================================

Autonomous workflow that reads a Jira bug ticket, analyses the indexed
codebase for relevant files, uses an LLM to generate a fix plan, and
posts the plan back as a Jira comment.

Steps:
  1. Read Ticket — JIRA_GET_ISSUE via ComposioToolExecutor
  2. Analyze     — CodeGraph symbol search for relevant code
  3. Plan        — LLM generates a fix plan from ticket + code context
  Post          — JIRA_ADD_COMMENT with the fix plan (or failure summary)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.routing import RequestEnvelope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """Outcome of a single triage run."""
    success: bool
    issue_key: str = ""
    fix_plan: str = ""
    relevant_files: List[str] = field(default_factory=list)
    error: str = ""
    steps_completed: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

class JiraBugTriageRecipe:
    """
    End-to-end bug triage: Read ticket → Analyse code → Plan fix → Comment.

    Usage::

        recipe = JiraBugTriageRecipe()
        result = await recipe.execute(envelope, db, workspace_id)
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        envelope: RequestEnvelope,
        db: Session,
        workspace_id: UUID,
    ) -> TriageResult:
        """Run the full triage pipeline for *envelope*."""
        self.db = db
        self.workspace_id = workspace_id
        start = time.monotonic()
        result = TriageResult(success=False)
        issue_key = envelope.metadata.get("issue_key", "")
        result.issue_key = issue_key

        try:
            # Step 1 — Read ticket
            logger.info("[triage] Step 1: Reading ticket %s", issue_key)
            ticket = await self._step_read_ticket(envelope)
            result.steps_completed.append("read_ticket")

            # Step 2 — Analyse codebase
            logger.info("[triage] Step 2: Analysing codebase for %s", issue_key)
            symbols = await self._step_analyse(ticket)
            result.relevant_files = list({s.get("file_path", "") for s in symbols if s.get("file_path")})
            result.steps_completed.append("analyse")

            # Step 3 — Generate fix plan
            logger.info("[triage] Step 3: Generating fix plan for %s", issue_key)
            fix_plan = await self._step_plan(ticket, symbols)
            result.fix_plan = fix_plan
            result.steps_completed.append("plan")

            # Post fix plan as Jira comment
            logger.info("[triage] Posting fix plan to %s", issue_key)
            await self._post_comment(issue_key, fix_plan)
            result.steps_completed.append("comment")

            result.success = True
            logger.info("[triage] Completed triage for %s", issue_key)

        except Exception as exc:
            result.error = str(exc)
            logger.exception("[triage] Failed at step after %s: %s", result.steps_completed, exc)
            # Best-effort: post failure summary back to Jira
            await self._post_failure_comment(issue_key, result)

        result.execution_time_ms = (time.monotonic() - start) * 1000
        return result

    # ------------------------------------------------------------------
    # Step 1 — Read Ticket
    # ------------------------------------------------------------------

    async def _step_read_ticket(self, envelope: RequestEnvelope) -> Dict[str, Any]:
        """Fetch full issue details via Composio JIRA_GET_ISSUE."""
        from core.composio.tool_executor import ComposioToolExecutor

        issue_key = envelope.metadata.get("issue_key", "")
        if not issue_key:
            raise ValueError("envelope.metadata is missing 'issue_key'")

        executor = ComposioToolExecutor(self.db)
        result = await executor.execute(
            action="JIRA_GET_ISSUE",
            params={"issue_key": issue_key},
            agent_id=0,
            workspace_id=self.workspace_id,
            app_name="JIRA",
            skip_validation=True,
        )

        if not result.get("success"):
            raise RuntimeError(
                f"JIRA_GET_ISSUE failed for {issue_key}: {result.get('error', 'unknown')}"
            )

        data = result.get("data", {})
        fields = data.get("fields", data)

        return {
            "key": issue_key,
            "summary": fields.get("summary", envelope.metadata.get("summary", "")),
            "description": fields.get("description", envelope.metadata.get("description", "")),
            "issue_type": _extract_name(fields.get("issuetype")) or envelope.metadata.get("issue_type", ""),
            "priority": _extract_name(fields.get("priority")) or envelope.metadata.get("priority", ""),
            "project": _extract_key(fields.get("project")) or envelope.metadata.get("project", ""),
            "reporter": _extract_name(fields.get("reporter")) or envelope.metadata.get("reporter", ""),
            "labels": fields.get("labels", []),
            "status": _extract_name(fields.get("status", "")),
        }

    # ------------------------------------------------------------------
    # Step 2 — Analyse codebase
    # ------------------------------------------------------------------

    async def _step_analyse(self, ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search the indexed CodeGraph for symbols related to the ticket."""
        from modules.codegraph.codegraph_service import CodeGraphService

        keywords = self._extract_keywords(ticket)
        if not keywords:
            logger.warning("[triage] No keywords extracted from ticket %s", ticket.get("key"))
            return []

        code_graph = CodeGraphService(self.db)
        all_symbols: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for keyword in keywords:
            try:
                results = await code_graph.search_symbols(
                    project_name=ticket.get("project", ""),
                    query=keyword,
                    limit=5,
                    workspace_id=str(self.workspace_id),
                )
                for sym in results.get("results", []):
                    sym_id = sym.get("id")
                    if sym_id and sym_id not in seen_ids:
                        seen_ids.add(sym_id)
                        all_symbols.append(sym)
            except Exception as exc:
                logger.warning("[triage] CodeGraph search failed for '%s': %s", keyword, exc)

        logger.info("[triage] Found %d unique symbols across %d keyword searches", len(all_symbols), len(keywords))
        return all_symbols[:20]  # cap to keep LLM context manageable

    # ------------------------------------------------------------------
    # Step 3 — Plan fix via LLM
    # ------------------------------------------------------------------

    async def _step_plan(
        self, ticket: Dict[str, Any], symbols: List[Dict[str, Any]]
    ) -> str:
        """Ask the LLM for a concrete fix plan based on the ticket and code."""
        from core.llm.manager import create_llm_manager

        prompt = self._build_plan_prompt(ticket, symbols)
        llm = create_llm_manager(service_name="jira_triage")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer performing bug triage. "
                    "Given a Jira bug ticket and relevant code snippets, produce "
                    "a concise fix plan. Include: root cause hypothesis, files to "
                    "change, specific code changes, and a test strategy. "
                    "Format in Jira wiki markup."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = await llm.generate_response(messages)

        if not response or not response.content:
            raise RuntimeError("LLM returned empty fix plan")

        return response.content

    # ------------------------------------------------------------------
    # Jira comment helpers
    # ------------------------------------------------------------------

    async def _post_comment(self, issue_key: str, body: str) -> None:
        """Post *body* as a Jira comment on *issue_key*."""
        if not issue_key:
            logger.warning("[triage] Skipping comment — no issue key")
            return

        from core.composio.tool_executor import ComposioToolExecutor

        executor = ComposioToolExecutor(self.db)
        result = await executor.execute(
            action="JIRA_ADD_COMMENT",
            params={"issue_key": issue_key, "body": body},
            agent_id=0,
            workspace_id=self.workspace_id,
            app_name="JIRA",
            skip_validation=True,
        )

        if not result.get("success"):
            logger.error(
                "[triage] Failed to post comment on %s: %s",
                issue_key,
                result.get("error", "unknown"),
            )

    async def _post_failure_comment(self, issue_key: str, result: TriageResult) -> None:
        """Best-effort post of a failure summary to the Jira ticket."""
        if not issue_key:
            return
        body = (
            "{panel:title=Automatos Bug Triage — Failed|borderColor=#ff0000}\n"
            f"Triage could not be completed.\n\n"
            f"*Steps completed:* {', '.join(result.steps_completed) or 'none'}\n"
            f"*Error:* {result.error[:500]}\n"
            "{panel}"
        )
        try:
            await self._post_comment(issue_key, body)
        except Exception:
            logger.exception("[triage] Could not post failure comment on %s", issue_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(ticket: Dict[str, Any]) -> List[str]:
        """Pull search-worthy tokens from the ticket summary and description."""
        import re

        text = f"{ticket.get('summary', '')} {ticket.get('description', '')}"
        # Remove common Jira markup, URLs, and very short tokens
        text = re.sub(r"\{[^}]*\}", "", text)       # {noformat}, {code}, etc.
        text = re.sub(r"https?://\S+", "", text)     # URLs
        text = re.sub(r"[^a-zA-Z0-9_. ]+", " ", text)

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "that", "this", "it", "its", "and",
            "or", "but", "not", "no", "so", "if", "when", "then",
            "than", "also", "just", "only", "very", "too", "more",
        }

        words = text.lower().split()
        keywords = []
        seen: set = set()
        for w in words:
            w = w.strip(".")
            if len(w) < 3 or w in stopwords or w in seen:
                continue
            seen.add(w)
            keywords.append(w)

        return keywords[:10]

    @staticmethod
    def _build_plan_prompt(
        ticket: Dict[str, Any], symbols: List[Dict[str, Any]]
    ) -> str:
        """Build the user-role LLM prompt for Step 3."""
        lines = [
            "h3. Jira Ticket",
            f"*Key:* {ticket.get('key', 'N/A')}",
            f"*Summary:* {ticket.get('summary', 'N/A')}",
            f"*Type:* {ticket.get('issue_type', 'N/A')}",
            f"*Priority:* {ticket.get('priority', 'N/A')}",
            "",
            "*Description:*",
            ticket.get("description", "(no description)") or "(no description)",
            "",
        ]

        if symbols:
            lines.append("h3. Relevant Code")
            for sym in symbols[:10]:  # limit to keep prompt concise
                lines.append(
                    f"*{sym.get('symbol_type', '')}* `{sym.get('qualified_name', sym.get('name', ''))}`"
                    f" — {sym.get('file_path', '?')}:{sym.get('line_number', '?')}"
                )
                snippet = (sym.get("code_snippet") or "")[:600]
                if snippet:
                    lines.append("{code}")
                    lines.append(snippet)
                    lines.append("{code}")
                lines.append("")
        else:
            lines.append("_No relevant code symbols found in the index._")

        lines.append("")
        lines.append(
            "Please produce a fix plan covering: root cause, files to change, "
            "specific code changes, and a test strategy."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_name(obj: Any) -> str:
    """Safely extract 'name' or 'displayName' from a Jira field value."""
    if isinstance(obj, dict):
        return obj.get("name", "") or obj.get("displayName", "")
    if isinstance(obj, str):
        return obj
    return ""


def _extract_key(obj: Any) -> str:
    """Safely extract 'key' from a Jira project/entity dict."""
    if isinstance(obj, dict):
        return obj.get("key", "")
    if isinstance(obj, str):
        return obj
    return ""
