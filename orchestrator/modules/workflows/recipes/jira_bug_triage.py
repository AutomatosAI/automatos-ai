"""
Jira Bug Triage Recipe (PRD-50: US-012, US-013)
================================================

Autonomous workflow that reads a Jira bug ticket, analyses the indexed
codebase for relevant files, uses an LLM to generate a fix plan, applies
the fix via GitHub, opens a PR, and updates the Jira ticket.

Steps:
  1. Read Ticket — JIRA_GET_ISSUE via ComposioToolExecutor
  2. Analyze     — CodeGraph symbol search for relevant code
  3. Plan        — LLM generates a fix plan from ticket + code context
  4. Clone & Fix — GITHUB_CREATE_BRANCH + GITHUB_CREATE_OR_UPDATE_FILE
  5. Open PR     — GITHUB_CREATE_PULL_REQUEST
  6. Update Jira — JIRA_UPDATE_ISSUE + JIRA_ADD_COMMENT with PR link
"""

from __future__ import annotations

import json
import logging
import re
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
    branch_name: str = ""
    pr_url: str = ""
    pr_number: int = 0
    error: str = ""
    steps_completed: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

class JiraBugTriageRecipe:
    """
    End-to-end bug triage: Read → Analyse → Plan → Fix → PR → Update Jira.

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

            # Step 4 — Clone & Fix: create branch and apply code changes
            logger.info("[triage] Step 4: Creating branch and applying fix for %s", issue_key)
            branch_name, file_changes = await self._step_clone_and_fix(
                ticket, fix_plan, symbols
            )
            result.branch_name = branch_name
            result.steps_completed.append("clone_and_fix")

            # Step 5 — Open PR
            logger.info("[triage] Step 5: Opening PR for %s", issue_key)
            pr_url, pr_number = await self._step_open_pr(
                ticket, branch_name, fix_plan
            )
            result.pr_url = pr_url
            result.pr_number = pr_number
            result.steps_completed.append("open_pr")

            # Step 6 — Update Jira: transition status + post PR link
            logger.info("[triage] Step 6: Updating Jira ticket %s", issue_key)
            await self._step_update_jira(issue_key, pr_url, pr_number)
            result.steps_completed.append("update_jira")

            result.success = True
            logger.info("[triage] Completed full triage for %s (PR: %s)", issue_key, pr_url)

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
    # Step 4 — Clone & Fix: create branch and apply code changes
    # ------------------------------------------------------------------

    async def _step_clone_and_fix(
        self,
        ticket: Dict[str, Any],
        fix_plan: str,
        symbols: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Create a fix branch and apply code changes based on the plan.

        Returns (branch_name, list of file change dicts).
        """
        from core.composio.tool_executor import ComposioToolExecutor

        issue_key = ticket.get("key", "UNKNOWN")
        branch_name = f"fix/{issue_key.lower()}"
        repo = self._resolve_repo()
        executor = ComposioToolExecutor(self.db)

        # Create the fix branch from default branch
        create_branch_result = await executor.execute(
            action="GITHUB_CREATE_BRANCH",
            params={
                "owner": repo["owner"],
                "repo": repo["repo"],
                "new_branch_name": branch_name,
            },
            agent_id=0,
            workspace_id=self.workspace_id,
            app_name="GITHUB",
            skip_validation=True,
        )

        if not create_branch_result.get("success"):
            raise RuntimeError(
                f"GITHUB_CREATE_BRANCH failed: {create_branch_result.get('error', 'unknown')}"
            )
        logger.info("[triage] Created branch %s", branch_name)

        # Ask LLM to generate concrete file changes as JSON
        file_changes = await self._generate_file_changes(
            ticket, fix_plan, symbols
        )

        if not file_changes:
            raise RuntimeError("LLM returned no file changes for the fix plan")

        # Apply each file change via GitHub API
        for change in file_changes:
            file_path = change.get("file_path", "")
            content = change.get("content", "")
            message = change.get("commit_message", f"fix({issue_key}): update {file_path}")

            if not file_path or not content:
                logger.warning("[triage] Skipping empty file change: %s", change)
                continue

            update_result = await executor.execute(
                action="GITHUB_CREATE_OR_UPDATE_FILE",
                params={
                    "owner": repo["owner"],
                    "repo": repo["repo"],
                    "path": file_path,
                    "message": message,
                    "content": content,
                    "branch": branch_name,
                },
                agent_id=0,
                workspace_id=self.workspace_id,
                app_name="GITHUB",
                skip_validation=True,
            )

            if not update_result.get("success"):
                raise RuntimeError(
                    f"GITHUB_CREATE_OR_UPDATE_FILE failed for {file_path}: "
                    f"{update_result.get('error', 'unknown')}"
                )
            logger.info("[triage] Updated file %s on branch %s", file_path, branch_name)

        return branch_name, file_changes

    async def _generate_file_changes(
        self,
        ticket: Dict[str, Any],
        fix_plan: str,
        symbols: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to produce concrete file changes as a JSON array.

        Each element: {"file_path": str, "content": str, "commit_message": str}
        """
        from core.llm.manager import create_llm_manager

        # Build context from relevant code snippets
        code_context_lines: List[str] = []
        for sym in symbols[:10]:
            snippet = (sym.get("code_snippet") or "")[:600]
            if snippet:
                code_context_lines.append(
                    f"--- {sym.get('file_path', '?')} ---\n{snippet}\n"
                )
        code_context = "\n".join(code_context_lines) if code_context_lines else "(no code context)"

        prompt = (
            f"Jira ticket {ticket.get('key')}: {ticket.get('summary')}\n\n"
            f"Fix plan:\n{fix_plan}\n\n"
            f"Relevant code:\n{code_context}\n\n"
            "Generate the concrete file changes to implement this fix. "
            "Return ONLY a JSON array where each element has:\n"
            '  {"file_path": "path/to/file", "content": "full file content after fix", '
            '"commit_message": "short commit message"}\n\n'
            "Return valid JSON only, no markdown fences."
        )

        llm = create_llm_manager(service_name="jira_triage")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. Given a bug fix plan "
                    "and existing code, produce the exact file changes needed. "
                    "Return ONLY a JSON array. Each element must have file_path, "
                    "content (the full updated file content), and commit_message."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = await llm.generate_response(messages)
        if not response or not response.content:
            raise RuntimeError("LLM returned empty file changes")

        return self._parse_file_changes(response.content)

    @staticmethod
    def _parse_file_changes(text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into a list of file change dicts."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting JSON array from the text
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise RuntimeError("Could not parse file changes JSON from LLM response")

        if not isinstance(parsed, list):
            parsed = [parsed]

        # Validate structure
        valid: List[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, dict) and item.get("file_path") and item.get("content"):
                valid.append(item)

        return valid

    # ------------------------------------------------------------------
    # Step 5 — Open PR
    # ------------------------------------------------------------------

    async def _step_open_pr(
        self,
        ticket: Dict[str, Any],
        branch_name: str,
        fix_plan: str,
    ) -> tuple[str, int]:
        """Create a pull request on GitHub. Returns (pr_url, pr_number)."""
        from core.composio.tool_executor import ComposioToolExecutor

        issue_key = ticket.get("key", "UNKNOWN")
        repo = self._resolve_repo()
        executor = ComposioToolExecutor(self.db)

        pr_title = f"[{issue_key}] {ticket.get('summary', 'Bug fix')}"[:256]
        pr_body = (
            f"## {issue_key} — Automated Bug Fix\n\n"
            f"**Summary:** {ticket.get('summary', 'N/A')}\n"
            f"**Priority:** {ticket.get('priority', 'N/A')}\n"
            f"**Type:** {ticket.get('issue_type', 'N/A')}\n\n"
            f"### Fix Plan\n{fix_plan[:3000]}\n\n"
            f"---\n_Generated by Automatos Bug Triage_"
        )

        result = await executor.execute(
            action="GITHUB_CREATE_PULL_REQUEST",
            params={
                "owner": repo["owner"],
                "repo": repo["repo"],
                "title": pr_title,
                "body": pr_body,
                "head": branch_name,
                "base": repo.get("default_branch", "main"),
            },
            agent_id=0,
            workspace_id=self.workspace_id,
            app_name="GITHUB",
            skip_validation=True,
        )

        if not result.get("success"):
            raise RuntimeError(
                f"GITHUB_CREATE_PULL_REQUEST failed: {result.get('error', 'unknown')}"
            )

        data = result.get("data", {})
        pr_url = data.get("html_url", data.get("url", ""))
        pr_number = data.get("number", 0)

        logger.info("[triage] Opened PR #%d: %s", pr_number, pr_url)
        return pr_url, pr_number

    # ------------------------------------------------------------------
    # Step 6 — Update Jira
    # ------------------------------------------------------------------

    async def _step_update_jira(
        self, issue_key: str, pr_url: str, pr_number: int
    ) -> None:
        """Transition the Jira ticket status and add a comment with the PR link."""
        from core.composio.tool_executor import ComposioToolExecutor

        executor = ComposioToolExecutor(self.db)

        # Transition ticket status (e.g., "In Progress" or "In Review")
        update_result = await executor.execute(
            action="JIRA_UPDATE_ISSUE",
            params={
                "issue_key": issue_key,
                "status": "In Progress",
            },
            agent_id=0,
            workspace_id=self.workspace_id,
            app_name="JIRA",
            skip_validation=True,
        )

        if not update_result.get("success"):
            # Status transition is non-fatal — the ticket may not have
            # an "In Progress" transition available in its workflow.
            logger.warning(
                "[triage] JIRA_UPDATE_ISSUE status transition failed for %s: %s",
                issue_key,
                update_result.get("error", "unknown"),
            )

        # Post PR link as a comment
        comment_body = (
            "{panel:title=Automatos Bug Triage — PR Opened|borderColor=#00875a}\n"
            f"A fix has been prepared and a pull request opened.\n\n"
            f"*Pull Request:* [PR #{pr_number}|{pr_url}]\n"
            f"*Branch:* fix/{issue_key.lower()}\n\n"
            "Please review the PR and merge when ready.\n"
            "{panel}"
        )
        await self._post_comment(issue_key, comment_body)

    # ------------------------------------------------------------------
    # Repository resolution
    # ------------------------------------------------------------------

    def _resolve_repo(self) -> Dict[str, str]:
        """Resolve the target GitHub repository for this workspace.

        Reads from environment variables (GITHUB_REPO_OWNER, GITHUB_REPO_NAME)
        or falls back to metadata on the envelope if available.
        """
        import os

        owner = os.getenv("GITHUB_REPO_OWNER", "")
        repo = os.getenv("GITHUB_REPO_NAME", "")
        default_branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

        if not owner or not repo:
            raise RuntimeError(
                "GITHUB_REPO_OWNER and GITHUB_REPO_NAME environment variables "
                "must be set for the Bug Triage recipe to create branches and PRs."
            )

        return {
            "owner": owner,
            "repo": repo,
            "default_branch": default_branch,
        }

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
