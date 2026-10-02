/**
 * The session CLIs by name: each one's label and its ``llm_usage.provider`` slug.
 *
 * A mirror of the backend registry (``orchestrator/core/cli_presets.py``), kept in
 * step by ``lib/__tests__/session-clis.test.ts``. The picker renders the live
 * registry from the backend; this is for the views that have no fetch of their
 * own (analytics rows, a host's line in Settings). An unknown CLI keeps its id.
 */

export interface SessionCli {
  label: string
  usageSlug: string
}

export const SESSION_CLIS: Record<string, SessionCli> = {
  claude: { label: 'Claude Code', usageSlug: 'claude_code' },
  codex: { label: 'Codex', usageSlug: 'codex' },
  copilot: { label: 'GitHub Copilot', usageSlug: 'copilot_cli' },
}

export function sessionCli(id: string | null | undefined): SessionCli {
  const key = String(id || '').trim().toLowerCase() || 'claude'
  return SESSION_CLIS[key] ?? { label: key, usageSlug: key }
}
