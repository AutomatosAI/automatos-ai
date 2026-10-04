/**
 * #942 — the platform tools a CLI-runtime agent's sessions get, as `GET /api/agents/{id}` reports them.
 *
 * A session always has the core tools. On top of them the owner picks tool groups per agent,
 * stored as `agent.configuration.session_tool_groups` (absent = every group). The detail also
 * reports what the agent's sessions cannot reach: a skill that names an API agent's tool, or a
 * workspace capability (a connected database, a built Knowledge Graph, …) no enabled group covers.
 */

/** The group ids the backend accepts; any other id is refused with a 422. */
export type SessionToolGroupId = 'data' | 'graph' | 'documents' | 'playbooks' | 'missions' | 'reports'

/** One group the owner can turn on for this agent's sessions. */
export interface SessionToolGroup {
  id: SessionToolGroupId | string
  label: string
  description: string
  /** the session tools the group adds */
  tools: string[]
}

/** `session_tool_groups` on the agent detail; absent on a backend older than #942. */
export interface SessionToolGroups {
  /** the groups on for this agent (or the `?groups=` override of one preview response) */
  enabled: string[]
  /** true when the agent stores no choice, so every group is on */
  is_default: boolean
  /** every group, in display order */
  available: SessionToolGroup[]
}

/**
 * PRD-245 S1.5: a skill of the agent whose body calls tools a session lacks under that name.
 * `kind` arrives from #942 on; an older backend sends the same shape without it.
 */
export interface SkillToolGap {
  kind?: 'skill'
  skill: string
  tools?: string[] | null
  instead?: Record<string, string> | null
}

/** #942: something this workspace has connected that the agent's sessions cannot use. */
export interface WorkspaceCapabilityGap {
  kind: 'workspace'
  /** e.g. 'database', 'graph', 'playbooks' */
  capability: string
  /** the group that would cover it */
  group: string
  /** one sentence for the owner, written by the backend */
  message: string
  fix?: { enable_group?: string | null } | null
}

/** One entry of the agent detail's `session_tool_gaps`. */
export type SessionToolGapEntry = SkillToolGap | WorkspaceCapabilityGap
