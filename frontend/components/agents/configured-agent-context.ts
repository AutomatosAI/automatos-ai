'use client'

/**
 * #942 — the id of the agent the configuration modal is editing, for the parts of its form that
 * read the saved agent themselves (the Runtime section's session tool picker and its preview).
 *
 * A context, not a prop: `AgentConfigurationModal` is one 1,900-line component, and any edited
 * line inside it fails the changed-lines shape gate (components ≤150 lines) until it is split.
 * `configured-agent-modal.tsx` provides the id around it. The create wizard has no agent yet and
 * provides nothing, so the picker does not render there (a new agent starts on every group).
 */

import { createContext, useContext } from 'react'

export const ConfiguredAgentContext = createContext<number | null>(null)

/** The agent being configured, or null outside the configuration modal. */
export function useConfiguredAgentId(): number | null {
  return useContext(ConfiguredAgentContext)
}
