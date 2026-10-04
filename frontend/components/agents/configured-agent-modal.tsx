'use client'

/**
 * #942 — the agent configuration modal with the id of the agent it edits in context, so the
 * Runtime section can read that agent's session tool groups and preview a new selection
 * (see `configured-agent-context.ts` for why this is not a prop).
 */

import type { ComponentProps } from 'react'
import { AgentConfigurationModal as ConfigurationModal } from './agent-configuration-modal'
import { ConfiguredAgentContext } from './configured-agent-context'

export function AgentConfigurationModal(props: ComponentProps<typeof ConfigurationModal>) {
  return (
    <ConfiguredAgentContext.Provider value={props.agentId}>
      <ConfigurationModal {...props} />
    </ConfiguredAgentContext.Provider>
  )
}
