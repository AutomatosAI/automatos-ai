'use client'

/**
 * PRD-274 — a saved agent's two profile choices, under the Runtime group of its Model tab: the
 * runtime profile (Full / Lean) and the memory profile. So a product agent's whole setup is one
 * stop: open the agent, Runtime profile: Lean, Memory profile: a profile, Save.
 *
 * The saved agent is read once here and shared by both; each pick rides the form through
 * `onChange(field, value)`, and the modal's existing save sends it (`runtimeConfiguration`).
 */

import { MemoryProfileField } from './memory-profile-field'
import type { RuntimeFields } from './runtime-section'
import { RuntimeProfileField } from './runtime-profile-field'
import { useSavedAgent } from './use-saved-agent'

interface AgentProfileFieldsProps {
  agentId: number
  value: RuntimeFields
  onChange: <K extends keyof RuntimeFields>(field: K, value: RuntimeFields[K]) => void
}

export function AgentProfileFields({ agentId, value, onChange }: AgentProfileFieldsProps) {
  const agent = useSavedAgent(agentId)
  return (
    <>
      <RuntimeProfileField
        agent={agent}
        value={value.runtime_profile}
        onChange={(profile) => onChange('runtime_profile', profile)}
      />
      <MemoryProfileField
        agent={agent}
        value={value.memory_profile}
        onChange={(slug) => onChange('memory_profile', slug)}
      />
    </>
  )
}

export default AgentProfileFields
