/**
 * Who made a deliverable, as its card says it: the agent that made it, else where it came
 * from. A Socials render, a Socials upload and some mission files carry no agent, and "Unknown
 * agent" on every one of them said nothing (4 Oct 2026).
 *
 * Issue #947: three more writers can leave the agent out, and their cards still said "Unknown
 * agent": an image or document an agent's tool generated with no agent in its context
 * (`agent_output`, handlers_blog / generate_document_tool), a document generated from a
 * template in the app (`document`, api/document_generation) and a trigger's file (`trigger`).
 */
const MADE_BY_SOURCE: Record<string, string> = {
  social_post: 'Socials',
  upload: 'Uploaded',
  mission: 'Mission',
  playbook: 'Playbook',
  heartbeat: 'Heartbeat',
  task: 'Task',
  chat: 'Chat',
  agent_output: 'Generated',
  document: 'Templates',
  trigger: 'Trigger',
}
export const MADE_BY_FALLBACK = 'Unknown agent'

export function madeBy(agentName: string | null | undefined, sourceType: string | null | undefined): string {
  if (agentName) return agentName
  return (sourceType && MADE_BY_SOURCE[sourceType]) || MADE_BY_FALLBACK
}
