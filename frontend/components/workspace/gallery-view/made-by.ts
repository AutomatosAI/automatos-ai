/**
 * Who made a deliverable, as its card says it: the agent that made it, else where it came
 * from. A Socials render, a Socials upload and some mission files carry no agent, and "Unknown
 * agent" on every one of them said nothing (4 Oct 2026).
 */
const MADE_BY_SOURCE: Record<string, string> = {
  social_post: 'Socials',
  upload: 'Uploaded',
  mission: 'Mission',
  playbook: 'Playbook',
  heartbeat: 'Heartbeat',
  task: 'Task',
  chat: 'Chat',
}
export const MADE_BY_FALLBACK = 'Unknown agent'

export function madeBy(agentName: string | null | undefined, sourceType: string | null | undefined): string {
  if (agentName) return agentName
  return (sourceType && MADE_BY_SOURCE[sourceType]) || MADE_BY_FALLBACK
}
