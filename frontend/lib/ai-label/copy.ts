/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1)) — the words of the in-app AI label (PRD §8).
 *
 * Every in-app chat shows "Auto is an AI", or "{agent name} is an AI" while another agent
 * answers. The label is the host's (components/ai-label/ai-label.tsx): no theme, manifest field
 * or setting changes these words or turns the label off.
 */

/** Who answers when no other agent is picked. */
export const DEFAULT_AGENT_NAME = 'Auto'

/** An agent was picked whose name is not loaded yet: the label still says it is an AI. */
export const UNNAMED_AGENT = 'This agent'

/** What the chat says, in place of a send, when the label can't be seen. */
export const AI_LABEL_PAUSED_TEXT = 'This theme hides the AI notice, so chat is paused. Switch theme in Settings.'

/** The label's words for the agent answering: its name, or Auto when it has none. */
export function aiLabelText(agentName?: string | null): string {
  const name = agentName?.trim()
  return `${name || DEFAULT_AGENT_NAME} is an AI`
}
