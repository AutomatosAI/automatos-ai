import type { GrantOwner } from './api-client'

/**
 * F091-E1: whose job a card belongs to, as one line — "Scout (agent #294) ·
 * Ticket #612 · Cafe questions". Night 3's cards read "Agent #294 ·
 * board_task:612", and the owner could not tell who was asking or what the
 * answer would move.
 */
export function agentLabel(owner?: GrantOwner | null, fallbackId?: number | null): string {
  const agent = owner?.agent
  if (agent?.name) return `${agent.name} (agent #${agent.id})`
  const id = agent?.id ?? fallbackId
  return id ? `Agent #${id}` : 'An agent'
}

export function ticketLabel(owner?: GrantOwner | null): string | null {
  const ticket = owner?.ticket
  if (!ticket) return null
  return ticket.title ? `Ticket #${ticket.id} · ${ticket.title}` : `Ticket #${ticket.id}`
}

export function ownerLine(owner?: GrantOwner | null): string | null {
  const parts = [owner?.agent ? agentLabel(owner) : null, ticketLabel(owner)].filter(Boolean)
  return parts.length ? parts.join(' · ') : null
}
