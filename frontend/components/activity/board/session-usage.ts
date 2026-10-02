/**
 * What a session's turn used, in its plan's own units: tokens — and, for GitHub
 * Copilot, the AI credits and premium requests its plan counts (PRD-253 S3.1).
 * Never a price: a session runs on the operator's subscription.
 */

function count(value: unknown): string {
  const n = Number(value)
  return Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(value)
}

export function sessionUsageText(usage: Record<string, any> | null | undefined): string | null {
  if (!usage || usage.total_tokens == null) return null
  const parts = [`${count(usage.total_tokens)}${usage.model ? ` on ${usage.model}` : ''}`]
  if (usage.ai_credits != null) parts.push(`${count(usage.ai_credits)} AI credits`)
  if (usage.premium_requests != null) parts.push(`${count(usage.premium_requests)} premium requests`)
  return `${parts.join(' · ')} · plan usage, no cost`
}
