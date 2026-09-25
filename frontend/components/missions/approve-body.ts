import type { MissionApproveRequest } from '@/types/missions'

export const BUDGET_MIN_TOKENS = 1000
export const BUDGET_TOO_SMALL = 'The budget is a number of tokens, at least 1,000.'

/**
 * F165 (night 5): what the mission page's Approve button sends. The approve
 * route takes the two overrides and nothing else; plan edits go to
 * PATCH /api/missions/{id}/plan before approval (the chat's approval widget
 * does that). `body` is undefined when nothing changed.
 */
export function approveBody(
  maxConcurrentOverride: string | null,
  tokenBudgetOverride: string,
  currentMaxConcurrent: number,
): { body?: MissionApproveRequest; error?: string } {
  const concurrency = maxConcurrentOverride != null ? parseInt(maxConcurrentOverride, 10) : undefined
  const newConcurrency = concurrency != null && concurrency !== currentMaxConcurrent ? concurrency : undefined
  const budgetText = tokenBudgetOverride.trim()
  const budget = budgetText ? Number(budgetText) : undefined
  if (budget != null && (!Number.isInteger(budget) || budget < BUDGET_MIN_TOKENS)) {
    return { error: BUDGET_TOO_SMALL }
  }
  if (newConcurrency == null && budget == null) return {}
  return {
    body: {
      ...(newConcurrency != null ? { max_concurrent_override: newConcurrency } : {}),
      ...(budget != null ? { token_budget_override: budget } : {}),
    },
  }
}
