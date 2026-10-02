/**
 * PRD-253 S3.1 — a session's usage in its plan's own units, never a price.
 */
import { describe, expect, it } from 'vitest'

import { sessionUsageText } from '@/components/activity/board/session-usage'

describe('sessionUsageText', () => {
  it('shows tokens and the model, as before', () => {
    expect(sessionUsageText({ total_tokens: 1500, model: 'sonnet' })).toBe('1,500 on sonnet · plan usage, no cost')
  })

  it('adds GitHub Copilot\'s AI credits and premium requests when the CLI books them', () => {
    expect(sessionUsageText({ total_tokens: 900, model: 'auto', ai_credits: 1.25, premium_requests: 2 }))
      .toBe('900 on auto · 1.25 AI credits · 2 premium requests · plan usage, no cost')
  })

  it('says nothing for a turn with no usage', () => {
    expect(sessionUsageText(null)).toBeNull()
    expect(sessionUsageText({})).toBeNull()
  })
})
