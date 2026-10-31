import { describe, it, expect } from 'vitest'
import { blockedLine, parseBlockedReason } from '../blocked-reason'

// PRD-268 US-C20b: the board's gate writes the rule that asked, in words, after the grant prefix.
const SENTENCE = 'Ask before work that costs more than your approval ceiling'
const HELD = `Awaiting human approval (grant #42): ${SENTENCE}`
const OLD = "Awaiting human approval (grant #11): board task requires approval under 'always_ask' policy"
const OLD_REUSED = 'Awaiting human approval (grant #12): pending approval grant already exists'

describe('parseBlockedReason: the rule sentence', () => {
  it('returns the sentence after the prefix, and the grant id', () => {
    expect(parseBlockedReason(HELD)).toEqual({ grantId: 42, text: HELD, sentence: SENTENCE })
  })

  it('still parses the grant id from a reason written before the sentence, which has none', () => {
    for (const [old, grantId] of [[OLD, 11], [OLD_REUSED, 12]] as const) {
      const info = parseBlockedReason(old)
      expect(info.grantId).toBe(grantId)
      expect(info.sentence).toBeUndefined()
    }
  })

  it('gives no sentence for a reason with no approval prefix', () => {
    expect(parseBlockedReason('Waiting for a host').sentence).toBeUndefined()
    expect(parseBlockedReason('Awaiting human approval (grant #7): ').sentence).toBeUndefined()
    expect(parseBlockedReason(null)).toEqual({ grantId: null, text: '' })
  })
})

describe('blockedLine: what the ticket shows under its heading', () => {
  it('shows the sentence, not the raw text', () => {
    expect(blockedLine(HELD)).toEqual({ grantId: 42, line: SENTENCE })
    expect(blockedLine(HELD).line).not.toContain('Awaiting human approval')
  })

  it('never shows an old policy key, and keeps its grant', () => {
    expect(blockedLine(OLD)).toEqual({ grantId: 11, line: '' })
  })

  it('shows any other reason as written', () => {
    expect(blockedLine('Waiting for a host')).toEqual({ grantId: null, line: 'Waiting for a host' })
    expect(blockedLine('Held by grant #9 until noon')).toEqual({ grantId: 9, line: 'Held by grant #9 until noon' })
    expect(blockedLine(undefined)).toEqual({ grantId: null, line: '' })
  })
})
