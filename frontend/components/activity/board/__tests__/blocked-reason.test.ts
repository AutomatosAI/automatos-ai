import { describe, it, expect } from 'vitest'
import { parseBlockedReason } from '../blocked-reason'

describe('parseBlockedReason', () => {
  it('finds the grant a blocked ticket is waiting for', () => {
    expect(parseBlockedReason("Awaiting human approval (grant #11): board task requires approval under 'always_ask' policy"))
      .toEqual({ grantId: 11, text: expect.stringContaining('grant #11') })
  })
  it('copes with no reason or no grant', () => {
    expect(parseBlockedReason(undefined)).toEqual({ grantId: null, text: '' })
    expect(parseBlockedReason('Waiting for a host')).toEqual({ grantId: null, text: 'Waiting for a host' })
  })
})
