/**
 * PRD-261 C-03b — reading a scan report, the approve rule and the note's digest line; C-05b — a
 * stored signature in words.
 */
import { describe, expect, it } from 'vitest'

import {
  SCAN_BLOCKED,
  SCAN_NOT_RUN,
  SIGNATURE_INVALID,
  SIGNATURE_VERIFIED,
  approveRefusal,
  noteDigestLine,
  noteNamesDigest,
  reportChecksOf,
  storedSignatureWords,
} from '../review-words'
import { DIGEST, KEY_ID, RETIRED_KEY_ID, SCAN_REPORT_V2, kitSummary } from './review-fixtures'

const ALLOWED = { allowed: true, reason: null }

describe('reportChecksOf', () => {
  it('reads a v2 report: one level per check, in the server order, LLM not run flagged', () => {
    const checks = reportChecksOf(SCAN_REPORT_V2)
    expect(checks.map((check) => [check.key, check.level])).toEqual([
      ['sizes', 'pass'],
      ['cssScope', 'pass'],
      ['looks', 'pass'],
      ['contrast', 'not_measured'],
      ['uses', 'error'],
      ['llm', 'warning'],
    ])
    expect(checks.find((check) => check.key === 'llm')?.llmDidNotRun).toBe(true)
    expect(checks.find((check) => check.key === 'uses')?.problems).toEqual(['useSecret is not an SDK name.'])
  })

  it('an LLM check that ran is not flagged; its findings are read defensively', () => {
    const [llm] = reportChecksOf({
      checks: { llm: { level: 'pass', status: 'ran', riskScore: 5, findings: [{ severity: 'low' }, 'junk'] } },
    })
    expect(llm.llmDidNotRun).toBe(false)
    expect(llm.riskScore).toBe(5)
    expect(llm.findings).toEqual([{ severity: 'low', category: 'finding', description: '', file: null, evidence: null }])
  })

  it('a v1 row (status, no level) shows its status, never Pass for a check that did not run', () => {
    const checks = reportChecksOf({ checks: { automated: { status: 'not_run', reason: 'not yet' } } })
    expect(checks).toMatchObject([{ key: 'automated', level: 'not_run', reason: 'not yet' }])
  })

  it('an unreadable report has no checks', () => {
    expect(reportChecksOf(null)).toEqual([])
    expect(reportChecksOf({ checks: [] })).toEqual([])
  })
})

describe('approveRefusal', () => {
  it('refuses a blocked or not-run scan whatever the approval state says', () => {
    expect(approveRefusal('blocked', ALLOWED)).toBe(SCAN_BLOCKED)
    expect(approveRefusal('not_run', ALLOWED)).toBe(SCAN_NOT_RUN)
  })

  it("passes the server's reason through, and allows an allowed reviewed scan", () => {
    expect(approveRefusal('pass', { allowed: false, reason: 'needs a note' })).toBe('needs a note')
    expect(approveRefusal('review_required', ALLOWED)).toBeNull()
    expect(approveRefusal('pass', ALLOWED)).toBeNull()
  })
})

describe('noteNamesDigest', () => {
  it("finds the kit's digest line, with the kit's two spaces", () => {
    expect(noteDigestLine(DIGEST)).toBe(`Package digest:  ${DIGEST}`)
    expect(noteNamesDigest(kitSummary(DIGEST), DIGEST)).toBe(true)
    expect(noteNamesDigest(kitSummary(DIGEST).replace(/\n/g, '\r\n'), DIGEST)).toBe(true)
  })

  it('refuses another digest, a single space and a digest only mentioned in passing', () => {
    expect(noteNamesDigest(kitSummary('e'.repeat(64)), DIGEST)).toBe(false)
    expect(noteNamesDigest(`Package digest: ${DIGEST}`, DIGEST)).toBe(false)
    expect(noteNamesDigest(`see ${DIGEST}`, DIGEST)).toBe(false)
  })
})

describe('storedSignatureWords (C-05b)', () => {
  it('a signature an active key verifies names that key', () => {
    expect(storedSignatureWords(true, KEY_ID)).toBe(`Verified with the active Automatos key ${KEY_ID}.`)
    expect(storedSignatureWords(true, null)).toBe(SIGNATURE_VERIFIED)
  })

  it('one no active key verifies says so, and names the retired or revoked key that does', () => {
    expect(storedSignatureWords(false, null)).toBe(SIGNATURE_INVALID)
    expect(storedSignatureWords(false, RETIRED_KEY_ID)).toBe(
      `${SIGNATURE_INVALID} Only ${RETIRED_KEY_ID}, a retired or revoked key, verifies it.`,
    )
  })
})
