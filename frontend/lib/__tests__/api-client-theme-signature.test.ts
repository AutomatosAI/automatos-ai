/**
 * PRD-261 C-05b (W-L1) — the signature fields of the Marketplace theme answers, as lib/api-client.ts
 * types them, held to the orchestrator's code (read here as text, as api-hint.test.ts reads
 * step_up.py and the route-manifest check reads the manifest):
 *
 * - signature_verify.py's Verdict states are ThemeSignatureState's, no more and no fewer;
 * - Verdict.view() carries exactly ThemeSignatureVerdict's keys, and listings.py puts it under
 *   `signature` on the card (listing_summary, which the detail spreads) and on each versions entry
 *   (_version_entry);
 * - review.py's report carries `signatureVerified` and `keyId`;
 * - signing.py's store_signature answers exactly ThemeSignatureResult's `signature` keys.
 *
 * The typed samples are checked by tsc (the Frontend CI lane's baselined count): a field the type
 * lacks is an excess property there, and one the sample lacks is a missing one, so each sample's
 * keys are the type's keys, and the runtime half compares them with the Python dicts.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import type {
  ThemeListing,
  ThemeListingDetail,
  ThemeListingVersion,
  ThemeReviewReport,
  ThemeSignatureResult,
  ThemeSignatureState,
  ThemeSignatureVerdict,
} from '@/lib/api-client'

const THEMES_PY = path.resolve(__dirname, '..', '..', '..', 'orchestrator', 'services', 'themes')
const KEY_ID = 'ed25519:0123456789abcdef'
const REQUEST_ID = '0b9e3c55-8d0e-4c64-9d3c-5a2a1e7c4f21'
const SIGNATURE = `${'A'.repeat(86)}==`

/** Every state ThemeSignatureState names; the two checks below fail tsc on one too many or too few. */
const STATES = ['verified', 'invalid', 'unsigned'] as const satisfies readonly ThemeSignatureState[]
type Unlisted = Exclude<ThemeSignatureState, (typeof STATES)[number]>
const EVERY_STATE_LISTED: [Unlisted] extends [never] ? true : never = true

const VERIFIED: ThemeSignatureVerdict = { state: 'verified', keyId: KEY_ID }
const UNSIGNED: ThemeSignatureVerdict = { state: 'unsigned', keyId: null }
const CARD: Pick<ThemeListing, 'signature'> = { signature: VERIFIED }
const DETAIL: Pick<ThemeListingDetail, 'signature'> = { signature: VERIFIED }
const VERSION: Pick<ThemeListingVersion, 'signature'> = { signature: UNSIGNED }
const REPORT: Pick<ThemeReviewReport, 'signatureVerified' | 'keyId'> = { signatureVerified: true, keyId: KEY_ID }
const STORED: ThemeSignatureResult['signature'] = {
  requestId: REQUEST_ID, signature: SIGNATURE, algorithm: 'ed25519', verified: true, keyId: KEY_ID,
}

function readThemesModule(name: string): string {
  return readFileSync(path.join(THEMES_PY, name), 'utf8')
}

/**
 * A function's source, from its `def` line to the next statement at its own indent or less (a name,
 * a decorator or a keyword; a `)` closing its own parameter list is not one).
 */
function pythonFunction(source: string, name: string): string {
  const head = new RegExp(`^( *)def ${name}\\(`, 'm').exec(source)
  expect(head, `def ${name} not found`).not.toBeNull()
  const rest = source.slice(head!.index)
  const indent = head![1].length
  const next = new RegExp(`\\n {0,${indent}}[A-Za-z_@]`).exec(rest.slice(1))
  return next === null ? rest : rest.slice(0, next.index + 1)
}

/** The keys of the dict a function's (last) `return {...}` builds. */
function returnedKeys(body: string): string[] {
  const returned = /return (\{[^{}]*\})\s*$/.exec(body.trimEnd())
  expect(returned, 'no `return {...}` at the end of the function').not.toBeNull()
  return Array.from(returned![1].matchAll(/"(\w+)":/g), (match) => match[1]).sort()
}

const keysOf = (sample: object): string[] => Object.keys(sample).sort()

describe("the signature states are signature_verify.py's", () => {
  it('lists every Verdict state and only those', () => {
    const line = /^STATE_\w+(?:, STATE_\w+)* = (.+)$/m.exec(readThemesModule('signature_verify.py'))
    expect(line, 'the STATE_* assignment in signature_verify.py').not.toBeNull()
    const python = Array.from(line![1].matchAll(/"(\w+)"/g), (match) => match[1]).sort()
    expect([...STATES].sort()).toEqual(python)
    expect(EVERY_STATE_LISTED).toBe(true)
  })
})

describe('the listings carry signature {state, keyId} (listings.py, C-05b)', () => {
  const listings = readThemesModule('listings.py')

  it("ThemeSignatureVerdict's keys are Verdict.view()'s", () => {
    expect(keysOf(VERIFIED)).toEqual(returnedKeys(pythonFunction(readThemesModule('signature_verify.py'), 'view')))
    expect(keysOf(UNSIGNED)).toEqual(keysOf(VERIFIED))
  })

  it('on the card, which the detail spreads, and on each versions entry', () => {
    expect(pythonFunction(listings, 'listing_summary')).toMatch(/"signature": [^\n]*\.view\(\)/)
    expect(pythonFunction(listings, 'listing_detail')).toContain('**summary')
    expect(pythonFunction(listings, '_version_entry')).toMatch(/"signature": [^\n]*\.view\(\)/)
    expect([keysOf(CARD), keysOf(DETAIL), keysOf(VERSION)]).toEqual([['signature'], ['signature'], ['signature']])
  })
})

describe('the admin answers name the key (review.py, signing.py, C-05b)', () => {
  it("the report carries signatureVerified and keyId", () => {
    const report = pythonFunction(readThemesModule('review.py'), 'report')
    for (const key of keysOf(REPORT)) expect(report, key).toContain(`"${key}":`)
  })

  it("a stored signature's answer has exactly ThemeSignatureResult's signature keys", () => {
    expect(keysOf(STORED)).toEqual(returnedKeys(pythonFunction(readThemesModule('signing.py'), 'store_signature')))
  })
})
