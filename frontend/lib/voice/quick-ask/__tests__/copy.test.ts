// @vitest-environment node
/**
 * Theme voice, web quick-ask — the AI notice speaks the backend's own words. No route serves the greeting to the
 * browser, so copy.ts keeps one copy; this test fails when orchestrator/modules/policy/ai_disclosure.py
 * VOICE_GREETING or config.py's VOICE_AI_GREETING default changes and the frontend copy does not.
 */
import fs from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { AI_VOICE_GREETING, QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'

const ORCHESTRATOR = path.resolve(__dirname, '../../../../../orchestrator')

function read(relative: string): string {
  return fs.readFileSync(path.join(ORCHESTRATOR, relative), 'utf8')
}

describe('the quick-ask copy', () => {
  it('greets with the backend’s VOICE_GREETING', () => {
    expect(read('modules/policy/ai_disclosure.py')).toContain(`VOICE_GREETING = "${AI_VOICE_GREETING}"`)
  })

  it('greets with the default of config.py VOICE_AI_GREETING', () => {
    expect(read('config.py')).toMatch(new RegExp(`"VOICE_AI_GREETING",\\s*"${AI_VOICE_GREETING.replace('.', '\\.')}"`))
  })

  it('says every error as a sentence, never a code or an object', () => {
    for (const sentence of Object.values(QUICK_ASK_ERRORS)) {
      expect(sentence).toMatch(/^[A-Z].*[.!?]$/)
      expect(sentence).not.toMatch(/[{}_]/)
    }
  })
})
