/**
 * PRD-180 S3 (F035) — the placebo model selector is gone; chat runs on the real
 * default (vitest proxy for the browser AC; the loop is headless).
 *
 * The per-message model picker was a placebo: the backend never read the chosen
 * model. Owner decision: DELETE it (a control that does nothing corrodes trust in
 * every real one). These guards pin:
 *   1. No component imports the deleted chatbot ModelSelector, and the file is gone.
 *   2. No chat surface hardcodes `initialChatModel="gpt-4"` — chat initialises
 *      from the real configured default (`LLM_DEFAULTS.model_id`).
 *   3. The chat request no longer sends a `selectedChatModel` override.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'

const FE = resolve(__dirname, '../../..')
const read = (rel: string) => readFileSync(resolve(FE, rel), 'utf8')

describe('placebo model selector removed (F035)', () => {
  it('deletes the chatbot ModelSelector component file', () => {
    expect(existsSync(resolve(FE, 'components/chatbot/model-selector.tsx'))).toBe(false)
  })

  it('no chatbot surface imports the deleted ModelSelector', () => {
    for (const f of [
      'components/chatbot/multimodal-input.tsx',
      'components/chatbot/chat.tsx',
    ]) {
      expect(read(f)).not.toContain("from './model-selector'")
    }
    // The input no longer threads the dead model props either.
    const input = read('components/chatbot/multimodal-input.tsx')
    expect(input).not.toContain('onModelChange')
    expect(input).not.toContain('<ModelSelector')
  })

  it('no chat surface hardcodes initialChatModel="gpt-4"', () => {
    for (const f of [
      'components/chatbot/chat-page-content.tsx',
      'components/chatbot/chat.tsx',
      'app/chat/page.tsx',
      'app/chat/[id]/page.tsx',
    ]) {
      const src = read(f)
      expect(src).not.toContain('initialChatModel')
      expect(src).not.toContain('"gpt-4"')
    }
  })

  it('the real default (LLM_DEFAULTS) is the chat model source of truth', () => {
    // LLM_DEFAULTS must NOT be gpt-4 — the fabricated seed value.
    const defaults = read('lib/llm-defaults.ts')
    expect(defaults).toContain('model_id')
    expect(defaults).not.toContain("model_id: 'gpt-4'")
  })

  it('the chat request no longer sends a selectedChatModel override', () => {
    expect(read('lib/chat/hooks.ts')).not.toContain('selectedChatModel')
  })
})
