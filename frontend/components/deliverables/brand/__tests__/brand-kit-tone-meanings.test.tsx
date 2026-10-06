/**
 * PRD-255 Wave 1, US-007 — the voice on the Brand kit page: each tone word with the line that
 * says what it means for this brand, edited in place and kept when the words change.
 */
import { useState } from 'react'
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

import { BrandKitSocial } from '@/components/documents/blocks/BrandKitSocial'
import type { BrandVoice } from '@/components/documents/blocks/types'

let latest: BrandVoice | null = null

function Harness({ start }: { start: BrandVoice }) {
  const [voice, setVoice] = useState(start)
  latest = voice
  return <BrandKitSocial voice={voice} onHandlesChange={() => undefined} onVoiceChange={setVoice} />
}

const VOICE: BrandVoice = {
  tone: [{ word: 'warm', meaning: 'friendly, never gushing' }, { word: 'plain', meaning: '' }, { word: 'sure', meaning: '' }],
  banned_phrases: [],
  sign_off: '',
}

afterEach(() => { cleanup(); latest = null })

describe('tone words with meanings', () => {
  it('shows a meaning box per tone word, filled with the meaning it has', () => {
    render(<Harness start={VOICE} />)
    expect(screen.getByLabelText('What “warm” means')).toHaveValue('friendly, never gushing')
    expect(screen.getByLabelText('What “plain” means')).toHaveValue('')
  })

  it('stores a meaning on its word, and keeps it when the words change', () => {
    render(<Harness start={VOICE} />)
    fireEvent.change(screen.getByLabelText('What “plain” means'), { target: { value: 'short words, no jargon' } })
    expect(latest?.tone[1]).toEqual({ word: 'plain', meaning: 'short words, no jargon' })

    fireEvent.change(screen.getByLabelText(/^Tone words/), { target: { value: 'plain, warm, curious' } })
    expect(latest?.tone).toEqual([
      { word: 'plain', meaning: 'short words, no jargon' },
      { word: 'warm', meaning: 'friendly, never gushing' },
      { word: 'curious', meaning: '' },
    ])
  })

  it('shows no meaning boxes while there are no tone words', () => {
    render(<Harness start={{ ...VOICE, tone: [] }} />)
    expect(screen.queryByLabelText(/^What .* means$/)).toBeNull()
  })
})
