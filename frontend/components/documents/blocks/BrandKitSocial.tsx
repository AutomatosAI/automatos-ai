'use client'

// The brand kit's social handles and voice (PRD-251 D5, S1.3). A handle is kept per
// Composio toolkit and checked by the server against that network's rule
// (modules/documents/brand_kit.py HANDLE_RULES). The voice is three to five tone
// words, or none, and the phrases the brand never uses: the agents that draft its
// posts read both.
import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { FieldHelp } from '@/components/ui/help-tooltip'
import type { BrandVoice } from './types'

// The Socials channels, keyed by Composio toolkit.
export const SOCIAL_HANDLE_FIELDS = [
  { toolkit: 'linkedin', label: 'LinkedIn', prefix: 'in/ or company/', placeholder: 'acme' },
  { toolkit: 'twitter', label: 'X', prefix: '@', placeholder: 'acme' },
  { toolkit: 'instagram', label: 'Instagram', prefix: '@', placeholder: 'acme' },
  { toolkit: 'tiktok', label: 'TikTok', prefix: '@', placeholder: 'acme' },
  { toolkit: 'youtube', label: 'YouTube', prefix: '@', placeholder: 'acme' },
] as const

// The server's rule (modules/documents/brand_kit.py MIN_TONE_WORDS / MAX_TONE_WORDS).
export const MIN_TONE_WORDS = 3
export const MAX_TONE_WORDS = 5

/** The trimmed, non-blank entries of ``text`` split at ``separator``, each once (case-insensitive). */
export function parseList(text: string, separator: RegExp): string[] {
  const seen = new Set<string>()
  return text
    .split(separator)
    .map((entry) => entry.trim())
    .filter((entry) => {
      const key = entry.toLowerCase()
      if (!entry || seen.has(key)) return false
      seen.add(key)
      return true
    })
}

/** Why these tone words cannot be saved, or null: three to five of them, or none. */
export function toneWordsProblem(tone: string[]): string | null {
  const count = tone.length
  if (count === 0 || (count >= MIN_TONE_WORDS && count <= MAX_TONE_WORDS)) return null
  return `Give ${MIN_TONE_WORDS} to ${MAX_TONE_WORDS} tone words, or none (${count} now).`
}

interface BrandKitSocialProps {
  handles: Record<string, string>
  voice: BrandVoice
  onHandlesChange: (handles: Record<string, string>) => void
  onVoiceChange: (voice: BrandVoice) => void
}

export function BrandKitSocial({ handles, voice, onHandlesChange, onVoiceChange }: BrandKitSocialProps) {
  // The text as typed; the kit holds the parsed list.
  const [toneText, setToneText] = useState(() => voice.tone.join(', '))
  const [phrasesText, setPhrasesText] = useState(() => voice.banned_phrases.join('\n'))
  const problem = toneWordsProblem(voice.tone)

  return (
    <>
      <div className="rounded-md border p-3">
        <p className="mb-2 flex items-center text-xs font-medium text-muted-foreground">
          Social handles <FieldHelp id="deliverables.brand_kit.handles" />
        </p>
        <div className="grid grid-cols-2 gap-3">
          {SOCIAL_HANDLE_FIELDS.map(({ toolkit, label, prefix, placeholder }) => (
            <div key={toolkit}>
              <Label htmlFor={`brand-handle-${toolkit}`} className="text-xs">
                {label}
              </Label>
              <div className="flex items-center gap-1.5">
                <span className="shrink-0 text-xs text-muted-foreground">{prefix}</span>
                <Input
                  id={`brand-handle-${toolkit}`}
                  value={handles[toolkit] ?? ''}
                  placeholder={placeholder}
                  onChange={(e) => onHandlesChange({ ...handles, [toolkit]: e.target.value })}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-md border p-3">
        <p className="mb-2 flex items-center text-xs font-medium text-muted-foreground">
          Brand voice <FieldHelp id="deliverables.brand_kit.voice" />
        </p>
        <Label htmlFor="brand-voice-tone" className="text-xs">
          Tone words (three to five, separated by commas)
        </Label>
        <Input
          id="brand-voice-tone"
          value={toneText}
          placeholder="warm, plain-spoken, confident"
          aria-invalid={problem ? true : undefined}
          onChange={(e) => {
            setToneText(e.target.value)
            onVoiceChange({ ...voice, tone: parseList(e.target.value, /,/) })
          }}
        />
        {problem && (
          <p className="mt-1 text-xs text-destructive" role="alert">
            {problem}
          </p>
        )}
        <Label htmlFor="brand-voice-banned" className="mt-3 block text-xs">
          Phrases the brand never uses (one a line)
        </Label>
        <Textarea
          id="brand-voice-banned"
          rows={3}
          value={phrasesText}
          placeholder={'game-changer\nrevolutionary'}
          onChange={(e) => {
            setPhrasesText(e.target.value)
            onVoiceChange({ ...voice, banned_phrases: parseList(e.target.value, /\n/) })
          }}
        />
      </div>
    </>
  )
}
