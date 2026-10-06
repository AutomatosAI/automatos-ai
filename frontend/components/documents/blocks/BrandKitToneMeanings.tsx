'use client'

// PRD-255 US-007 — what each tone word means for this brand: one line per word (the
// agents' rules block carries it). A word with no meaning is fine.
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { ToneWord } from './types'

// modules/documents/brand_system.py MAX_TONE_MEANING_CHARS: the server refuses a longer one.
export const MAX_TONE_MEANING_CHARS = 120

interface BrandKitToneMeaningsProps {
  tone: ToneWord[]
  onChange: (tone: ToneWord[]) => void
}

export function BrandKitToneMeanings({ tone, onChange }: BrandKitToneMeaningsProps) {
  if (!tone.length) return null
  return (
    <div className="mt-2 space-y-2">
      {tone.map((entry, index) => (
        <div key={`${entry.word}-${index}`}>
          <Label htmlFor={`brand-tone-meaning-${index}`} className="text-xs">
            What &ldquo;{entry.word}&rdquo; means
          </Label>
          <Input
            id={`brand-tone-meaning-${index}`}
            value={entry.meaning}
            maxLength={MAX_TONE_MEANING_CHARS}
            placeholder="One line, such as: friendly, never gushing"
            onChange={(e) => onChange(tone.map((t, i) => (i === index ? { ...t, meaning: e.target.value } : t)))}
          />
        </div>
      ))}
    </div>
  )
}
