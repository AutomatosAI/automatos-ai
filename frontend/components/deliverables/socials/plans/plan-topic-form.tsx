'use client'

/**
 * PRD-251B US-B203 — a topic added by hand: its title, angle, formats and one fact with its
 * source (a fact without one is refused by the bank; a topic may have no facts yet).
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { SocialFactSourceKind, SocialTopicInput } from '@/lib/socials-plan-types'
import { FORMAT_CHOICES } from './plan-step-cadence'

const SELECT = 'h-[38px] rounded-md border border-input bg-background px-2 text-sm'
const SOURCE_KINDS: ReadonlyArray<{ value: SocialFactSourceKind; label: string }> = [
  { value: 'web', label: 'A web page' },
  { value: 'knowledge', label: 'A document' },
  { value: 'deliverable', label: 'A Deliverable' },
  { value: 'github', label: 'GitHub' },
  { value: 'note', label: 'A note' },
]

interface TopicFormProps {
  busy: boolean
  onSave: (input: SocialTopicInput) => void
  onCancel: () => void
}

export function topicInput(title: string, angle: string, formats: string[], fact: { text: string; kind: SocialFactSourceKind; ref: string; label: string }): SocialTopicInput {
  const facts = fact.text.trim() ? [{ text: fact.text.trim(), source: { kind: fact.kind, ref: fact.ref.trim(), label: fact.label.trim() } }] : []
  return { title: title.trim(), angle: angle.trim() || null, formats, facts }
}

export function PlanTopicForm({ busy, onSave, onCancel }: TopicFormProps) {
  const [title, setTitle] = useState('')
  const [angle, setAngle] = useState('')
  const [formats, setFormats] = useState<string[]>([])
  const [fact, setFact] = useState({ text: '', kind: 'web' as SocialFactSourceKind, ref: '', label: '' })
  const toggle = (format: string) => setFormats((now) => (now.includes(format) ? now.filter((f) => f !== format) : [...now, format]))
  return (
    <form aria-label="Add a topic" className="flex flex-col gap-3 rounded-xl border border-border bg-card p-3.5"
      onSubmit={(e) => { e.preventDefault(); onSave(topicInput(title, angle, formats, fact)) }}>
      <div className="flex flex-col gap-1">
        <Label htmlFor="topic-title">Title</Label>
        <Input id="topic-title" value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} required />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="topic-angle">Angle</Label>
        <Input id="topic-angle" value={angle} maxLength={1000} onChange={(e) => setAngle(e.target.value)} />
      </div>
      <div role="group" aria-label="Formats it suits" className="flex flex-wrap gap-1.5">
        {FORMAT_CHOICES.map((choice) => (
          <button key={choice.value} type="button" aria-pressed={formats.includes(choice.value)}
            className={`h-[32px] rounded-full border border-border px-3 text-[13px] ${formats.includes(choice.value) ? 'border-accent bg-accent/15' : 'text-muted-foreground'}`}
            onClick={() => toggle(choice.value)}>
            {choice.label}
          </button>
        ))}
      </div>
      <div className="grid gap-2 sm:grid-cols-[minmax(0,2fr)_150px_minmax(0,1fr)_minmax(0,1fr)]">
        <Input aria-label="Fact" placeholder="A fact, in one sentence" value={fact.text} onChange={(e) => setFact({ ...fact, text: e.target.value })} />
        <select aria-label="Its source" className={SELECT} value={fact.kind} onChange={(e) => setFact({ ...fact, kind: e.target.value as SocialFactSourceKind })}>
          {SOURCE_KINDS.map((kind) => <option key={kind.value} value={kind.value}>{kind.label}</option>)}
        </select>
        <Input aria-label="Source address or id" placeholder="Address or id" value={fact.ref} onChange={(e) => setFact({ ...fact, ref: e.target.value })} />
        <Input aria-label="Source name" placeholder="Its name" value={fact.label} onChange={(e) => setFact({ ...fact, label: e.target.value })} />
      </div>
      <div className="flex gap-2">
        <Button type="submit" disabled={busy || !title.trim()}>Add to the bank</Button>
        <Button type="button" variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </form>
  )
}
