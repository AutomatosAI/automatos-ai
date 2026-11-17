'use client'

/**
 * PRD-251B US-B109 — the editor's Text on the image (it was "Claims and sources"): the
 * template's fields, each with its example and what goes in it, and each claim with its
 * source or the red Unsourced chip (the PRD-251 variables form and claim picker, D7).
 * "Add a source" opens the first claim still without one.
 *
 * The fields are Auto's to write, so the card stays closed while every one is empty: it says
 * how many there are and that Auto writes them from the brief. It opens once a field holds
 * a value (Auto's or the person's), or when the person opens it to write their own, and it
 * never closes under someone who is typing in it.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import type { SocialClaimSource, SocialPostVariable, SocialTemplateVariable } from '@/lib/api-client'
import { SocialsVariablesForm } from '../socials-variables-form'
import { EditorCard, Hint } from './editor-ui'

export const UNSOURCED_HINT = 'A claim without a source needs a second confirmation from whoever approves.'
export const TEXT_CARD = 'Text on the image'
export const NO_TEMPLATE_HINT = 'The words on the image come from a template. Pick one in Look to see its fields, or let Auto pick one and write them when you render.'
export const HIDE_FIELDS = 'Hide fields'

/** "Show 11 fields": the closed card's way in. */
export function showFieldsLabel(count: number): string {
  return count === 1 ? 'Show 1 field' : `Show ${count} fields`
}

/** The closed card: how many fields the template has, and who writes them. */
export function fieldsLeftToAuto(count: number): string {
  const fields = count === 1 ? 'This template has 1 field' : `This template has ${count} fields`
  return `${fields}. Auto writes them from your brief when you write with Auto or render. Open them only to use your own words: what you write is kept.`
}

/** The claims the schema declares that have no source yet, in schema order. */
export function unsourcedClaims(schema: Record<string, SocialTemplateVariable>, sources: Record<string, SocialClaimSource>): string[] {
  return Object.entries(schema)
    .filter(([name, spec]) => spec.claim && !sources[name])
    .map(([name]) => name)
}

interface EditorClaimsCardProps {
  schema: Record<string, SocialTemplateVariable> | null
  variables: Record<string, SocialPostVariable>
  sources: Record<string, SocialClaimSource>
  /** The chosen template's sample text: each empty field's example. */
  examples?: Record<string, unknown>
  onChange: (variables: Record<string, SocialPostVariable>, sources: Record<string, SocialClaimSource>) => void
}

export function EditorClaimsCard({ schema, variables, sources, examples, onChange }: EditorClaimsCardProps) {
  const [opened, setOpened] = useState(0)
  // null: the card follows its fields (closed while all are empty); true or false: the person's choice.
  const [toggled, setToggled] = useState<boolean | null>(null)
  const names = schema ? Object.keys(schema) : []
  const open = toggled ?? names.some((name) => variables[name] !== undefined)
  const missing = schema ? unsourcedClaims(schema, sources) : []
  const hasClaims = !!schema && Object.values(schema).some((spec) => spec.claim)
  const addSource = () => {
    setToggled(true)
    setOpened((n) => n + 1)
    document.getElementById(`socials-claim-source-${missing[0]}`)?.scrollIntoView?.({ block: 'center' })
  }
  const action = names.length > 0 ? (
    <div className="flex items-center gap-2">
      {hasClaims && (
        <Button type="button" size="sm" variant="secondary" onClick={addSource} disabled={missing.length === 0}>
          Add a source
        </Button>
      )}
      <Button type="button" size="sm" variant="ghost" aria-expanded={open} onClick={() => setToggled(!open)}>
        {open ? HIDE_FIELDS : showFieldsLabel(names.length)}
      </Button>
    </div>
  ) : undefined
  // A change made in the form keeps it open: emptying the last filled field must not close it mid-edit.
  const edited: EditorClaimsCardProps['onChange'] = (nextVariables, nextSources) => {
    setToggled(true)
    onChange(nextVariables, nextSources)
  }
  const form = schema && open && (
    <SocialsVariablesForm
      key={opened} schema={schema} variables={variables} sources={sources} examples={examples} onChange={edited}
      openFirstUnsourced={opened > 0}
    />
  )
  return (
    <EditorCard label={TEXT_CARD} action={action}>
      {names.length === 0 && <Hint>{NO_TEMPLATE_HINT}</Hint>}
      {names.length > 0 && !open && <Hint>{fieldsLeftToAuto(names.length)}</Hint>}
      {names.length > 0 && form}
      {hasClaims && <Hint>{UNSOURCED_HINT}</Hint>}
    </EditorCard>
  )
}
