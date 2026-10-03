'use client'

/**
 * PRD-251B US-B109 — the editor's Text on the image (it was "Claims and sources"): the
 * template's fields, each with its example, what goes in it and whether the render needs it,
 * and each claim with its source or the red Unsourced chip (the PRD-251 variables form and
 * claim picker, D7). "Add a source" opens the first claim still without one.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import type { SocialClaimSource, SocialPostVariable, SocialTemplateVariable } from '@/lib/api-client'
import { SocialsVariablesForm } from '../socials-variables-form'
import { EditorCard, Hint } from './editor-ui'

export const UNSOURCED_HINT = 'A claim without a source needs a second confirmation from whoever approves.'
export const TEXT_CARD = 'Text on the image'
export const NO_TEMPLATE_HINT = 'The words on the image come from a template. Pick one in Look to see its fields, or let Auto pick one and write them when you render.'

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
  const missing = schema ? unsourcedClaims(schema, sources) : []
  const hasClaims = !!schema && Object.values(schema).some((spec) => spec.claim)
  const addSource = () => {
    setOpened((n) => n + 1)
    document.getElementById(`socials-claim-source-${missing[0]}`)?.scrollIntoView?.({ block: 'center' })
  }
  const action = hasClaims ? (
    <Button type="button" size="sm" variant="secondary" onClick={addSource} disabled={missing.length === 0}>
      Add a source
    </Button>
  ) : undefined
  return (
    <EditorCard label={TEXT_CARD} action={action}>
      {schema && Object.keys(schema).length > 0 ? (
        <SocialsVariablesForm
          key={opened} schema={schema} variables={variables} sources={sources} examples={examples} onChange={onChange}
          openFirstUnsourced={opened > 0}
        />
      ) : (
        <Hint>{NO_TEMPLATE_HINT}</Hint>
      )}
      {hasClaims && <Hint>{UNSOURCED_HINT}</Hint>}
    </EditorCard>
  )
}
