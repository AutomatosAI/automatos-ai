'use client'

/**
 * PRD-251 S2.2b (US-208) — the composer's variables: one field per variable of the
 * template's variables_schema (text, number or a switch), with a Form ⇄ JSON
 * toggle like the Template Studio's PreviewDataForm. A claim variable (D7) shows
 * its source, or the red Unsourced chip until one is picked.
 */
import { useState } from 'react'
import { Braces } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialClaimSource, SocialPostVariable, SocialTemplateVariable } from '@/lib/api-client'
import { ClaimSource } from './socials-claim-source'

type Variables = Record<string, SocialPostVariable>
type Sources = Record<string, SocialClaimSource>
const LONG_TEXT_CHARS = 120

/** `variables` with `name` set to `raw` as its type reads it, or without it when empty. */
export function withValue(variables: Variables, name: string, spec: SocialTemplateVariable, raw: string | boolean): Variables {
  const { [name]: _dropped, ...rest } = variables
  if (raw === '') return rest
  const value = spec.type === 'number' ? Number(raw) : raw
  if (spec.type === 'number' && !Number.isFinite(value)) return rest
  return { ...rest, [name]: { value, claim: spec.claim === true } }
}

interface VariableFieldProps {
  name: string
  spec: SocialTemplateVariable
  variables: Variables
  onChange: (variables: Variables) => void
}

function VariableField({ name, spec, variables, onChange }: VariableFieldProps) {
  const id = `socials-variable-${name}`
  const value = variables[name]?.value
  const label = spec.label || name
  if (spec.type === 'boolean') {
    return (
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={value === true} onChange={(e) => onChange(withValue(variables, name, spec, e.target.checked))} />
        {label}
      </label>
    )
  }
  const text = value == null ? '' : String(value)
  const long = spec.type === 'text' && (spec.max_chars ?? LONG_TEXT_CHARS * 2) > LONG_TEXT_CHARS
  const set = (raw: string) => onChange(withValue(variables, name, spec, raw))
  return (
    <div className="space-y-1">
      <Label htmlFor={id}>{label}</Label>
      {long ? (
        <Textarea id={id} value={text} rows={2} maxLength={spec.max_chars} placeholder={String(spec.default ?? '')} onChange={(e) => set(e.target.value)} />
      ) : (
        <Input
          id={id}
          type={spec.type === 'number' ? 'number' : 'text'}
          value={text}
          maxLength={spec.max_chars}
          min={spec.min}
          max={spec.max}
          placeholder={String(spec.default ?? '')}
          onChange={(e) => set(e.target.value)}
        />
      )}
    </div>
  )
}

function JsonView({ variables, onChange }: { variables: Variables; onChange: (variables: Variables) => void }) {
  const [text, setText] = useState(() => JSON.stringify(variables, null, 2))
  const [error, setError] = useState<string | null>(null)
  const apply = (next: string) => {
    setText(next)
    try {
      const parsed = JSON.parse(next)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('Must be a JSON object')
      setError(null)
      onChange(parsed as Variables)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid JSON')
    }
  }
  return (
    <div className="space-y-1">
      <Textarea value={text} onChange={(e) => apply(e.target.value)} className="min-h-[140px] font-mono text-xs" aria-label="Variables as JSON" />
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}

interface SocialsVariablesFormProps {
  schema: Record<string, SocialTemplateVariable>
  variables: Variables
  sources: Sources
  onChange: (variables: Variables, sources: Sources) => void
}

export function SocialsVariablesForm({ schema, variables, sources, onChange }: SocialsVariablesFormProps) {
  const [json, setJson] = useState(false)
  const names = Object.keys(schema)
  const setSource = (name: string, source: SocialClaimSource | null) => {
    const { [name]: _dropped, ...rest } = sources
    onChange(variables, source ? { ...rest, [name]: source } : rest)
  }
  return (
    <section aria-label="Variables" className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-foreground">Variables</h4>
        <Button type="button" variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => setJson(!json)}>
          <Braces className="h-3.5 w-3.5" aria-hidden /> {json ? 'Form view' : 'JSON view'}
        </Button>
      </div>
      {names.length === 0 && <p className="text-sm text-muted-foreground">This template has no variables.</p>}
      {json ? (
        <JsonView variables={variables} onChange={(next) => onChange(next, sources)} />
      ) : (
        names.map((name) => (
          <div key={name} className="space-y-1">
            <VariableField name={name} spec={schema[name]} variables={variables} onChange={(next) => onChange(next, sources)} />
            {schema[name].claim && <ClaimSource name={name} source={sources[name] ?? null} onChange={(s) => setSource(name, s)} />}
          </div>
        ))
      )}
    </section>
  )
}
