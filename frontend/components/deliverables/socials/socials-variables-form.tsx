'use client'

/**
 * PRD-251 S2.2b (US-208) — the composer's variables: one field per variable of the
 * template's variables_schema (text, number or a switch), with a Form ⇄ JSON
 * toggle like the Template Studio's PreviewDataForm. A claim variable (D7) shows
 * its source, or the red Unsourced chip until one is picked.
 *
 * Each field says what goes in it: the template's sample text greyed in as an example,
 * the field's own description under it, a * on a field the render needs (one with no
 * default), and a count against its limit.
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
export const FIELDS_LEGEND = 'Fields marked * are needed before the post can render. Grey text in a field is an example.'

/** Whether the render needs a value: a text or number field with no default (core/social_templates.py). */
export function isRequired(spec: SocialTemplateVariable): boolean {
  return spec.type !== 'boolean' && spec.default === undefined
}

/**
 * F377 (night 11): the fields a video at `length` seconds asks for. A shorter cut shows only
 * the fields `fieldsByLength` lists for it (core/social_cuts.py), so the others leave the form
 * and none of them is marked needed; a length without a cut (or an image) keeps every field.
 */
export function schemaAtLength(
  schema: Record<string, SocialTemplateVariable>,
  fieldsByLength: Record<string, string[]> | undefined,
  length: number | null | undefined,
): Record<string, SocialTemplateVariable> {
  const shown = length == null ? undefined : fieldsByLength?.[String(length)]
  if (!shown) return schema
  const kept = new Set(shown)
  return Object.fromEntries(Object.entries(schema).filter(([name]) => kept.has(name)))
}

/** The grey text in an empty field: the template's sample value as an example, else its default. */
export function exampleOf(spec: SocialTemplateVariable, example: unknown): string {
  if (typeof example === 'string' ? example.trim() : example !== undefined && example !== null) return `e.g. ${String(example)}`
  return spec.default === undefined ? '' : String(spec.default)
}

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
  /** The template's sample value for this field: greyed in as an example. */
  example?: unknown
  onChange: (variables: Variables) => void
}

function FieldLabel({ id, label, required }: { id: string; label: string; required: boolean }) {
  return (
    <Label htmlFor={id}>
      {label}
      {required && (
        <>
          <span aria-hidden className="ml-0.5 text-destructive">*</span>
          <span className="sr-only"> (needed to render)</span>
        </>
      )}
    </Label>
  )
}

/** Under the field: what goes in it, and how much of its limit is used. */
function FieldHelp({ id, description, used, limit }: { id: string; description?: string; used: number; limit?: number }) {
  if (!description && !limit) return null
  return (
    <div className="flex items-start justify-between gap-3 text-xs text-muted-foreground">
      <p id={id}>{description}</p>
      {limit ? <span className="shrink-0 tabular-nums">{used}/{limit}</span> : null}
    </div>
  )
}

function VariableField({ name, spec, variables, example, onChange }: VariableFieldProps) {
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
  const field = {
    id, value: text, maxLength: spec.max_chars, placeholder: exampleOf(spec, example),
    'aria-describedby': spec.description ? `${id}-help` : undefined,
  }
  return (
    <div className="space-y-1">
      <FieldLabel id={id} label={label} required={isRequired(spec)} />
      {long ? (
        <Textarea {...field} rows={2} onChange={(e) => set(e.target.value)} />
      ) : (
        <Input {...field} type={spec.type === 'number' ? 'number' : 'text'} min={spec.min} max={spec.max} onChange={(e) => set(e.target.value)} />
      )}
      <FieldHelp id={`${id}-help`} description={spec.description} used={text.length} limit={spec.max_chars} />
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
  /** PRD-251B US-B109: open the search of the first claim with no source ("Add a source"). */
  openFirstUnsourced?: boolean
  /** The template's sample text per field: each empty field's grey example. */
  examples?: Record<string, unknown>
}

export function SocialsVariablesForm({ schema, variables, sources, onChange, openFirstUnsourced = false, examples = {} }: SocialsVariablesFormProps) {
  const [json, setJson] = useState(false)
  const names = Object.keys(schema)
  const firstUnsourced = openFirstUnsourced ? names.find((name) => schema[name].claim && !sources[name]) : undefined
  const setSource = (name: string, source: SocialClaimSource | null) => {
    const { [name]: _dropped, ...rest } = sources
    onChange(variables, source ? { ...rest, [name]: source } : rest)
  }
  return (
    <section aria-label="Fields" className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-foreground">Fields</h4>
        <Button type="button" variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={() => setJson(!json)}>
          <Braces className="h-3.5 w-3.5" aria-hidden /> {json ? 'Form view' : 'JSON view'}
        </Button>
      </div>
      {names.length === 0 && <p className="text-sm text-muted-foreground">This template has no fields.</p>}
      {names.length > 0 && !json && <p className="text-xs text-muted-foreground">{FIELDS_LEGEND}</p>}
      {json ? (
        <JsonView variables={variables} onChange={(next) => onChange(next, sources)} />
      ) : (
        names.map((name) => (
          <div key={name} className="space-y-1">
            <VariableField name={name} spec={schema[name]} variables={variables} example={examples[name]} onChange={(next) => onChange(next, sources)} />
            {schema[name].claim && (
              <ClaimSource
                name={name}
                source={sources[name] ?? null}
                onChange={(s) => setSource(name, s)}
                initiallyPicking={name === firstUnsourced}
              />
            )}
          </div>
        ))
      )}
    </section>
  )
}
