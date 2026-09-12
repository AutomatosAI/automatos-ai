'use client'

import React, { useState } from 'react'
import { Braces, Palette } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { fieldLabel, getDataField, setDataField } from './templateFields'

interface PreviewDataFormProps {
  // The data.* names the template references (prefix stripped).
  fields: string[]
  // The `data` object (what generate_document's `data` carries).
  data: Record<string, any>
  onChange: (data: Record<string, any>) => void
  // Catalog chips with no value on file (brand/company/user) → send the author to the Brand Kit.
  missingOnFile: string[]
  onOpenBrandKit: () => void
  title?: string
}

// One input per data.* chip, so an author sees exactly the contract an agent must fill
// (PRD-242 S5). The raw JSON stays one toggle away for nested/legacy shapes.
export function PreviewDataForm({
  fields,
  data,
  onChange,
  missingOnFile,
  onOpenBrandKit,
  title = 'Fill-in fields',
}: PreviewDataFormProps) {
  const [advanced, setAdvanced] = useState(false)
  const [jsonText, setJsonText] = useState(() => JSON.stringify(data, null, 2))
  const [jsonError, setJsonError] = useState<string | null>(null)

  const openAdvanced = () => {
    setJsonText(JSON.stringify(data, null, 2))
    setJsonError(null)
    setAdvanced(true)
  }

  const applyJson = (text: string) => {
    setJsonText(text)
    try {
      const parsed = JSON.parse(text)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setJsonError('Must be a JSON object')
        return
      }
      setJsonError(null)
      onChange(parsed)
    } catch (e: any) {
      setJsonError(e?.message || 'Invalid JSON')
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="flex items-center text-xs text-muted-foreground">
          {title}
          <FieldHelp id="deliverables.templates.editor.fill_in_fields" />
        </Label>
        <Button type="button" variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={advanced ? () => setAdvanced(false) : openAdvanced}>
          <Braces className="h-3.5 w-3.5" /> {advanced ? 'Form view' : 'JSON view'}
        </Button>
      </div>

      {missingOnFile.length > 0 && (
        <Alert className="border-warning/40 bg-warning/5 py-2">
          <AlertDescription className="flex flex-wrap items-center gap-2 text-xs">
            <span>
              Not on file yet: <span className="font-mono">{missingOnFile.join(', ')}</span> — a document with these chips is blocked until they are filled.
            </span>
            <Button type="button" size="sm" variant="outline" className="h-6 gap-1 text-xs" onClick={onOpenBrandKit}>
              <Palette className="h-3 w-3" /> Open Brand Kit
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {advanced ? (
        <div className="space-y-1">
          <Textarea
            value={jsonText}
            onChange={(e) => applyJson(e.target.value)}
            className="min-h-[120px] font-mono text-xs"
            aria-label="Preview data as JSON"
          />
          {jsonError ? (
            <p className="text-xs text-destructive">{jsonError}</p>
          ) : (
            <p className="text-xs text-muted-foreground">This is the object an agent passes as <code>data</code> to generate_document.</p>
          )}
        </div>
      ) : fields.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          This template has no <code>data.*</code> chips — add one with “Insert variable → data” to create a fill-in field.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {fields.map((field) => {
            const value = getDataField(data, field)
            const long = value.length > 80 || /summary|body|details|content|notes|appendix/i.test(field)
            return (
              <div key={field} className={long ? 'sm:col-span-2' : ''}>
                <Label className="text-xs">
                  {fieldLabel(field)} <span className="font-mono text-[10px] text-muted-foreground">data.{field}</span>
                </Label>
                {long ? (
                  <Textarea value={value} onChange={(e) => onChange(setDataField(data, field, e.target.value))} className="min-h-[64px] text-sm" />
                ) : (
                  <Input value={value} onChange={(e) => onChange(setDataField(data, field, e.target.value))} />
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
