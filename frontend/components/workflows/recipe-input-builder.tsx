'use client'

import * as React from 'react'
import { Plus, Trash2, GripVertical } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

export interface InputField {
  name: string
  type: 'string' | 'number' | 'boolean'
  description: string
  required: boolean
  default: string
}

interface RecipeInputBuilderProps {
  value: string // JSON string of inputs schema
  onChange: (value: string) => void
  onValidation?: (valid: boolean) => void
}

function parseInputsSchema(jsonStr: string): InputField[] {
  try {
    const parsed = JSON.parse(jsonStr)
    if (!parsed || typeof parsed !== 'object') return []
    return Object.entries(parsed).map(([name, def]: [string, any]) => ({
      name,
      type: def?.type || 'string',
      description: def?.description || '',
      required: !!def?.required,
      default: def?.default ?? '',
    }))
  } catch {
    return []
  }
}

function fieldsToSchema(fields: InputField[]): string {
  const schema: Record<string, any> = {}
  for (const f of fields) {
    if (!f.name.trim()) continue
    schema[f.name.trim()] = {
      type: f.type,
      description: f.description,
      required: f.required,
      ...(f.default ? { default: f.default } : {}),
    }
  }
  return JSON.stringify(schema)
}

export function RecipeInputBuilder({ value, onChange, onValidation }: RecipeInputBuilderProps) {
  const [fields, setFields] = React.useState<InputField[]>(() => parseInputsSchema(value))

  // Sync back to parent
  React.useEffect(() => {
    const schema = fieldsToSchema(fields)
    onChange(schema)
    // Validate: all fields must have a name
    const valid = fields.every(f => f.name.trim().length > 0)
    onValidation?.(valid)
  }, [fields]) // eslint-disable-line react-hooks/exhaustive-deps

  const addField = () => {
    setFields([...fields, {
      name: '',
      type: 'string',
      description: '',
      required: false,
      default: '',
    }])
  }

  const removeField = (index: number) => {
    setFields(fields.filter((_, i) => i !== index))
  }

  const updateField = (index: number, updates: Partial<InputField>) => {
    setFields(fields.map((f, i) => i === index ? { ...f, ...updates } : f))
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Recipe Inputs
        </Label>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={addField}
          className="gap-1.5 text-xs"
        >
          <Plus className="w-3.5 h-3.5" />
          Add Input
        </Button>
      </div>

      {fields.length === 0 && (
        <p className="text-xs text-muted-foreground py-4 text-center">
          No inputs defined. Click &quot;Add Input&quot; to define parameters this recipe accepts.
        </p>
      )}

      <div className="space-y-3">
        {fields.map((field, index) => (
          <div
            key={index}
            className="glass-card rounded-xl p-4 space-y-3 border border-border/20"
          >
            <div className="flex items-start gap-3">
              <div className="pt-2 text-muted-foreground/40">
                <GripVertical className="w-4 h-4" />
              </div>

              <div className="flex-1 grid grid-cols-2 gap-3">
                {/* Name */}
                <div>
                  <Label className="text-xs text-muted-foreground">Name</Label>
                  <Input
                    value={field.name}
                    onChange={(e) => updateField(index, {
                      name: e.target.value.replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase()
                    })}
                    placeholder="issue_key"
                    className="bg-secondary/50 rounded-lg text-sm h-8"
                  />
                </div>

                {/* Type */}
                <div>
                  <Label className="text-xs text-muted-foreground">Type</Label>
                  <Select
                    value={field.type}
                    onValueChange={(val) => updateField(index, { type: val as InputField['type'] })}
                  >
                    <SelectTrigger className="bg-secondary/50 rounded-lg text-sm h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="string">Text</SelectItem>
                      <SelectItem value="number">Number</SelectItem>
                      <SelectItem value="boolean">Boolean</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Description */}
                <div className="col-span-2">
                  <Label className="text-xs text-muted-foreground">Description</Label>
                  <Input
                    value={field.description}
                    onChange={(e) => updateField(index, { description: e.target.value })}
                    placeholder="The Jira ticket to process"
                    className="bg-secondary/50 rounded-lg text-sm h-8"
                  />
                </div>

                {/* Default value + Required toggle */}
                <div>
                  <Label className="text-xs text-muted-foreground">Default Value</Label>
                  <Input
                    value={field.default}
                    onChange={(e) => updateField(index, { default: e.target.value })}
                    placeholder="(optional)"
                    className="bg-secondary/50 rounded-lg text-sm h-8"
                  />
                </div>

                <div className="flex items-end gap-2 pb-0.5">
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={field.required}
                      onCheckedChange={(checked) => updateField(index, { required: checked })}
                      className="scale-75"
                    />
                    <Label className="text-xs text-muted-foreground">Required</Label>
                  </div>
                </div>
              </div>

              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => removeField(index)}
                className="text-muted-foreground hover:text-destructive h-8 w-8 mt-4"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
