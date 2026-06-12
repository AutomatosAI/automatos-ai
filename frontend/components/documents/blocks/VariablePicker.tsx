'use client'

import React, { useState } from 'react'
import { Braces } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import type { VariableEntry } from './types'

interface VariablePickerProps {
  variables: VariableEntry[]
  onInsert: (path: string) => void
  disabled?: boolean
  label?: string
}

// Inserts a {{path}} chip. Grouped by category, searchable, shows the resolved value
// (or sample) so authors see what a chip will render to (PRD-167 S5).
export function VariablePicker({ variables, onInsert, disabled, label = 'Variable' }: VariablePickerProps) {
  const [open, setOpen] = useState(false)

  const byCategory = variables.reduce<Record<string, VariableEntry[]>>((acc, v) => {
    ;(acc[v.category] ||= []).push(v)
    return acc
  }, {})

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button type="button" variant="outline" size="sm" disabled={disabled} className="h-7 gap-1.5">
          <Braces className="h-3.5 w-3.5" />
          {label}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-72 p-0" align="start">
        <Command>
          <CommandInput placeholder="Search variables…" />
          <CommandList>
            <CommandEmpty>No variables found.</CommandEmpty>
            {Object.entries(byCategory).map(([category, entries]) => (
              <CommandGroup key={category} heading={category}>
                {entries.map((entry) => (
                  <CommandItem
                    key={entry.path}
                    value={`${entry.path} ${entry.label}`}
                    onSelect={() => {
                      onInsert(entry.path)
                      setOpen(false)
                    }}
                  >
                    <div className="flex flex-col">
                      <span className="font-mono text-xs">{`{{${entry.path}}}`}</span>
                      <span className="text-xs text-muted-foreground">
                        {entry.label}
                        {entry.value ? ` · ${entry.value}` : entry.sample ? ` · e.g. ${entry.sample}` : ''}
                      </span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
