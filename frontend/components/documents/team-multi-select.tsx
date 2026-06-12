'use client'

/**
 * PRD-158 S2 — TeamMultiSelect.
 *
 * Reusable multi-select for a document's team_access, sourced from /api/teams
 * (no more free-text). Selected values are canonical normalized_name strings, so
 * what's persisted is always normalized. Admins can create a team inline.
 */

import { useState } from 'react'
import { Plus, ChevronsUpDown, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useTeams, useCreateTeam } from '@/hooks/use-teams'

interface TeamMultiSelectProps {
  /** Selected teams as canonical normalized_name values. */
  value: string[]
  onChange: (teams: string[]) => void
  disabled?: boolean
  allowCreate?: boolean
  placeholder?: string
}

export function TeamMultiSelect({
  value,
  onChange,
  disabled = false,
  allowCreate = true,
  placeholder = 'Select teams…',
}: TeamMultiSelectProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const { data: teams = [], isLoading } = useTeams()
  const createTeam = useCreateTeam()

  const toggle = (normalized: string) => {
    onChange(
      value.includes(normalized)
        ? value.filter((t) => t !== normalized)
        : [...value, normalized]
    )
  }

  const term = search.trim().toLowerCase()
  const filtered = teams.filter((t) => t.name.toLowerCase().includes(term))
  const exactExists = teams.some((t) => t.normalized_name === term)
  const displayName = (normalized: string) =>
    teams.find((t) => t.normalized_name === normalized)?.name || normalized

  const handleCreate = async () => {
    const name = search.trim()
    if (!name) return
    try {
      const team = await createTeam.mutateAsync(name)
      if (team?.normalized_name && !value.includes(team.normalized_name)) {
        onChange([...value, team.normalized_name])
      }
      setSearch('')
    } catch {
      /* toast handled in the hook */
    }
  }

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            disabled={disabled}
            className="w-full justify-between font-normal"
          >
            <span className="truncate text-muted-foreground">
              {value.length === 0
                ? placeholder
                : `${value.length} team${value.length > 1 ? 's' : ''} selected`}
            </span>
            <ChevronsUpDown className="w-4 h-4 opacity-50 shrink-0" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[var(--radix-popover-trigger-width)] min-w-[14rem] p-0" align="start">
          <div className="p-2 border-b border-border">
            <Input
              autoFocus
              placeholder={allowCreate ? 'Search or create…' : 'Search…'}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="h-8"
            />
          </div>
          <div className="max-h-60 overflow-y-auto p-1">
            {isLoading && (
              <p className="px-2 py-3 text-sm text-muted-foreground">Loading teams…</p>
            )}
            {!isLoading && filtered.length === 0 && !(allowCreate && term) && (
              <p className="px-2 py-3 text-sm text-muted-foreground">No teams</p>
            )}
            {filtered.map((team) => (
              <button
                key={team.id}
                type="button"
                onClick={() => toggle(team.normalized_name)}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-accent"
              >
                <Checkbox
                  checked={value.includes(team.normalized_name)}
                  className="pointer-events-none"
                />
                <span className="truncate">{team.name}</span>
              </button>
            ))}
            {allowCreate && term && !exactExists && (
              <button
                type="button"
                onClick={handleCreate}
                disabled={createTeam.isPending}
                className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm text-primary hover:bg-accent disabled:opacity-50"
              >
                <Plus className="w-4 h-4" /> Create &quot;{search.trim()}&quot;
              </button>
            )}
          </div>
        </PopoverContent>
      </Popover>

      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((normalized) => (
            <Badge
              key={normalized}
              variant="outline"
              className="cursor-pointer"
              onClick={() => toggle(normalized)}
            >
              {displayName(normalized)} <X className="w-3 h-3 ml-1" />
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}
