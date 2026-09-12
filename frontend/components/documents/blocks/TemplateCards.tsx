'use client'

import React from 'react'
import { Copy, FileText, Pencil, Play, Sparkles, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { EmptyState } from '@/components/shared'
import { UseWithAutoPopover } from './UseWithAutoPopover'
import type { TemplateSummary } from './types'

interface TemplateCardsProps {
  templates: TemplateSummary[]
  onEdit: (t: TemplateSummary) => void
  onCopy: (t: TemplateSummary) => void
  onGenerate: (t: TemplateSummary) => void
  onDelete: (t: TemplateSummary) => void
  onCreate: () => void
}

function IconAction({
  label,
  onClick,
  children,
  destructive,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
  destructive?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className={destructive ? 'h-7 w-7 text-destructive hover:text-destructive' : 'h-7 w-7'}
          aria-label={label}
          onClick={onClick}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function FieldsLine({ t }: { t: TemplateSummary }) {
  if (t.has_blocks && t.data_fields.length === 0) {
    return <p className="text-xs text-muted-foreground">No fill-in fields — renders from your brand kit and profile alone.</p>
  }
  if (!t.has_blocks) {
    return (
      <p className="text-xs text-muted-foreground">
        Built-in layout · fills from its sample data structure (not block-editable — copy it to start a block version).
      </p>
    )
  }
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-xs text-muted-foreground">Needs:</span>
      {t.data_fields.map((f) => (
        <code key={f} className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
          {f}
        </code>
      ))}
    </div>
  )
}

// Gallery cards (PRD-242 S5): what each template IS (starter / block / legacy), what it
// NEEDS (data.* chips), and every action a non-technical user might want, in one place.
export function TemplateCards({ templates, onEdit, onCopy, onGenerate, onDelete, onCreate }: TemplateCardsProps) {
  if (templates.length === 0) {
    return (
      <EmptyState
        icon={FileText}
        title="No templates yet"
        description="Start from a blank page, or ask Auto to draft one — every document your agents generate can use it."
        action={
          <Button onClick={onCreate}>
            <Sparkles className="mr-2 h-4 w-4" /> New template
          </Button>
        }
      />
    )
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {templates.map((t) => (
          <Card key={t.id} className="group flex flex-col transition-colors hover:border-primary/30">
            <CardContent className="flex flex-1 flex-col p-5">
              <div className="mb-2 flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <h3 className="truncate font-semibold">{t.name}</h3>
                  <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">{t.description || 'No description'}</p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-1">
                  <Badge variant="outline" className="uppercase">{String(t.format)}</Badge>
                  {t.is_starter && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Badge variant="secondary" className="text-[10px]">Starter</Badge>
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        A platform starter. Copy it to customise — your copy is yours to edit or delete; the starter stays.
                      </TooltipContent>
                    </Tooltip>
                  )}
                </div>
              </div>

              <div className="mb-3">
                <FieldsLine t={t} />
              </div>

              <div className="mt-auto flex items-center justify-between gap-2 pt-2">
                <Badge variant="secondary" className="text-xs">{t.category}</Badge>
                <div className="flex items-center gap-0.5">
                  {t.has_blocks && !t.is_starter && (
                    <IconAction label="Edit" onClick={() => onEdit(t)}>
                      <Pencil className="h-3.5 w-3.5" />
                    </IconAction>
                  )}
                  <IconAction label={t.has_blocks ? 'Copy to customise' : 'Copy as a block template'} onClick={() => onCopy(t)}>
                    <Copy className="h-3.5 w-3.5" />
                  </IconAction>
                  <IconAction label="Generate a document now" onClick={() => onGenerate(t)}>
                    <Play className="h-3.5 w-3.5" />
                  </IconAction>
                  {!t.is_starter && (
                    <IconAction label="Delete" destructive onClick={() => onDelete(t)}>
                      <Trash2 className="h-3.5 w-3.5" />
                    </IconAction>
                  )}
                  <UseWithAutoPopover template={t} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </TooltipProvider>
  )
}
