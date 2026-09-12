'use client'

import React, { useEffect, useState } from 'react'
import { BookOpen, Bot, Braces, ChevronDown, ChevronUp, Palette, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

const DISMISSED_KEY = 'automatos.templates.guide.dismissed'

function readDismissed(): boolean {
  try {
    return localStorage.getItem(DISMISSED_KEY) === '1'
  } catch {
    return false
  }
}

function writeDismissed(value: boolean) {
  try {
    if (value) localStorage.setItem(DISMISSED_KEY, '1')
    else localStorage.removeItem(DISMISSED_KEY)
  } catch {
    /* private mode / blocked storage — the guide simply shows again next time */
  }
}

const STEPS: { icon: React.ComponentType<{ className?: string }>; title: string; body: string }[] = [
  {
    icon: Palette,
    title: '1 · Set your Brand Kit once',
    body: 'Upload your logo and pick your colours. Every document rendered from any template picks them up — {{brand.*}} and {{company.*}} chips fill in from here and from your business profile.',
  },
  {
    icon: Braces,
    title: '2 · Build a template from blocks',
    body: 'Headings, text, tables, your logo. Drop in chips: {{user.name}} and {{date.long}} fill themselves; {{data.title}} and other data.* chips are the fields an agent (or you) supplies when a document is generated.',
  },
  {
    icon: Bot,
    title: '3 · Use it from chat, a schedule, or a playbook',
    body: 'Tell Auto “generate the Weekly Report with the … template” — it lists your templates, reads which data.* fields each needs, fills them from its research, and saves the PDF/DOCX to Deliverables with a share link you can email.',
  },
]

interface TemplateGuideProps {
  onOpenBrandKit: () => void
  className?: string
}

// "How templates work" — a dismissable, per-browser panel (PRD-242 S5). The three
// steps are the whole feature in one screen so a first-time user never has to guess.
export function TemplateGuide({ onOpenBrandKit, className }: TemplateGuideProps) {
  const [dismissed, setDismissed] = useState(true) // assume dismissed until storage is read (no flash)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    setDismissed(readDismissed())
  }, [])

  if (dismissed) {
    return (
      <div className={cn('flex justify-end', className)}>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs text-muted-foreground"
          onClick={() => {
            writeDismissed(false)
            setDismissed(false)
          }}
        >
          <BookOpen className="h-3.5 w-3.5" /> How templates work
        </Button>
      </div>
    )
  }

  return (
    <Card className={cn('border-primary/20 bg-primary/[0.03]', className)}>
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">How templates work</h3>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              aria-label={collapsed ? 'Expand guide' : 'Collapse guide'}
              onClick={() => setCollapsed((c) => !c)}
            >
              {collapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              aria-label="Dismiss guide"
              onClick={() => {
                writeDismissed(true)
                setDismissed(true)
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {!collapsed && (
          <>
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              {STEPS.map(({ icon: Icon, title, body }) => (
                <div key={title} className="flex gap-3">
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-background border">
                    <Icon className="h-4 w-4 text-primary" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium">{title}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Button size="sm" variant="outline" onClick={onOpenBrandKit}>
                <Palette className="mr-2 h-4 w-4" /> Start with the Brand Kit
              </Button>
              <p className="text-xs text-muted-foreground">
                A document with an unfilled chip is never delivered — the preview shows every gap before you save.
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
