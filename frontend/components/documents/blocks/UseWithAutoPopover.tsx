'use client'

import React, { useState } from 'react'
import { Bot, Check, Copy } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { autoPrompt, emailPrompt, playbookStepJson, schedulePrompt } from './promptSnippets'
import type { TemplateSummary } from './types'

interface UseWithAutoPopoverProps {
  template: Pick<TemplateSummary, 'id' | 'name' | 'format' | 'data_fields'>
  size?: 'sm' | 'default'
}

function CopyBlock({ text, hint }: { text: string; hint: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast.success('Copied')
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error('Could not copy — select the text and copy it manually')
    }
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">{hint}</p>
      <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-md border bg-muted/40 p-2 text-[11px] leading-relaxed">
        {text}
      </pre>
      <Button type="button" size="sm" variant="outline" className="h-7 gap-1.5" onClick={copy}>
        {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />} Copy
      </Button>
    </div>
  )
}

// The "how do I actually use this?" answer, per template (PRD-242 S5): a prompt for
// chat, one for a schedule, one that emails the result, and a playbook step.
export function UseWithAutoPopover({ template, size = 'sm' }: UseWithAutoPopoverProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button type="button" variant="outline" size={size} className={size === 'sm' ? 'h-7 gap-1.5' : 'gap-2'}>
          <Bot className="h-3.5 w-3.5" /> Use with Auto
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[26rem] max-w-[95vw] p-3">
        <p className="mb-2 text-sm font-medium">Use “{template.name}” from…</p>
        <Tabs defaultValue="chat">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="schedule">Schedule</TabsTrigger>
            <TabsTrigger value="email">Email</TabsTrigger>
            <TabsTrigger value="playbook">Playbook</TabsTrigger>
          </TabsList>
          <TabsContent value="chat" className="mt-3">
            <CopyBlock
              hint="Paste into the chat with Auto (or any agent with document tools). Replace the placeholder with what to research."
              text={autoPrompt(template)}
            />
          </TabsContent>
          <TabsContent value="schedule" className="mt-3">
            <CopyBlock
              hint="Auto files it as a scheduled task; each run produces a fresh document in Deliverables."
              text={schedulePrompt(template)}
            />
          </TabsContent>
          <TabsContent value="email" className="mt-3">
            <CopyBlock
              hint="Needs a connected email app (Composio Gmail/Outlook). The agent sends the 7-day share link — no sign-in needed to open it."
              text={emailPrompt(template)}
            />
          </TabsContent>
          <TabsContent value="playbook" className="mt-3">
            <CopyBlock
              hint="A deterministic playbook step: a research step (step 1) feeds this one. Pass it as a step in the playbook's JSON definition."
              text={playbookStepJson(template)}
            />
          </TabsContent>
        </Tabs>
      </PopoverContent>
    </Popover>
  )
}
