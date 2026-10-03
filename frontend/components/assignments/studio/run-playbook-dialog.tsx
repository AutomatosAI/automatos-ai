'use client'

/**
 * RunPlaybookDialog: the Run button's dialog in the Studio Assignments hub
 * (F242, night 7b). The library's Run, and any link to a playbook
 * (?tab=playbooks&id=<template id or number>), open it.
 *
 * The playbook runs now, with a "Wait for me" switch. The switch starts at the
 * playbook's own setting (execution_config.wait_for_me), the choice its timer's
 * runs and Auto's runs follow. Switched on, this run's card stops in Review for
 * the owner's check instead of closing itself (#0185 went straight to Done).
 * A refusal (a step with no agent, F270) is shown in the server's words.
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useExecutePlaybook, useWorkflowPlaybook } from '@/hooks/use-playbook-api'

export interface RunnablePlaybook {
  id?: number | string
  template_id?: string
  name?: string
  description?: string | null
  execution_config?: { wait_for_me?: boolean } | null
}

/** The playbook's own "wait for me": where the switch starts. */
export function waitsByDefault(playbook: RunnablePlaybook | undefined): boolean {
  return playbook?.execution_config?.wait_for_me === true
}

/** Where a started run is watched (the same page the featured strip opens). */
export function runPageHref(executionId: string, address: string): string {
  return `/activity/execution?id=${encodeURIComponent(executionId)}&recipeId=${encodeURIComponent(address)}`
}

interface RunPlaybookDialogProps {
  /** The playbook's address: its template id or its number (F277). */
  address: string
  onClose: () => void
}

export function RunPlaybookDialog({ address, onClose }: RunPlaybookDialogProps) {
  const router = useRouter()
  const { data, isLoading, isError } = useWorkflowPlaybook(address)
  const playbook = data as RunnablePlaybook | undefined
  const [choice, setChoice] = useState<boolean | null>(null)
  const waitForMe = choice ?? waitsByDefault(playbook)
  const execute = useExecutePlaybook()

  const run = async () => {
    if (!playbook) return
    try {
      const started = (await execute.mutateAsync({ playbookId: address, waitForMe })) as {
        recipe_execution_id?: string
      } | null
      toast('Playbook started', {
        description: waitForMe
          ? `"${playbook.name}" is running. Its card will wait for your check.`
          : `"${playbook.name}" is running.`,
      })
      if (started?.recipe_execution_id) router.push(runPageHref(started.recipe_execution_id, address) as any)
      else onClose()
    } catch (err) {
      toast.error('The playbook did not start', {
        description: err instanceof Error ? err.message : 'Try again in a moment.',
      })
    }
  }

  const about = isError
    ? 'This playbook could not be found in your workspace.'
    : playbook?.description || (isLoading ? 'Loading the playbook…' : '')

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent size="sm">
        <DialogHeader>
          <DialogTitle>{playbook?.name ? `Run ${playbook.name}` : 'Run playbook'}</DialogTitle>
          <DialogDescription>{about}</DialogDescription>
        </DialogHeader>
        <div className="flex items-start justify-between gap-4 rounded-md border border-border p-3">
          <div className="space-y-1">
            <Label htmlFor="run-playbook-wait-for-me">Wait for me</Label>
            <p className="text-xs text-muted-foreground">
              The run&apos;s card stops in Review for your check, instead of closing itself.
            </p>
          </div>
          <Switch
            id="run-playbook-wait-for-me"
            checked={waitForMe}
            onCheckedChange={setChoice}
            disabled={!playbook}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={run} disabled={!playbook || execute.isLoading}>
            {execute.isLoading ? 'Starting…' : 'Run now'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
