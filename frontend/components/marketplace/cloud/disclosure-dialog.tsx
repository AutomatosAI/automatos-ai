'use client'

/**
 * PRD-270 W2 US-270-14c (§5.6 "Before install the user sees the prompt text and the tools and apps the item
 * will use", K18, K19, K30) — the disclosure: what an Automatos Cloud agent, Playbook, package or plugin
 * will do and use, read before it is added.
 *
 * `CloudDisclosureDialog({kind, id, open, onAdd, onClose})` reads the item's local disclosure when it opens
 * (hooks/use-cloud-items.ts `useCloudDisclosure`, by the key its install takes) and shows its name,
 * publisher, version, when it was synced and both badges, then its sections (disclosure-sections.tsx).
 * "Add to my workspace" is the view's one primary; it calls `onAdd` once per press and stays pressed while
 * the install `onAdd` returns is pending, and Close (or the dialog's own close) calls `onClose`. While it
 * reads, a loading line; a failed read, or an answer it can't show whole (another kind's, or one missing
 * what a section shows), says so and offers no Add. The layer (cloud-items-layer.tsx, US-270-14d) owns what
 * Add and Close do.
 */
import { useState } from 'react'
import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@/components/ui/dialog'
import { useCloudDisclosure } from '@/hooks/use-cloud-items'
import type { CloudItemKey, CloudItemKind } from '@/lib/marketplace/cloud-item-rows'
import type { CloudDisclosure } from '@/lib/marketplace/cloud-items'

import { CloudBadgePair } from './cloud-item-badges'
import { DisclosureSections } from './disclosure-sections'
import {
  ADDING, ADD_TO_WORKSPACE, CLOSE, CLOUD_ITEM_TITLE, DISCLOSURE_LEDE, DISCLOSURE_UNREADABLE, READING_DISCLOSURE,
  publishedLine, syncedOn,
} from './item-copy'

interface CloudDisclosureDialogProps {
  kind: CloudItemKind
  /** The key its install takes: an agent's or a Playbook's id, a package's slug, a plugin's uuid. */
  id: CloudItemKey
  open: boolean
  /** Called once per press; while a promise it returns is pending, Add stays pressed. */
  onAdd: () => void | Promise<unknown>
  onClose: () => void
}

type StateProps = Omit<CloudDisclosureDialogProps, 'open'>

function CloseButton({ onClose }: { onClose: () => void }) {
  return (
    <Button variant="outline" onClick={onClose} data-testid="cloud-disclosure-close">
      {CLOSE}
    </Button>
  )
}

/** The loading line, or the failed read's sentence: Close only, never Add. */
function DisclosureMessage({ kind, failed, onClose }: { kind: CloudItemKind; failed: boolean; onClose: () => void }) {
  return (
    <>
      <DialogHeader>
        <DialogTitle>{failed ? CLOUD_ITEM_TITLE : READING_DISCLOSURE}</DialogTitle>
        <DialogDescription className={failed ? 'text-destructive' : undefined}>
          {failed ? DISCLOSURE_UNREADABLE : DISCLOSURE_LEDE[kind]}
        </DialogDescription>
      </DialogHeader>
      <DialogFooter><CloseButton onClose={onClose} /></DialogFooter>
    </>
  )
}

function DisclosureHead({ disclosure }: { disclosure: CloudDisclosure }) {
  return (
    <DialogHeader className="space-y-2 border-b border-border/30 pb-4">
      <DialogTitle className="break-words pr-6 text-xl">{disclosure.name}</DialogTitle>
      <DialogDescription>{DISCLOSURE_LEDE[disclosure.kind]}</DialogDescription>
      <p className="text-sm text-muted-foreground">
        {publishedLine(disclosure.publisher, disclosure.version, syncedOn(disclosure.syncedAt))}
      </p>
      <CloudBadgePair />
    </DialogHeader>
  )
}

function DisclosureBody({ disclosure, onAdd, onClose }: { disclosure: CloudDisclosure } & Omit<StateProps, 'kind' | 'id'>) {
  const [adding, setAdding] = useState(false)
  const add = () => {
    if (adding) return
    setAdding(true)
    const settled: unknown = onAdd()
    // An install the layer awaits: Add can be pressed again once it ends (the layer reports a failure).
    if (settled instanceof Promise) settled.then(() => setAdding(false), () => setAdding(false))
  }
  return (
    <>
      <DisclosureHead disclosure={disclosure} />
      <DisclosureSections disclosure={disclosure} />
      <DialogFooter className="gap-2 border-t border-border/30 pt-4 sm:gap-0">
        <CloseButton onClose={onClose} />
        <Button onClick={add} disabled={adding} data-testid="cloud-disclosure-add">
          <Download className="mr-2 h-4 w-4" aria-hidden />
          {adding ? ADDING : ADD_TO_WORKSPACE}
        </Button>
      </DialogFooter>
    </>
  )
}

function DisclosureState({ kind, id, onAdd, onClose }: StateProps) {
  const disclosure = useCloudDisclosure(kind, id)
  // A read that never starts (the hosted edition) is not loading: it ends as a failed one.
  if (disclosure.isInitialLoading) {
    return <DisclosureMessage kind={kind} failed={false} onClose={onClose} />
  }
  // The read refuses an answer for another kind, or one it can't show whole (cloudDisclosureOf).
  if (disclosure.isError || !disclosure.data) {
    return <DisclosureMessage kind={kind} failed onClose={onClose} />
  }
  return <DisclosureBody disclosure={disclosure.data} onAdd={onAdd} onClose={onClose} />
}

export function CloudDisclosureDialog({ kind, id, open, onAdd, onClose }: CloudDisclosureDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent size="md" className="max-h-[85vh] overflow-y-auto" data-testid="cloud-disclosure">
        {open && <DisclosureState key={`${kind}:${String(id)}`} kind={kind} id={id} onAdd={onAdd} onClose={onClose} />}
      </DialogContent>
    </Dialog>
  )
}
