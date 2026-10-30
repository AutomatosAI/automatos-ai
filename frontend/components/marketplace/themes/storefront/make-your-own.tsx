'use client'

/**
 * PRD-261 — "Make your own": the Design Kit, and "Submit a theme package for review", which opens
 * C-01b's Publish theme form (publish-theme.tsx) in a dialog, unchanged: every member may submit,
 * and the submit route decides (members:read, the Marketplace switch, step-up). The strip ends on
 * this card; the empty storefront shows the same actions on its stage.
 */
import { useState } from 'react'
import { ExternalLink, PenLine } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'

import { PUBLISH_THEME, PublishThemeForm } from '../publish-theme'
import { DESIGN_KIT_URL, MAKE_YOUR_OWN, MAKE_YOUR_OWN_BODY, OPEN_DESIGN_KIT, SUBMIT_PACKAGE } from './copy'

function PublishDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      {/* The form's own intro describes it (PUBLISH_INTRO), so the dialog needs no second description. */}
      <DialogContent size="md" aria-describedby={undefined} data-testid="publish-theme-dialog">
        <DialogHeader>
          <DialogTitle>{PUBLISH_THEME}</DialogTitle>
        </DialogHeader>
        <PublishThemeForm />
      </DialogContent>
    </Dialog>
  )
}

/** The Design Kit link and the submit button (with its dialog). */
export function MakeYourOwnActions() {
  const [publishing, setPublishing] = useState(false)
  return (
    <div className="flex flex-col gap-2.5">
      <Button asChild variant="secondary" className="min-h-11">
        <a href={DESIGN_KIT_URL} target="_blank" rel="noopener noreferrer">
          {OPEN_DESIGN_KIT}
          <ExternalLink className="ml-2 h-4 w-4" aria-hidden />
        </a>
      </Button>
      <Button variant="link" className="min-h-11 whitespace-normal text-muted-foreground" onClick={() => setPublishing(true)}>
        {SUBMIT_PACKAGE}
      </Button>
      <PublishDialog open={publishing} onClose={() => setPublishing(false)} />
    </div>
  )
}

/** The strip's last card. */
export function MakeYourOwnCard() {
  return (
    <div
      className="flex min-h-[340px] w-[300px] shrink-0 snap-start flex-col justify-between gap-5 rounded-2xl border-2 border-dashed border-border p-7 sm:w-[360px]"
      data-testid="storefront-make-your-own"
    >
      <div className="flex flex-col gap-3">
        <PenLine className="h-8 w-8 text-foreground" aria-hidden />
        <p className="text-lg font-semibold text-foreground">{MAKE_YOUR_OWN}</p>
        <p className="text-sm leading-relaxed text-muted-foreground">{MAKE_YOUR_OWN_BODY}</p>
      </div>
      <MakeYourOwnActions />
    </div>
  )
}
