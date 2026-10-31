'use client'

/**
 * PRD-270 US-270-08b (§5.7, K20) — a visitor's item detail: what the public read says of it (its
 * publisher, description, category, tags, the apps it needs, its own plain facts, a package's
 * members or an LLM's capabilities) and Add to my workspace, which goes to sign-in. Text only:
 * nothing here is the item's configuration, prompt or steps (the public read has none of them).
 *
 * The one Marketplace page opens it signed out for a package, an application, an LLM, a plugin or a
 * skill (public-item-detail.tsx): their member dialogs read member routes or install from inside.
 * An icon is drawn only as an `https` logo (the app's img-src decides where from) or a short text
 * such as an emoji, else the tab's own icon.
 */
import type { LucideIcon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { ToolLogo } from '@/components/ui/tool-logo'
import { HTTPS_PREFIX } from '@/lib/marketplace/public-items'
import type { PublicItem } from '@/lib/marketplace/public-items'

import { CAPABILITIES, IN_THIS_PACKAGE, REQUIRED_APPS, TAGS, byPublisher } from './copy'
import { VisitorAdd } from './visitor-add'

const ICON_SIZE = 40

function ItemIcon({ item, fallback: Fallback }: { item: PublicItem; fallback: LucideIcon }) {
  if (item.icon?.startsWith(HTTPS_PREFIX)) return <ToolLogo logo={item.icon} name={item.name} size={ICON_SIZE} />
  if (item.icon) {
    return (
      <span className="flex h-10 w-10 items-center justify-center text-2xl" aria-hidden>
        {item.icon}
      </span>
    )
  }
  return <Fallback className="h-10 w-10 shrink-0 text-primary" aria-hidden />
}

interface VisitorItemDetailProps {
  /** The item shown; null closes the dialog. */
  readonly item: PublicItem | null
  /** The Marketplace tab the item's link names (`?tab=`). */
  readonly tab: string
  readonly icon: LucideIcon
  readonly onClose: () => void
}

function Words({ heading, words, outline = false }: { heading: string; words: readonly string[]; outline?: boolean }) {
  if (words.length === 0) return null
  return (
    <section aria-label={heading} className="space-y-2">
      <p className="text-sm font-medium text-foreground">{heading}</p>
      <div className="flex flex-wrap gap-1.5">
        {words.map((word, index) => (
          <Badge key={`${index}:${word}`} variant={outline ? 'outline' : 'secondary'}>
            {word}
          </Badge>
        ))}
      </div>
    </section>
  )
}

function Facts({ item }: { item: PublicItem }) {
  const facts = item.category ? [{ label: 'Category', value: item.category }, ...item.facts] : item.facts
  if (facts.length === 0) return null
  return (
    <dl className="space-y-1.5">
      {facts.map((fact) => (
        <div key={fact.label} className="flex gap-2 text-sm">
          <dt className="w-32 shrink-0 text-muted-foreground">{fact.label}</dt>
          <dd className="min-w-0 break-words text-foreground">{fact.value}</dd>
        </div>
      ))}
    </dl>
  )
}

function DetailBody({ item, tab, icon }: { item: PublicItem; tab: string; icon: LucideIcon }) {
  const includesHeading = tab === 'llms' ? CAPABILITIES : IN_THIS_PACKAGE
  return (
    <div className="space-y-5">
      <DialogHeader>
        <div className="flex items-center gap-3">
          <ItemIcon item={item} fallback={icon} />
          <div className="min-w-0 space-y-1">
            <DialogTitle className="break-words">{item.name}</DialogTitle>
            {item.publisher && <p className="text-sm text-muted-foreground">{byPublisher(item.publisher)}</p>}
          </div>
        </div>
        {item.description && <DialogDescription className="whitespace-pre-line">{item.description}</DialogDescription>}
      </DialogHeader>
      <VisitorAdd tab={tab} item={item.key} />
      <Facts item={item} />
      <Words heading={REQUIRED_APPS} words={item.requiredApps} outline />
      <Words heading={includesHeading} words={item.includes} />
      <Words heading={TAGS} words={item.tags} />
    </div>
  )
}

/** An item with no description has no DialogDescription, so the dialog says it describes nothing. */
const NO_DESCRIPTION = { 'aria-describedby': undefined }

export function VisitorItemDetail({ item, tab, icon, onClose }: VisitorItemDetailProps) {
  return (
    <Dialog open={item !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        className="max-h-[85vh] max-w-2xl overflow-y-auto"
        data-testid="visitor-item-detail"
        {...(item?.description ? {} : NO_DESCRIPTION)}
      >
        {item && <DetailBody key={item.key} item={item} tab={tab} icon={icon} />}
      </DialogContent>
    </Dialog>
  )
}
