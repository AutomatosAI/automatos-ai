'use client'

/**
 * PRD-270 US-270-08a (§5.7, K19) — the Marketplace for a signed-out visitor on SaaS.
 *
 * The slim visitor header and the Marketplace's own heading, outside the app shell (no MainLayout:
 * no sidebar, workspace card, notifications or ⌘K workspace search). Its tabs read only the public
 * catalogue routes; nothing here reads or names a workspace.
 */
import { PageHeader } from '@/components/shared/page-header'
import { VisitorHeader } from './visitor-header'

export function VisitorMarketplace() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <VisitorHeader />
      <main className="mx-auto max-w-6xl space-y-8 px-4 py-8 md:px-6">
        <PageHeader
          title="Community"
          titleAccent="Marketplace"
          eyebrow="Automatos AI · Marketplace"
          lede="Themes, agents, Playbooks, packages and tools for Automatos. Browse everything here; sign in to add any of it to your workspace."
        />
      </main>
    </div>
  )
}
