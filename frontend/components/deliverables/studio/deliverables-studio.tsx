'use client'

/**
 * PRD-244 W5b (D8) — Deliverables in the Studio style: the editorial head,
 * in-page tabs driven by ?tab= (the hub's pattern), and the same tab bodies
 * the Classic route mounts. Classic is untouched.
 */
import { useCallback, useMemo } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { ArrowLeft, Loader2 } from 'lucide-react'
import { OutputsFeed } from '@/components/deliverables/outputs-feed'
import { DeliverablesBlog } from '@/components/deliverables/deliverables-blogs'
import { TemplateStudio } from '@/components/documents/blocks/TemplateStudio'
import { GalleryView } from '@/components/workspace/gallery-view'
import { useWorkspace } from '@/components/workspace-provider'
import { DEFAULT_FILTERS, FEED_DEFAULT_FILTERS, type FilterState } from '@/hooks/use-deliverables-api'
import { deliverableLabel, isDeliverableType } from '@/components/icons/deliverable-icon'
import { DELIVERABLE_TABS, type DeliverableTab } from '@/lib/deliverables/tabs'

const TAB_LABELS: Record<DeliverableTab, string> = {
  outputs: 'Outputs',
  blogs: 'Blogs',
  templates: 'Templates',
}
/** The file explorer is its own route; it sits on the strip as the last tab. */
const EXPLORER_HREF = '/deliverables/explorer'

function resolveTab(param: string | null): DeliverableTab {
  return param && (DELIVERABLE_TABS as readonly string[]).includes(param) ? (param as DeliverableTab) : 'outputs'
}

export function DeliverablesStudio() {
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()
  const router = useRouter()

  const tab = resolveTab(searchParams?.get('tab') ?? null)
  const artifactTypeParam = searchParams?.get('artifact_type') ?? null

  const drilldownTitle = useMemo(
    () => (artifactTypeParam && isDeliverableType(artifactTypeParam) ? deliverableLabel(artifactTypeParam) : null),
    [artifactTypeParam],
  )
  const drilldownFilters = useMemo<FilterState>(
    () => (artifactTypeParam ? { ...FEED_DEFAULT_FILTERS, artifact_type: artifactTypeParam } : DEFAULT_FILTERS),
    [artifactTypeParam],
  )

  const selectTab = useCallback(
    (next: DeliverableTab) => router.replace(`/deliverables?tab=${next}` as any),
    [router],
  )
  const backToFeed = useCallback(() => router.replace('/deliverables?tab=outputs' as any), [router])

  if (isLoading || !workspace) {
    return (
      <div className="cc-page" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" aria-label="Loading" />
      </div>
    )
  }

  return (
    <div className="cc-page">
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">Outputs · the work that landed · {TAB_LABELS[tab]}</p>
          <h1 className="cc-h1">Deliverables</h1>
          <p className="cc-sub">
            Every file, report, draft, and template your agents produced. Open, share, fork into a
            new mission, or post to the books.
          </p>
        </div>
      </div>

      <nav className="cc-tabs" aria-label="Deliverables sections">
        {DELIVERABLE_TABS.map((key) => (
          <button
            key={key}
            type="button"
            className={`cc-tab${tab === key ? ' active' : ''}`}
            aria-current={tab === key ? 'page' : undefined}
            onClick={() => selectTab(key)}
          >
            <span>{TAB_LABELS[key]}</span>
          </button>
        ))}
        <button type="button" className="cc-tab" onClick={() => router.push(EXPLORER_HREF as any)}>
          <span>Explorer</span>
        </button>
      </nav>

      {tab === 'outputs' &&
        (drilldownTitle ? (
          <section style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="cc-toolbar">
              <button type="button" className="cc-btn" onClick={backToFeed}>
                <ArrowLeft style={{ width: 12, height: 12 }} />
                Back to feed
              </button>
              <span className="cc-eyebrow-sm">Showing all {drilldownTitle}</span>
            </div>
            <GalleryView workspaceId={workspace.id} initialFilters={drilldownFilters} />
          </section>
        ) : (
          <OutputsFeed />
        ))}
      {tab === 'blogs' && <DeliverablesBlog variant="studio" />}
      {tab === 'templates' && <TemplateStudio />}
    </div>
  )
}
