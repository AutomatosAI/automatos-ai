'use client'

import { useCallback, useMemo } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import {
  ArrowLeft,
  Loader2,
  LayoutGrid,
  FileText,
  BookOpen,
  FolderTree,
  Share2,
} from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader, FilterTabs, TabsContent, type FilterTab } from '@/components/shared'
import { Button } from '@/components/ui/button'
import { OutputsFeed } from '@/components/deliverables/outputs-feed'
import { DeliverablesBlog } from '@/components/deliverables/deliverables-blogs'
import { TemplateStudio } from '@/components/documents/blocks/TemplateStudio'
import { GalleryView } from '@/components/workspace/gallery-view'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { DeliverablesStudio } from '@/components/deliverables/studio/deliverables-studio'
import { SocialsTab } from '@/components/deliverables/socials/socials-tab'
import {
  DEFAULT_FILTERS,
  FEED_DEFAULT_FILTERS,
  type FilterState,
} from '@/hooks/use-deliverables-api'
import { deliverableLabel, isDeliverableType } from '@/components/icons/deliverable-icon'

import { resolveDeliverableTab, visibleDeliverableTabs, type DeliverableTab } from '@/lib/deliverables/tabs'

const TAB_META: Record<DeliverableTab, Omit<FilterTab, 'value'>> = {
  outputs: { label: 'Outputs', icon: LayoutGrid },
  blogs: { label: 'Blogs', icon: FileText },
  templates: { label: 'Templates', icon: BookOpen },
  socials: { label: 'Socials', icon: Share2 },
}
const EXPLORER_TAB: FilterTab = { value: 'explorer', label: 'Explorer', icon: FolderTree }

export default function DeliverablesPage() {
  usePageAPI('workspace')
  const isStudio = useIsStudio()
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()
  const router = useRouter()

  // PRD-251 D1: the Socials tab exists only while the platform offers Socials.
  const socialsAvailable = workspace?.socials?.available === true
  const activeTab = resolveDeliverableTab(searchParams?.get('tab') ?? null, socialsAvailable)
  const filterTabs = useMemo<FilterTab[]>(
    () => [
      ...visibleDeliverableTabs(socialsAvailable).map((value) => ({ value, ...TAB_META[value] })),
      EXPLORER_TAB,
    ],
    [socialsAvailable],
  )
  const artifactTypeParam = searchParams?.get('artifact_type') ?? null

  const drilldownTitle = useMemo(() => {
    if (artifactTypeParam && isDeliverableType(artifactTypeParam)) {
      return deliverableLabel(artifactTypeParam)
    }
    return null
  }, [artifactTypeParam])

  const drilldownFilters = useMemo<FilterState>(() => {
    if (!artifactTypeParam) return DEFAULT_FILTERS
    return {
      ...FEED_DEFAULT_FILTERS,
      artifact_type: artifactTypeParam,
    }
  }, [artifactTypeParam])

  const handleTabChange = useCallback(
    (value: string) => {
      if (value === 'explorer') {
        router.push('/deliverables/explorer')
        return
      }
      router.replace(`/deliverables?tab=${value}`)
    },
    [router],
  )

  const handleBackToFeed = useCallback(() => {
    router.replace('/deliverables?tab=outputs')
  }, [router])

  // PRD-244 W5b (two styles): the Studio style renders the Studio page on
  // desktop; the Classic style keeps this page. Below 1024 px both styles use
  // Classic until the mobile pass (PRD-245).
  if (isStudio) {
    return (
      <MainLayout fullBleed>
        <DeliverablesStudio />
      </MainLayout>
    )
  }

  return (
    <MainLayout>
      {isLoading || !workspace ? (
        <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="space-y-6">
          <PageHeader
            title="Your"
            titleAccent="Deliverables"
            eyebrow="Outputs · the work that landed"
            lede="Every file, report, draft, and template your agents produced. Open, share, fork into a new mission, or post to the books."
          />
          <FilterTabs
            tabs={filterTabs}
            value={activeTab}
            onValueChange={handleTabChange}
          >
            <TabsContent value="outputs">
              <div className="mx-auto max-w-[1600px]">
                {drilldownTitle ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="gap-1.5 text-muted-foreground hover:text-foreground"
                        onClick={handleBackToFeed}
                      >
                        <ArrowLeft className="h-3.5 w-3.5" />
                        Back to feed
                      </Button>
                      <h2 className="text-sm font-medium text-muted-foreground">
                        Showing all <span className="text-foreground">{drilldownTitle}</span>
                      </h2>
                    </div>
                    <GalleryView
                      workspaceId={workspace.id}
                      initialFilters={drilldownFilters}
                    />
                  </div>
                ) : (
                  <OutputsFeed />
                )}
              </div>
            </TabsContent>

            <TabsContent value="blogs">
              <div className="mx-auto max-w-[1600px]">
                <DeliverablesBlog />
              </div>
            </TabsContent>

            <TabsContent value="templates">
              <div className="mx-auto max-w-[1600px]">
                <TemplateStudio />
              </div>
            </TabsContent>

            {socialsAvailable && (
              <TabsContent value="socials">
                <div className="mx-auto max-w-[1600px]">
                  <SocialsTab />
                </div>
              </TabsContent>
            )}
          </FilterTabs>
        </div>
      )}
    </MainLayout>
  )
}
