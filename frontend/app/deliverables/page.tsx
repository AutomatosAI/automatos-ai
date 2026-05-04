'use client'

import { useCallback, useRef } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import {
  Loader2,
  LayoutGrid,
  FileText,
  BookOpen,
  FolderTree,
} from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader, FilterTabs, TabsContent } from '@/components/shared'
import { CreatedToday } from '@/components/deliverables/created-today'
import { DeliverablesBlog } from '@/components/deliverables/deliverables-blogs'
import { TemplateManager } from '@/components/documents/template-manager'
import { GalleryView } from '@/components/workspace/gallery-view'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'

type DeliverableTab = 'outputs' | 'blogs' | 'templates'

const VALID_TABS: ReadonlyArray<DeliverableTab> = ['outputs', 'blogs', 'templates']

function resolveTab(param: string | null): DeliverableTab {
  if (param && VALID_TABS.includes(param as DeliverableTab)) return param as DeliverableTab
  return 'outputs'
}

export default function DeliverablesPage() {
  usePageAPI('workspace')
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()
  const router = useRouter()
  const galleryRef = useRef<HTMLDivElement>(null)

  const activeTab = resolveTab(searchParams?.get('tab') ?? null)

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

  const handleBrowseRecent = useCallback(() => {
    galleryRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  return (
    <MainLayout>
      {isLoading || !workspace ? (
        <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="space-y-6">
          <PageHeader
            title=""
            titleAccent="Deliverables"
            subtitle="Files, reports & agent output"
          />
          <FilterTabs
            tabs={[
              { value: 'outputs', label: 'Outputs', icon: LayoutGrid },
              { value: 'blogs', label: 'Blogs', icon: FileText },
              { value: 'templates', label: 'Templates', icon: BookOpen },
              { value: 'explorer', label: 'Explorer', icon: FolderTree },
            ]}
            value={activeTab}
            onValueChange={handleTabChange}
            dataTour="deliverables-tabs"
          >
            <TabsContent value="outputs">
              <div className="mx-auto max-w-[1600px] space-y-6">
                <CreatedToday onBrowseRecent={handleBrowseRecent} />
                <div ref={galleryRef}>
                  <GalleryView workspaceId={workspace.id} />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="blogs">
              <div className="mx-auto max-w-[1600px]">
                <DeliverablesBlog />
              </div>
            </TabsContent>

            <TabsContent value="templates">
              <div className="mx-auto max-w-[1600px]">
                <TemplateManager />
              </div>
            </TabsContent>
          </FilterTabs>
        </div>
      )}
    </MainLayout>
  )
}
