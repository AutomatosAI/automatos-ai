'use client'

import { useCallback, useRef } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import {
  Package,
  Loader2,
  LayoutGrid,
  FileText,
  BookOpen,
  FolderTree,
} from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
import { CreatedToday } from '@/components/deliverables/created-today'
import { DeliverablesBlog } from '@/components/deliverables/deliverables-blogs'
import { TemplatesTab } from '@/components/workflows/templates-tab'
import { GalleryView } from '@/components/workspace/gallery-view'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

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

  const handleUseTemplate = useCallback(
    (template: { template_id?: string; name?: string }) => {
      router.push(
        `/assignments?tab=playbooks&templateId=${encodeURIComponent(template.template_id ?? '')}`,
      )
    },
    [router],
  )

  return (
    <MainLayout>
      {isLoading || !workspace ? (
        <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="h-[calc(100vh-4rem)] flex flex-col">
          <div className="flex items-center gap-3 px-4 py-2 border-b border-border/30">
            <Package className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Deliverables</span>
            <span className="text-xs text-muted-foreground">
              Files, reports & agent output
            </span>
          </div>
          <Tabs
            value={activeTab}
            onValueChange={handleTabChange}
            className="flex-1 flex flex-col min-h-0"
          >
            <div className="px-4 pt-2">
              <TabsList data-tour="deliverables-tabs" className="bg-secondary/50">
                <TabsTrigger value="outputs" className="flex items-center gap-1.5">
                  <LayoutGrid className="h-4 w-4" />
                  Outputs
                </TabsTrigger>
                <TabsTrigger value="blogs" className="flex items-center gap-1.5">
                  <FileText className="h-4 w-4" />
                  Blogs
                </TabsTrigger>
                <TabsTrigger value="templates" className="flex items-center gap-1.5">
                  <BookOpen className="h-4 w-4" />
                  Templates
                </TabsTrigger>
                <TabsTrigger value="explorer" className="flex items-center gap-1.5">
                  <FolderTree className="h-4 w-4" />
                  Explorer
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="outputs" className="flex-1 min-h-0 mt-0">
              <div className="h-full overflow-y-auto p-4">
                <div className="mx-auto max-w-[1600px] space-y-6">
                  <CreatedToday onBrowseRecent={handleBrowseRecent} />
                  <div ref={galleryRef}>
                    <GalleryView workspaceId={workspace.id} />
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="blogs" className="flex-1 min-h-0 mt-0">
              <div className="h-full overflow-y-auto p-4">
                <div className="mx-auto max-w-[1600px]">
                  <DeliverablesBlog />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="templates" className="flex-1 min-h-0 mt-0">
              <div className="h-full overflow-y-auto p-4">
                <div className="mx-auto max-w-[1600px]">
                  <TemplatesTab onUseTemplate={handleUseTemplate} />
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </MainLayout>
  )
}
