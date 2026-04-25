'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { Package, Loader2 } from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
import { ActivityFeed } from '@/components/activity/activity-feed'
import { GalleryView } from '@/components/workspace/gallery-view'
import { WorkspaceExplorer } from '@/components/workspace/WorkspaceExplorer'
import {
  WorkspaceViewToggle,
  type WorkspaceView,
} from '@/components/workspace/workspace-view-toggle'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'

function resolveInitialView(
  viewParam: string | null,
  hasPath: boolean,
): WorkspaceView {
  if (hasPath) return 'explorer'
  if (viewParam === 'explorer' || viewParam === 'activity') return viewParam
  return 'gallery'
}

export default function DeliverablesPage() {
  usePageAPI('workspace')
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()

  const viewParam = searchParams?.get('view') ?? null
  const pathParam = searchParams?.get('path') ?? null

  const [view, setView] = useState<WorkspaceView>(() =>
    resolveInitialView(viewParam, Boolean(pathParam)),
  )

  useEffect(() => {
    setView(resolveInitialView(viewParam, Boolean(pathParam)))
  }, [viewParam, pathParam])

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
            <div className="ml-auto">
              <WorkspaceViewToggle view={view} onViewChange={setView} />
            </div>
          </div>
          <div className="flex-1 min-h-0">
            {view === 'gallery' && (
              <div className="h-full overflow-y-auto p-4">
                <GalleryView
                  workspaceId={workspace.id}
                  className="mx-auto max-w-[1600px]"
                />
              </div>
            )}
            {view === 'explorer' && (
              <WorkspaceExplorer
                workspaceId={workspace.id}
                initialFilePath={pathParam}
                className="h-full"
              />
            )}
            {view === 'activity' && (
              <div className="h-full overflow-y-auto">
                <ActivityFeed />
              </div>
            )}
          </div>
        </div>
      )}
    </MainLayout>
  )
}
