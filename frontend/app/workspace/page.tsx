'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { Activity, HardDrive, Loader2 } from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
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
  // If a file path is in the URL, the user is deep-linking into a specific
  // file — force Explorer so Code Canvas routing still works.
  if (hasPath) return 'explorer'
  if (viewParam === 'explorer' || viewParam === 'activity') return viewParam
  return 'gallery'
}

export default function WorkspacePage() {
  usePageAPI('workspace')
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()

  const viewParam = searchParams?.get('view') ?? null
  const pathParam = searchParams?.get('path') ?? null

  const [view, setView] = useState<WorkspaceView>(() =>
    resolveInitialView(viewParam, Boolean(pathParam)),
  )

  // Re-sync when the URL changes (e.g. "Open in Canvas" from the preview).
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
            <HardDrive className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Workspace</span>
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
              <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-muted-foreground">
                <Activity className="h-12 w-12" strokeWidth={1.5} />
                <div className="text-base font-medium text-foreground">
                  Activity view coming soon
                </div>
                <div className="max-w-sm text-sm">
                  A live timeline of agent actions in this workspace will land
                  in a later release.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </MainLayout>
  )
}
