'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { WorkspaceExplorer } from '@/components/workspace/WorkspaceExplorer'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'
import { HardDrive, Loader2 } from 'lucide-react'

export default function WorkspacePage() {
  usePageAPI('workspace')
  const { workspace, isLoading } = useWorkspace()

  return (
    <MainLayout>
      {isLoading || !workspace ? (
        <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="h-[calc(100vh-4rem)] flex flex-col">
          <div className="flex items-center gap-2 px-4 py-2 border-b border-border/30">
            <HardDrive className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Workspace</span>
            <span className="text-xs text-muted-foreground">
              Files, reports & agent output
            </span>
          </div>
          <div className="flex-1 min-h-0">
            <WorkspaceExplorer
              workspaceId={workspace.id}
              className="h-full"
            />
          </div>
        </div>
      )}
    </MainLayout>
  )
}
