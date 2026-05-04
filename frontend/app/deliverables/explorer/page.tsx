'use client'

import { useEffect } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { ArrowLeft, FolderTree, Loader2 } from 'lucide-react'

import { MainLayout } from '@/components/layout/main-layout'
import { WorkspaceExplorer } from '@/components/workspace/WorkspaceExplorer'
import { useWorkspace } from '@/components/workspace-provider'
import { usePageAPI } from '@/hooks/use-page-api'
import { Button } from '@/components/ui/button'
import { PageHeader } from '@/components/shared'

/**
 * /deliverables/explorer — Full-page workspace file browser.
 *
 * Renders the existing WorkspaceExplorer (file tree + editor + terminal)
 * in viewport-full mode. The Deliverables tab strip is hidden here — this
 * is a mode switch, not a tab within the Deliverables page.
 *
 * Supports ?path=... deep-linking (passed through to WorkspaceExplorer).
 */
export default function ExplorerPage() {
  usePageAPI('workspace')
  const { workspace, isLoading } = useWorkspace()
  const searchParams = useSearchParams()
  const router = useRouter()

  const pathParam = searchParams?.get('path') ?? null

  // ESC keybinding → return to Deliverables
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        router.push('/deliverables?tab=outputs')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [router])

  return (
    <MainLayout>
      {isLoading || !workspace ? (
        <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <div className="h-[calc(100vh-4rem)] flex flex-col">
          {/* Header — Back button + title */}
          <div className="px-4 py-2 border-b border-border/30">
            <PageHeader
              title="Workspace"
              titleAccent="Explorer"
              subtitle="Browse and edit workspace files"
              actions={
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-1.5 text-muted-foreground hover:text-foreground"
                  onClick={() => router.push('/deliverables?tab=outputs')}
                >
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Back to Deliverables
                </Button>
              }
            />
          </div>

          {/* WorkspaceExplorer fills remaining space */}
          <div className="flex-1 min-h-0">
            <WorkspaceExplorer
              workspaceId={workspace.id}
              initialFilePath={pathParam}
              className="h-full"
            />
          </div>
        </div>
      )}
    </MainLayout>
  )
}
