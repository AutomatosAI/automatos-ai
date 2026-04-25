'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { FolderTree } from 'lucide-react'

/**
 * Stub route for /deliverables/explorer.
 * US-007 will replace this with the full-page WorkspaceExplorer.
 */
export default function ExplorerPage() {
  return (
    <MainLayout>
      <div className="flex flex-col items-center justify-center h-[calc(100vh-4rem)] text-muted-foreground gap-3">
        <FolderTree className="h-8 w-8" />
        <p className="text-sm">Explorer — full-page mode coming in US-007</p>
      </div>
    </MainLayout>
  )
}
