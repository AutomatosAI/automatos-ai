

import { MainLayout } from '@/components/layout/main-layout'
// TODO: Tools component needs to be moved from duplicate folder
// import { ToolsDashboard } from '@/components/tools/tools-dashboard'

export default function ToolsPage() {
  return (
    <MainLayout>
      <div className="p-6">
        <h1 className="text-2xl font-bold">Tools Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Tools component needs to be migrated from duplicate folder.
        </p>
      </div>
    </MainLayout>
  )
}
