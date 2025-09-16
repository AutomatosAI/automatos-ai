import { MainLayout } from '@/components/layout/main-layout'
import { EnhancedDashboard } from '@/components/dashboard/enhanced-dashboard'

export default function Home() {
  return (
    <MainLayout>
      <EnhancedDashboard />
    </MainLayout>
  )
}
