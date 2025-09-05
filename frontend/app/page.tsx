
import { EnhancedMainLayout } from '@/components/layout/enhanced-main-layout'
import { EnhancedDashboard } from '@/components/dashboard/enhanced-dashboard'

export default function Home() {
  return (
    <EnhancedMainLayout>
      <EnhancedDashboard />
    </EnhancedMainLayout>
  )
}
