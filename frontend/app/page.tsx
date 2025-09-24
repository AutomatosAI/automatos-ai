import { MainLayout } from '@/components/layout/main-layout'
import { SimpleDashboard } from '@/components/dashboard/simple-dashboard'

export default function Home() {
  return (
    <MainLayout>
      <SimpleDashboard />
    </MainLayout>
  )
}
