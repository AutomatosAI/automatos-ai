'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { SettingsPanel } from '@/components/settings/SettingsPanel'
import { usePageAPI } from '@/hooks/use-page-api'

export default function SettingsPage() {
  usePageAPI('settings')
  
  return (
    <MainLayout>
      <SettingsPanel />
    </MainLayout>
  )
}


