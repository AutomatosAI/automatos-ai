'use client'

/**
 * Notification preferences settings page (PRD-128 US-009).
 *
 * Standalone route so the NotificationBell dropdown "Settings" link and
 * the settings navigation can both deep-link here.
 */

import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared'
import { NotificationsSettingsTab } from '@/components/settings/NotificationsSettingsTab'
import { usePageAPI } from '@/hooks/use-page-api'

export default function NotificationSettingsPage() {
  usePageAPI('settings')

  return (
    <MainLayout>
      <div className="space-y-6">
        <PageHeader
          title="Notification"
          titleAccent="Preferences"
          subtitle="Choose where each event type notifies you."
        />
        <NotificationsSettingsTab />
      </div>
    </MainLayout>
  )
}
