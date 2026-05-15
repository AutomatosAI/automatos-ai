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
          eyebrow="Account · alerts"
          lede="Decide which event types page you, which go to Slack, and which sit quietly in the log. You can change these any time."
        />
        <NotificationsSettingsTab />
      </div>
    </MainLayout>
  )
}
