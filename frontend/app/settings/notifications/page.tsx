'use client'

/**
 * Notification preferences settings page (PRD-128 US-009).
 *
 * Standalone route so the NotificationBell dropdown "Settings" link and
 * the settings navigation can both deep-link here.
 */

import { MainLayout } from '@/components/layout/main-layout'
import { NotificationsSettingsTab } from '@/components/settings/NotificationsSettingsTab'
import { usePageAPI } from '@/hooks/use-page-api'

export default function NotificationSettingsPage() {
  usePageAPI('settings')

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">Notifications</h1>
          <p className="text-muted-foreground mt-1">
            Choose where each event type notifies you.
          </p>
        </div>
        <NotificationsSettingsTab />
      </div>
    </MainLayout>
  )
}
