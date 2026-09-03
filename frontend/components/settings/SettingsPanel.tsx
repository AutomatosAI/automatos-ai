'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Settings, Key, Webhook, KeyRound, Radio, Brain, Puzzle, Bell, UserCircle, ArrowRight, TerminalSquare } from 'lucide-react'
import { CredentialsTab } from './CredentialsTab'
import SystemSettingsTab from './SystemSettingsTab'
import SystemLLMSettingsTab from './SystemLLMSettingsTab'
import WebhooksSettingsTab from './WebhooksSettingsTab'
import { ApiKeysSettingsTab } from './ApiKeysSettingsTab'
import { ChannelsSettingsTab } from './ChannelsSettingsTab'
import { ApiKeyManager } from './ApiKeyManager'
import { WidgetSdkTab } from './WidgetSdkTab'
import { NotificationsSettingsTab } from './NotificationsSettingsTab'
import { SessionModeTab } from './SessionModeTab'
import { useSystemRole } from '@/contexts/role-context'
import { isLocal } from '@/lib/auth-edition'
import { PageHeader, FilterTabs, TabsContent } from '@/components/shared'

/**
 * PRD-233 S6/S7 — the settings tabs that exist in the local edition. Profile is
 * local-only here (saas profiles live under Clerk's own surface); Webhooks,
 * Channels and Widget SDK are hosted-edition surfaces — inbound webhooks and
 * channel callbacks need a public URL, the widget embed needs the hosted
 * loader — so they are hidden by this explicit list, never by role (the local
 * operator is super_admin by design).
 */
export const LOCAL_EDITION_SETTINGS_TABS: ReadonlySet<string> = new Set([
  'profile',
  'system-settings',
  'orchestrator',
  'api-keys',
  'credentials',
  'notifications',
  'session-mode',  // PRD-234 S4 — local only: the CLI host lane
])

export function SettingsPanel() {
  const { isAdmin } = useSystemRole()
  const [activeTab, setActiveTab] = useState(isAdmin ? 'system-settings' : 'orchestrator')

  const allTabs = [
    ...(isLocal ? [{ value: 'profile', label: 'Profile', icon: UserCircle }] : []),
    ...(isLocal ? [{ value: 'session-mode', label: 'Session mode', icon: TerminalSquare }] : []),
    ...(isAdmin ? [{ value: 'system-settings', label: 'System Settings', icon: Settings }] : []),
    { value: 'orchestrator', label: 'Orchestrator', icon: Brain },
    { value: 'webhooks', label: 'Webhooks', icon: Webhook },
    { value: 'api-keys', label: 'API Keys', icon: KeyRound },
    { value: 'credentials', label: 'Credentials', icon: Key },
    { value: 'channels', label: 'Channels', icon: Radio },
    { value: 'notifications', label: 'Notifications', icon: Bell },
    { value: 'widget-sdk', label: 'Widget SDK', icon: Puzzle },
  ]
  const tabs = isLocal ? allTabs.filter((tab) => LOCAL_EDITION_SETTINGS_TABS.has(tab.value)) : allTabs

  return (
    <div className="space-y-6">
      <div>
        <PageHeader
          title="System"
          titleAccent="Settings"
          eyebrow="Workspace · configuration"
          lede="Defaults that apply across every agent, mission, and routing decision. Workspace inherits to teams; teams to individuals."
        />
      </div>

      <FilterTabs
        tabs={tabs}
        value={activeTab}
        onValueChange={setActiveTab}
      >
        {/* PRD-233 S6: local edition — the operator's profile lives at /settings/profile */}
        {isLocal && (
          <TabsContent value="profile">
            <Link
              href="/settings/profile"
              className="glass-card flex items-center justify-between rounded-2xl border border-border p-6 transition-colors hover:bg-secondary/40"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/15">
                  <UserCircle className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-base font-semibold text-foreground">Your profile</p>
                  <p className="text-sm text-muted-foreground">
                    Name, username and avatar for this instance&apos;s operator. Auto greets you by the name you set here.
                  </p>
                </div>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground" />
            </Link>
          </TabsContent>
        )}

        {/* System Settings Tab — Admin only */}
        {isAdmin && (
          <TabsContent value="system-settings">
            <SystemSettingsTab />
          </TabsContent>
        )}

        {/* PRD-54/55: Orchestrator Tab (self-loading) */}
        {/* PRD-234 S4: local edition — session mode (CLI host) status + pairing */}
        {isLocal && (
          <TabsContent value="session-mode">
            <SessionModeTab />
          </TabsContent>
        )}

        <TabsContent value="orchestrator">
          <SystemLLMSettingsTab />
        </TabsContent>

        {/* Webhooks Tab */}
        <TabsContent value="webhooks">
          <WebhooksSettingsTab />
        </TabsContent>

        {/* PRD-54: BYOK API Keys Tab */}
        <TabsContent value="api-keys">
          <ApiKeysSettingsTab />
        </TabsContent>

        {/* PRD-18: Credentials Management Tab */}
        <TabsContent value="credentials">
          <CredentialsTab />
        </TabsContent>

        {/* PRD-55: Channels Tab */}
        <TabsContent value="channels">
          <ChannelsSettingsTab />
        </TabsContent>

        {/* PRD-128: Notification Preferences */}
        <TabsContent value="notifications">
          <NotificationsSettingsTab />
        </TabsContent>

        {/* PRD-008-A: Widget SDK — sites, behaviour, callback, API keys */}
        <TabsContent value="widget-sdk">
          <WidgetSdkTab />
        </TabsContent>
      </FilterTabs>
    </div>
  )
}


