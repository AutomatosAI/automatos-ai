'use client'

import { useState } from 'react'
import { Settings, Key, Webhook, KeyRound, Radio, Brain, Puzzle, Volume2, Bell } from 'lucide-react'
import { CredentialsTab } from './CredentialsTab'
import SystemSettingsTab from './SystemSettingsTab'
import SystemLLMSettingsTab from './SystemLLMSettingsTab'
import WebhooksSettingsTab from './WebhooksSettingsTab'
import { ApiKeysSettingsTab } from './ApiKeysSettingsTab'
import { ChannelsSettingsTab } from './ChannelsSettingsTab'
import { ApiKeyManager } from './ApiKeyManager'
import { VoiceProfilesSettingsTab } from './VoiceProfilesSettingsTab'
import { NotificationsSettingsTab } from './NotificationsSettingsTab'
import { useSystemRole } from '@/contexts/role-context'
import { PageHeader, FilterTabs, TabsContent } from '@/components/shared'

export function SettingsPanel() {
  const { isAdmin } = useSystemRole()
  const [activeTab, setActiveTab] = useState(isAdmin ? 'system-settings' : 'orchestrator')

  const tabs = [
    ...(isAdmin ? [{ value: 'system-settings', label: 'System Settings', icon: Settings }] : []),
    { value: 'orchestrator', label: 'Orchestrator', icon: Brain },
    { value: 'webhooks', label: 'Webhooks', icon: Webhook },
    { value: 'api-keys', label: 'API Keys', icon: KeyRound },
    { value: 'credentials', label: 'Credentials', icon: Key },
    { value: 'channels', label: 'Channels', icon: Radio },
    { value: 'notifications', label: 'Notifications', icon: Bell },
    { value: 'voice-profiles', label: 'Voices', icon: Volume2 },
    { value: 'widget-sdk', label: 'Widget SDK', icon: Puzzle },
  ]

  return (
    <div className="space-y-6">
      <div data-tour="settings-page-header">
        <PageHeader
          title="System"
          titleAccent="Settings"
          subtitle="Manage system configuration, credentials, and security"
        />
      </div>

      <FilterTabs
        tabs={tabs}
        value={activeTab}
        onValueChange={setActiveTab}
        dataTour="settings-tabs"
      >
        {/* System Settings Tab — Admin only */}
        {isAdmin && (
          <TabsContent value="system-settings">
            <SystemSettingsTab />
          </TabsContent>
        )}

        {/* PRD-54/55: Orchestrator Tab (self-loading) */}
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
        <TabsContent value="credentials" data-tour="settings-credentials-tab">
          <CredentialsTab />
        </TabsContent>

        {/* PRD-55: Channels Tab */}
        <TabsContent value="channels">
          <ChannelsSettingsTab />
        </TabsContent>

        {/* PRD-128: Notification Preferences */}
        <TabsContent value="notifications" data-tour="settings-notifications-tab">
          <NotificationsSettingsTab />
        </TabsContent>

        {/* PRD-74: Voice Profiles */}
        <TabsContent value="voice-profiles">
          <VoiceProfilesSettingsTab />
        </TabsContent>

        {/* PRD-38.4: Widget SDK API Keys */}
        <TabsContent value="widget-sdk">
          <ApiKeyManager />
        </TabsContent>
      </FilterTabs>
    </div>
  )
}


