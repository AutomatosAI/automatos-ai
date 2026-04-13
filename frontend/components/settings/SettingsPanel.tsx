'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Settings, Key, Webhook, KeyRound, Radio, Brain, Puzzle, Volume2, Bell, Rocket } from 'lucide-react'
import { CredentialsTab } from './CredentialsTab'
import SystemSettingsTab from './SystemSettingsTab'
import SystemLLMSettingsTab from './SystemLLMSettingsTab'
import WebhooksSettingsTab from './WebhooksSettingsTab'
import { ApiKeysSettingsTab } from './ApiKeysSettingsTab'
import { ChannelsSettingsTab } from './ChannelsSettingsTab'
import { ApiKeyManager } from './ApiKeyManager'
import { VoiceProfilesSettingsTab } from './VoiceProfilesSettingsTab'
import { useSystemRole } from '@/contexts/role-context'

export function SettingsPanel() {
  const { isAdmin } = useSystemRole()

  return (
    <div className="space-y-6">
      <div data-tour="settings-page-header">
        <h1 className="text-2xl md:text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage system configuration, credentials, and security
        </p>
      </div>

      <Tabs defaultValue={isAdmin ? "system-settings" : "orchestrator"} className="space-y-6">
        <TabsList data-tour="settings-tabs" className="w-full justify-start gap-1">
          {isAdmin && (
            <TabsTrigger value="system-settings">
              <Settings className="w-4 h-4 mr-1 shrink-0" />
              <span className="hidden sm:inline">System</span> Settings
            </TabsTrigger>
          )}
          <TabsTrigger value="orchestrator">
            <Brain className="w-4 h-4 mr-1 shrink-0" />
            Orchestrator
          </TabsTrigger>
          <TabsTrigger value="webhooks">
            <Webhook className="w-4 h-4 mr-1 shrink-0" />
            Webhooks
          </TabsTrigger>
          <TabsTrigger value="api-keys">
            <KeyRound className="w-4 h-4 mr-1 shrink-0" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="credentials" data-tour="settings-credentials-tab">
            <Key className="w-4 h-4 mr-1 shrink-0" />
            Credentials
          </TabsTrigger>
          <TabsTrigger value="channels">
            <Radio className="w-4 h-4 mr-1 shrink-0" />
            Channels
          </TabsTrigger>
          <TabsTrigger value="notifications" data-tour="settings-notifications-tab">
            <Bell className="w-4 h-4 mr-1 shrink-0" />
            Notifications
          </TabsTrigger>
          <TabsTrigger value="voice-profiles">
            <Volume2 className="w-4 h-4 mr-1 shrink-0" />
            Voices
          </TabsTrigger>
          <TabsTrigger value="widget-sdk">
            <Puzzle className="w-4 h-4 mr-1 shrink-0" />
            Widget SDK
          </TabsTrigger>
          <TabsTrigger value="onboarding-agents">
            <Rocket className="w-4 h-4 mr-1 shrink-0" />
            Onboarding
          </TabsTrigger>
        </TabsList>

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

        {/* PRD-74: Voice Profiles */}
        <TabsContent value="voice-profiles">
          <VoiceProfilesSettingsTab />
        </TabsContent>

        {/* PRD-38.4: Widget SDK API Keys */}
        <TabsContent value="widget-sdk">
          <ApiKeyManager />
        </TabsContent>

        {/* Mission Zero: Onboarding Agents */}
        <TabsContent value="onboarding-agents">
          <OnboardingAgentsTab />
        </TabsContent>
      </Tabs>
    </div>
  )
}


