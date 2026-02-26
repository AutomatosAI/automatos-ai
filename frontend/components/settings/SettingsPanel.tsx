'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Settings, Key, Shield, Webhook, KeyRound, Radio, Brain, FileText, Puzzle } from 'lucide-react'
import { CredentialsTab } from './CredentialsTab'
import { CredentialAuditTab } from './CredentialAuditTab'
import SystemSettingsTab from './SystemSettingsTab'
import SystemLLMSettingsTab from './SystemLLMSettingsTab'
import WebhooksSettingsTab from './WebhooksSettingsTab'
import { ApiKeysSettingsTab } from './ApiKeysSettingsTab'
import { ChannelsSettingsTab } from './ChannelsSettingsTab'
import { SystemPromptsTab } from './SystemPromptsTab'
import { ApiKeyManager } from './ApiKeyManager'

export function SettingsPanel() {

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage system configuration, credentials, and security
        </p>
      </div>

      <Tabs defaultValue="system-settings" className="space-y-6">
        <TabsList className="w-full justify-start gap-1">
          <TabsTrigger value="system-settings">
            <Settings className="w-4 h-4 mr-1 shrink-0" />
            <span className="hidden sm:inline">System</span> Settings
          </TabsTrigger>
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
          <TabsTrigger value="credentials">
            <Key className="w-4 h-4 mr-1 shrink-0" />
            Credentials
          </TabsTrigger>
          <TabsTrigger value="audit">
            <Shield className="w-4 h-4 mr-1 shrink-0" />
            Audit <span className="hidden sm:inline">Logs</span>
          </TabsTrigger>
          <TabsTrigger value="channels">
            <Radio className="w-4 h-4 mr-1 shrink-0" />
            Channels
          </TabsTrigger>
          <TabsTrigger value="prompts">
            <FileText className="w-4 h-4 mr-1 shrink-0" />
            <span className="hidden sm:inline">System</span> Prompts
          </TabsTrigger>
          <TabsTrigger value="widget-sdk">
            <Puzzle className="w-4 h-4 mr-1 shrink-0" />
            Widget SDK
          </TabsTrigger>
        </TabsList>

        {/* System Settings Tab */}
        <TabsContent value="system-settings">
          <SystemSettingsTab />
        </TabsContent>

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

        {/* PRD-18: Audit Logs Tab */}
        <TabsContent value="audit">
          <CredentialAuditTab />
        </TabsContent>

        {/* PRD-55: Channels Tab */}
        <TabsContent value="channels">
          <ChannelsSettingsTab />
        </TabsContent>

        {/* PRD-58: System Prompts Tab */}
        <TabsContent value="prompts">
          <SystemPromptsTab />
        </TabsContent>

        {/* PRD-38.4: Widget SDK API Keys */}
        <TabsContent value="widget-sdk">
          <ApiKeyManager />
        </TabsContent>
      </Tabs>
    </div>
  )
}


