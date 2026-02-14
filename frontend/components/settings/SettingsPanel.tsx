'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Settings, Key, FileText, Shield, Webhook } from 'lucide-react'
import { CredentialsTab } from './CredentialsTab'
import { CredentialTypesTab } from './CredentialTypesTab'
import { CredentialAuditTab } from './CredentialAuditTab'
import SystemSettingsTab from './SystemSettingsTab'
import WebhooksSettingsTab from './WebhooksSettingsTab'

export function SettingsPanel() {

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Manage system configuration, credentials, and security
        </p>
      </div>

      <Tabs defaultValue="system-settings" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="system-settings">
            <Settings className="w-4 h-4 mr-2" />
            System Settings
          </TabsTrigger>
          <TabsTrigger value="webhooks">
            <Webhook className="w-4 h-4 mr-2" />
            Webhooks
          </TabsTrigger>
          <TabsTrigger value="credentials">
            <Key className="w-4 h-4 mr-2" />
            Credentials
          </TabsTrigger>
          <TabsTrigger value="credential-types">
            <FileText className="w-4 h-4 mr-2" />
            Credential Types
          </TabsTrigger>
          <TabsTrigger value="audit">
            <Shield className="w-4 h-4 mr-2" />
            Audit Logs
          </TabsTrigger>
        </TabsList>

        {/* System Settings Tab */}
        <TabsContent value="system-settings">
          <SystemSettingsTab />
        </TabsContent>

        {/* Webhooks Tab */}
        <TabsContent value="webhooks">
          <WebhooksSettingsTab />
        </TabsContent>

        {/* PRD-18: Credentials Management Tab */}
        <TabsContent value="credentials">
          <CredentialsTab />
        </TabsContent>

        {/* PRD-18: Credential Types Browser Tab */}
        <TabsContent value="credential-types">
          <CredentialTypesTab />
        </TabsContent>

        {/* PRD-18: Audit Logs Tab */}
        <TabsContent value="audit">
          <CredentialAuditTab />
        </TabsContent>
      </Tabs>
    </div>
  )
}


