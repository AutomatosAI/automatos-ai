/**
 * Tool Configuration Modal
 * ========================
 * Configure credentials and view integration features
 */

'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Zap,
  Key,
  Info,
  AlertTriangle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { DynamicCredentialForm } from '@/components/settings/DynamicCredentialForm'
import { ToolLogo } from '@/components/ui/tool-logo'
import {
  getCredentialTypeByName,
  listCredentials,
  type CredentialType,
  type Credential
} from '@/lib/api/credentials'
import { apiClient } from '@/lib/api-client'
import {
  useAppActions,
  useInitiateConnection,
  useConnectedApps,
  useDisconnectApp,
  type ComposioAction
} from '@/hooks/use-composio-api'

interface Tool {
  id: number
  name: string
  description: string
  category: string
  icon: string
  provider: string
  status: string
  version: string
  tags: string[]
  capabilities: any
  credentials_schema: any
  metadata: any
  integration_url?: string
  requiredCredentials?: string[]
  composio_app_name?: string  // PRD-36: Composio app identifier
  source?: 'composio' | 'internal'  // Tool source
}

interface ToolConfigModalProps {
  open: boolean
  onClose: () => void
  tool: Tool | null
}

export function ToolConfigModal({ open, onClose, tool }: ToolConfigModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    provider: '',
    version: '1.0.0',
    icon: '⚡',
    category: 'developer',
    status: 'active',
    tags: '',
  })
  const [activeTab, setActiveTab] = useState('credentials')
  const [credentialType, setCredentialType] = useState<CredentialType | null>(null)
  const [existingCredential, setExistingCredential] = useState<Credential | null>(null)
  const [credentialsLoading, setCredentialsLoading] = useState(false)
  const [resetFormKey, setResetFormKey] = useState(0)
  const [fullTool, setFullTool] = useState<Tool | null>(null)

  // PRD-36: Composio Integration
  const isComposioApp = !!(tool?.composio_app_name || tool?.metadata?.composio_app_name || tool?.source === 'composio')
  const composioAppName = tool?.composio_app_name || tool?.metadata?.composio_app_name || ''
  // IMPORTANT: do not hit Composio endpoints unless the modal is actually open.
  const { data: connections = [] } = useConnectedApps({ enabled: open && isComposioApp })
  const initiateConnectionMutation = useInitiateConnection()
  const disconnectMutation = useDisconnectApp()

  const isConnected = connections.some(
    (c) => c.app_name.toUpperCase() === composioAppName.toUpperCase() && c.status === 'active'
  )
  const { data: appActions = [] } = useAppActions(composioAppName, { enabled: open && isComposioApp && isConnected })

  // Fetch full tool details when modal opens
  useEffect(() => {
    const fetchFullTool = async () => {
      if (tool && tool.id && open) {
        try {
          // Tool details are already provided by the Tools list.
          setFullTool(tool)
        } catch (error: any) {
          console.error('❌ Failed to fetch full tool details:', {
            error: error?.message || error,
            status: error?.status,
            response: error?.response
          })
          // Fallback to provided tool, but log what we have
          console.log('⚠️ Falling back to provided tool:', {
            hasTool: !!tool,
            toolId: tool?.id,
            toolName: tool?.name,
            hasMetadata: !!tool?.metadata,
            metadata: tool?.metadata
          })
          setFullTool(tool)
        }
      } else if (tool) {
        console.log('Using provided tool (no fetch needed):', tool.name)
        setFullTool(tool)
      }
    }
    fetchFullTool()
  }, [tool, open])

  useEffect(() => {
    const currentTool = fullTool || tool
    if (currentTool) {
      setFormData({
        name: currentTool.name || '',
        description: currentTool.description || '',
        provider: currentTool.provider || '',
        version: currentTool.version || '1.0.0',
        icon: currentTool.icon || '⚡',
        category: currentTool.category || 'developer',
        status: currentTool.status || 'active',
        tags: (currentTool.tags || []).join(', '),
      })
      setActiveTab('credentials')
      setResetFormKey((prev) => prev + 1)
    }
  }, [fullTool, tool])

  const resolveCredentialType = async (rawType: string): Promise<CredentialType | null> => {
    if (!rawType) return null

    const normalized = toSnakeCase(rawType)
    // Try various naming patterns to find the credential type
    const candidates = [
      rawType,                                    // Original: "slack"
      normalized,                                 // Normalized: "slack"
      `${rawType}_api`,                          // "slack_api"
      `${normalized}_api`,                       // "slack_api"
      `${rawType}_credentials`,                  // "slack_credentials"
      `${normalized}_credentials`,               // "slack_credentials"
      `${rawType}Api`,                           // "slackApi" (camelCase)
      `${rawType}OAuth2Api`,                     // "slackOAuth2Api"
      `${rawType}_oauth2_api`,                   // "slack_oauth2_api"
      // Try with capitalized first letter
      rawType.charAt(0).toUpperCase() + rawType.slice(1), // "Slack"
      normalized.charAt(0).toUpperCase() + normalized.slice(1), // "Slack"
    ].filter(Boolean)

    for (const candidate of candidates) {
      try {
        const type = await getCredentialTypeByName(candidate)
        if (type) return type
      } catch (error) {
        continue
      }
    }

    return null
  }

  useEffect(() => {
    const loadCredentialsForTool = async () => {
      // Use fullTool if available, otherwise fallback to tool
      const currentTool = fullTool || tool

      if (!currentTool) {
        setCredentialType(null)
        setExistingCredential(null)
        return
      }

      // Debug: Log the tool metadata to see what we're working with
      console.log('🔍 Tool metadata:', {
        toolName: currentTool.name,
        metadata: currentTool.metadata,
        hasMetadata: !!currentTool.metadata,
        credentialType: currentTool.metadata?.credential_type,
        autoEnable: currentTool.metadata?.auto_enable_on_credential,
        usingFullTool: !!fullTool
      })

      // Priority: credential_type (explicit) > auto_enable_on_credential (needs resolution)
      const explicitCredentialType = currentTool.metadata?.credential_type
      const autoEnableValue = currentTool.metadata?.auto_enable_on_credential

      if (!explicitCredentialType && !autoEnableValue) {
        console.warn(`⚠️  Tool "${currentTool.name}" has no credential_type or auto_enable_on_credential in metadata`)
        setCredentialType(null)
        setExistingCredential(null)
        setCredentialsLoading(false)
        return
      }

      try {
        setCredentialsLoading(true)
        let type: CredentialType | null = null

        // If we have an explicit credential_type, try to use it directly first
        if (explicitCredentialType) {
          console.log(`🔍 Trying direct lookup for credential_type: "${explicitCredentialType}"`)
          try {
            type = await getCredentialTypeByName(explicitCredentialType)
            console.log(`✅ Found credential type directly:`, type?.name || 'unknown')
          } catch (error: any) {
            // If direct lookup fails, fall through to resolution
            console.error(`❌ Direct lookup failed for credential_type "${explicitCredentialType}":`, {
              error: error?.message || error,
              status: error?.status,
              response: error?.response
            })
          }
        }

        // If no type found yet, try resolution with credential_type or auto_enable_on_credential
        if (!type) {
          const credentialTypeName = explicitCredentialType || autoEnableValue
          if (credentialTypeName) {
            console.log(`🔍 Trying resolution for: "${credentialTypeName}"`)
            try {
              type = await resolveCredentialType(credentialTypeName)
              if (type) {
                console.log(`✅ Found credential type via resolution:`, type.name)
              } else {
                console.warn(`❌ Resolution returned null for: "${credentialTypeName}"`)
              }
            } catch (error: any) {
              console.error(`❌ Resolution threw error for "${credentialTypeName}":`, {
                error: error?.message || error,
                status: error?.status
              })
            }
          }
        }

        if (!type) {
          console.error(`❌ Could not find credential type for tool "${currentTool.name}"`)
          setCredentialType(null)
          setExistingCredential(null)
          return
        }
        setCredentialType(type)

        // Check for existing credentials of this type
        try {
          const response = await listCredentials({
            credential_type_id: type.id,
            active_only: true
          }) as any  // Response is paginated: { items: [...], total, ... }

          // Handle paginated response
          const existingCreds = Array.isArray(response) ? response : (response?.items || [])

          // Use the first matching credential if one exists
          if (existingCreds && existingCreds.length > 0) {
            setExistingCredential(existingCreds[0])
          } else {
            setExistingCredential(null)
          }
        } catch (err) {
          console.error('Failed to check existing credentials:', err)
          setExistingCredential(null)
        }
      } catch (error) {
        console.error('Failed to load credentials for tool:', error)
        setCredentialType(null)
        setExistingCredential(null)
      } finally {
        setCredentialsLoading(false)
      }
    }

    if (open && (fullTool || tool)) {
      loadCredentialsForTool()
    }
  }, [tool, fullTool, open])

  const handleClose = () => {
    onClose()
  }

  const currentTool = fullTool || tool
  const capabilities = useMemo(
    () => extractCapabilities(currentTool?.capabilities),
    [currentTool?.capabilities]
  )
  const credentialList = useMemo(() => {
    if (!currentTool) return []
    return (
      currentTool.requiredCredentials ||
      Object.keys((currentTool.credentials_schema || {}).required || {})
    )
  }, [currentTool, currentTool?.requiredCredentials, currentTool?.credentials_schema])

  if (!currentTool) return null

  const handleCredentialSaved = async () => {
    setResetFormKey((prev) => prev + 1)
    // Reload credentials to get the updated/created credential
    if (credentialType) {
      try {
        const response = await listCredentials({
          credential_type_id: credentialType.id,
          active_only: true
        }) as any
        const existingCreds = Array.isArray(response) ? response : (response?.items || [])
        if (existingCreds && existingCreds.length > 0) {
          setExistingCredential(existingCreds[0])
        }
      } catch (err) {
        console.error('Failed to reload credentials:', err)
      }
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent size="lg">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Zap className="w-6 h-6 text-brand-primary" />
            <span>Tool: <span className="gradient-text">{currentTool.name}</span></span>
          </DialogTitle>
        </DialogHeader>

        <div className="pr-2">
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className={`grid w-full ${isComposioApp ? 'grid-cols-3' : 'grid-cols-2'} bg-secondary/50`}>
                    <TabsTrigger value="credentials">1. Credentials</TabsTrigger>
                    <TabsTrigger value="details">2. Details</TabsTrigger>
                    {isComposioApp && (
                      <TabsTrigger value="features" className="text-[hsl(var(--agent))]">
                        3. Features
                      </TabsTrigger>
                    )}
                  </TabsList>

                  <TabsContent value="details" className="mt-6 space-y-6 max-h-[60vh] overflow-y-auto">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-6"
                    >
                      <div className="space-y-4">
                        <h4 className="font-medium text-sm">Basic Information</h4>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">Tool Name</p>
                            <p className="font-medium">{formData.name}</p>
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">Provider</p>
                            <p className="font-medium">{formData.provider}</p>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <p className="text-xs text-muted-foreground">Description</p>
                          <p className="text-sm text-muted-foreground">{formData.description}</p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">Category</p>
                            <Badge variant="outline" className="capitalize">
                              {formData.category}
                            </Badge>
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">Version</p>
                            <p className="font-medium">{formData.version}</p>
                          </div>

                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground">Status</p>
                            <Badge variant="secondary" className="capitalize">
                              {formData.status}
                            </Badge>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <p className="text-xs text-muted-foreground">Icon</p>
                          <p className="text-2xl">{formData.icon}</p>
                        </div>
                      </div>

                      {capabilities.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="font-medium text-sm">Capabilities</h4>
                          <div className="flex flex-wrap gap-2">
                            {capabilities.map((capability) => (
                              <Badge key={capability} variant="outline" className="text-xs">
                                {capability}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {credentialList.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="font-medium text-sm flex items-center gap-2">
                            <Key className="w-4 h-4" />
                            Credentials Required
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {credentialList.map((cred) => (
                              <Badge key={cred} variant="secondary" className="text-xs">
                                {cred}
                              </Badge>
                            ))}
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Info className="w-3 h-3" />
                            Configure credentials in the Credentials tab.
                          </div>
                        </div>
                      )}
                    </motion.div>
                  </TabsContent>

                  <TabsContent value="credentials" className="mt-6 space-y-6 max-h-[60vh] overflow-y-auto pr-2">
                    {credentialsLoading ? (
                      <div className="text-sm text-muted-foreground">Loading credential type...</div>
                    ) : !credentialType ? (
                      <div className="rounded-md border border-border/50 bg-secondary/20 p-4 text-sm text-muted-foreground">
                        No credential type is linked to this tool yet.
                      </div>
                    ) : (
                      <div className="space-y-6">
                        {/* Credential Type Info Section */}
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-center gap-3">
                            <ToolLogo
                              logo={credentialType.logo || undefined}
                              name={credentialType.display_name || credentialType.name}
                              size={40}
                              fallbackIcon={credentialType.icon || undefined}
                              showBackground={true}
                            />
                            <div className="space-y-1">
                              <h4 className="font-medium text-sm">{credentialType.display_name}</h4>
                              <p className="text-xs text-muted-foreground">
                                {credentialType.description || 'Provide credentials for this tool.'}
                              </p>
                            </div>
                          </div>
                          <Badge variant="outline" className="text-xs bg-secondary/50 text-muted-foreground border-border/50">
                            Required
                          </Badge>
                        </div>

                        {/* Existing Credential Status Box */}
                        {existingCredential && (
                          <div className="p-4 bg-[hsl(var(--info))]/10 border border-[hsl(var(--info))]/20 rounded-lg">
                            <p className="text-sm text-[hsl(var(--info))]">
                              Updating existing credential: <strong className="text-[hsl(var(--info))]">{existingCredential.name}</strong>
                            </p>
                            <p className="text-xs text-muted-foreground mt-1">
                              Last updated: {new Date(existingCredential.updated_at).toLocaleString('en-GB', {
                                day: '2-digit',
                                month: '2-digit',
                                year: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit'
                              })}
                            </p>
                          </div>
                        )}

                        {/* Configuration Section */}
                        <div className="space-y-4">
                          <h4 className="font-medium text-sm">{credentialType.display_name} Configuration</h4>

                          {/* Security Warning */}
                          {existingCredential && (
                            <div className="flex items-start gap-2 p-3 bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/20 rounded-lg">
                              <AlertTriangle className="w-4 h-4 text-[hsl(var(--warning))] mt-0.5 flex-shrink-0" />
                              <p className="text-sm text-[hsl(var(--warning))]">
                                For security reasons, existing values are not shown. Re-enter values to update.
                              </p>
                            </div>
                          )}

                          {/* Credential Form */}
                          <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                            <DynamicCredentialForm
                              key={resetFormKey}
                              credentialId={existingCredential?.id}
                              lockCredentialTypeId={credentialType.id}
                              lockedCredentialType={credentialType}
                              hideMetadataSection={true}
                              defaultName={existingCredential?.name || `${currentTool.name} Credential`}
                              defaultEnvironment="production"
                              onSuccess={handleCredentialSaved}
                              onCancel={onClose}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                  </TabsContent>

                  {/* PRD-36: Composio Features Tab */}
                  {isComposioApp && (
                    <TabsContent value="features" className="mt-6 space-y-6 max-h-[60vh] overflow-y-auto pr-2">
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-6"
                      >
                        {/* Connection Status */}
                        <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-[hsl(var(--success))]' : 'bg-muted-foreground'}`} />
                              <div>
                                <p className="font-medium">{composioAppName}</p>
                                <p className="text-sm text-muted-foreground">
                                  {isConnected ? 'Connected - Ready to use' : 'Not connected - Click to connect'}
                                </p>
                              </div>
                            </div>
                            {isConnected ? (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => disconnectMutation.mutate(composioAppName)}
                                className="text-destructive hover:text-destructive/80"
                              >
                                Disconnect
                              </Button>
                            ) : (
                              <Button
                                variant="outline"
                                onClick={() => {
                                  initiateConnectionMutation.mutate({
                                    appName: composioAppName,
                                    callbackUrl: `${window.location.origin}/tools/callback?connected=${composioAppName}`,
                                  })
                                }}
                              >
                                Connect {composioAppName}
                              </Button>
                            )}
                          </div>
                        </div>

                        {/* Available Actions */}
                        <div className="space-y-3">
                          <h4 className="font-medium text-sm">Available Actions ({appActions.length})</h4>
                          {appActions.length === 0 ? (
                            <p className="text-sm text-muted-foreground">
                              {isConnected ? 'Loading actions...' : 'Connect the app to see available actions'}
                            </p>
                          ) : (
                            <div className="grid grid-cols-1 gap-2 max-h-60 overflow-y-auto">
                              {appActions.map((action: ComposioAction) => (
                                <div
                                  key={action.name}
                                  className="flex items-center justify-between p-3 bg-background/50 rounded-lg border border-border/50"
                                >
                                  <div>
                                    <p className="font-medium text-sm">{action.display_name}</p>
                                    {action.description && (
                                      <p className="text-xs text-muted-foreground">{action.description}</p>
                                    )}
                                  </div>
                                  <Badge variant="outline" className={action.enabled ? 'text-[hsl(var(--success))]' : 'text-muted-foreground'}>
                                    {action.enabled ? 'Enabled' : 'Disabled'}
                                  </Badge>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </motion.div>
                    </TabsContent>
                  )}
                </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function extractCapabilities(capabilities: any): string[] {
  if (!capabilities) return []
  if (Array.isArray(capabilities)) return capabilities

  const knownKeys = ['methods', 'tools', 'actions', 'operations', 'capabilities']
  for (const key of knownKeys) {
    const value = capabilities[key]
    if (Array.isArray(value)) return value
    if (value && typeof value === 'object') return Object.keys(value)
  }

  if (typeof capabilities === 'object') {
    return Object.keys(capabilities)
  }

  return []
}

function toSnakeCase(value: string): string {
  if (!value) return value
  return value
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s\-]+/g, '_')
    .toLowerCase()
}
