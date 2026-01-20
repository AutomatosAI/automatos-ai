/**
 * Edit MCP Tool Modal
 * ===================
 * Complete edit functionality matching create modal
 */

'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Zap, 
  Key,
  Info,
  X,
  AlertTriangle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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
  mcp_server_url?: string
  requiredCredentials?: string[]
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

  // Fetch full tool details when modal opens
  useEffect(() => {
    const fetchFullTool = async () => {
      if (tool && tool.id && open) {
        try {
          console.log(`🔍 Fetching full tool details for ID: ${tool.id}, Name: ${tool.name}`)
          const response = await apiClient.getMCPTool(tool.id)
          // Handle different response structures
          const fullToolData = response?.data || response
          console.log(`✅ Received tool data:`, {
            id: fullToolData?.id,
            name: fullToolData?.name,
            hasMetadata: !!fullToolData?.metadata,
            metadata: fullToolData?.metadata,
            credentialType: fullToolData?.metadata?.credential_type,
            rawResponse: response
          })
          setFullTool(fullToolData)
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
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          />

          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card w-full max-w-5xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <Zap className="w-6 h-6 text-brand-primary" />
                  <span>Tool: {currentTool.name}</span>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={handleClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="pr-2">
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2 bg-secondary/50">
                    <TabsTrigger value="credentials">1. Credentials</TabsTrigger>
                    <TabsTrigger value="details">2. Details</TabsTrigger>
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
                          <Badge variant="outline" className="text-xs bg-gray-800/50 text-gray-300 border-gray-700">
                            Required
                          </Badge>
                        </div>

                        {/* Existing Credential Status Box */}
                        {existingCredential && (
                          <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                            <p className="text-sm text-blue-400">
                              Updating existing credential: <strong className="text-blue-300">{existingCredential.name}</strong>
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
                            <div className="flex items-start gap-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                              <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                              <p className="text-sm text-yellow-400">
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
                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
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
