'use client'

import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ToolLogo } from '@/components/ui/tool-logo'
import { 
  ExternalLink, 
  BookOpen, 
  Settings, 
  Download, 
  Tag, 
  Shield,
  Zap,
  Database,
  Cloud,
  MessageSquare,
  Terminal,
  Container,
  CreditCard,
  Activity,
  HardDrive,
  Globe,
  Key,
  X
} from 'lucide-react'

interface ToolDetailsModalProps {
  open: boolean
  onClose: () => void
  tool: any
  onInstall?: () => void
  onConfigure?: () => void
  onUninstall?: () => void
  loading?: boolean
}

const categoryIcons: Record<string, any> = {
  'AI/ML': Zap,
  'Database': Database,
  'Cloud': Cloud,
  'Communication': MessageSquare,
  'Developer': Terminal,
  'Infrastructure': Container,
  'Business': CreditCard,
  'Monitoring': Activity,
  'Storage': HardDrive,
  'Integration': Globe,
  'Security': Shield
}

export function ToolDetailsModal({ 
  open, 
  onClose, 
  tool, 
  onInstall, 
  onConfigure, 
  onUninstall, 
  loading = false 
}: ToolDetailsModalProps) {
  if (!tool) return null

  const CategoryIcon = categoryIcons[tool.category] || Settings
  const capabilities = tool.capabilities || {}
  const credentialsSchema = tool.credentials_schema || {}
  const metadata = tool.metadata || {}
  const capabilityList = extractCapabilities(capabilities)
  const credentialList =
    tool.requiredCredentials ||
    Object.keys(credentialsSchema.required || {}) ||
    []

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <ToolLogo
                    logo={tool.logo}
                    name={tool.name}
                    size={48}
                    fallbackIcon={tool.icon}
                    showBackground={true}
                  />
                  <div>
                    <div className="flex items-center gap-2">
                      {tool.name}
                      <Badge variant="outline" className="text-xs">
                        {tool.category}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground font-normal">
                      {tool.provider} • v{tool.version}
                    </p>
                  </div>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="space-y-6 max-h-[70vh] overflow-y-auto pr-2">
                <div className="space-y-6">
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                      <CategoryIcon className="w-4 h-4" />
                      Description
                    </h3>
                    <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {tool.description}
                      </p>
                    </div>
                  </div>

                  {tool.tags && tool.tags.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Tag className="w-4 h-4" />
                        Tags
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {tool.tags.map((tag: string) => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {capabilityList.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        Capabilities
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {capabilityList.map((capability: string) => (
                          <Badge key={capability} variant="outline" className="text-xs">
                            {capability}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {credentialList.length > 0 && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Key className="w-4 h-4" />
                        Required Credentials
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {credentialList.map((cred: string) => (
                          <Badge key={cred} variant="secondary" className="text-xs">
                            {cred}
                          </Badge>
                        ))}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Add credentials in the Configure step for this tool.
                      </p>
                    </div>
                  )}

                  {metadata.auto_enable_on_credential && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Settings className="w-4 h-4" />
                        Auto-Activation
                      </h3>
                      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                        <p className="text-sm text-blue-400">
                          This tool will automatically activate when you add a <strong>{metadata.auto_enable_on_credential}</strong> credential.
                        </p>
                      </div>
                    </div>
                  )}

                  {metadata.documentation && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <BookOpen className="w-4 h-4" />
                        Documentation
                      </h3>
                      <Button
                        variant="outline"
                        onClick={() => window.open(metadata.documentation, '_blank')}
                        className="w-full hover:border-blue-500/50 hover:text-blue-400"
                      >
                        <BookOpen className="w-4 h-4 mr-2" />
                        View Documentation
                        <ExternalLink className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  )}

                  <Separator />

                  <div className="flex gap-3">
                    {tool.isInstalled ? (
                      <>
                        <Button 
                          variant="outline" 
                          className="flex-1 hover:border-blue-500/50"
                          onClick={onConfigure}
                        >
                          <Settings className="w-4 h-4 mr-2" />
                          Configure
                        </Button>
                        <Button 
                          variant="outline"
                          onClick={onUninstall}
                          className="hover:border-red-500/50 text-red-400"
                          disabled={loading}
                        >
                          Remove
                        </Button>
                      </>
                    ) : (
                      <Button 
                        onClick={onInstall}
                        disabled={loading}
                        className="flex-1 bg-brand-primary hover:bg-brand-primary/90"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Install Tool
                      </Button>
                    )}
                  </div>
                </div>
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