'use client'

import React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  ExternalLink, 
  BookOpen, 
  Settings, 
  Download, 
  Star, 
  Tag, 
  Code, 
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
  Key
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

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-2xl">
            <div className="w-12 h-12 rounded-lg bg-secondary/50 flex items-center justify-center">
              <span className="text-2xl">{tool.icon}</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                {tool.name}
                <Badge variant="outline" className="text-sm">
                  {tool.category}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground font-normal">
                {tool.provider} • v{tool.version}
              </p>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Description */}
          <div>
            <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
              <CategoryIcon className="w-5 h-5" />
              Description
            </h3>
            <p className="text-muted-foreground leading-relaxed">
              {tool.description}
            </p>
          </div>

          {/* Tags */}
          {tool.tags && tool.tags.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Tag className="w-5 h-5" />
                Tags
              </h3>
              <div className="flex flex-wrap gap-2">
                {tool.tags.map((tag: string) => (
                  <Badge key={tag} variant="outline" className="text-sm">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Capabilities */}
          {Object.keys(capabilities).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Capabilities
              </h3>
              <div className="bg-secondary/30 rounded-lg p-4">
                <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {JSON.stringify(capabilities, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Credentials Schema */}
          {Object.keys(credentialsSchema).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Key className="w-5 h-5" />
                Required Credentials
              </h3>
              <div className="bg-secondary/30 rounded-lg p-4">
                <pre className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {JSON.stringify(credentialsSchema, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Auto-activation Info */}
          {metadata.auto_enable_on_credential && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Auto-Activation
              </h3>
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <p className="text-sm text-blue-400">
                  This tool will automatically activate when you add a <strong>{metadata.auto_enable_on_credential}</strong> credential.
                </p>
              </div>
            </div>
          )}

          {/* Documentation */}
          {metadata.documentation && (
            <div>
              <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
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

          {/* Actions */}
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
      </DialogContent>
    </Dialog>
  )
}