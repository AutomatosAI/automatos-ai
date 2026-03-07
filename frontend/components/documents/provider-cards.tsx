'use client'

import React from 'react'
import { motion } from 'framer-motion'
import {
  Cloud,
  Upload,
  HardDrive,
  Database,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileText,
  Package
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ToolLogo } from '@/components/ui/tool-logo'

interface ProviderStats {
  documentCount: number
  chunkCount: number
  lastSyncedAt?: string
  syncStatus: 'idle' | 'syncing' | 'error'
  rootFolder?: string
}

interface Provider {
  id: string
  name: string
  type: 'manual' | 'googledrive' | 'dropbox' | 'onedrive' | 'box'
  icon: React.ComponentType<any>
  color: string
  stats: ProviderStats
  connected: boolean
  connectionId?: number
}

interface ProviderCardsProps {
  providers: Provider[]
  onProviderClick: (provider: Provider) => void
}

const providerIcons: Record<string, { icon: React.ComponentType<any>, color: string }> = {
  manual: { icon: HardDrive, color: 'text-blue-400' },
  googledrive: { icon: Cloud, color: 'text-green-400' },
  dropbox: { icon: Database, color: 'text-blue-500' },
  onedrive: { icon: Cloud, color: 'text-blue-600' },
  box: { icon: Database, color: 'text-indigo-400' }
}

// Brand logos for cloud providers
const providerLogos: Record<string, string | undefined> = {
  googledrive: '/logos/GoogleDrive.png',
  dropbox: '/logos/Dropbox.png',
  onedrive: '/logos/OneDrive.png',
  box: '/logos/Box.png',
  manual: undefined // Use icon for local storage
}

export function ProviderCards({ providers, onProviderClick }: ProviderCardsProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Storage Providers</h2>
        <p className="text-muted-foreground mt-1">
          Browse and manage documents across all connected storage locations
        </p>
      </div>

      {/* Provider Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {providers.map((provider, index) => {
          const { icon: Icon, color } = providerIcons[provider.type] || providerIcons.manual
          const logo = providerLogos[provider.type]
          const { stats } = provider

          return (
            <motion.div
              key={provider.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="h-full"
            >
              <Card
                className="glass-card hover:border-primary/20 transition-all duration-300 cursor-pointer group h-full min-h-[280px] flex flex-col"
                onClick={() => onProviderClick(provider)}
              >
                <CardContent className="p-6 flex flex-col h-full">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {logo ? (
                        <div className="group-hover:scale-110 transition-transform">
                          <ToolLogo
                            logo={logo}
                            name={provider.name}
                            size={48}
                            showBackground={false}
                          />
                        </div>
                      ) : (
                        <div className={`w-12 h-12 rounded-xl bg-secondary/50 flex items-center justify-center group-hover:scale-110 transition-transform`}>
                          <Icon className={`w-6 h-6 ${color}`} />
                        </div>
                      )}
                      <div>
                        <h3 className="font-semibold text-lg">{provider.name}</h3>
                        <div className="flex items-center gap-2 mt-1">
                          {provider.connected ? (
                            <>
                              <CheckCircle className="w-4 h-4 text-green-400" />
                              <span className="text-xs text-green-400">Connected</span>
                            </>
                          ) : (
                            <>
                              <AlertCircle className="w-4 h-4 text-muted-foreground" />
                              <span className="text-xs text-muted-foreground">Not connected</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>

                  {/* Content */}
                  {provider.connected ? (
                    <div className="space-y-3 flex-1 flex flex-col">
                      {/* Stats */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-secondary/20 rounded-lg p-3">
                          <div className="flex items-center gap-2 mb-1">
                            <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">Documents</span>
                          </div>
                          <p className="text-lg font-semibold">{stats.documentCount}</p>
                        </div>
                        <div className="bg-secondary/20 rounded-lg p-3">
                          <div className="flex items-center gap-2 mb-1">
                            <Package className="w-3.5 h-3.5 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">Chunks</span>
                          </div>
                          <p className="text-lg font-semibold">{stats.chunkCount}</p>
                        </div>
                      </div>

                      {/* Root Folder */}
                      {stats.rootFolder && (
                        <div className="flex-1">
                          <p className="text-xs text-muted-foreground mb-1">Syncing from:</p>
                          <p className="text-sm font-medium truncate">{stats.rootFolder}</p>
                        </div>
                      )}

                      {/* Sync Status */}
                      <div className="mt-auto pt-2">
                        {stats.syncStatus === 'syncing' ? (
                          <Badge variant="outline" className="text-xs w-full justify-center">
                            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                            Syncing...
                          </Badge>
                        ) : stats.syncStatus === 'error' ? (
                          <Badge variant="destructive" className="text-xs w-full justify-center">
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Sync Error
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="text-xs w-full justify-center">
                            {stats.lastSyncedAt ? `Synced ${stats.lastSyncedAt}` : 'Ready to sync'}
                          </Badge>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex flex-col items-center justify-center py-4">
                      <p className="text-sm text-muted-foreground mb-3">
                        Connect to sync documents
                      </p>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          // TODO: Navigate to /tools to connect
                          window.location.href = '/tools'
                        }}
                      >
                        Connect Provider
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Empty State */}
      {providers.length === 0 && (
        <Card className="glass-card p-12 text-center">
          <Cloud className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Storage Providers</h3>
          <p className="text-muted-foreground mb-6">
            Connect cloud storage providers to sync your documents
          </p>
          <Button onClick={() => window.location.href = '/tools'}>
            Connect Provider
          </Button>
        </Card>
      )}
    </div>
  )
}
