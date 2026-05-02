'use client'

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Search, ExternalLink, CheckCircle, AlertTriangle, Clock, Settings, Store } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useConnectedApps, type ComposioApp } from '@/hooks/use-composio-api'
import { PageHeader, SearchInput } from '@/components/shared'

export function MyToolsDashboard() {
  const [searchQuery, setSearchQuery] = useState('')
  const { data: connections = [], isLoading } = useConnectedApps()

  // Filter connections by search
  const filteredConnections = connections.filter((conn) => {
    if (!searchQuery.trim()) return true
    const query = searchQuery.toLowerCase()
    return (
      conn.app_name.toLowerCase().includes(query) ||
      conn.connection_id?.toLowerCase().includes(query)
    )
  })

  // Get status stats
  const activeCount = connections.filter((c) => c.status === 'active').length
  const errorCount = connections.filter((c) => c.status === 'failed').length

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return (
          <Badge className="bg-[hsl(var(--success))]/20 text-[hsl(var(--success))] border-[hsl(var(--success))]/30">
            <CheckCircle className="h-3 w-3 mr-1" />
            Active
          </Badge>
        )
      case 'failed':
        return (
          <Badge className="bg-[hsl(var(--destructive))]/20 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/30">
            <AlertTriangle className="h-3 w-3 mr-1" />
            Error
          </Badge>
        )
      default:
        return (
          <Badge className="bg-muted/20 text-muted-foreground border-border">
            <Clock className="h-3 w-3 mr-1" />
            Disconnected
          </Badge>
        )
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Page Header */}
      <div className="mb-6">
        <PageHeader
          title="My"
          titleAccent="Tools"
          subtitle="Manage your connected tools and integrations"
          actions={
            <Link href="/marketplace?tab=tools">
              <Button variant="outline">
                <Store className="h-4 w-4 mr-2" />
                Browse Marketplace
              </Button>
            </Link>
          }
        />
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card className="bg-secondary border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-foreground">{connections.length}</div>
                <div className="text-sm text-muted-foreground">Total Connected</div>
              </div>
              <CheckCircle className="h-8 w-8 text-[hsl(var(--info))]" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-secondary border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-foreground">{activeCount}</div>
                <div className="text-sm text-muted-foreground">Active</div>
              </div>
              <CheckCircle className="h-8 w-8 text-[hsl(var(--success))]" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-secondary border-border">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-foreground">{errorCount}</div>
                <div className="text-sm text-muted-foreground">Errors</div>
              </div>
              <AlertTriangle className="h-8 w-8 text-[hsl(var(--destructive))]" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="mb-6">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search connected tools..."
        />
      </div>

      {/* Connected Tools List */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-32 bg-secondary animate-pulse rounded-lg" />
          ))}
        </div>
      ) : filteredConnections.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Store className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground mb-4">
            {searchQuery ? 'No tools found matching your search' : 'No tools connected yet'}
          </p>
          <Link href="/marketplace?tab=tools">
            <Button variant="outline">
              Browse Marketplace
            </Button>
          </Link>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredConnections.map((connection) => (
            <Card
              key={connection.connection_id || connection.app_name}
              className="glass-card card-glow hover:border-primary/20 transition-all duration-300"
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-foreground capitalize">
                      {connection.app_name.toLowerCase().replace(/_/g, ' ')}
                    </h3>
                    {connection.connection_id && (
                      <p className="text-xs text-muted-foreground mt-1 truncate">
                        {connection.connection_id}
                      </p>
                    )}
                  </div>
                  {getStatusBadge(connection.status)}
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <div className="text-xs text-muted-foreground space-y-1">
                  {connection.created_at && (
                    <div>
                      Connected: {new Date(connection.created_at).toLocaleDateString()}
                    </div>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1 border-border text-muted-foreground hover:bg-secondary"
                  >
                    <Settings className="h-3 w-3 mr-1" />
                    Settings
                  </Button>
                  {connection.status === 'active' && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="border-border text-muted-foreground hover:bg-secondary"
                    >
                      <ExternalLink className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
