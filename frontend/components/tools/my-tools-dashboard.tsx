'use client'

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Search, ExternalLink, CheckCircle, AlertTriangle, Clock, Settings, Store } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useConnectedApps, type ComposioApp } from '@/hooks/use-composio-api'

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
          <Badge className="bg-green-500/20 text-green-500 border-green-500/30">
            <CheckCircle className="h-3 w-3 mr-1" />
            Active
          </Badge>
        )
      case 'failed':
        return (
          <Badge className="bg-red-500/20 text-red-500 border-red-500/30">
            <AlertTriangle className="h-3 w-3 mr-1" />
            Error
          </Badge>
        )
      default:
        return (
          <Badge className="bg-gray-500/20 text-gray-500 border-gray-500/30">
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
        <div className="flex items-center justify-between mb-2">
          <h1 className="text-3xl font-bold">
            My <span className="text-orange-500">Tools</span>
          </h1>
          <Link href="/marketplace?tab=tools">
            <Button className="bg-orange-500 hover:bg-orange-600 text-white">
              <Store className="h-4 w-4 mr-2" />
              Browse Marketplace
            </Button>
          </Link>
        </div>
        <p className="text-gray-400">
          Manage your connected tools and integrations
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{connections.length}</div>
                <div className="text-sm text-gray-400">Total Connected</div>
              </div>
              <CheckCircle className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{activeCount}</div>
                <div className="text-sm text-gray-400">Active</div>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-[#1a1a1a] border-gray-800">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-white">{errorCount}</div>
                <div className="text-sm text-gray-400">Errors</div>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <Input
            type="text"
            placeholder="Search connected tools..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-[#1a1a1a] border-gray-800 text-white placeholder:text-gray-500"
          />
        </div>
      </div>

      {/* Connected Tools List */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-800 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : filteredConnections.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Store className="h-12 w-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">
            {searchQuery ? 'No tools found matching your search' : 'No tools connected yet'}
          </p>
          <Link href="/marketplace?tab=tools">
            <Button className="bg-orange-500 hover:bg-orange-600 text-white">
              Browse Marketplace
            </Button>
          </Link>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredConnections.map((connection) => (
            <Card
              key={connection.connection_id || connection.app_name}
              className="bg-[#1a1a1a] border-gray-800 hover:border-orange-500/50 transition-all duration-200"
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-white capitalize">
                      {connection.app_name.toLowerCase().replace(/_/g, ' ')}
                    </h3>
                    {connection.connection_id && (
                      <p className="text-xs text-gray-500 mt-1 truncate">
                        {connection.connection_id}
                      </p>
                    )}
                  </div>
                  {getStatusBadge(connection.status)}
                </div>
              </CardHeader>

              <CardContent className="space-y-3">
                <div className="text-xs text-gray-400 space-y-1">
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
                    className="flex-1 border-gray-700 text-gray-300 hover:bg-gray-800"
                  >
                    <Settings className="h-3 w-3 mr-1" />
                    Settings
                  </Button>
                  {connection.status === 'active' && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="border-gray-700 text-gray-300 hover:bg-gray-800"
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
