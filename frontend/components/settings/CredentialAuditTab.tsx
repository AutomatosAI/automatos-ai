'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { 
  CheckCircle, XCircle, AlertCircle, Clock, Filter
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  getCredentialAuditLogs,
  type CredentialAuditLog
} from '@/lib/api/credentials'

const ACTION_COLORS: Record<string, string> = {
  created: 'bg-green-500/20 text-green-400',
  updated: 'bg-blue-500/20 text-blue-400',
  deleted: 'bg-red-500/20 text-red-400',
  accessed: 'bg-purple-500/20 text-purple-400',
  tested: 'bg-yellow-500/20 text-yellow-400',
  access_denied: 'bg-red-500/20 text-red-400',
  access_failed: 'bg-red-500/20 text-red-400'
}

export function CredentialAuditTab() {
  const [logs, setLogs] = useState<CredentialAuditLog[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [actionFilter, setActionFilter] = useState<string>('all')

  useEffect(() => {
    loadLogs()
  }, [actionFilter])

  const loadLogs = async () => {
    try {
      setLoading(true)
      const data = await getCredentialAuditLogs({
        action: actionFilter === 'all' ? undefined : actionFilter,
        limit: 100
      })
      setLogs(data)
    } catch (error) {
      console.error('Failed to load audit logs:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredLogs = logs.filter(log =>
    (log.credential_id && log.credential_id.toString().includes(searchTerm)) ||
    log.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (log.user_id && log.user_id.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  const getSuccessIcon = (success: boolean | null) => {
    if (success === null) return Clock
    return success ? CheckCircle : XCircle
  }

  const getSuccessColor = (success: boolean | null) => {
    if (success === null) return 'text-gray-400'
    return success ? 'text-green-400' : 'text-red-400'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Audit Logs</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Track all credential access and modifications
        </p>
      </div>

      {/* Filters */}
      <Card className="glass-card">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <Input
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1"
            />
            <Select value={actionFilter} onValueChange={setActionFilter}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Action" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Actions</SelectItem>
                <SelectItem value="created">Created</SelectItem>
                <SelectItem value="updated">Updated</SelectItem>
                <SelectItem value="deleted">Deleted</SelectItem>
                <SelectItem value="accessed">Accessed</SelectItem>
                <SelectItem value="tested">Tested</SelectItem>
                <SelectItem value="access_denied">Access Denied</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Audit Logs */}
      {loading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading audit logs...</p>
        </div>
      ) : filteredLogs.length === 0 ? (
        <Card className="glass-card">
          <CardContent className="pt-12 pb-12 text-center">
            <AlertCircle className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Audit Logs</h3>
            <p className="text-sm text-muted-foreground">
              {searchTerm || actionFilter !== 'all'
                ? 'No logs match your filters'
                : 'Audit logs will appear here as credentials are used'}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {filteredLogs.map((log) => {
            const SuccessIcon = getSuccessIcon(log.success)
            const successColor = getSuccessColor(log.success)
            
            return (
              <Card key={log.id} className="glass-card">
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      <SuccessIcon className={`w-5 h-5 mt-0.5 ${successColor}`} />
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge className={ACTION_COLORS[log.action] || 'bg-gray-500/20'}>
                            {log.action.toUpperCase()}
                          </Badge>
                          <span className="font-semibold">Credential #{log.credential_id}</span>
                        </div>
                        
                        <div className="text-sm text-muted-foreground mt-1 space-y-1">
                          {log.user_id && (
                            <p>User: {log.user_id}</p>
                          )}
                          {log.ip_address && (
                            <p>IP: {log.ip_address}</p>
                          )}
                          {log.error_message && (
                            <p className="text-red-400">Error: {log.error_message}</p>
                          )}
                        </div>

                        {log.metadata && Object.keys(log.metadata).length > 0 && (
                          <details className="mt-2">
                            <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                              View metadata
                            </summary>
                            <pre className="text-xs bg-black/30 p-2 rounded mt-2 overflow-x-auto">
                              {JSON.stringify(log.metadata, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    </div>

                    <time className="text-xs text-muted-foreground whitespace-nowrap">
                      {new Date(log.timestamp || log.created_at || '').toLocaleString()}
                    </time>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}

