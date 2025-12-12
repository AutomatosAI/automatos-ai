'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Plus, Key, Trash2, Edit, TestTube, Eye, EyeOff, AlertCircle,
  CheckCircle2, XCircle, Clock, Database, Brain, Server, Mail, Code
} from 'lucide-react'
import { ToolLogo } from '@/components/ui/tool-logo'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { DynamicCredentialForm } from './DynamicCredentialForm'
import { EnhancedPagination } from '@/components/ui/pagination'
import {
  listCredentials,
  deleteCredential,
  testCredential,
  getCredentialType,
  type Credential,
  type CredentialType,
  getCredentialCategories
} from '@/lib/api/credentials'

const CATEGORY_ICONS: Record<string, any> = {
  database: Database,
  ai: Brain,
  infrastructure: Server,
  communication: Mail,
  code: Code,
  api: Key,
  cloud: Server
}

const TEST_STATUS_BADGES = {
  passed: { color: 'bg-green-500/20 text-green-400', icon: CheckCircle2 },
  failed: { color: 'bg-red-500/20 text-red-400', icon: XCircle },
  pending: { color: 'bg-yellow-500/20 text-yellow-400', icon: Clock },
  not_tested: { color: 'bg-gray-500/20 text-gray-400', icon: AlertCircle }
}

export function CredentialsTab() {
  const [credentials, setCredentials] = useState<Credential[]>([])
  const [credentialTypes, setCredentialTypes] = useState<Map<number, CredentialType>>(new Map())
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all')
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [selectedCredential, setSelectedCredential] = useState<Credential | null>(null)
  const [testingCredential, setTestingCredential] = useState<number | null>(null)

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(20)
  const [paginationData, setPaginationData] = useState<any>(null)

  useEffect(() => {
    loadCredentials()
  }, [environmentFilter, currentPage])

  // Debounced search effect
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (currentPage !== 1) {
        setCurrentPage(1) // Reset to first page when searching
      } else {
        loadCredentials()
      }
    }, 500)

    return () => clearTimeout(timeoutId)
  }, [searchTerm])

  const loadCredentialTypes = async (creds: Credential[]) => {
    const typeIds = [...new Set(creds.map(c => c.credential_type_id))]
    const types = new Map<number, CredentialType>()

    for (const typeId of typeIds) {
      try {
        const type = await getCredentialType(typeId)
        types.set(typeId, type)
      } catch (error) {
        console.error(`Failed to load credential type ${typeId}:`, error)
      }
    }

    setCredentialTypes(types)
  }

  const loadCredentials = async () => {
    try {
      setLoading(true)
      const data = await listCredentials({
        environment: environmentFilter === 'all' ? undefined : environmentFilter,
        active_only: true,
        skip: (currentPage - 1) * pageSize,
        limit: pageSize,
        search: searchTerm || undefined
      })

      // Ensure data is an array
      if (Array.isArray(data)) {
        setCredentials(data)
        setPaginationData(null)
        // Load credential types for logos
        await loadCredentialTypes(data)
      } else if (data.items && Array.isArray(data.items)) {
        // Handle paginated response
        setCredentials(data.items)
        setPaginationData({
          total: data.total,
          skip: data.skip,
          limit: data.limit,
          pages: data.pages,
          current_page: data.current_page
        })
        // Load credential types for logos
        await loadCredentialTypes(data.items)
      } else {
        // Fallback to empty array
        console.warn('Unexpected credentials response format:', data)
        setCredentials([])
        setPaginationData(null)
      }
    } catch (error) {
      console.error('Failed to load credentials:', error)
      setCredentials([]) // Set to empty array on error
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (credentialId: number, credentialName: string) => {
    if (!confirm(`Are you sure you want to delete credential "${credentialName}"?\n\nThis action cannot be undone.`)) {
      return
    }

    try {
      await deleteCredential(credentialId)
      await loadCredentials()
    } catch (error) {
      console.error('Failed to delete credential:', error)
      alert('Failed to delete credential')
    }
  }

  const handleTest = async (credentialId: number) => {
    try {
      setTestingCredential(credentialId)
      const result = await testCredential(credentialId)

      if (result.success) {
        alert(`✅ Test Successful\n\n${result.message}`)
      } else {
        alert(`❌ Test Failed\n\n${result.message}`)
      }

      await loadCredentials() // Refresh to show updated test status
    } catch (error) {
      console.error('Failed to test credential:', error)
      alert('Failed to test credential')
    } finally {
      setTestingCredential(null)
    }
  }

  // Since we're now using server-side search, we don't need client-side filtering
  const filteredCredentials = credentials

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Credentials</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Manage encrypted credentials for services and integrations
          </p>
        </div>

        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Add Credential
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Create New Credential</DialogTitle>
              <DialogDescription>
                Add a new encrypted credential for use with agents and workflows
              </DialogDescription>
            </DialogHeader>
            <DynamicCredentialForm
              onSuccess={() => {
                setShowCreateDialog(false)
                loadCredentials()
              }}
              onCancel={() => setShowCreateDialog(false)}
            />
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters */}
      <Card className="glass-card">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <Input
              placeholder="Search credentials..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1"
            />
            <Select value={environmentFilter} onValueChange={setEnvironmentFilter}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Environment" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Environments</SelectItem>
                <SelectItem value="development">Development</SelectItem>
                <SelectItem value="staging">Staging</SelectItem>
                <SelectItem value="production">Production</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Credentials List */}
      {loading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading credentials...</p>
        </div>
      ) : filteredCredentials.length === 0 ? (
        <Card className="glass-card">
          <CardContent className="pt-12 pb-12 text-center">
            <Key className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Credentials Found</h3>
            <p className="text-sm text-muted-foreground mb-4">
              {searchTerm || environmentFilter !== 'all'
                ? 'Try adjusting your filters'
                : 'Get started by adding your first credential'}
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="w-4 h-4 mr-2" />
              Add Credential
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCredentials.map((credential) => {
            const TestStatusBadge = TEST_STATUS_BADGES[credential.test_status || 'not_tested']
            const StatusIcon = TestStatusBadge.icon
            const credType = credentialTypes.get(credential.credential_type_id)

            return (
              <Card key={credential.id} className="glass-card hover:border-primary/50 transition-all">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3 flex-1">
                      <ToolLogo
                        logo={credType?.logo || undefined}
                        name={credential.credential_type_display_name}
                        size={40}
                        fallbackIcon={credType?.icon || undefined}
                        showBackground={true}
                      />
                      <div className="flex-1">
                        <CardTitle className="text-lg">
                          {credential.name}
                        </CardTitle>
                        <CardDescription className="mt-1">
                          {credential.credential_type_display_name}
                        </CardDescription>
                      </div>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {credential.environment}
                    </Badge>
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  {/* Description */}
                  {credential.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {credential.description}
                    </p>
                  )}

                  {/* Test Status */}
                  <div className="flex items-center gap-2">
                    <Badge className={TestStatusBadge.color}>
                      <StatusIcon className="w-3 h-3 mr-1" />
                      {credential.test_status?.replace('_', ' ').toUpperCase() || 'NOT TESTED'}
                    </Badge>
                    {credential.last_tested && (
                      <span className="text-xs text-muted-foreground">
                        {new Date(credential.last_tested).toLocaleDateString()}
                      </span>
                    )}
                  </div>

                  {/* Fields Info */}
                  <div className="text-xs text-muted-foreground">
                    Fields: {credential.field_names.join(', ')}
                  </div>

                  {/* Tags */}
                  {credential.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {credential.tags.map((tag, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2 pt-2 border-t border-border/30">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleTest(credential.id)}
                      disabled={testingCredential === credential.id}
                      className="flex-1"
                    >
                      <TestTube className="w-3 h-3 mr-1" />
                      {testingCredential === credential.id ? 'Testing...' : 'Test'}
                    </Button>

                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setSelectedCredential(credential)}
                    >
                      <Edit className="w-3 h-3" />
                    </Button>

                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleDelete(credential.id, credential.name)}
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>

                  {/* Expiry Warning */}
                  {credential.expires_at && (
                    <div className="text-xs text-yellow-500 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />
                      Expires {new Date(credential.expires_at).toLocaleDateString()}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {/* Pagination */}
      {paginationData && (
        <EnhancedPagination
          data={paginationData}
          onPageChange={(page) => setCurrentPage(page)}
          className="mt-6"
        />
      )}

      {/* Edit Dialog */}
      {selectedCredential && (
        <Dialog open={!!selectedCredential} onOpenChange={() => setSelectedCredential(null)}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Edit Credential</DialogTitle>
              <DialogDescription>
                Update {selectedCredential.name}
              </DialogDescription>
            </DialogHeader>
            <DynamicCredentialForm
              credentialId={selectedCredential.id}
              onSuccess={() => {
                setSelectedCredential(null)
                loadCredentials()
              }}
              onCancel={() => setSelectedCredential(null)}
            />
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}

