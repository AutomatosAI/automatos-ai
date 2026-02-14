'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { Key, Plus, Trash2, TestTube, Loader2, CheckCircle, XCircle, Shield } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ApiKeyOut {
  id: number
  provider: string
  display_name: string | null
  masked_key: string
  is_active: boolean
  last_used_at: string | null
  usage_count: number
  created_at: string
}

interface AddKeyPayload {
  provider: string
  api_key: string
  display_name: string
}

const PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'azure', label: 'Azure' },
  { value: 'grok', label: 'Grok' },
] as const

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ApiKeysSettingsTab() {
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<ApiKeyOut | null>(null)

  // Form state
  const [provider, setProvider] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [displayName, setDisplayName] = useState('')

  // Track which key is currently being tested
  const [testingKeyId, setTestingKeyId] = useState<number | null>(null)

  // ---- Queries & Mutations ------------------------------------------------

  const {
    data: keys = [],
    isLoading,
    isError,
  } = useQuery<ApiKeyOut[]>({
    queryKey: ['api-keys'],
    queryFn: () => apiClient.get<ApiKeyOut[]>('/api/keys'),
  })

  const addKeyMutation = useMutation({
    mutationFn: (payload: AddKeyPayload) => apiClient.post('/api/keys', payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
      toast.success('API key added successfully')
      resetForm()
      setDialogOpen(false)
    },
    onError: (err: Error) => {
      toast.error(`Failed to add key: ${err.message}`)
    },
  })

  const deleteKeyMutation = useMutation({
    mutationFn: (id: number) => apiClient.delete(`/api/keys/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
      toast.success('API key deleted')
      setDeleteTarget(null)
    },
    onError: (err: Error) => {
      toast.error(`Failed to delete key: ${err.message}`)
    },
  })

  const testKeyMutation = useMutation({
    mutationFn: (id: number) => {
      setTestingKeyId(id)
      return apiClient.post<{ success: boolean; message?: string }>(`/api/keys/${id}/test`)
    },
    onSuccess: (data: any) => {
      if (data?.success) {
        toast.success('Key is valid', { icon: <CheckCircle className="h-4 w-4 text-green-400" /> })
      } else {
        toast.error(data?.message || 'Key test failed', { icon: <XCircle className="h-4 w-4 text-red-400" /> })
      }
      setTestingKeyId(null)
    },
    onError: (err: Error) => {
      toast.error(`Test failed: ${err.message}`)
      setTestingKeyId(null)
    },
  })

  // ---- Helpers ------------------------------------------------------------

  function resetForm() {
    setProvider('')
    setApiKey('')
    setDisplayName('')
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!provider || !apiKey) {
      toast.error('Provider and API key are required')
      return
    }
    addKeyMutation.mutate({ provider, api_key: apiKey, display_name: displayName })
  }

  function providerLabel(value: string) {
    return PROVIDERS.find((p) => p.value === value)?.label ?? value
  }

  function formatDate(iso: string | null) {
    if (!iso) return '--'
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
  }

  // ---- Render -------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="glass-card border-border/40">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-primary/10 p-2">
                <Shield className="h-5 w-5 text-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">API Keys</CardTitle>
                <CardDescription className="mt-1">
                  Bring Your Own Key (BYOK) — connect your own LLM provider keys for
                  chat, agents, and workflows. Keys are encrypted at rest and never
                  leave your workspace.
                </CardDescription>
              </div>
            </div>

            {/* Add Key Dialog */}
            <Dialog
              open={dialogOpen}
              onOpenChange={(open) => {
                setDialogOpen(open)
                if (!open) resetForm()
              }}
            >
              <DialogTrigger asChild>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Add API Key
                </Button>
              </DialogTrigger>

              <DialogContent className="sm:max-w-md">
                <form onSubmit={handleSubmit}>
                  <DialogHeader>
                    <DialogTitle>Add API Key</DialogTitle>
                    <DialogDescription>
                      Select a provider and paste your key. It will be encrypted before
                      storage.
                    </DialogDescription>
                  </DialogHeader>

                  <div className="space-y-4 py-4">
                    {/* Provider */}
                    <div className="space-y-2">
                      <Label htmlFor="provider">Provider</Label>
                      <Select value={provider} onValueChange={setProvider}>
                        <SelectTrigger id="provider">
                          <SelectValue placeholder="Select provider" />
                        </SelectTrigger>
                        <SelectContent>
                          {PROVIDERS.map((p) => (
                            <SelectItem key={p.value} value={p.value}>
                              {p.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* API Key */}
                    <div className="space-y-2">
                      <Label htmlFor="api-key">API Key</Label>
                      <Input
                        id="api-key"
                        type="password"
                        placeholder="sk-..."
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        autoComplete="off"
                      />
                    </div>

                    {/* Display Name */}
                    <div className="space-y-2">
                      <Label htmlFor="display-name">Display Name</Label>
                      <Input
                        id="display-name"
                        placeholder="e.g. Production GPT-4 Key"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                      />
                      <p className="text-xs text-muted-foreground">
                        Optional label to help identify this key.
                      </p>
                    </div>
                  </div>

                  <DialogFooter>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setDialogOpen(false)}
                    >
                      Cancel
                    </Button>
                    <Button type="submit" disabled={addKeyMutation.isPending}>
                      {addKeyMutation.isPending && (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      )}
                      Add Key
                    </Button>
                  </DialogFooter>
                </form>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
      </Card>

      {/* Keys Table */}
      <Card className="glass-card border-border/40">
        <CardContent className="p-0">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading keys...</span>
            </div>
          ) : isError ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <XCircle className="h-8 w-8 mb-2 text-red-400" />
              <p>Failed to load API keys.</p>
              <Button
                variant="outline"
                size="sm"
                className="mt-3"
                onClick={() => queryClient.invalidateQueries({ queryKey: ['api-keys'] })}
              >
                Retry
              </Button>
            </div>
          ) : keys.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Key className="h-10 w-10 mb-3 opacity-40" />
              <p className="font-medium">No API keys yet</p>
              <p className="text-sm mt-1">Add a provider key to get started.</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-border/40">
                  <TableHead>Provider</TableHead>
                  <TableHead>Display Name</TableHead>
                  <TableHead>Key</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Last Used</TableHead>
                  <TableHead className="text-right">Usage</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {keys.map((key) => (
                  <TableRow key={key.id} className="border-border/30">
                    <TableCell className="font-medium">
                      {providerLabel(key.provider)}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {key.display_name || '--'}
                    </TableCell>
                    <TableCell>
                      <code className="rounded bg-muted/50 px-2 py-0.5 text-xs font-mono">
                        {key.masked_key}
                      </code>
                    </TableCell>
                    <TableCell>
                      {key.is_active ? (
                        <Badge className="bg-green-500/15 text-green-400 border-green-500/30">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Active
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-red-500/15 text-red-400 border-red-500/30">
                          <XCircle className="h-3 w-3 mr-1" />
                          Inactive
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {formatDate(key.last_used_at)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-sm">
                      {key.usage_count.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        {/* Test */}
                        <Button
                          variant="ghost"
                          size="sm"
                          disabled={testingKeyId === key.id}
                          onClick={() => testKeyMutation.mutate(key.id)}
                          title="Test key"
                        >
                          {testingKeyId === key.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <TestTube className="h-4 w-4" />
                          )}
                        </Button>

                        {/* Delete */}
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          onClick={() => setDeleteTarget(key)}
                          title="Delete key"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Delete API Key</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete{' '}
              <span className="font-medium text-foreground">
                {deleteTarget?.display_name || deleteTarget?.masked_key}
              </span>
              ? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={deleteKeyMutation.isPending}
              onClick={() => deleteTarget && deleteKeyMutation.mutate(deleteTarget.id)}
            >
              {deleteKeyMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
