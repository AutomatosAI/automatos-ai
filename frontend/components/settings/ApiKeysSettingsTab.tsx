'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  Key, Plus, Trash2, TestTube, Loader2, XCircle, Shield,
} from 'lucide-react'
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
import { Switch } from '@/components/ui/switch'

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

interface PlatformKeyStatus {
  platform_keys: Record<string, { configured: boolean }>
}

interface ByokPreferences {
  byok_overrides: Record<string, boolean>
}

const PROVIDERS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google' },
  { value: 'openrouter', label: 'OpenRouter' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'azure', label: 'Azure OpenAI' },
  { value: 'bedrock', label: 'AWS Bedrock' },
  { value: 'grok', label: 'Grok / xAI' },
  { value: 'cohere', label: 'Cohere' },
  { value: 'huggingface', label: 'HuggingFace' },
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

  const { data: platformStatus } = useQuery<PlatformKeyStatus>({
    queryKey: ['platform-key-status'],
    queryFn: () => apiClient.get<PlatformKeyStatus>('/api/keys/platform-status'),
  })

  const { data: byokPrefs } = useQuery<ByokPreferences>({
    queryKey: ['byok-preferences'],
    queryFn: () => apiClient.get<ByokPreferences>('/api/workspaces/current/byok-preferences'),
  })

  const addKeyMutation = useMutation({
    mutationFn: (payload: AddKeyPayload) => apiClient.post('/api/keys', payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
      queryClient.invalidateQueries({ queryKey: ['byok-preferences'] })
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
      if (data?.valid) {
        toast.success('Key is valid')
      } else {
        toast.error(data?.message || 'Key test failed')
      }
      setTestingKeyId(null)
    },
    onError: (err: Error) => {
      toast.error(`Test failed: ${err.message}`)
      setTestingKeyId(null)
    },
  })

  const saveByokMutation = useMutation({
    mutationFn: (overrides: Record<string, boolean>) =>
      apiClient.put('/api/workspaces/current/byok-preferences', { byok_overrides: overrides }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['byok-preferences'] })
    },
    onError: (err: Error) => {
      toast.error(`Failed to save preference: ${err.message}`)
    },
  })

  // ---- Helpers ------------------------------------------------------------

  const byokOverrides = byokPrefs?.byok_overrides ?? {}
  const platformKeys = platformStatus?.platform_keys ?? {}

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

  function getProviderKeys(prov: string) {
    return keys.filter((k) => k.provider === prov)
  }

  function toggleByok(prov: string, enabled: boolean) {
    saveByokMutation.mutate({ ...byokOverrides, [prov]: enabled })
  }

  function getActiveSource(prov: string): 'byok' | 'platform' | 'none' {
    const hasUserKey = keys.some((k) => k.provider === prov && k.is_active)
    const byokEnabled = byokOverrides[prov] ?? false
    const hasPlatform = platformKeys[prov]?.configured ?? false

    if (byokEnabled && hasUserKey) return 'byok'
    if (hasPlatform) return 'platform'
    return 'none'
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
                  Manage your personal (BYOK) API keys. Toggle per-provider
                  to control which key is used for chat and agent execution.
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

      {/* Your API Keys (BYOK management) */}
      <Card className="glass-card border-border/40">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-base">Your API Keys</CardTitle>
          </div>
          <CardDescription className="text-xs">
            Add your own keys per provider. Toggle to override platform keys.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          {isLoading ? (
            <div className="flex items-center justify-center py-10">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">Loading keys...</span>
            </div>
          ) : isError ? (
            <div className="flex flex-col items-center justify-center py-10 text-muted-foreground">
              <XCircle className="h-6 w-6 mb-2 text-destructive" />
              <p className="text-sm">Failed to load API keys.</p>
              <Button
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => queryClient.invalidateQueries({ queryKey: ['api-keys'] })}
              >
                Retry
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {PROVIDERS.map((p) => {
                const provKeys = getProviderKeys(p.value)
                const hasKey = provKeys.length > 0
                const byokOn = byokOverrides[p.value] ?? false
                const activeSource = getActiveSource(p.value)

                return (
                  <div
                    key={p.value}
                    className="rounded-md border border-border/30 px-3 py-3"
                  >
                    {/* Provider row */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-sm font-medium w-24">{p.label}</span>

                        {/* BYOK toggle */}
                        <Switch
                          checked={byokOn}
                          onCheckedChange={(checked) => toggleByok(p.value, checked)}
                          disabled={!hasKey}
                          aria-label={`Use your ${p.label} key`}
                        />

                        {/* Active source indicator */}
                        {activeSource === 'byok' ? (
                          <Badge className="bg-info/15 text-info border-info/30 text-xs">
                            Using: Your Key
                          </Badge>
                        ) : activeSource === 'platform' ? (
                          <Badge className="bg-success/15 text-success border-success/30 text-xs">
                            Using: Platform Key
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="text-xs opacity-50">
                            No key available
                          </Badge>
                        )}
                      </div>

                      {/* Key info + actions */}
                      <div className="flex items-center gap-2">
                        {hasKey && (
                          <>
                            <code className="rounded bg-muted/50 px-2 py-0.5 text-xs font-mono">
                              {provKeys[0].masked_key}
                            </code>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 w-7 p-0"
                              disabled={testingKeyId === provKeys[0].id}
                              onClick={() => testKeyMutation.mutate(provKeys[0].id)}
                              title="Test key"
                            >
                              {testingKeyId === provKeys[0].id ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <TestTube className="h-3.5 w-3.5" />
                              )}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 w-7 p-0 text-destructive hover:text-destructive"
                              onClick={() => setDeleteTarget(provKeys[0])}
                              title="Delete key"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Delete BYOK Key Confirmation Dialog */}
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
