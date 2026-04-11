'use client'

/**
 * PRD-54: Platform API Keys Card (admin-only)
 * ============================================
 *
 * Shared keys configured by the admin — available to all workspaces.
 * Previously lived under Settings > API Keys (user-facing tab) but was
 * moved to Settings > System Settings > API Keys so end users only see
 * the BYOK surface.
 */

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  Building2, CheckCircle, Loader2, Settings, Trash2,
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
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

interface PlatformKeyStatus {
  platform_keys: Record<string, { configured: boolean }>
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

export function PlatformApiKeysCard() {
  const queryClient = useQueryClient()

  const [dialogOpen, setDialogOpen] = useState(false)
  const [provider, setProvider] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [removeTarget, setRemoveTarget] = useState<string | null>(null)

  const { data: platformStatus } = useQuery<PlatformKeyStatus>({
    queryKey: ['platform-key-status'],
    queryFn: () => apiClient.get<PlatformKeyStatus>('/api/keys/platform-status'),
  })

  const setKeyMutation = useMutation({
    mutationFn: (payload: { provider: string; api_key: string }) =>
      apiClient.put('/api/keys/platform', payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platform-key-status'] })
      toast.success('Platform key saved')
      setDialogOpen(false)
      setProvider('')
      setApiKey('')
    },
    onError: (err: Error) => {
      toast.error(`Failed to save platform key: ${err.message}`)
    },
  })

  const removeKeyMutation = useMutation({
    mutationFn: (prov: string) =>
      apiClient.delete(`/api/keys/platform/${prov}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['platform-key-status'] })
      toast.success('Platform key removed')
      setRemoveTarget(null)
    },
    onError: (err: Error) => {
      toast.error(`Failed to remove platform key: ${err.message}`)
    },
  })

  const platformKeys = platformStatus?.platform_keys ?? {}

  function openConfigure(providerValue: string) {
    setProvider(providerValue)
    setApiKey('')
    setDialogOpen(true)
  }

  function providerLabel(value: string) {
    return PROVIDERS.find((p) => p.value === value)?.label ?? value
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!provider || !apiKey) {
      toast.error('API key is required')
      return
    }
    setKeyMutation.mutate({ provider, api_key: apiKey })
  }

  return (
    <>
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            Platform API Keys
          </CardTitle>
          <CardDescription>
            Shared LLM provider keys configured by the admin. Available to all
            workspaces when the user has not enabled a BYOK override.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="grid gap-2">
            {PROVIDERS.map((p) => {
              const configured = platformKeys[p.value]?.configured ?? false
              return (
                <div
                  key={p.value}
                  className="flex items-center justify-between rounded-md border border-border/30 px-3 py-2"
                >
                  <span className="text-sm font-medium">{p.label}</span>
                  <div className="flex items-center gap-2">
                    {configured ? (
                      <>
                        <Badge className="bg-green-500/15 text-green-400 border-green-500/30 text-xs">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Configured
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() => openConfigure(p.value)}
                        >
                          <Settings className="h-3 w-3 mr-1" />
                          Update
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-xs text-destructive hover:text-destructive"
                          onClick={() => setRemoveTarget(p.value)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </>
                    ) : (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-3 text-xs"
                        onClick={() => openConfigure(p.value)}
                      >
                        Configure
                      </Button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Configure / Update dialog */}
      <Dialog
        open={dialogOpen}
        onOpenChange={(open) => {
          setDialogOpen(open)
          if (!open) {
            setProvider('')
            setApiKey('')
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <form onSubmit={handleSubmit}>
            <DialogHeader>
              <DialogTitle>
                {platformKeys[provider]?.configured ? 'Update' : 'Configure'} Platform Key
              </DialogTitle>
              <DialogDescription>
                Set the platform-level API key for{' '}
                <span className="font-medium text-foreground">{providerLabel(provider)}</span>.
                This key is shared across all workspaces.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="platform-api-key">API Key</Label>
                <Input
                  id="platform-api-key"
                  type="password"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  autoComplete="off"
                />
                <p className="text-xs text-muted-foreground">
                  The key will be encrypted before storage.
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
              <Button type="submit" disabled={setKeyMutation.isPending || !apiKey}>
                {setKeyMutation.isPending && (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                )}
                Save Platform Key
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Remove confirmation dialog */}
      <Dialog open={!!removeTarget} onOpenChange={(open) => !open && setRemoveTarget(null)}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Remove Platform Key</DialogTitle>
            <DialogDescription>
              Remove the platform-level API key for{' '}
              <span className="font-medium text-foreground">
                {providerLabel(removeTarget || '')}
              </span>
              ? Workspaces without their own BYOK key will lose access to this provider.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRemoveTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={removeKeyMutation.isPending}
              onClick={() => removeTarget && removeKeyMutation.mutate(removeTarget)}
            >
              {removeKeyMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Remove
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
