'use client'

/**
 * Add API Key dialog (PRD-54, issue #830).
 *
 * `POST /api/keys` validates the key before trusting it: a bad key still
 * comes back 201, stored `is_active=False`, with the outcome in the
 * response's `validation` field. On a failed validation this dialog shows
 * the provider's own error and stays open, instead of claiming success for
 * a key that was never going to work.
 */
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { Loader2, Plus } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
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
import { endpointPlaceholder, keyPlaceholder, providerNotes, useProviderRegistry } from '@/hooks/use-provider-registry'
import type { AddKeyPayload, ApiKeyOut, ProviderOption } from './api-keys-types'

interface AddApiKeyDialogProps {
  providers: ProviderOption[]
}

export function AddApiKeyDialog({ providers }: AddApiKeyDialogProps) {
  const queryClient = useQueryClient()
  const registry = useProviderRegistry()

  const [dialogOpen, setDialogOpen] = useState(false)
  const [provider, setProvider] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [endpoint, setEndpoint] = useState('')
  const selectedNotes = providerNotes(registry, provider)
  const endpointHint = endpointPlaceholder(registry, provider)

  function resetForm() {
    setProvider('')
    setApiKey('')
    setDisplayName('')
    setEndpoint('')
  }

  const addKeyMutation = useMutation({
    mutationFn: (payload: AddKeyPayload) => apiClient.post<ApiKeyOut>('/api/keys', payload),
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
      queryClient.invalidateQueries({ queryKey: ['byok-preferences'] })
      if (saved.validation && !saved.validation.valid) {
        // Stored inactive — never claim success for a key that failed. Keep
        // the dialog open so the user can see the error and fix the key.
        toast.error(`Key saved but failed validation: ${saved.validation.message}`)
        return
      }
      toast.success('API key added successfully')
      resetForm()
      setDialogOpen(false)
    },
    onError: (err: Error) => {
      toast.error(`Failed to add key: ${err.message}`)
    },
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!provider || !apiKey) {
      toast.error('Provider and API key are required')
      return
    }
    const base_url = endpointHint && endpoint.trim() ? endpoint.trim() : undefined
    addKeyMutation.mutate({ provider, api_key: apiKey, display_name: displayName, base_url })
  }

  return (
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
              Select a provider and paste your key. It will be encrypted before storage.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="provider">Provider</Label>
              <Select value={provider} onValueChange={setProvider}>
                <SelectTrigger id="provider">
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  {providers.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="api-key">API Key</Label>
              <Input
                id="api-key"
                type="password"
                placeholder={provider ? keyPlaceholder(registry, provider) : 'sk-...'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                autoComplete="off"
              />
              {selectedNotes.length > 0 && (
                <div className="space-y-1" data-testid="provider-notes">
                  {selectedNotes.map((note) => (
                    <p key={note} className="text-xs text-muted-foreground">
                      {note}
                    </p>
                  ))}
                </div>
              )}
            </div>

            {endpointHint && (
              <div className="space-y-2">
                <Label htmlFor="endpoint">Endpoint</Label>
                <Input
                  id="endpoint"
                  type="url"
                  placeholder={endpointHint}
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  autoComplete="off"
                />
              </div>
            )}

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
            <Button type="button" variant="outline" onClick={() => setDialogOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={addKeyMutation.isPending}>
              {addKeyMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Add Key
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
