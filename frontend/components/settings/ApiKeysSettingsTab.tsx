'use client'

import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Key, Loader2, XCircle, Shield } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { providersForByok, useProviderRegistry } from '@/hooks/use-provider-registry'
import { AddApiKeyDialog } from './AddApiKeyDialog'
import { ApiKeyProviderRow } from './ApiKeyProviderRow'
import { DeleteApiKeyDialog } from './DeleteApiKeyDialog'
import { useApiKeysMutations } from './use-api-keys-mutations'
import type { ApiKeyOut, ByokPreferences, PlatformKeyStatus } from './api-keys-types'

export function ApiKeysSettingsTab() {
  const queryClient = useQueryClient()
  const [deleteTarget, setDeleteTarget] = useState<ApiKeyOut | null>(null)
  const { deleteKeyMutation, testKeyMutation, saveByokMutation, testingKeyId } = useApiKeysMutations()

  // PRD-236: the provider list is the backend registry (static fallback while loading)
  const registry = useProviderRegistry()
  const PROVIDERS = useMemo(
    () => providersForByok(registry).map((p) => ({ value: p.slug, label: p.label })),
    [registry],
  )

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

  const byokOverrides = byokPrefs?.byok_overrides ?? {}
  const platformKeys = platformStatus?.platform_keys ?? {}

  function handleDelete(id: number) {
    deleteKeyMutation.mutate(id, { onSuccess: () => setDeleteTarget(null) })
  }

  return (
    <div className="space-y-6">
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
            <AddApiKeyDialog providers={PROVIDERS} />
          </div>
        </CardHeader>
      </Card>

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
              {PROVIDERS.map((p) => (
                <ApiKeyProviderRow
                  key={p.value}
                  provider={p}
                  keys={keys.filter((k) => k.provider === p.value)}
                  byokOn={byokOverrides[p.value] ?? false}
                  hasPlatformKey={platformKeys[p.value]?.configured ?? false}
                  testingKeyId={testingKeyId}
                  onToggleByok={(enabled) => saveByokMutation.mutate({ ...byokOverrides, [p.value]: enabled })}
                  onTest={(id) => testKeyMutation.mutate(id)}
                  onDelete={setDeleteTarget}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <DeleteApiKeyDialog
        target={deleteTarget}
        isPending={deleteKeyMutation.isPending}
        onClose={() => setDeleteTarget(null)}
        onConfirm={handleDelete}
      />
    </div>
  )
}
