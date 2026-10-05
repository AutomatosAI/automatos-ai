'use client'

/**
 * One provider's row in the "Your API Keys" list (PRD-54, issue #830).
 *
 * A key that failed save-time validation is stored `is_active=false` and
 * never resolves, but it still appears here — marked Inactive, with Test
 * still reachable — instead of looking like no key was ever saved.
 */
import { Loader2, TestTube, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import type { ApiKeyOut, ProviderOption } from './api-keys-types'

type ActiveSource = 'byok' | 'platform' | 'none'

interface ApiKeyProviderRowProps {
  provider: ProviderOption
  keys: ApiKeyOut[]
  byokOn: boolean
  hasPlatformKey: boolean
  testingKeyId: number | null
  onToggleByok: (enabled: boolean) => void
  onTest: (id: number) => void
  onDelete: (key: ApiKeyOut) => void
}

const SOURCE_BADGE: Record<ActiveSource, { label: string; className: string }> = {
  byok: { label: 'Using: Your Key', className: 'bg-info/15 text-info border-info/30' },
  platform: { label: 'Using: Platform Key', className: 'bg-success/15 text-success border-success/30' },
  none: { label: 'No key available', className: 'opacity-50' },
}

/** Only an active (validation-passed) key can ever be the one in use. */
function activeSourceFor(hasActiveUserKey: boolean, byokOn: boolean, hasPlatformKey: boolean): ActiveSource {
  if (byokOn && hasActiveUserKey) return 'byok'
  if (hasPlatformKey) return 'platform'
  return 'none'
}

export function ApiKeyProviderRow({
  provider, keys, byokOn, hasPlatformKey, testingKeyId, onToggleByok, onTest, onDelete,
}: ApiKeyProviderRowProps) {
  const key = keys[0]
  const hasKey = !!key
  const hasActiveUserKey = keys.some((k) => k.is_active)
  const activeSource = activeSourceFor(hasActiveUserKey, byokOn, hasPlatformKey)
  const badge = SOURCE_BADGE[activeSource]
  const testing = hasKey && testingKeyId === key.id

  return (
    <div className="rounded-md border border-border/30 px-3 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium w-24">{provider.label}</span>
          <Switch
            checked={byokOn}
            onCheckedChange={onToggleByok}
            disabled={!hasKey}
            aria-label={`Use your ${provider.label} key`}
          />
          <Badge
            variant={activeSource === 'none' ? 'secondary' : undefined}
            className={`text-xs ${badge.className}`}
          >
            {badge.label}
          </Badge>
        </div>

        {hasKey && (
          <div className="flex items-center gap-2">
            <code className="rounded bg-muted/50 px-2 py-0.5 text-xs font-mono">
              {key.masked_key}
            </code>
            {!key.is_active && (
              <Badge
                variant="outline"
                className="text-xs text-warning border-warning/40"
                data-testid={`key-inactive-${provider.value}`}
              >
                Inactive
              </Badge>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              disabled={testing}
              onClick={() => onTest(key.id)}
              title="Test key"
            >
              {testing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <TestTube className="h-3.5 w-3.5" />
              )}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-destructive hover:text-destructive"
              onClick={() => onDelete(key)}
              title="Delete key"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
