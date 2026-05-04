'use client'

import { useState } from 'react'
import {
  CreditCard,
  RefreshCw,
  Clock,
  CalendarDays,
  Calendar,
  Settings,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useOpenRouterCredits, useOpenRouterKeyInfo, useTriggerOpenRouterSync } from '@/hooks/use-unified-analytics'

function formatCurrency(n: number): string {
  if (n >= 1000) return `$${(n / 1000).toFixed(1)}K`
  return `$${n.toFixed(4)}`
}

function getUsageColor(percentage: number): string {
  if (percentage >= 90) return 'bg-[hsl(var(--destructive))]'
  if (percentage >= 70) return 'bg-[hsl(var(--warning))]'
  return 'bg-[hsl(var(--success))]'
}

function getUsageTextColor(percentage: number): string {
  if (percentage >= 90) return 'text-[hsl(var(--destructive))]'
  if (percentage >= 70) return 'text-[hsl(var(--warning))]'
  return 'text-[hsl(var(--success))]'
}

export function AnalyticsOpenRouterCredits() {
  const { data: credits, isLoading: creditsLoading } = useOpenRouterCredits()
  const { data: keyInfo, isLoading: keyInfoLoading } = useOpenRouterKeyInfo()
  const syncMutation = useTriggerOpenRouterSync()

  const isLoading = creditsLoading || keyInfoLoading

  // No OpenRouter key configured
  if (!isLoading && credits === null) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CreditCard className="w-5 h-5 text-[hsl(var(--agent))]" />
            OpenRouter Credits
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <CreditCard className="w-10 h-10 mx-auto mb-3 opacity-50 text-muted-foreground" />
            <p className="text-sm text-muted-foreground mb-3">
              No OpenRouter API key configured
            </p>
            <a href="/settings" className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
              <Settings className="w-4 h-4" />
              Configure in Settings
            </a>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CreditCard className="w-5 h-5 text-[hsl(var(--agent))]" />
            OpenRouter Credits
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-8 w-32" />
          <Skeleton className="h-3 w-full" />
          <div className="grid grid-cols-3 gap-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const totalCredits = credits?.total_credits ?? 0
  const totalUsage = credits?.total_usage ?? 0
  const remaining = totalCredits - totalUsage
  const usagePercent = totalCredits > 0
    ? Math.min(100, (totalUsage / totalCredits) * 100)
    : 0

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <CreditCard className="w-5 h-5 text-[hsl(var(--agent))]" />
            OpenRouter Credits
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => syncMutation.mutate()}
            disabled={syncMutation.isLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${syncMutation.isLoading ? 'animate-spin' : ''}`} />
            {syncMutation.isLoading ? 'Syncing...' : 'Sync Activity'}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Credits Balance */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-2xl font-bold">{formatCurrency(remaining)}</p>
              <p className="text-xs text-muted-foreground">Remaining Balance</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">
                {formatCurrency(totalUsage)} used of {formatCurrency(totalCredits)}
              </p>
            </div>
          </div>

          {/* Progress bar */}
          {totalCredits > 0 && (
            <div className="space-y-1">
              <div className="h-3 rounded-full bg-secondary overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${getUsageColor(usagePercent)}`}
                  style={{ width: `${usagePercent}%` }}
                />
              </div>
              <p className={`text-xs ${getUsageTextColor(usagePercent)}`}>
                {usagePercent.toFixed(1)}% of credits used
              </p>
            </div>
          )}
        </div>

        {/* Usage Breakdown */}
        {keyInfo && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-3 rounded-lg bg-secondary/20">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="w-4 h-4 text-[hsl(var(--info))]" />
                <span className="text-xs text-muted-foreground">Daily</span>
              </div>
              <p className="text-sm font-medium">{formatCurrency(keyInfo.usage_daily)}</p>
            </div>
            <div className="p-3 rounded-lg bg-secondary/20">
              <div className="flex items-center gap-2 mb-1">
                <CalendarDays className="w-4 h-4 text-[hsl(var(--success))]" />
                <span className="text-xs text-muted-foreground">Weekly</span>
              </div>
              <p className="text-sm font-medium">{formatCurrency(keyInfo.usage_weekly)}</p>
            </div>
            <div className="p-3 rounded-lg bg-secondary/20">
              <div className="flex items-center gap-2 mb-1">
                <Calendar className="w-4 h-4 text-primary" />
                <span className="text-xs text-muted-foreground">Monthly</span>
              </div>
              <p className="text-sm font-medium">{formatCurrency(keyInfo.usage_monthly)}</p>
            </div>
          </div>
        )}

        {/* Limit Info */}
        {keyInfo?.limit_remaining !== null && keyInfo?.limit_remaining !== undefined && (
          <div className="text-xs text-muted-foreground border-t border-border/50 pt-3">
            Rate limit remaining: {formatCurrency(keyInfo.limit_remaining)}
            {keyInfo.is_free_tier && (
              <span className="ml-2 text-[hsl(var(--warning))]">(Free tier)</span>
            )}
          </div>
        )}

        {/* Sync result feedback */}
        {syncMutation.isSuccess && (
          <div className="text-xs text-[hsl(var(--success))] border-t border-border/50 pt-3">
            Synced {syncMutation.data.synced} records
            {syncMutation.data.skipped > 0 && ` (${syncMutation.data.skipped} skipped)`}
          </div>
        )}
        {syncMutation.isError && (
          <div className="text-xs text-[hsl(var(--destructive))] border-t border-border/50 pt-3">
            Sync failed: {syncMutation.error.message}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
