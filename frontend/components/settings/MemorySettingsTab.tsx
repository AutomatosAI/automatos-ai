/**
 * Memory Management Settings Tab Component
 * =========================================
 *
 * Configure memory storage limits, context budgets, circuit breaker,
 * cache TTLs, and decay/promotion thresholds.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Save, RotateCcw, Database, Brain, Shield, Clock, TrendingDown } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'
import { InlineHelp } from '@/components/ui/help-tooltip'

interface MemorySettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

export default function MemorySettingsTab({
  settings,
  onSave,
  saving,
  onReset
}: MemorySettingsTabProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

  React.useEffect(() => {
    const initialData: Record<string, string> = {}
    settings.forEach(setting => {
      initialData[setting.key] = setting.value !== null && setting.value !== undefined
        ? setting.value
        : (setting.default_value || '')
    })
    setFormData(initialData)
  }, [settings])

  const handleInputChange = (key: string, value: string) => {
    setFormData(prev => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    onSave(formData)
  }

  const handleReset = () => {
    const defaultData: Record<string, string> = {}
    settings.forEach(setting => {
      defaultData[setting.key] = setting.default_value || ''
    })
    setFormData(defaultData)
    onReset()
  }

  return (
    <div className="space-y-6">
      {/* Storage Limits */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Storage Limits
          </CardTitle>
          <CardDescription>
            Control how much conversation content is stored per memory save. Higher values preserve richer context for recall.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="store_max_chars" className="flex items-center gap-1">Max Characters Per Message <InlineHelp id="settings.memory.store_max_chars" size="sm" /></Label>
              <Input
                id="store_max_chars"
                type="number"
                min="500"
                max="20000"
                step="500"
                value={formData.store_max_chars || '6000'}
                onChange={(e) => handleInputChange('store_max_chars', e.target.value)}
                placeholder="6000"
              />
              <p className="text-xs text-muted-foreground">
                Characters per side (user + assistant) sent to Mem0 for fact extraction. ~1500 words at 6000.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="daily_log_max_chars" className="flex items-center gap-1">Daily Log Max Characters <InlineHelp id="settings.memory.daily_log_max_chars" size="sm" /></Label>
              <Input
                id="daily_log_max_chars"
                type="number"
                min="500"
                max="10000"
                step="500"
                value={formData.daily_log_max_chars || '2000'}
                onChange={(e) => handleInputChange('daily_log_max_chars', e.target.value)}
                placeholder="2000"
              />
              <p className="text-xs text-muted-foreground">
                Max size for daily activity summary entries.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Context Injection Budgets */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Context Injection Budgets
          </CardTitle>
          <CardDescription>
            Token budgets controlling how much memory is injected into agent prompts. Total is split across layers.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="context_budget_total" className="flex items-center gap-1">Total Budget (tokens) <InlineHelp id="settings.memory.context_budget_total" size="sm" /></Label>
              <Input
                id="context_budget_total"
                type="number"
                min="1000"
                max="16000"
                step="500"
                value={formData.context_budget_total || '4000'}
                onChange={(e) => handleInputChange('context_budget_total', e.target.value)}
                placeholder="4000"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="context_budget_session" className="flex items-center gap-1">Session (L1) Budget <InlineHelp id="settings.memory.context_budget_session" size="sm" /></Label>
              <Input
                id="context_budget_session"
                type="number"
                min="100"
                max="4000"
                step="100"
                value={formData.context_budget_session || '500'}
                onChange={(e) => handleInputChange('context_budget_session', e.target.value)}
                placeholder="500"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="context_budget_long_term" className="flex items-center gap-1">Long-Term (L3) Budget <InlineHelp id="settings.memory.context_budget_long_term" size="sm" /></Label>
              <Input
                id="context_budget_long_term"
                type="number"
                min="200"
                max="4000"
                step="100"
                value={formData.context_budget_long_term || '800'}
                onChange={(e) => handleInputChange('context_budget_long_term', e.target.value)}
                placeholder="800"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="context_budget_temporal" className="flex items-center gap-1">Temporal (L2) Budget <InlineHelp id="settings.memory.context_budget_temporal" size="sm" /></Label>
              <Input
                id="context_budget_temporal"
                type="number"
                min="200"
                max="4000"
                step="100"
                value={formData.context_budget_temporal || '600'}
                onChange={(e) => handleInputChange('context_budget_temporal', e.target.value)}
                placeholder="600"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="context_budget_daily" className="flex items-center gap-1">Daily Logs Budget <InlineHelp id="settings.memory.context_budget_daily" size="sm" /></Label>
              <Input
                id="context_budget_daily"
                type="number"
                min="100"
                max="2000"
                step="100"
                value={formData.context_budget_daily || '400'}
                onChange={(e) => handleInputChange('context_budget_daily', e.target.value)}
                placeholder="400"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Retrieval Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Retrieval
          </CardTitle>
          <CardDescription>
            How many memories are fetched and injected per query.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="search_result_limit" className="flex items-center gap-1">Search Result Limit <InlineHelp id="settings.memory.search_result_limit" size="sm" /></Label>
              <Input
                id="search_result_limit"
                type="number"
                min="1"
                max="50"
                value={formData.search_result_limit || '8'}
                onChange={(e) => handleInputChange('search_result_limit', e.target.value)}
                placeholder="8"
              />
              <p className="text-xs text-muted-foreground">
                Max memories returned per search query.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="long_term_search_limit" className="flex items-center gap-1">Long-Term Fetch Limit <InlineHelp id="settings.memory.long_term_search_limit" size="sm" /></Label>
              <Input
                id="long_term_search_limit"
                type="number"
                min="1"
                max="20"
                value={formData.long_term_search_limit || '5'}
                onChange={(e) => handleInputChange('long_term_search_limit', e.target.value)}
                placeholder="5"
              />
              <p className="text-xs text-muted-foreground">
                Mem0 memories fetched for context injection.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Circuit Breaker & Timeouts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Circuit Breaker & Timeouts
          </CardTitle>
          <CardDescription>
            Resilience settings for the Mem0 connection. Circuit breaker prevents cascading failures when Mem0 is down.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="circuit_breaker_threshold" className="flex items-center gap-1">Failure Threshold <InlineHelp id="settings.memory.circuit_breaker_threshold" size="sm" /></Label>
              <Input
                id="circuit_breaker_threshold"
                type="number"
                min="2"
                max="20"
                value={formData.circuit_breaker_threshold || '5'}
                onChange={(e) => handleInputChange('circuit_breaker_threshold', e.target.value)}
                placeholder="5"
              />
              <p className="text-xs text-muted-foreground">
                Consecutive failures before circuit opens.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="circuit_breaker_cooldown" className="flex items-center gap-1">Cooldown (seconds) <InlineHelp id="settings.memory.circuit_breaker_cooldown" size="sm" /></Label>
              <Input
                id="circuit_breaker_cooldown"
                type="number"
                min="10"
                max="600"
                step="10"
                value={formData.circuit_breaker_cooldown || '60'}
                onChange={(e) => handleInputChange('circuit_breaker_cooldown', e.target.value)}
                placeholder="60"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="request_timeout" className="flex items-center gap-1">Request Timeout (seconds) <InlineHelp id="settings.memory.request_timeout" size="sm" /></Label>
              <Input
                id="request_timeout"
                type="number"
                min="5"
                max="60"
                step="5"
                value={formData.request_timeout || '15'}
                onChange={(e) => handleInputChange('request_timeout', e.target.value)}
                placeholder="15"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Cache & Session TTLs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Cache & Session TTLs
          </CardTitle>
          <CardDescription>
            Time-to-live settings for caches and session memory.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="cache_ttl" className="flex items-center gap-1">Search Cache TTL (seconds) <InlineHelp id="settings.memory.cache_ttl" size="sm" /></Label>
              <Input
                id="cache_ttl"
                type="number"
                min="30"
                max="1800"
                step="30"
                value={formData.cache_ttl || '300'}
                onChange={(e) => handleInputChange('cache_ttl', e.target.value)}
                placeholder="300"
              />
              <p className="text-xs text-muted-foreground">
                How long Mem0 search results are cached. Lower = fresher results, more API calls.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="session_ttl" className="flex items-center gap-1">Session TTL (seconds) <InlineHelp id="settings.memory.session_ttl" size="sm" /></Label>
              <Input
                id="session_ttl"
                type="number"
                min="3600"
                max="604800"
                step="3600"
                value={formData.session_ttl || '86400'}
                onChange={(e) => handleInputChange('session_ttl', e.target.value)}
                placeholder="86400"
              />
              <p className="text-xs text-muted-foreground">
                L1 session memory lifetime in Redis. Default 24 hours (86400s).
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Decay & Promotion */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="h-5 w-5" />
            Decay & Promotion
          </CardTitle>
          <CardDescription>
            Controls how memories age out and how important memories get promoted to long-term storage.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="decay_rate" className="flex items-center gap-1">Decay Rate <InlineHelp id="settings.memory.decay_rate" size="sm" /></Label>
              <Input
                id="decay_rate"
                type="number"
                min="0.01"
                max="0.5"
                step="0.01"
                value={formData.decay_rate || '0.1'}
                onChange={(e) => handleInputChange('decay_rate', e.target.value)}
                placeholder="0.1"
              />
              <p className="text-xs text-muted-foreground">
                Importance decay per cycle (0.0-1.0). Higher = faster forgetting.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="promotion_min_importance" className="flex items-center gap-1">Promotion Importance <InlineHelp id="settings.memory.promotion_min_importance" size="sm" /></Label>
              <Input
                id="promotion_min_importance"
                type="number"
                min="0.3"
                max="1.0"
                step="0.05"
                value={formData.promotion_min_importance || '0.7'}
                onChange={(e) => handleInputChange('promotion_min_importance', e.target.value)}
                placeholder="0.7"
              />
              <p className="text-xs text-muted-foreground">
                Min importance score for L2 to L3 promotion.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="promotion_min_access_count" className="flex items-center gap-1">Min Access Count <InlineHelp id="settings.memory.promotion_min_access_count" size="sm" /></Label>
              <Input
                id="promotion_min_access_count"
                type="number"
                min="1"
                max="20"
                value={formData.promotion_min_access_count || '3'}
                onChange={(e) => handleInputChange('promotion_min_access_count', e.target.value)}
                placeholder="3"
              />
              <p className="text-xs text-muted-foreground">
                Times a memory must be accessed before promotion.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Save / Reset */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={handleReset} disabled={saving}>
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset to Defaults
        </Button>
        <Button onClick={handleSave} disabled={saving}>
          {saving ? (
            <><span className="animate-spin mr-2">...</span> Saving...</>
          ) : (
            <><Save className="mr-2 h-4 w-4" /> Save Memory Settings</>
          )}
        </Button>
      </div>
    </div>
  )
}
