'use client'

import { useEffect, useState } from 'react'
import {
  Loader2, RefreshCw, CheckCircle2, AlertCircle,
  ShoppingBag, Network,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  getShopifySyncStatus,
  startShopifySync,
  getShopifyOrdersSyncStatus,
  startShopifyOrdersSync,
  type ShopifySyncStatus,
  type ShopifyOrdersSyncStatus,
} from '@/lib/sites/api'
import type { Site } from '@/lib/sites/types'

interface SyncPanelProps {
  site: Site
  workspaceId: string | null
}

// One platform-agnostic panel that drives ALL catalog→knowledge-graph syncs.
// Today only Shopify lights up; WooCommerce / BigCommerce / Amazon plug in by
// adding a section here with the same shape: status row + last-sync stats +
// "Sync now" button + auth-required guard.
export function SyncPanel({ site, workspaceId }: SyncPanelProps) {
  return (
    <div className="space-y-4">
      {/* Header card describing what this tab does — keeps merchants oriented
          as we add more platforms here. */}
      <Card className="glass-card">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Catalog sync</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Pulls your product catalog into the workspace knowledge graph so the
          agent answers product questions from real data instead of guessing.
          The graph stays fresh automatically via webhooks; use the button
          below for a manual full re-sync (e.g. after a CSV import).
        </CardContent>
      </Card>

      {/* Per-platform cards. Connectivity is workspace-level (Composio), not
          site.type — InbuildUK has type='custom' but is fully wired to
          Shopify via Composio. So we always render the Shopify card; the card
          itself checks if Shopify is actually connected and guides the
          merchant if not. Add WooCommerce/BigCommerce/Amazon cards here as
          their integrations land. */}
      <ShopifySyncCard site={site} workspaceId={workspaceId} />
      <ShopifyOrdersSyncCard workspaceId={workspaceId} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Shopify card
// ---------------------------------------------------------------------------

function ShopifySyncCard({ site, workspaceId }: { site: Site; workspaceId: string | null }) {
  const [status, setStatus] = useState<ShopifySyncStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function refreshStatus() {
    if (!workspaceId) {
      setLoading(false)
      return
    }
    try {
      const s = await getShopifySyncStatus(workspaceId)
      setStatus(s)
      setError(null)
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load sync status')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refreshStatus()
  }, [workspaceId])

  // No front-end readiness gate — Composio connection status lives at the
  // workspace level (composio_connections table) and isn't exposed on the
  // Site row. The backend's POST /api/shopify/sync/products/start returns a
  // clear 400/503 if Composio isn't wired; that error renders in the panel
  // below. Better to let merchants click and see truth than block them on a
  // stale site-level flag.

  // Detect a "running" status that's actually stuck (started_at older than
  // 5 minutes). The bulk-op should complete in ~45s; anything beyond 5 min
  // means the previous request crashed without writing a terminal status.
  // We let the merchant retry by treating the row as effectively idle.
  const isStuckRunning =
    status?.status === 'running' &&
    !!status.started_at &&
    Date.now() / 1000 - status.started_at > 300

  async function handleSync() {
    if (!workspaceId) return
    setSyncing(true)
    setError(null)
    try {
      const result = await startShopifySync(workspaceId)
      setStatus(result)
    } catch (e: any) {
      setError(e?.message ?? 'Sync failed')
    } finally {
      setSyncing(false)
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <ShoppingBag className="w-4 h-4" /> Shopify catalog
          {status?.status === 'complete' && (
            <Badge variant="outline" className="ml-auto text-emerald-400 border-emerald-400/30">
              <CheckCircle2 className="w-3 h-3 mr-1" /> Synced
            </Badge>
          )}
          {status?.status === 'running' && (
            <Badge variant="outline" className="ml-auto text-amber-400 border-amber-400/30">
              <Loader2 className="w-3 h-3 mr-1 animate-spin" /> Running
            </Badge>
          )}
          {status?.status === 'error' && (
            <Badge variant="outline" className="ml-auto text-destructive border-destructive/30">
              <AlertCircle className="w-3 h-3 mr-1" /> Error
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {loading && (
          <div className="flex items-center text-muted-foreground">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading sync status…
          </div>
        )}

        {!loading && status?.status === 'complete' && (
          <div className="grid grid-cols-2 gap-y-1.5 text-xs">
            <span className="text-muted-foreground">Products + variants indexed</span>
            <span className="text-foreground">{status.node_count?.toLocaleString() ?? '—'}</span>
            <span className="text-muted-foreground">Relations</span>
            <span className="text-foreground">{status.edge_count?.toLocaleString() ?? '—'}</span>
            <span className="text-muted-foreground">Last sync duration</span>
            <span className="text-foreground">
              {status.duration_seconds ? `${status.duration_seconds.toFixed(1)}s` : '—'}
            </span>
          </div>
        )}

        {!loading && status?.status === 'never_synced' && (
          <p className="text-xs text-muted-foreground">
            Catalog hasn't been synced yet. Click <span className="text-foreground">Sync now</span> to
            build the knowledge graph for this store.
          </p>
        )}

        {!loading && status?.status === 'error' && (
          <div className="text-xs text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
            {status.error || 'Previous sync failed.'}
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
            {error}
          </div>
        )}


        {isStuckRunning && (
          <div className="text-xs text-amber-400 bg-amber-400/10 border border-amber-400/30 rounded p-2">
            Previous sync didn't finish cleanly. You can re-run it now.
          </div>
        )}

        <Button
          onClick={handleSync}
          disabled={!workspaceId || syncing || (status?.status === 'running' && !isStuckRunning)}
          size="sm"
          className="gap-2"
        >
          {syncing ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Syncing 1,000+ products… (~45s)
            </>
          ) : (
            <>
              <RefreshCw className="w-3.5 h-3.5" /> Sync now
            </>
          )}
        </Button>
        <p className="text-[11px] text-muted-foreground">
          The sync pulls every product, variant, metafield, and collection from Shopify in one
          Bulk Operation. Webhooks keep the graph fresh between manual syncs.
        </p>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Orders card — PRD-009 Phase 2
// ---------------------------------------------------------------------------
// Pulls order history (PII-stripped — only product co-occurrence, NO
// customer / email / phone / address). Adds FREQUENTLY_BOUGHT_WITH edges
// between products so the agent can recommend "customers who bought X
// also bought Y". Requires the catalog sync to have run first.

// Window presets — merchant picks before clicking Sync. days=0 means
// all-time (omit the date filter in the bulk-op GraphQL).
const WINDOW_OPTIONS: Array<{ days: number; label: string; hint: string }> = [
  { days: 90,  label: '90 days', hint: 'Last quarter' },
  { days: 180, label: '180 days', hint: 'Last 6 months' },
  { days: 365, label: '1 year',  hint: 'Last 12 months' },
  { days: 0,   label: 'All time', hint: 'Every order ever' },
]

function ShopifyOrdersSyncCard({ workspaceId }: { workspaceId: string | null }) {
  const [status, setStatus] = useState<ShopifyOrdersSyncStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [syncing, setSyncing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [windowDays, setWindowDays] = useState<number>(90)

  async function refreshStatus() {
    if (!workspaceId) { setLoading(false); return }
    try {
      const s = await getShopifyOrdersSyncStatus(workspaceId)
      setStatus(s); setError(null)
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load orders sync status')
    } finally { setLoading(false) }
  }

  useEffect(() => { refreshStatus() }, [workspaceId])

  // Same stuck-state detection as catalog card
  const isStuckRunning =
    status?.status === 'running' &&
    !!status.started_at &&
    Date.now() / 1000 - status.started_at > 600  // orders can be slower than catalog

  async function handleSync() {
    if (!workspaceId) return
    setSyncing(true); setError(null)
    try {
      // Auto-tune min_support to the window size: tiny window → 1 (any
      // co-occurrence counts), big window → higher floor so noise filters
      // out. Floor at 2 except for explicitly tight windows.
      const minSupport = windowDays === 0 || windowDays >= 365 ? 3 : 2
      const result = await startShopifyOrdersSync(workspaceId, { days: windowDays, minSupport })
      setStatus(result)
    } catch (e: any) {
      setError(e?.message ?? 'Sync failed')
    } finally { setSyncing(false) }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Network className="w-4 h-4" /> Order patterns
          {status?.status === 'complete' && (
            <Badge variant="outline" className="ml-auto text-emerald-400 border-emerald-400/30">
              <CheckCircle2 className="w-3 h-3 mr-1" /> Synced
            </Badge>
          )}
          {status?.status === 'running' && !isStuckRunning && (
            <Badge variant="outline" className="ml-auto text-amber-400 border-amber-400/30">
              <Loader2 className="w-3 h-3 mr-1 animate-spin" /> Running
            </Badge>
          )}
          {status?.status === 'error' && (
            <Badge variant="outline" className="ml-auto text-destructive border-destructive/30">
              <AlertCircle className="w-3 h-3 mr-1" /> Error
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p className="text-xs text-muted-foreground">
          Adds <strong>frequently-bought-together</strong> relationships between products by analysing
          your order history. Only product IDs are read — no customer data, email,
          address, or phone touches the graph.
        </p>

        {/* Window selector. For test stores: pick All time. For high-volume
            stores: shorter windows = faster syncs, less drift from current
            ranges. */}
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground/70">
            Analyse orders from the last…
          </div>
          <div className="inline-flex bg-black/30 backdrop-blur-sm border border-white/10 rounded-md overflow-hidden">
            {WINDOW_OPTIONS.map(({ days, label, hint }, i) => (
              <button
                key={days}
                onClick={() => setWindowDays(days)}
                disabled={syncing}
                title={hint}
                className={`px-3 py-1.5 text-xs transition ${i > 0 ? 'border-l border-white/10' : ''} ${
                  windowDays === days
                    ? 'bg-white/15 text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {loading && (
          <div className="flex items-center text-muted-foreground">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading status…
          </div>
        )}

        {!loading && status?.status === 'complete' && (
          <div className="grid grid-cols-2 gap-y-1.5 text-xs">
            <span className="text-muted-foreground">FBT relations added</span>
            <span className="text-foreground">{status.fbt_edges_added?.toLocaleString() ?? '—'}</span>
            <span className="text-muted-foreground">Orders analysed</span>
            <span className="text-foreground">{status.total_orders_analysed?.toLocaleString() ?? '—'}</span>
            <span className="text-muted-foreground">Window</span>
            <span className="text-foreground">{status.days_window ?? status.days ?? 90} days</span>
            <span className="text-muted-foreground">Last sync duration</span>
            <span className="text-foreground">
              {status.duration_seconds ? `${status.duration_seconds.toFixed(1)}s` : '—'}
            </span>
          </div>
        )}

        {!loading && status?.status === 'never_synced' && (
          <p className="text-xs text-muted-foreground">
            Orders haven't been synced yet. Click <span className="text-foreground">Sync now</span> to
            build customer co-purchase patterns. Sync the catalog first if you haven't.
          </p>
        )}

        {!loading && status?.status === 'error' && (
          <div className="text-xs text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
            {status.error || 'Previous sync failed.'}
          </div>
        )}

        {error && (
          <div className="text-xs text-destructive bg-destructive/10 border border-destructive/30 rounded p-2">
            {error}
          </div>
        )}

        {isStuckRunning && (
          <div className="text-xs text-amber-400 bg-amber-400/10 border border-amber-400/30 rounded p-2">
            Previous sync didn't finish cleanly. You can re-run it now.
          </div>
        )}

        <Button
          onClick={handleSync}
          disabled={!workspaceId || syncing || (status?.status === 'running' && !isStuckRunning)}
          size="sm"
          className="gap-2"
        >
          {syncing ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Analysing orders… (1-2 min)
            </>
          ) : (
            <>
              <RefreshCw className="w-3.5 h-3.5" /> Sync orders
            </>
          )}
        </Button>
        <p className="text-[11px] text-muted-foreground">
          GDPR-safe by design: only product IDs and order counts touch the graph.
          Customer records stay in Shopify as the system of record.
        </p>
      </CardContent>
    </Card>
  )
}
