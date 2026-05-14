'use client'

import { useEffect, useState } from 'react'
import { Loader2, Globe, ShoppingBag, Code2, AlertCircle } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ApiKeyManager } from './ApiKeyManager'
import { ProactivePanel } from '@/components/sites/ProactivePanel'
import { CallbackPanel } from '@/components/sites/CallbackPanel'
import { CartIdlePanel } from '@/components/sites/CartIdlePanel'
import { ShopifyTab } from '@/components/sites/ShopifyTab'
import { listSites } from '@/lib/sites/api'
import type { Site, SiteType } from '@/lib/sites/types'

const TYPE_LABEL: Record<SiteType, string> = {
  shopify: 'Shopify',
  wix: 'Wix',
  woocommerce: 'WooCommerce',
  custom: 'Custom embed',
}

const TYPE_ICON: Record<SiteType, typeof Globe> = {
  shopify: ShoppingBag,
  wix: Globe,
  woocommerce: ShoppingBag,
  custom: Code2,
}

type Section = 'api-keys' | 'behaviour' | 'callback' | 'shopify'

export function WidgetSdkTab() {
  const [sites, setSites] = useState<Site[] | null>(null)
  const [sitesError, setSitesError] = useState<string | null>(null)
  const [activeSiteId, setActiveSiteId] = useState<string | null>(null)
  const [section, setSection] = useState<Section>('api-keys')

  useEffect(() => {
    let cancelled = false
    listSites()
      .then((s) => {
        if (cancelled) return
        setSites(s)
        if (s.length > 0) setActiveSiteId((curr) => curr ?? s[0].id)
      })
      .catch((e) => {
        if (cancelled) return
        setSitesError(e?.message ?? 'Failed to load sites')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const activeSite = sites?.find((s) => s.id === activeSiteId) ?? null

  const sections: Array<{ key: Section; label: string; show: boolean }> = [
    { key: 'api-keys', label: 'API keys', show: true },
    { key: 'behaviour', label: 'Behaviour', show: true },
    { key: 'callback', label: 'Callback', show: true },
    { key: 'shopify', label: 'Shopify', show: activeSite?.type === 'shopify' },
  ]

  const renderSiteRequired = (label: string) => {
    if (sitesError) {
      return (
        <Card>
          <CardContent className="py-6 text-sm text-red-700 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Couldn't load sites</div>
              <div className="text-xs text-red-600 mt-0.5">{sitesError}</div>
              <div className="text-xs text-gray-500 mt-2">
                Sign out and back in if this persists — the dashboard session may have expired.
              </div>
            </div>
          </CardContent>
        </Card>
      )
    }
    if (!sites) {
      return (
        <div className="flex items-center text-gray-500 text-sm py-4">
          <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading sites…
        </div>
      )
    }
    if (sites.length === 0) {
      return (
        <Card>
          <CardContent className="py-8 text-center text-sm text-gray-600 space-y-2">
            <p>No sites yet — install the Automatos app on a Shopify store and your first Site appears automatically.</p>
            <p className="text-xs text-gray-500">
              {label} configuration unlocks once a site exists.
            </p>
          </CardContent>
        </Card>
      )
    }
    return null
  }

  return (
    <div className="space-y-4">
      {sites && sites.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {sites.map((s) => {
            const Icon = TYPE_ICON[s.type] ?? Globe
            const selected = s.id === activeSiteId
            return (
              <button
                key={s.id}
                onClick={() => setActiveSiteId(s.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm border transition ${
                  selected
                    ? 'border-indigo-300 bg-indigo-50 text-indigo-800'
                    : 'border-gray-200 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {s.display_name}
                <Badge variant="outline" className="text-[10px] py-0 px-1.5">
                  {TYPE_LABEL[s.type] ?? s.type}
                </Badge>
              </button>
            )
          })}
        </div>
      )}

      <div className="border-b flex gap-1">
        {sections.filter((s) => s.show).map((s) => (
          <button
            key={s.key}
            onClick={() => setSection(s.key)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition ${
              section === s.key
                ? 'border-indigo-500 text-indigo-700'
                : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {section === 'api-keys' && <ApiKeyManager />}

      {section === 'behaviour' && (
        activeSite ? (
          <div className="space-y-4">
            <ProactivePanel
              site={activeSite}
              onUpdated={(updated) =>
                setSites((prev) => prev?.map((x) => (x.id === updated.id ? updated : x)) ?? null)
              }
            />
            <CartIdlePanel
              site={activeSite}
              onUpdated={(updated) =>
                setSites((prev) => prev?.map((x) => (x.id === updated.id ? updated : x)) ?? null)
              }
            />
          </div>
        ) : renderSiteRequired('Behaviour')
      )}

      {section === 'callback' && (
        activeSite ? (
          <CallbackPanel
            site={activeSite}
            onUpdated={(updated) =>
              setSites((prev) => prev?.map((x) => (x.id === updated.id ? updated : x)) ?? null)
            }
          />
        ) : renderSiteRequired('Callback')
      )}

      {section === 'shopify' && activeSite?.type === 'shopify' && (
        <ShopifyTab site={activeSite} />
      )}
    </div>
  )
}
