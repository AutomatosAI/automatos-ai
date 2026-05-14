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

type Section = 'behaviour' | 'callback' | 'api-keys' | 'shopify'

export function WidgetSdkTab() {
  const [sites, setSites] = useState<Site[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeSiteId, setActiveSiteId] = useState<string | null>(null)
  const [section, setSection] = useState<Section>('behaviour')

  useEffect(() => {
    listSites()
      .then((s) => {
        setSites(s)
        if (s.length > 0 && !activeSiteId) setActiveSiteId(s[0].id)
      })
      .catch((e) => setError(e?.message ?? 'Failed to load sites'))
  }, [activeSiteId])

  const activeSite = sites?.find((s) => s.id === activeSiteId) ?? null

  const sections: Array<{ key: Section; label: string; show: boolean }> = [
    { key: 'behaviour', label: 'Behaviour', show: true },
    { key: 'callback', label: 'Callback', show: true },
    { key: 'api-keys', label: 'API keys', show: true },
    { key: 'shopify', label: 'Shopify', show: activeSite?.type === 'shopify' },
  ]

  return (
    <div className="space-y-6">
      {error && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded text-sm">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!sites && !error && (
        <div className="flex items-center text-gray-500 text-sm">
          <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading sites…
        </div>
      )}

      {sites && sites.length === 0 && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-gray-600 space-y-2">
            <p>No sites yet — install the Automatos app on a Shopify store and your first Site appears automatically.</p>
            <p className="text-xs text-gray-500">
              Wix, WooCommerce, and custom embed adapters are next on the roadmap.
            </p>
          </CardContent>
        </Card>
      )}

      {sites && sites.length > 0 && activeSite && (
        <>
          {sites.length > 1 && (
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

          {section === 'behaviour' && (
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
          )}

          {section === 'callback' && (
            <CallbackPanel
              site={activeSite}
              onUpdated={(updated) =>
                setSites((prev) => prev?.map((x) => (x.id === updated.id ? updated : x)) ?? null)
              }
            />
          )}

          {section === 'api-keys' && <ApiKeyManager />}

          {section === 'shopify' && activeSite.type === 'shopify' && (
            <ShopifyTab site={activeSite} />
          )}
        </>
      )}
    </div>
  )
}
