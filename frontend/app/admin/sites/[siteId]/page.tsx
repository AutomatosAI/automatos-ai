'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { MainLayout } from '@/components/layout/main-layout';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { usePageAPI } from '@/hooks/use-page-api';
import { getSite } from '@/lib/sites/api';
import type { Site } from '@/lib/sites/types';
import { ProactivePanel } from '@/components/sites/ProactivePanel';
import { CallbackPanel } from '@/components/sites/CallbackPanel';
import { CartIdlePanel } from '@/components/sites/CartIdlePanel';
import { ShopifyTab } from '@/components/sites/ShopifyTab';

type TabKey = 'widget' | 'destinations' | 'shopify';

export default function SiteDetailPage() {
  usePageAPI('site_detail');
  const params = useParams();
  const siteId = params?.siteId as string;
  const [site, setSite] = useState<Site | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>('widget');

  useEffect(() => {
    if (!siteId) return;
    getSite(siteId)
      .then(setSite)
      .catch((e) => setError(e?.message ?? 'Failed to load Site'));
  }, [siteId]);

  return (
    <MainLayout>
      <div className="container mx-auto p-6 max-w-4xl">
        <Link
          href="/admin/sites"
          className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" /> Back to Sites
        </Link>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4 text-sm">
            {error}
          </div>
        )}

        {!site && !error && (
          <div className="flex items-center text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading…
          </div>
        )}

        {site && (
          <>
            <div className="mb-6">
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-semibold">{site.display_name}</h1>
                <Badge variant="outline" className="ml-2 text-xs">
                  {site.type}
                </Badge>
              </div>
              {site.external_id && (
                <p className="text-sm text-gray-500 mt-1">{site.external_id}</p>
              )}
            </div>

            <div className="border-b mb-4 flex gap-1">
              {(['widget', 'destinations', 'shopify'] as TabKey[])
                .filter((t) => t !== 'shopify' || site.type === 'shopify')
                .map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-4 py-2 text-sm border-b-2 -mb-px transition ${
                      tab === t
                        ? 'border-indigo-500 text-indigo-700'
                        : 'border-transparent text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    {t === 'widget' ? 'Widget' : t === 'destinations' ? 'Destinations' : 'Shopify'}
                  </button>
                ))}
            </div>

            {tab === 'widget' && (
              <div className="space-y-4">
                <ProactivePanel site={site} onUpdated={setSite} />
                <CartIdlePanel site={site} onUpdated={setSite} />
              </div>
            )}
            {tab === 'destinations' && (
              <CallbackPanel site={site} onUpdated={setSite} />
            )}
            {tab === 'shopify' && site.type === 'shopify' && (
              <ShopifyTab site={site} />
            )}
          </>
        )}
      </div>
    </MainLayout>
  );
}
