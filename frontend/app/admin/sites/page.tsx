'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Globe,
  ShoppingBag,
  Code2,
  CheckCircle2,
  PauseCircle,
  AlertTriangle,
  ChevronRight,
  Loader2,
} from 'lucide-react';
import { MainLayout } from '@/components/layout/main-layout';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { usePageAPI } from '@/hooks/use-page-api';
import { listSites } from '@/lib/sites/api';
import type { Site, SiteType, SiteStatus } from '@/lib/sites/types';


const TYPE_LABEL: Record<SiteType, string> = {
  shopify: 'Shopify',
  wix: 'Wix',
  woocommerce: 'WooCommerce',
  custom: 'Custom embed',
};

const TYPE_ICON: Record<SiteType, typeof Globe> = {
  shopify: ShoppingBag,
  wix: Globe,
  woocommerce: ShoppingBag,
  custom: Code2,
};


function StatusBadge({ status }: { status: SiteStatus }) {
  if (status === 'active') {
    return (
      <Badge variant="outline" className="text-emerald-700 border-emerald-200 bg-emerald-50">
        <CheckCircle2 className="w-3 h-3 mr-1" /> Active
      </Badge>
    );
  }
  if (status === 'paused') {
    return (
      <Badge variant="outline" className="text-amber-700 border-amber-200 bg-amber-50">
        <PauseCircle className="w-3 h-3 mr-1" /> Paused
      </Badge>
    );
  }
  if (status === 'error') {
    return (
      <Badge variant="outline" className="text-red-700 border-red-200 bg-red-50">
        <AlertTriangle className="w-3 h-3 mr-1" /> Error
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="text-gray-600 border-gray-200">
      Disconnected
    </Badge>
  );
}


export default function SitesListPage() {
  usePageAPI('sites');
  const [sites, setSites] = useState<Site[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSites()
      .then(setSites)
      .catch((e) => setError(e?.message ?? 'Failed to load sites'));
  }, []);

  return (
    <MainLayout>
      <div className="container mx-auto p-6 max-w-5xl">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold">Sites</h1>
          <p className="text-sm text-gray-500 mt-1">
            Every place your widget runs. Connect a Shopify store today; Wix and custom embeds coming next.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4 text-sm">
            {error}
          </div>
        )}

        {!sites && !error && (
          <div className="flex items-center text-gray-500 text-sm">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading sites…
          </div>
        )}

        {sites && sites.length === 0 && (
          <Card>
            <CardContent className="py-10 text-center text-gray-500">
              No sites yet. Connect a Shopify store via the install flow to provision your first Site.
            </CardContent>
          </Card>
        )}

        {sites && sites.length > 0 && (
          <Card>
            <CardContent className="p-0">
              <ul className="divide-y">
                {sites.map((site) => {
                  const Icon = TYPE_ICON[site.type] ?? Globe;
                  return (
                    <li key={site.id}>
                      <Link
                        href={`/admin/sites/${site.id}`}
                        className="flex items-center justify-between px-5 py-4 hover:bg-gray-50 transition"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <Icon className="w-5 h-5 text-gray-400 shrink-0" />
                          <div className="min-w-0">
                            <p className="font-medium text-sm text-gray-900 truncate">
                              {site.display_name}
                            </p>
                            <p className="text-xs text-gray-500 mt-0.5">
                              {TYPE_LABEL[site.type] ?? site.type}
                              {site.external_id ? ` · ${site.external_id}` : ''}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <StatusBadge status={site.status} />
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        </div>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </MainLayout>
  );
}
