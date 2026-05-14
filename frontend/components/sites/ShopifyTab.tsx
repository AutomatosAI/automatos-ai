'use client';

import { ExternalLink, ShoppingBag, Database } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Site } from '@/lib/sites/types';


export function ShopifyTab({ site }: { site: Site }) {
  if (site.type !== 'shopify' || !site.external_id) {
    return (
      <Card>
        <CardContent className="py-6 text-center text-gray-500 text-sm">
          This Site is not a Shopify store.
        </CardContent>
      </Card>
    );
  }

  const themeEditorUrl = `https://${site.external_id}/admin/themes/current/editor`;
  const adminUrl = `https://${site.external_id}/admin`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <ShoppingBag className="w-4 h-4" /> Shopify store
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Domain</span>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded">
              {site.external_id}
            </code>
          </div>
          <div className="flex items-center gap-2">
            <Button asChild size="sm" variant="outline">
              <a href={themeEditorUrl} target="_blank" rel="noopener noreferrer">
                Open theme editor <ExternalLink className="w-3 h-3 ml-1" />
              </a>
            </Button>
            <Button asChild size="sm" variant="outline">
              <a href={adminUrl} target="_blank" rel="noopener noreferrer">
                Open Shopify admin <ExternalLink className="w-3 h-3 ml-1" />
              </a>
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Database className="w-4 h-4" /> Capabilities
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {Object.entries(site.capabilities).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-gray-600 text-xs">{key}</span>
                <Badge
                  variant="outline"
                  className={
                    value
                      ? 'text-emerald-700 border-emerald-200 bg-emerald-50'
                      : 'text-gray-500 border-gray-200'
                  }
                >
                  {value ? 'yes' : 'no'}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Catalog sync</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500">
            Catalog knowledge sync ships with PRD-009 (parked until PRD-008-A ships).
            For now the agent grounds responses on real-time Composio queries
            against your Shopify catalog.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
