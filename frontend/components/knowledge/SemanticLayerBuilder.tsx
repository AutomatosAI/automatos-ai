'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Textarea } from '@/components/ui/textarea';
import { AlertCircle, CheckCircle, Plus, Save, Trash2 } from 'lucide-react';
import { apiClient } from '@/lib/api-client';

/**
 * Business definitions ("semantic layer") editor — PRD-199 S2.
 *
 * Instructions-first: the free-text business-definitions document is the
 * field the SQL generator treats as authoritative (rendered first in the
 * generation prompt), so it is the primary surface. Metric and dimension
 * rows are optional extras beneath it.
 *
 * Replaces the PRD-21 builder that could neither load (GET route didn't
 * exist — every load 405'd to console) nor save (the service crashed on
 * two phantom methods, and the route swallowed that into a silent "saved").
 * Honest state by construction: "Saved" renders only on a confirmed
 * `success` response; failures render the error. Whether this surface
 * grows a richer builder is gated on the S4 ΔS measurement (§8-Q1) — this
 * is the floor, not the ceiling.
 */

interface MetricRow {
  name: string;
  sql: string;
  description: string;
}

interface DimensionRow {
  category: string;
  name: string;
  sql: string;
}

interface SemanticLayerBuilderProps {
  sourceId: string;
  sourceName: string;
  dialect: string;
  className?: string;
}

interface SemanticDocResponse {
  success?: boolean;
  instructions?: string;
  metrics?: Record<string, { sql?: string; description?: string }>;
  dimensions?: Record<string, Record<string, string>>;
}

function metricsToRows(doc: SemanticDocResponse): MetricRow[] {
  return Object.entries(doc.metrics ?? {}).map(([name, m]) => ({
    name,
    sql: m?.sql ?? '',
    description: m?.description ?? '',
  }));
}

function dimensionsToRows(doc: SemanticDocResponse): DimensionRow[] {
  return Object.entries(doc.dimensions ?? {}).flatMap(([category, dims]) =>
    Object.entries(dims ?? {}).map(([name, sql]) => ({ category, name, sql })),
  );
}

export function SemanticLayerBuilder({
  sourceId,
  sourceName,
  dialect,
  className = '',
}: SemanticLayerBuilderProps) {
  const [instructions, setInstructions] = useState('');
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [dimensions, setDimensions] = useState<DimensionRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const doc = await apiClient.get<SemanticDocResponse>(
        `/api/knowledge/sources/database/${sourceId}/semantic`,
      );
      setInstructions(doc.instructions ?? '');
      setMetrics(metricsToRows(doc));
      setDimensions(dimensionsToRows(doc));
    } catch {
      setError('Could not load the stored business definitions.');
    }
    setLoading(false);
  }, [sourceId]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const data = await apiClient.post<SemanticDocResponse>(
        `/api/knowledge/sources/database/${sourceId}/semantic`,
        {
          instructions,
          metrics: metrics
            .filter((m) => m.name.trim() && m.sql.trim())
            .map((m) => ({ name: m.name.trim(), sql: m.sql.trim(), description: m.description })),
          dimensions: dimensions
            .filter((d) => d.category.trim() && d.name.trim() && d.sql.trim())
            .map((d) => ({ category: d.category.trim(), name: d.name.trim(), sql: d.sql.trim() })),
        },
      );
      if (data?.success) {
        setSavedAt(new Date().toLocaleTimeString());
      } else {
        setError('Save failed — the server did not confirm the write.');
      }
    } catch {
      setError('Save failed — the definitions were not stored.');
    }
    setSaving(false);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Business definitions</CardTitle>
              <CardDescription className="mt-1">
                Authoritative guidance the SQL generator follows for {sourceName} ({dialect}) — e.g.
                &quot;active = status NOT IN (&apos;churned&apos;,&apos;deleted&apos;); fiscal year starts in
                February&quot;.
              </CardDescription>
            </div>
            <Button onClick={handleSave} disabled={saving || loading}>
              <Save className="w-4 h-4 mr-2" />
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="w-4 h-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {savedAt && !error && (
            <Alert>
              <CheckCircle className="w-4 h-4" />
              <AlertDescription>Saved at {savedAt}.</AlertDescription>
            </Alert>
          )}
          <div>
            <Label htmlFor="semantic-instructions">Definitions</Label>
            <Textarea
              id="semantic-instructions"
              value={instructions}
              onChange={(e) => setInstructions(e.target.value)}
              placeholder={
                loading
                  ? 'Loading…'
                  : 'Business rules, vocabulary, and definitions the generator must follow…'
              }
              rows={10}
              disabled={loading}
              className="font-mono text-sm mt-1"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Metrics (optional)</CardTitle>
          <CardDescription>
            Named calculations the generator can reuse — e.g. revenue → SUM(orders.total).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {metrics.map((m, i) => (
            <div key={i} className="grid grid-cols-[1fr_2fr_2fr_auto] gap-2 items-center">
              <Input
                value={m.name}
                placeholder="name"
                onChange={(e) =>
                  setMetrics(metrics.map((row, j) => (j === i ? { ...row, name: e.target.value } : row)))
                }
              />
              <Input
                value={m.sql}
                placeholder="SQL expression"
                className="font-mono text-sm"
                onChange={(e) =>
                  setMetrics(metrics.map((row, j) => (j === i ? { ...row, sql: e.target.value } : row)))
                }
              />
              <Input
                value={m.description}
                placeholder="description"
                onChange={(e) =>
                  setMetrics(
                    metrics.map((row, j) => (j === i ? { ...row, description: e.target.value } : row)),
                  )
                }
              />
              <Button
                size="sm"
                variant="ghost"
                aria-label="Remove metric"
                onClick={() => setMetrics(metrics.filter((_, j) => j !== i))}
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          ))}
          <Button
            size="sm"
            variant="outline"
            onClick={() => setMetrics([...metrics, { name: '', sql: '', description: '' }])}
          >
            <Plus className="w-4 h-4 mr-1" /> Add metric
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Dimensions (optional)</CardTitle>
          <CardDescription>
            Common groupings by category — e.g. time.month → date_trunc(&apos;month&apos;, created_at).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {dimensions.map((d, i) => (
            <div key={i} className="grid grid-cols-[1fr_1fr_2fr_auto] gap-2 items-center">
              <Input
                value={d.category}
                placeholder="category"
                onChange={(e) =>
                  setDimensions(
                    dimensions.map((row, j) => (j === i ? { ...row, category: e.target.value } : row)),
                  )
                }
              />
              <Input
                value={d.name}
                placeholder="name"
                onChange={(e) =>
                  setDimensions(dimensions.map((row, j) => (j === i ? { ...row, name: e.target.value } : row)))
                }
              />
              <Input
                value={d.sql}
                placeholder="SQL expression"
                className="font-mono text-sm"
                onChange={(e) =>
                  setDimensions(dimensions.map((row, j) => (j === i ? { ...row, sql: e.target.value } : row)))
                }
              />
              <Button
                size="sm"
                variant="ghost"
                aria-label="Remove dimension"
                onClick={() => setDimensions(dimensions.filter((_, j) => j !== i))}
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          ))}
          <Button
            size="sm"
            variant="outline"
            onClick={() => setDimensions([...dimensions, { category: '', name: '', sql: '' }])}
          >
            <Plus className="w-4 h-4 mr-1" /> Add dimension
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
