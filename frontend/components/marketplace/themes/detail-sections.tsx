'use client'

/**
 * PRD-261 C-04b — the parts of a listing's detail (C-04a `listing_detail`): previews per preset
 * (PRD §7 item 5), what the theme changes (pages, routes, base, tone, minimum width, `uses`), the
 * versions map with which ones this host runs (PRD §4.2) and its README and CHANGELOG. Every URL
 * is the files route on the app's own origin (`themeFileUrl` drops anything else).
 *
 * C-05k: the facts include the shown version's Signature (listings.py `signature`, C-05b), in the
 * words of theme-labels.ts: "Signed by Automatos" only for `verified`. The Official badge is not
 * this: it comes only from `publisher.official` (listing-badges.tsx ListingBadges, QB-21).
 */
import { SIGNATURE_FACT, signatureWords } from '@/components/settings/themes/theme-labels'
import { Badge } from '@/components/ui/badge'
import type { ThemeListingDetail, ThemePresetPreviews } from '@/lib/api-client'

import { NOT_COMPATIBLE, themeFileUrl } from './listing-format'

const DEFAULT_PRESET = 'default look'

function Preview({ url, label }: { url: string; label: string }) {
  const src = themeFileUrl(url)
  if (!src) return null
  // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={label} className="aspect-video w-full rounded-lg border border-border/40 object-cover" loading="lazy" />
}

function PresetRow({ preset }: { preset: ThemePresetPreviews }) {
  return (
    <div className="space-y-2" data-testid={`theme-preset-${preset.id}`}>
      <p className="text-sm font-medium text-foreground">
        {preset.id}
        {preset.default && <span className="ml-2 text-xs text-muted-foreground">({DEFAULT_PRESET})</span>}
      </p>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {preset.previews.map((preview) => (
          <Preview key={preview.url} url={preview.url} label={`${preset.id} ${preview.screen ?? ''}`.trim()} />
        ))}
      </div>
    </div>
  )
}

export function DetailPreviews({ detail }: { detail: ThemeListingDetail }) {
  const presets = detail.presets.filter((preset) => preset.previews.length > 0)
  if (presets.length === 0 && detail.otherPreviews.length === 0) return null
  return (
    <section aria-label="Previews" className="space-y-4">
      {presets.map((preset) => (
        <PresetRow key={preset.id} preset={preset} />
      ))}
      {detail.otherPreviews.length > 0 && (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {detail.otherPreviews.map((url) => (
            <Preview key={url} url={url} label={detail.name ?? detail.themeId} />
          ))}
        </div>
      )}
    </section>
  )
}

function Fact({ label, value }: { label: string; value: string | null }) {
  if (!value) return null
  return (
    <div className="flex gap-2 text-sm">
      <dt className="w-28 shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words text-foreground">{value}</dd>
    </div>
  )
}

function listOrNull(values: readonly string[]): string | null {
  return values.length > 0 ? values.join(', ') : null
}

export function DetailFacts({ detail }: { detail: ThemeListingDetail }) {
  return (
    <dl className="space-y-1.5" aria-label="What this theme changes">
      <Fact label="Publisher" value={detail.publisher.name} />
      <Fact label="Theme id" value={detail.themeId} />
      <Fact label={SIGNATURE_FACT} value={signatureWords(detail.signature)} />
      <Fact label="Pages" value={listOrNull(detail.pages)} />
      <Fact label="Its own pages" value={listOrNull(detail.routes)} />
      <Fact label="Built on" value={detail.base} />
      <Fact label="Light or dark" value={detail.tone} />
      <Fact label="Minimum width" value={detail.minWidth ? `${detail.minWidth}px` : null} />
      <Fact label="Uses" value={listOrNull(detail.uses)} />
    </dl>
  )
}

export function DetailVersions({ detail }: { detail: ThemeListingDetail }) {
  const versions = Object.entries(detail.versions)
  if (versions.length === 0) return null
  return (
    <section aria-label="Versions" className="space-y-2">
      <p className="text-sm font-medium text-foreground">Versions</p>
      <ul className="space-y-1">
        {versions.map(([version, entry]) => (
          <li key={entry.packageId} className="flex flex-wrap items-center gap-2 text-sm" data-testid={`theme-version-${version}`}>
            <span className="font-mono text-foreground">{version}</span>
            {version === detail.latest && <Badge variant="secondary">Newest for this Automatos</Badge>}
            {!entry.compatible && <Badge variant="outline">{NOT_COMPATIBLE}</Badge>}
            {entry.revoked && <span className="text-xs text-warning">{entry.revoked.reason}</span>}
          </li>
        ))}
      </ul>
    </section>
  )
}

export function DetailDocs({ detail }: { detail: ThemeListingDetail }) {
  const readme = themeFileUrl(detail.readme)
  const changelog = themeFileUrl(detail.changelog)
  if (!readme && !changelog) return null
  return (
    <div className="flex flex-wrap gap-4 text-sm">
      {readme && (
        <a href={readme} target="_blank" rel="noopener noreferrer" className="text-primary underline-offset-4 hover:underline">
          Read me
        </a>
      )}
      {changelog && (
        <a href={changelog} target="_blank" rel="noopener noreferrer" className="text-primary underline-offset-4 hover:underline">
          Changelog
        </a>
      )}
    </div>
  )
}
