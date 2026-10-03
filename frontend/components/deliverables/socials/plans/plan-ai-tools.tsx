'use client'

/**
 * PRD-251B US-B207 (step 4) with US-B304 — the AI tools a plan's posts are made with: the
 * workspace's default per media type and its caps, as the Brand kit's AI tools set them,
 * with the link there. A plan has no tools of its own.
 */
import Link from 'next/link'

import type { SocialMediaToolsResponse, SocialMediaType } from '@/lib/brand-style-types'
import { BRAND_KIT_HREF } from '@/lib/deliverables/tabs'
import { useSocialMediaTools } from '@/hooks/use-media-tools'
import { MEDIA_TYPE_LABELS } from '../../brand/brand-ai-tools'

export const AI_TOOLS_NOTE = 'Paid AI tools are connected in Composio and chosen in the Brand kit; templates and Kokoro need nothing.'

/** "AI images: fal.ai · Footage: Off · …": each default by the label it is offered under. */
export function defaultsSummary(tools: SocialMediaToolsResponse): string {
  return (Object.keys(MEDIA_TYPE_LABELS) as SocialMediaType[])
    .map((type) => {
      const chosen = tools.defaults[type]
      const label = (tools.offered[type] ?? []).find((choice) => choice.value === chosen)?.label ?? chosen
      return `${MEDIA_TYPE_LABELS[type]}: ${label}`
    })
    .join(' · ')
}

export function PlanAiTools() {
  const tools = useSocialMediaTools()
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border px-3.5 py-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-sm text-muted-foreground">{AI_TOOLS_NOTE}</span>
        {tools.data && (
          <span className="text-[12.5px] text-foreground">
            {defaultsSummary(tools.data)}. Caps: ${tools.data.caps.monthly_usd.toFixed(2)} a month, ${tools.data.caps.per_post_usd.toFixed(2)} a post.
          </span>
        )}
      </div>
      <Link href={BRAND_KIT_HREF as any} className="text-sm font-medium text-foreground underline-offset-4 hover:underline">AI tools</Link>
    </div>
  )
}
