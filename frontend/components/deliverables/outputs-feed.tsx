/**
 * OutputsFeed — Netflix-style top-of-page layout for the Deliverables Outputs tab.
 *
 * Replaces the flat 2,393-card grid as the front door. Heartbeats are demoted
 * to a quiet system strip; everything else is sliced into typed Netflix rows
 * over a Today hero. The original grid lives on as the "See all →" drill-in.
 *
 * Layout:
 *   1. Today hero            (what did the team make today?)
 *   2. 7 type rows           (Slides, Reports, Images, Documents,
 *                             Spreadsheets, Code, Blog Posts)
 *   3. System diagnostics    (heartbeat counter strip)
 */

'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'

import { TodayHero } from './today-hero'
import { TypeRow } from './type-row'
import { SystemStrip } from './system-strip'
import type { DeliverableType } from '@/components/icons/deliverable-icon'

// Order optimised for likely-interesting-first: visual stuff up top,
// then text, ending with code & long-form posts.
const TYPE_ROW_ORDER: ReadonlyArray<{ type: DeliverableType; portrait?: boolean }> = [
  { type: 'slide', portrait: true },
  { type: 'report' },
  { type: 'image' },
  { type: 'document' },
  { type: 'spreadsheet' },
  { type: 'code' },
  { type: 'blog_post' },
]

export interface OutputsFeedProps {
  className?: string
}

export function OutputsFeed({ className }: OutputsFeedProps) {
  const router = useRouter()

  const handleSeeAll = useCallback(
    (type: DeliverableType) => {
      router.push(`/deliverables?tab=outputs&artifact_type=${encodeURIComponent(type)}`)
    },
    [router],
  )

  return (
    <div className={className}>
      <div className="space-y-8">
        <TodayHero />

        <div className="space-y-7">
          {TYPE_ROW_ORDER.map(({ type, portrait }) => (
            <TypeRow
              key={type}
              type={type}
              portrait={portrait}
              onSeeAll={handleSeeAll}
            />
          ))}
        </div>

        <SystemStrip />
      </div>
    </div>
  )
}
