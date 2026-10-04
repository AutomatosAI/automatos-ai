'use client'

/**
 * PRD-251C US-C408 — Posted: what went out, newest first (GET /api/socials/posted). Each post
 * shows its plan and topic, when it went out, each channel's receipt (a link to the post on
 * the platform) and its numbers once they are read. Filtered by plan (the URL's &plan=, so
 * the weekly note links straight to a plan's posts), channel and format.
 */
import { useState } from 'react'

import { useSocialChannels } from '@/hooks/use-socials-composer'
import { useSocialPlans } from '@/hooks/use-socials-plans'
import { useSocialPosted } from '@/hooks/use-socials-results'
import type { SocialPostedPost } from '@/lib/socials-results-types'
import { FORMAT_FILTERS, NOTHING_POSTED, numbersLine, receiptLabel, wentOutLabel } from './posted-model'
import type { GoTo } from './studio-route'

const SELECT = 'h-[38px] rounded-md border border-input bg-background px-2 text-sm'

function PostedRow({ post, onOpen }: { post: SocialPostedPost; onOpen: () => void }) {
  return (
    <li aria-label={post.title} className="flex flex-col gap-1.5 rounded-xl border border-border bg-card p-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <button type="button" className="text-left font-medium text-foreground hover:underline" onClick={onOpen}>{post.title}</button>
        <span className="text-[12.5px] text-muted-foreground">{wentOutLabel(post.went_out_at)}</span>
      </div>
      <p className="m-0 text-[13px] text-muted-foreground">
        {[post.plan_name, post.topic, post.format].filter(Boolean).join(' · ')}
      </p>
      <p className="m-0 text-[13px] text-foreground">{numbersLine(post.numbers)}</p>
      <div className="flex flex-wrap gap-3 text-[13px]" aria-label="Receipts">
        {post.receipts.map((receipt) => (receipt.permalink ? (
          <a key={`${receipt.toolkit}-${receipt.post_kind}`} href={receipt.permalink} target="_blank" rel="noreferrer" className="underline">
            {receiptLabel(receipt)}
          </a>
        ) : (
          <span key={`${receipt.toolkit}-${receipt.post_kind}`} className="text-muted-foreground">{receiptLabel(receipt)}</span>
        )))}
      </div>
    </li>
  )
}

export function SocialsPosted({ planId, go }: { planId: string | null; go: GoTo }) {
  const [channel, setChannel] = useState('')
  const [format, setFormat] = useState('')
  const { data: plans } = useSocialPlans()
  const { data: channels } = useSocialChannels()
  const posted = useSocialPosted({ planId, channel: channel || null, format: format || null })
  const posts = posted.data?.posts ?? []
  return (
    <section aria-label="Posted" className="flex flex-col gap-4">
      <div role="group" aria-label="Filters" className="flex flex-wrap gap-2">
        <select aria-label="Plan" className={SELECT} value={planId ?? ''} onChange={(e) => go({ plan: e.target.value || null })}>
          <option value="">Every plan and post</option>
          {(plans?.plans ?? []).map((plan) => <option key={plan.id} value={plan.id}>{plan.name}</option>)}
        </select>
        <select aria-label="Channel" className={SELECT} value={channel} onChange={(e) => setChannel(e.target.value)}>
          <option value="">Every channel</option>
          {(channels ?? []).map((item) => <option key={item.toolkit} value={item.toolkit}>{item.label}</option>)}
        </select>
        <select aria-label="Format" className={SELECT} value={format} onChange={(e) => setFormat(e.target.value)}>
          {FORMAT_FILTERS.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
        </select>
      </div>
      {posted.isLoading ? (
        <p className="m-0 text-sm text-muted-foreground">Loading what went out…</p>
      ) : posts.length === 0 ? (
        <p className="m-0 text-sm text-muted-foreground">{NOTHING_POSTED}</p>
      ) : (
        <ul aria-label="Posts that went out" className="flex flex-col gap-2.5">
          {posts.map((post) => <PostedRow key={post.id} post={post} onOpen={() => go({ view: 'calendar', post: post.id, plan: null })} />)}
        </ul>
      )}
    </section>
  )
}
