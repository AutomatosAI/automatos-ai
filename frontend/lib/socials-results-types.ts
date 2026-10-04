/**
 * PRD-251C Wave 4 — what Socials learns once posts go out: a post's numbers (C7, US-C402), the
 * Posted view (US-C408), Auto's proposals for a plan (US-C404), the plan's health (US-C407) and
 * the owner's voice examples (C8, US-C406). Shapes as the server answers them.
 */

/** Every number a channel may give: only those it gave are present, never a guessed zero. */
export type SocialNumberKey =
  | 'views' | 'reach' | 'likes' | 'comments' | 'shares' | 'saves' | 'replies' | 'reposts' | 'quotes' | 'reactions'

export interface SocialPostNumbers {
  /** Summed across the post's channels, for the numbers they gave. */
  numbers: Partial<Record<SocialNumberKey, number>>
  by_channel: Record<string, Partial<Record<SocialNumberKey, number>>>
  /** Everyone who acted on the post: likes, comments, shares, saves, replies, reposts, quotes, reactions. */
  engagement: number
  /** The latest reading taken: 1 or 7 days after it went out. */
  reading: number
  read_at: string | null
}

export interface SocialPostedReceipt {
  toolkit: string
  post_kind: string
  permalink: string | null
  remote_id: string | null
  published_at: string | null
}

/** GET /api/socials/posted: one post that went out. */
export interface SocialPostedPost {
  id: string
  title: string
  format: string
  plan_id: string | null
  plan_name: string | null
  topic: string | null
  went_out_at: string | null
  receipts: SocialPostedReceipt[]
  /** null until its first reading. */
  numbers: SocialPostNumbers | null
}

export interface SocialPostedResponse {
  posts: SocialPostedPost[]
  total: number
}

export interface SocialPostedFilters {
  planId?: string | null
  channel?: string | null
  format?: string | null
}
