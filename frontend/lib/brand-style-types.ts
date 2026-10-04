/**
 * PRD-251B Wave 3 — the shapes of the brand kit's style references and style profile
 * (/api/documents/brand-kit/references, orchestrator/api/document_brand_references.py),
 * the AI tools section (/api/socials/media-tools, orchestrator/api/socials_media_tools.py)
 * and a slot's AI options. apiClient (lib/api-client.ts) calls the routes; this file only
 * names what they carry.
 */

export type BrandReferenceStance = 'like' | 'avoid'

/** One style reference: its image is served by `url` (auth headers needed), never by its storage path. */
export interface BrandReference {
  id: string
  note: string
  stance: BrandReferenceStance
  content_type: string | null
  bytes: number | null
  created_at: string | null
  url: string
}

/** What Auto read from the references (US-B303); `null` until it has read any. */
export interface BrandStyleProfile {
  palette: string[]
  mood: string[]
  composition: string
  avoid: string
  read_at?: string
  /** The references the profile was read from: when they differ, a new read is on its way. */
  reference_ids?: string[]
}

export interface BrandStyleResponse {
  references: BrandReference[]
  profile: BrandStyleProfile | null
  /** Whether liked images go to an AI tool whose action takes a reference image. */
  send_liked: boolean
  limits: { references: number; bytes: number }
}

export interface BrandReferenceChange {
  note?: string
  stance?: BrandReferenceStance
}

/** The media types an AI tools default is set for (US-B304). */
export type SocialMediaType = 'images' | 'ai_images' | 'footage' | 'voice'

/** A toolkit row of the AI tools section: connected, to connect in Composio, unavailable, or built in. */
export interface SocialMediaToolkitRow {
  toolkit: string
  label: string
  kind: string
  status: 'available' | 'connect' | 'unavailable' | 'builtin'
  reason?: string
  makes?: string[]
}

export interface SocialMediaChoice {
  value: string
  label: string
}

export interface SocialMediaToolsResponse {
  toolkits: SocialMediaToolkitRow[]
  offered: Record<SocialMediaType, SocialMediaChoice[]>
  defaults: Record<SocialMediaType, string>
  caps: { monthly_usd: number; per_post_usd: number; problem: string | null }
  spend: { month_usd: number; period_end: string }
  /** PRD-251C (US-C302): what a still or a clip is booked at when its toolkit prices nothing ahead. */
  shot_usd?: { image: number; video: number }
}

/** PUT /api/socials/media-tools: a cap of null goes back to the platform's default. */
export interface SocialMediaToolsInput {
  defaults?: Partial<Record<SocialMediaType, string>>
  monthly_cap_usd?: number | null
  per_post_cap_usd?: number | null
}

/** One AI option made for an image slot (US-B305): a file of the post, picked to become the slot's. */
export interface SocialAiOption {
  name: string
  prompt: string
  toolkit: string
  model: string
  deliverable_id?: string | null
  content_type: string
  bytes: number
  estimate_usd: number
  generated_at: string
}

export type SocialAiOptionsState = 'making' | 'ready' | 'failed'
