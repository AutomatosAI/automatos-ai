/**
 * PRD-251B Wave 2 — the shapes of a Socials plan, its slots and its content bank, as
 * /api/socials/plans answers them (orchestrator/api/socials_plans.py, socials_topics.py),
 * and the music library (GET /api/socials/music). apiClient (lib/api-client.ts) calls the
 * routes; this file only names what they carry.
 */

export type SocialPlanStatus = 'active' | 'paused' | 'ended'
export type SocialLatePolicy = 'skip' | 'next_slot'
export type SocialWeekday = 'mon' | 'tue' | 'wed' | 'thu' | 'fri' | 'sat' | 'sun'
/** PRD-251C (C1): a plan's posts are made on their day, a week at a time, or a month at a time. */
export type SocialPlanRhythm = 'daily' | 'weekly' | 'monthly'

/** One cadence row: a channel group posting one format on some days at one time. */
export interface SocialPlanCadenceRow {
  id: string
  channels: string[]
  format: string
  length_seconds: number | null
  /** null lets Auto pick the template when the post is made. */
  template_id: string | null
  days: SocialWeekday[]
  /** HH:MM in the plan's timezone. */
  time: string
  /** PRD-251C (US-C301): 'story' posts the row's image or video as an Instagram story; absent for the format's own kind. */
  kind?: SocialPlanRowKind | null
  /** PRD-251C (US-C302): the row's own visual, over the plan's mix; absent follows the mix. */
  visual?: SocialPlanRowVisual | null
}

/** PRD-251C (C6): where a row's visuals come from (the plan mix's sources). */
export type SocialPlanVisualSource = 'templates' | 'library' | 'ai_images' | 'ai_footage'

export interface SocialPlanRowVisual {
  source: SocialPlanVisualSource
  /** The Composio toolkit that makes the row's AI media; null: the workspace's default. */
  toolkit?: string | null
}

/** PRD-251C (C6): what a cadence row may post as instead of its format's own kind. */
export type SocialPlanRowKind = 'story'

export interface SocialPlanSources {
  knowledge: boolean
  deliverables: boolean
  website: boolean
  github: boolean
  notes: string
  never_say: string[]
}

export interface SocialPlanMake {
  time: string
  video_days_early: number
  image_days_early: number
  max_per_day: number | null
  /** Shares of 100 among templates, library, ai_images and ai_footage. */
  visual_mix: Record<string, number>
  /** PRD-251C: a plan saved before it has no rhythm, and is daily. */
  rhythm?: SocialPlanRhythm
  /** A weekly plan's batch day, a monthly plan's batch date (1-28). */
  batch_day?: SocialWeekday
  batch_date?: number
  /** HH:MM, the plan's timezone: the evening-before reminder. */
  remind_at?: string
}

export interface SocialPlanResearch {
  enabled: boolean
  day: SocialWeekday
  time: string
  /** PRD-251C: research adds no topic close to a post of the last this-many days (60 unless set). */
  repeat_after_days?: number
  last_run_at?: string | null
  last_run_id?: string | null
}

export interface SocialPlan {
  id: string
  workspace_id: string
  name: string
  kind: 'plan'
  status: SocialPlanStatus
  goal: string | null
  audience: string | null
  starts_on: string | null
  ends_on: string | null
  timezone: string | null
  cadence: SocialPlanCadenceRow[]
  sources: SocialPlanSources
  make: SocialPlanMake
  research: SocialPlanResearch
  late_policy: SocialLatePolicy
  approval_mode: 'per_post' | 'series'
  slot_overrides: Record<string, { skip?: boolean; to?: string }>
  created_by: string
  created_at: string | null
  updated_at: string | null
  /** The content bank's counts. */
  bank: { topics: number; unused: number }
  /** PRD-251C: when a weekly or monthly plan's next batch is made (ISO); null for a daily plan. */
  next_batch_at?: string | null
}

export interface SocialPlansResponse {
  plans: SocialPlan[]
  total: number
}

/** What a create or an update sends: every field a plan's form edits; a new row has no id yet. */
export interface SocialPlanInput {
  name?: string
  goal?: string | null
  audience?: string | null
  starts_on?: string
  ends_on?: string
  timezone?: string
  cadence?: Array<Omit<SocialPlanCadenceRow, 'id'> & { id?: string }>
  sources?: Partial<SocialPlanSources>
  make?: Partial<SocialPlanMake>
  research?: Partial<Pick<SocialPlanResearch, 'enabled' | 'day' | 'time' | 'repeat_after_days'>>
  late_policy?: SocialLatePolicy
  approval_mode?: 'per_post' | 'series'
}

/** The sources "Plan with Auto" lets research read: the person's choice, sent with their words. */
export interface SocialPlanAutoSources {
  knowledge: boolean
  website: boolean
  deliverables: boolean
}

/** What "Plan with Auto" sends (POST /api/socials/plans/draft): nothing is saved by it. */
export interface SocialPlanAutoRequest {
  request: string
  timezone: string
  sources: SocialPlanAutoSources
}

/** A post idea Auto heard, as a content bank topic: it joins the bank when the plan is saved. */
export interface SocialPlanAutoTopic {
  title: string
  angle: string | null
  formats: string[]
}

/** Auto's draft: the plan as the Plan page opens it, the ideas it heard, and each thing in its
 * answer a plan cannot carry (a channel not connected, a day or time it could not use). */
export interface SocialPlanAutoDraft {
  plan: {
    name: string
    goal: string
    audience: string
    starts_on: string
    ends_on: string
    timezone: string
    cadence: Array<Omit<SocialPlanCadenceRow, 'id'>>
    sources: SocialPlanSources
  }
  topics: SocialPlanAutoTopic[]
  warnings: string[]
}

export interface SocialPlanSlot {
  key: string
  row_id: string
  channels: string[]
  format: string
  length_seconds: number | null
  template_id: string | null
  local_date: string
  local_time: string
  /** UTC ISO. */
  at: string
  moved: boolean
  state: 'planned' | 'made'
  post: { id: string; title: string; status: string; at: string | null } | null
  /** The topic pinned to the slot's day, while the slot is not made. */
  topic: { id: string; title: string } | null
}

export interface SocialPlanSlotsResponse {
  plan_id: string
  slots: SocialPlanSlot[]
}

export type SocialFactSourceKind = 'knowledge' | 'deliverable' | 'web' | 'github' | 'note'

export interface SocialTopicFact {
  text: string
  source: { kind: SocialFactSourceKind; ref: string; label: string }
}

export interface SocialTopic {
  id: string
  workspace_id: string
  plan_id: string
  title: string
  angle: string | null
  facts: SocialTopicFact[]
  formats: string[]
  pinned_on: string | null
  used_post_id: string | null
  used_at: string | null
  origin: 'research' | 'person'
  created_by: string
  created_at: string | null
  updated_at: string | null
  /** PRD-251C: a post in the workspace's history close to this topic, and what the bank says of it. */
  repeat?: { post_id: string; note: string } | null
}

export interface SocialTopicsResponse {
  topics: SocialTopic[]
  total: number
  unused: number
  /** Why research cannot run in the workspace (not set up, or its playbook removed); null when it can. */
  research_note?: string | null
  /** PRD-251C: when the plan's research last started (ISO); null before its first run. */
  research_last_run_at?: string | null
}

export interface SocialTopicInput {
  title?: string
  angle?: string | null
  facts?: SocialTopicFact[]
  formats?: string[]
  pinned_on?: string | null
}

/** A track of media-render's music library: a post may pick it over its template's own. */
export interface SocialMusicTrack {
  id: string
  title: string
  artist: string
  style: string
  duration: number | null
  licence: string
  credit_required: boolean
}

export interface SocialMusicResponse {
  tracks: SocialMusicTrack[]
  /** false when the workspace has no renderer to read the library from. */
  available: boolean
}

/** A post's music: null is the template's own track; {track: null} none. */
export type SocialPostMusic = { track: string | null } | null
