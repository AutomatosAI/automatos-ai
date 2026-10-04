/**
 * PRD-251C (C6, US-C302) — a cadence row's own visual (pure): the visuals a row's format may
 * set over the plan's mix, the toolkits that make its AI media now (the Brand kit's AI tools,
 * GET /api/socials/media-tools), and the row's estimated AI spend over a month.
 *
 * The estimate counts the row's posts in 30 days, the AI shots each asks for (its template's
 * image or footage slots; one when Auto picks the template) and the share of them that is AI
 * (the row's own visual, else the plan's mix), priced as a render books a shot whose toolkit
 * prices nothing ahead (`shot_usd`). The render still prices, caps and books every shot (D13).
 */
import type { SocialTemplateSummary } from '@/lib/api-client'
import type { SocialMediaChoice, SocialMediaToolsResponse } from '@/lib/brand-style-types'
import type { SocialPlanRowVisual, SocialPlanVisualSource } from '@/lib/socials-plan-types'
import type { DraftRow } from './plan-model'

/** The select's value for a row that follows the plan's mix. */
export const PLAN_MIX = 'mix'
export const VISUAL_LABELS: Record<SocialPlanVisualSource, string> = {
  templates: "Template's own",
  library: 'From the Library',
  ai_images: 'AI images',
  ai_footage: 'AI footage',
}
const AI_SOURCES: ReadonlyArray<SocialPlanVisualSource> = ['ai_images', 'ai_footage']
const DAYS_PER_MONTH = 30
const DAYS_PER_WEEK = 7
const PERCENT = 100

export function isAi(source: SocialPlanVisualSource): boolean {
  return AI_SOURCES.includes(source)
}

/** The visuals a row of `format` may set: none for text, AI footage only on a video row. */
export function visualChoices(format: string): SocialPlanVisualSource[] {
  if (format === 'text') return []
  const choices: SocialPlanVisualSource[] = ['templates', 'library', 'ai_images']
  return format === 'video' ? [...choices, 'ai_footage'] : choices
}

/** The row's visual if `format` still takes it; null (the plan's mix) otherwise. */
export function keptVisual(format: string, visual: SocialPlanRowVisual | null | undefined): SocialPlanRowVisual | null {
  return visual && visualChoices(format).includes(visual.source) ? visual : null
}

/** The toolkits that make the source's AI media in the workspace now: its stills or its footage. */
export function toolkitChoices(source: SocialPlanVisualSource, tools: SocialMediaToolsResponse | undefined): SocialMediaChoice[] {
  const type = source === 'ai_images' ? 'ai_images' : source === 'ai_footage' ? 'footage' : null
  if (!type || !tools) return []
  return (tools.offered[type] ?? []).filter((choice) => choice.value !== 'ask' && choice.value !== 'off')
}

interface PerKind { image: number; video: number }

/** The AI shots a post of the row asks for, per kind: its template's slots, else one. */
function shotsPerPost(row: Pick<DraftRow, 'format' | 'templateId'>, templates: ReadonlyArray<SocialTemplateSummary>): PerKind {
  const template = templates.find((t) => t.id === row.templateId)
  if (!template) return { image: 1, video: row.format === 'video' ? 1 : 0 }
  const images = (template.image_slots ?? []).length
  return { image: images, video: Math.max(0, template.footage_slots.length - images) }
}

/** The share of the row's posts whose shots are AI-made, per kind. */
function aiShare(row: Pick<DraftRow, 'format' | 'visual'>, mix: Record<string, number>): PerKind {
  if (row.format === 'text') return { image: 0, video: 0 }
  if (row.visual) return { image: row.visual.source === 'ai_images' ? 1 : 0, video: row.visual.source === 'ai_footage' ? 1 : 0 }
  return { image: (mix.ai_images ?? 0) / PERCENT, video: row.format === 'video' ? (mix.ai_footage ?? 0) / PERCENT : 0 }
}

export interface RowSpend {
  shots: number
  usd: number
}

/** The row's AI shots over 30 days and what they are booked at. */
export function rowSpend(
  row: Pick<DraftRow, 'format' | 'templateId' | 'visual' | 'days'>,
  mix: Record<string, number>,
  rates: { image: number; video: number },
  templates: ReadonlyArray<SocialTemplateSummary>,
): RowSpend {
  const posts = (row.days.length * DAYS_PER_MONTH) / DAYS_PER_WEEK
  const perPost = shotsPerPost(row, templates)
  const share = aiShare(row, mix)
  const images = posts * share.image * perPost.image
  const videos = posts * share.video * perPost.video
  return { shots: Math.round(images + videos), usd: images * rates.image + videos * rates.video }
}

/** "AI media: about 13 shots a month (about $3.90)."; null for a row that makes none. */
export function spendLine(spend: RowSpend): string | null {
  if (!spend.shots) return null
  return `AI media: about ${spend.shots} shot${spend.shots === 1 ? '' : 's'} a month (about $${spend.usd.toFixed(2)}).`
}
