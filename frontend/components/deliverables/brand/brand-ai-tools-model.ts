/**
 * PRD-251B F252 — the line under a Brand kit AI tools dropdown that offers one choice: why it
 * is the only one, and what to connect for more. Templates, Ask each time, Off and Kokoro
 * need nothing, so a workspace with no paid tool connected sees one choice per media type.
 *
 * The toolkit rows are the server's (modules/socials/media_tools.py toolkit_rows): the
 * generation toolkits ("Images and footage", each with what it makes, or would make once
 * connected) and the voice toolkits ("Voice").
 */
import type { SocialMediaToolkitRow, SocialMediaToolsResponse, SocialMediaType } from '@/lib/brand-style-types'

const GENERATION_KIND = 'Images and footage'
const VOICE_KIND = 'Voice'

/** What a generation toolkit must make to serve each image or footage media type. */
const MAKES: Record<Exclude<SocialMediaType, 'voice'>, string> = { images: 'image', ai_images: 'image', footage: 'video' }

/** What connecting a toolkit would let the media type do. */
const FOR_MORE: Record<SocialMediaType, string> = {
  images: 'make images with AI',
  ai_images: 'choose an AI image tool',
  footage: 'make AI footage',
  voice: 'choose another voice',
}

/** A toolkit that would add a choice to ``type``: to connect, or connected but unavailable. */
function serves(row: SocialMediaToolkitRow, type: SocialMediaType): boolean {
  if (row.status !== 'connect' && row.status !== 'unavailable') return false
  if (type === 'voice') return row.kind === VOICE_KIND
  // An unavailable toolkit says why instead of what it makes.
  return row.kind === GENERATION_KIND && (row.status === 'unavailable' || (row.makes ?? []).includes(MAKES[type]))
}

/** "fal.ai", "fal.ai or Kie.ai", "fal.ai, Kie.ai or Higgsfield". */
export function orList(labels: ReadonlyArray<string>): string {
  if (labels.length <= 1) return labels.join('')
  return `${labels.slice(0, -1).join(', ')} or ${labels[labels.length - 1]}`
}

/** The line under ``type``'s dropdown, or null when it offers more than one choice. */
export function oneChoiceHint(type: SocialMediaType, tools: Pick<SocialMediaToolsResponse, 'offered' | 'toolkits'>): string | null {
  const offered = tools.offered[type] ?? []
  if (offered.length !== 1) return null
  const only = `Only ${offered[0].label} for now`
  const rows = tools.toolkits.filter((row) => serves(row, type))
  const toConnect = rows.filter((row) => row.status === 'connect').map((row) => row.label)
  if (toConnect.length > 0) return `${only}: connect ${orList(toConnect)} above to ${FOR_MORE[type]}.`
  const unavailable = rows.map((row) => `${row.label} is connected but unavailable (${row.reason ?? 'no reason given'})`)
  if (unavailable.length > 0) return `${only}: ${unavailable.join('; ')}.`
  return `${only}: no AI tool for this is set up on this platform.`
}
