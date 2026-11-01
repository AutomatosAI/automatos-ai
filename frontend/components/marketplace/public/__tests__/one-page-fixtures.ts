/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the hosted edition's public answers, in their
 * wire shapes, for the signed-out page's tests: one page of each tab's public read, the Featured
 * read, the counts and the themes catalogue. An apiClient double answers only these routes; any
 * other request (a member or workspace route) is refused, so it fails the test that made it.
 */
import { CATALOG_ENDPOINT, PUBLIC_COUNTS_ENDPOINT, PUBLIC_FEATURED_ENDPOINT } from '@/lib/marketplace/public-items'
import { ARCADE_NEW, HUD, catalog, publicAnswer, publicEntry } from '@/lib/marketplace/__tests__/public-fixtures'

export const PLUGIN_ID = '0b6c2f4e-8d1a-4c3b-9e7f-1a2b3c4d5e6f'
export const LLM_KEY = 'anthropic:claude-opus'

export const RESEARCHER = publicEntry({ key: '7', name: 'Research Assistant' })
export const WRITER = publicEntry({ key: '8', name: 'Writer', description: 'Drafts posts.', requiredApps: [] })
export const WEEKLY_REPORT = publicEntry({ key: '12', name: 'Weekly report', category: 'Reporting', icon: null, requiredApps: ['gmail'] })

export const PAGES: Readonly<Record<string, unknown>> = {
  agents: publicAnswer('agents', [RESEARCHER, WRITER]),
  playbooks: publicAnswer('playbooks', [WEEKLY_REPORT]),
  packages: publicAnswer('packages', [
    publicEntry({ key: 'socials', name: 'Socials', members: [{ type: 'agent', name: 'Social Writer' }], requiredApps: ['instagram'] }),
  ]),
  applications: publicAnswer('applications', [
    publicEntry({
      key: 'notion', name: 'Notion', icon: 'https://logos.composio.dev/api/notion', category: 'productivity',
      tags: ['productivity'], publisher: 'Automatos', requiredApps: [], actionCount: 12, triggerCount: 2,
    }),
  ]),
  llms: publicAnswer('llms', [
    publicEntry({
      key: LLM_KEY, name: 'Claude Opus', icon: null, publisher: 'Anthropic', provider: 'anthropic', vendor: 'anthropic',
      contextWindow: 200000, priceTier: 'premium', capabilities: [{ name: 'reasoning', rating: 'excellent' }],
    }),
  ]),
  plugins: publicAnswer('plugins', [
    publicEntry({ key: PLUGIN_ID, name: 'Code review', category: 'Engineering', version: '1.2.0', skillsCount: 3, publisher: 'Ada' }),
  ]),
  skills: publicAnswer('skills', [publicEntry({ key: 'pdf', name: 'PDF', version: '1.0.0', category: 'Documents' })]),
}

export const FEATURED = {
  featured: [
    { ...RESEARCHER, type: 'agents', version: '1.0.0', installCount: 40 },
    { ...WRITER, type: 'agents', version: '2.0.0', installCount: 12 },
  ],
  popular: [{ ...WEEKLY_REPORT, type: 'playbooks', version: '1.0.0', installCount: 61 }],
}

export const COUNTS = { items: 100, installs: 113 }

/** Every request the signed-out page may make: the public reads only. */
export const PUBLIC_REQUEST =
  /^\/api\/(marketplace\/catalog|public\/marketplace\/(featured|counts|[a-z]+\?limit=100&offset=0))$/
const READ = /^\/api\/public\/marketplace\/([a-z]+)\?limit=100&offset=0$/

/** The public routes' answers; anything else is refused. */
export function publicAnswerFor(url: string): Promise<unknown> {
  if (url === CATALOG_ENDPOINT) return Promise.resolve(catalog([ARCADE_NEW, HUD]))
  if (url === PUBLIC_FEATURED_ENDPOINT) return Promise.resolve(FEATURED)
  if (url === PUBLIC_COUNTS_ENDPOINT) return Promise.resolve(COUNTS)
  const page = READ.exec(url)
  const answer = page ? PAGES[page[1]] : undefined
  return answer ? Promise.resolve(answer) : Promise.reject(new Error(`not a public route: ${url}`))
}
