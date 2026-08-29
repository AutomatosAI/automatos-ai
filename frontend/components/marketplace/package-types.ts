/**
 * PRD-230 US-007 — shared types for the marketplace Packages tab + detail popup.
 *
 * A package is a curated bundle of EXISTING marketplace artifacts. These shapes
 * mirror the backend `MarketplacePackage.to_dict()` (read API) and the install
 * manifest returned by `platform_install_package` (`InstallManifest.to_dict()`).
 * Members carry OPTIONAL display metadata (`name`/`description`) alongside the
 * typed ref so the popup can render descriptions; the installer only reads
 * `type`/`ref`.
 */

export type MemberType = 'agent' | 'tool' | 'skill' | 'plugin' | 'playbook' | 'llm'

export interface PackageMember {
  type: MemberType | string
  ref?: string
  id?: string
  slug?: string
  name?: string
  description?: string
}

export interface RequiredConnect {
  app_name: string
  app_type?: string
  needs_oauth?: boolean
  /** Optional guidance shown in the setup summary (e.g. the Shopify two-step). */
  note?: string
}

export interface ReportTemplate {
  name?: string
  title?: string
  description?: string
}

export interface PackageSetupManifest {
  questions?: unknown[]
  required_connects?: Array<RequiredConnect | string>
  guide_steps?: Array<{ title?: string; description?: string } | string>
  report_templates?: Array<ReportTemplate | string>
}

export interface MarketplacePackage {
  id: string
  slug: string
  name: string
  description?: string
  vertical_tags: string[]
  matching: Record<string, unknown>
  members: PackageMember[]
  setup_manifest: PackageSetupManifest
  showcase: boolean
}

export interface InstallRegistration {
  type: string
  ref: string
  name: string
  status: string
  workspace_owned: boolean
}

/** The install response — a success manifest, or an honest non-install payload
 * (over-quota plan conversation / one-package-during-onboarding). */
export interface PackageInstallResult {
  success: boolean
  slug?: string
  message?: string
  registrations?: InstallRegistration[]
  required_connects?: RequiredConnect[]
  warnings?: string[]
  added_count?: number
  // Honest non-install paths (D6 / D9)
  over_quota?: boolean
  onboarding_restricted?: boolean
  plan_recommendation?: string
  package_agents?: number
  max_agents?: number
  error?: string
}

/** Member types in display order, with the label + accent used across the tab. */
export const MEMBER_TYPE_ORDER: MemberType[] = [
  'agent',
  'playbook',
  'skill',
  'tool',
  'plugin',
  'llm',
]

export const MEMBER_TYPE_LABELS: Record<string, { singular: string; plural: string }> = {
  agent: { singular: 'Agent', plural: 'Agents' },
  playbook: { singular: 'Playbook', plural: 'Playbooks' },
  skill: { singular: 'Skill', plural: 'Skills' },
  tool: { singular: 'Tool', plural: 'Tools' },
  plugin: { singular: 'Plugin', plural: 'Plugins' },
  llm: { singular: 'LLM', plural: 'LLMs' },
}

export function memberRef(m: PackageMember): string {
  return String(m.ref ?? m.id ?? m.slug ?? '')
}

export function memberLabel(m: PackageMember): string {
  return m.name || memberRef(m) || m.type
}

/** Group members by type, preserving the display order above (unknown types last). */
export function groupMembersByType(
  members: PackageMember[],
): Array<{ type: string; members: PackageMember[] }> {
  const groups = new Map<string, PackageMember[]>()
  for (const m of members || []) {
    const t = m.type || 'other'
    if (!groups.has(t)) groups.set(t, [])
    groups.get(t)!.push(m)
  }
  const ordered: Array<{ type: string; members: PackageMember[] }> = []
  for (const t of MEMBER_TYPE_ORDER) {
    if (groups.has(t)) {
      ordered.push({ type: t, members: groups.get(t)! })
      groups.delete(t)
    }
  }
  // Any unknown types after the known ones, in insertion order.
  for (const [t, ms] of groups) ordered.push({ type: t, members: ms })
  return ordered
}

export function connectLabel(c: RequiredConnect | string): string {
  return typeof c === 'string' ? c : c.app_name
}

export function reportLabel(r: ReportTemplate | string): string {
  if (typeof r === 'string') return r
  return r.title || r.name || ''
}
