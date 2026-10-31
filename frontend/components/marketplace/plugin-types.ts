/**
 * The Marketplace plugins tab's shapes (marketplace-plugins-tab.tsx): a plugin as the plugins list
 * answers it, and a plugin category. Moved out of the tab so it doesn't grow (PRD-270 W2 K3).
 */

export interface PluginSummary {
  id: string
  slug: string
  name: string
  version: string
  description: string | null
  category_id: string | null
  category_name: string | null
  tags: string[]
  skills_count: number
  commands_count: number
  agents_count: number
  hooks_count: number
  token_estimate: number
  enable_count: number
  is_featured: boolean
  security_status: string | null
  approval_status: string | null
  original_author: string | null
  license: string | null
  created_at: string | null
}

export interface PluginCategory {
  id: string
  slug: string
  name: string
  description: string | null
  icon: string | null
  sort_order: number
  parent_id: string | null
}
