'use client'

/**
 * PRD-130: Business Intake Wizard API hooks
 * ==========================================
 *
 * React Query hooks for the wizard endpoints. All endpoints are blocking
 * (no background queue in Phase 1) so callers should show a spinner while
 * scan/scrape mutations are in flight.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

export interface WizardGoal {
  id: string
  label: string
  description: string
}

export const WIZARD_GOALS: WizardGoal[] = [
  { id: 'manage',     label: 'Manage day-to-day operations', description: 'Orders, inventory, fulfillment, customer service' },
  { id: 'grow',       label: 'Grow revenue',                 description: 'Sales support, brand expansion, partnerships' },
  { id: 'market',     label: 'Run marketing',                description: 'Content, campaigns, voice consistency' },
  { id: 'advise',     label: 'Provide expert advice',        description: 'Technical sales support and consulting' },
  { id: 'social',     label: 'Manage social presence',       description: 'Posts, engagement, community' },
  { id: 'compliance', label: 'Ensure compliance',            description: 'Standards, regulations, certifications' },
]

export interface StartWizardBody {
  domain: string
  goals: string[]
}

export interface StartWizardResponse {
  profile_id: string
  domain: string
  status: string
  domain_verified: boolean
}

export interface ScanResponse {
  profile_id: string
  archetype: string | null
  confidence: number
  matched_signals: string[]
  total_urls: number
  must_have_urls: string[]
  recommended_urls: string[]
  sample_urls: string[]
}

export interface ScrapeResponse {
  profile_id: string
  pages_scraped: number
  pages_failed: number
  documents_ingested: number
  profile: BusinessProfilePayload
}

export interface BusinessProfilePayload {
  domain: string
  archetype: string | null
  company_name: string | null
  sectors: string[] | null
  brands: Array<Record<string, unknown>> | null
  standards: string[] | null
  voice_notes: string | null
  goals: string[] | null
  quality_findings: { errors?: string[]; notes?: string[] } | null
}

export interface PlanResponseAgent {
  slug: string
  name: string
  team: string
  job_title: string
  persona: string
  skills: string[]
  tools: string[]
  llm: string
  rationale: string
  citations: Array<{ id: string; label: string; type?: string; snippet?: string | null }>
}

export interface PlanResponse {
  profile_id: string
  draft_plan: {
    proposed_agents: PlanResponseAgent[]
    org_chart: Array<{ agent: string; reports_to: string | null }>
    integrations_needed: string[]
    open_questions: string[]
    graph_available: boolean
    graph_node_count: number
  }
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export function useStartWizard() {
  return useMutation<StartWizardResponse, Error, StartWizardBody>({
    mutationFn: (body) => apiClient.post('/api/wizard/start', body),
  })
}

export function useScanDomain() {
  return useMutation<ScanResponse, Error, string>({
    mutationFn: (profileId) => apiClient.post(`/api/wizard/scan/${profileId}`),
  })
}

export function useScrapeSelected() {
  return useMutation<ScrapeResponse, Error, { profileId: string; selectedUrls: string[] }>({
    mutationFn: ({ profileId, selectedUrls }) =>
      apiClient.post(`/api/wizard/scrape/${profileId}`, { selected_urls: selectedUrls }),
  })
}

export function usePatchProfile() {
  const qc = useQueryClient()
  return useMutation<
    { profile_id: string; updated_fields: string[]; status: string },
    Error,
    { profileId: string; patch: Partial<BusinessProfilePayload> }
  >({
    mutationFn: ({ profileId, patch }) =>
      apiClient.request(`/api/wizard/profile/${profileId}`, {
        method: 'PATCH',
        body: patch,
      } as RequestInit & { body?: unknown }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['wizard-profile'] })
    },
  })
}

export function useGenerateDraftPlan() {
  return useMutation<PlanResponse, Error, string>({
    mutationFn: (profileId) => apiClient.post(`/api/wizard/plan/${profileId}`),
  })
}
