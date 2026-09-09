'use client'

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/** PRD-235 W3 — is a Claude Code host online for this workspace? */
export interface CliHostHealth {
  online: boolean
  paired_hosts: number
  online_hosts: Array<{ id: string; name?: string | null; last_seen_at?: string | null; capabilities?: { max_terminals?: number | null } | null }>
  last_seen_at: string | null
  cli_agents: number
  waiting_tickets: number
  host_contract?: string
  expected_host_version?: string
}

export function useCliHostHealth(options?: { enabled?: boolean }) {
  return useQuery<CliHostHealth>({
    queryKey: ['cli-host-health'],
    queryFn: () => apiClient.get('/api/v1/cli-hosts/health'),
    refetchInterval: 30_000,
    staleTime: 15_000,
    refetchOnWindowFocus: true,
    retry: 1,
    enabled: options?.enabled ?? true,
  })
}
