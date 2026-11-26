/**
 * PRD-277 US-704 (D-12) — whether this install calls one of Automatos's hosted services, for a component.
 *
 * `useHostedService(service)` answers `true` (on), `false` (off) or `undefined` (the enterprise answer is
 * still loading, or couldn't be read): a caller shows the service only for `true`. On the local and hosted
 * builds the answer is the build's (lib/hosted-services.ts `editionHostedServices`): no query is mounted, so
 * a component there needs no QueryClientProvider and asks nothing. The enterprise build reads
 * `GET /api/hosted-services` once for every component (one cache), since the operator's settings change only
 * with a restart.
 */
import { useQuery } from '@tanstack/react-query'

import { apiClient } from '@/lib/api-client'
import { isEnterprise } from '@/lib/edition-name'
import { editionHostedServices, readHostedServices } from '@/lib/hosted-services'
import type { HostedService, HostedServices } from '@/lib/hosted-services'

export const HOSTED_SERVICES_QUERY_KEY = ['hosted-services'] as const
export const HOSTED_SERVICES_STALE_MS = 5 * 60_000

/** The build's answer (local and hosted). */
function useBuiltHostedService(service: HostedService): boolean | undefined {
  return editionHostedServices()?.[service]
}

/** The enterprise install's answer, read once and shared; unknown while loading or when it can't be read. */
function useAskedHostedService(service: HostedService): boolean | undefined {
  const asked = useQuery<HostedServices>({
    queryKey: HOSTED_SERVICES_QUERY_KEY,
    queryFn: () => readHostedServices(apiClient),
    staleTime: HOSTED_SERVICES_STALE_MS,
    retry: false,
  })
  return asked.data?.[service]
}

/** Whether this install calls `service`: picked once for the build, so a component's hooks never change. */
export const useHostedService: (service: HostedService) => boolean | undefined =
  isEnterprise ? useAskedHostedService : useBuiltHostedService
