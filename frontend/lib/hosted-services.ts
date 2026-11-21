/**
 * PRD-277 US-704 (D-12) — which of Automatos's hosted services this install calls.
 *
 *   themeRevocations → the theme revocation list (the pull, its schedule, its sync status, Check now);
 *   cloudSync        → Sync with Automatos Cloud (the Marketplace's button, Settings › Automatos Cloud);
 *   cloudItems       → the agents, Playbooks, packages and plugins brought from Automatos Cloud (their
 *                      badges, the disclosure before an install, the update notices).
 *
 * The local and hosted builds know the answers (`editionHostedServices`): local calls each one, the hosted
 * edition is Automatos Cloud and calls none, and nothing is asked. In the enterprise build each one is the
 * operator's own setting, so the web app reads `GET /api/hosted-services` (`readHostedServices`;
 * hooks/use-hosted-services.ts in components). The server enforces every service's setting itself.
 *
 * It imports nothing from lib/api-client.ts: the install gate (lib/marketplace/cloud-install-gate.ts), which
 * api-client imports, reads it through the client it is handed.
 */
import { isLocal } from '@/lib/auth-edition'
import { isEnterprise } from '@/lib/edition-name'

export type HostedService = 'themeRevocations' | 'cloudSync' | 'cloudItems'
export type HostedServices = Readonly<Record<HostedService, boolean>>

export const HOSTED_SERVICES_ENDPOINT = '/api/hosted-services'
export const HOSTED_SERVICE_NAMES: readonly HostedService[] = ['themeRevocations', 'cloudSync', 'cloudItems']

const ALL_ON: HostedServices = { themeRevocations: true, cloudSync: true, cloudItems: true }
const ALL_OFF: HostedServices = { themeRevocations: false, cloudSync: false, cloudItems: false }

/** What the build knows: every service on locally, none on the hosted edition; null in enterprise (ask). */
export function editionHostedServices(): HostedServices | null {
  if (isEnterprise) return null
  return isLocal ? ALL_ON : ALL_OFF
}

/** The answer of `GET /api/hosted-services`: a service is on only when the answer says `true` for it. */
export function hostedServicesOf(raw: unknown): HostedServices {
  const answer = (raw !== null && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  return {
    themeRevocations: answer.themeRevocations === true,
    cloudSync: answer.cloudSync === true,
    cloudItems: answer.cloudItems === true,
  }
}

/** Reads the enterprise install's answers through `client` (apiClient, or the client the gate is handed). */
export async function readHostedServices(client: { get<T = unknown>(endpoint: string): Promise<T> }): Promise<HostedServices> {
  return hostedServicesOf(await client.get<unknown>(HOSTED_SERVICES_ENDPOINT))
}
