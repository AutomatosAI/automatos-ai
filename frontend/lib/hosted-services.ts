/**
 * PRD-277 US-704 (D-12) — which of Automatos's hosted services this install calls.
 *
 *   themeRevocations → the theme revocation list (the pull, its schedule, its sync status, Check now);
 *   cloudSync        → Sync with Automatos Cloud (the Marketplace's button, Settings › Automatos Cloud);
 *   cloudItems       → whether a Sync also brings agents, Playbooks, packages and plugins (what Settings ›
 *                      Automatos Cloud says a Sync brings).
 *
 * A setting decides only what offers or describes a request to Automatos. What the install already holds is
 * shown whatever the settings say: the Cloud items' badges, the review before an install and the update
 * notices follow `holdsCloudItems()` (every edition but the hosted one), so an item brought while Cloud items
 * were on is still reviewed after they are switched off.
 *
 * The local and hosted builds know the answers (`editionHostedServices`): local calls each one, the hosted
 * edition is Automatos Cloud and calls none, and nothing is asked. In the enterprise build each one is the
 * operator's own setting, so the web app reads `GET /api/hosted-services` (`readHostedServices`;
 * hooks/use-hosted-services.ts in components). The server enforces every service's setting itself.
 *
 * It imports nothing from lib/api-client.ts: the install gate (lib/marketplace/cloud-install-gate.ts), which
 * api-client imports, reads `holdsCloudItems` from here.
 */
import { isLocal } from '@/lib/auth-edition'
import { isEnterprise } from '@/lib/edition-name'

export type HostedService = 'themeRevocations' | 'cloudSync' | 'cloudItems'
export type HostedServices = Readonly<Record<HostedService, boolean>>

export const HOSTED_SERVICES_ENDPOINT = '/api/hosted-services'
export const HOSTED_SERVICE_NAMES: readonly HostedService[] = ['themeRevocations', 'cloudSync', 'cloudItems']

const ALL_ON: HostedServices = { themeRevocations: true, cloudSync: true, cloudItems: true }
const ALL_OFF: HostedServices = { themeRevocations: false, cloudSync: false, cloudItems: false }

/**
 * Whether this install may hold items brought from Automatos Cloud: every edition but the hosted one, which is
 * Automatos Cloud. Read at each call, as `isLocal` was, so a test's edition can change between renders.
 */
export function holdsCloudItems(): boolean {
  return isLocal || isEnterprise
}

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
