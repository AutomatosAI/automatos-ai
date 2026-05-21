import { apiClient } from '@/lib/api-client';
import type { Site, SitesListResponse, SiteSettings } from './types';
import { CALLBACK_PLATFORMS, type ChannelConnection } from '@/lib/channels/types';

export async function listSites(): Promise<Site[]> {
  const resp = await apiClient.get<SitesListResponse>('/api/sites');
  return resp?.sites ?? [];
}

/**
 * Channels eligible as callback destinations — filtered to the
 * platforms a sales/support team actually monitors.
 */
export async function listCallbackChannels(): Promise<ChannelConnection[]> {
  const resp = await apiClient.get<ChannelConnection[]>('/api/channels');
  const all = Array.isArray(resp) ? resp : [];
  const allowed = new Set<string>(CALLBACK_PLATFORMS);
  return all
    .filter((c) => allowed.has(c.platform))
    .sort((a, b) => {
      const at = a.last_activity_at ? Date.parse(a.last_activity_at) : 0;
      const bt = b.last_activity_at ? Date.parse(b.last_activity_at) : 0;
      return bt - at;
    });
}

export async function getSite(siteId: string): Promise<Site> {
  return apiClient.get<Site>(`/api/sites/${siteId}`);
}

export async function updateSiteMeta(
  siteId: string,
  body: { display_name?: string; status?: string },
): Promise<Site> {
  return apiClient.patch<Site>(`/api/sites/${siteId}`, body);
}

export async function updateSiteSettings(
  siteId: string,
  settingsPatch: Partial<SiteSettings>,
): Promise<Site> {
  return apiClient.patch<Site>(`/api/sites/${siteId}/settings`, {
    settings: settingsPatch,
  });
}

/**
 * Result for a single destination from `sendCallbackTest`.
 * Mirrors the orchestrator's `CallbackTestDestinationResult` schema.
 */
export interface CallbackTestDestinationResult {
  destination_type: string;
  success: boolean;
  latency_ms: number;
  error?: string | null;
  platform?: string | null;
  target?: string | null;
}

export interface CallbackTestResponse {
  request_id: string;
  destinations_attempted: number;
  results: CallbackTestDestinationResult[];
}

/**
 * Fire a synthetic callback through the site's configured destinations.
 * Used by the dashboard's "Send test" button so merchants can prove their
 * Telegram / Slack / WhatsApp wiring works without needing a real shopper
 * submission.
 */
export async function sendCallbackTest(siteId: string): Promise<CallbackTestResponse> {
  return apiClient.post<CallbackTestResponse>(
    `/api/sites/${siteId}/callback/test`,
    {},
  );
}
