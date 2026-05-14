import { apiClient } from '@/lib/api-client';
import type { Site, SitesListResponse, SiteSettings } from './types';

export async function listSites(): Promise<Site[]> {
  const resp = await apiClient.get<SitesListResponse>('/api/sites');
  return resp?.sites ?? [];
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
