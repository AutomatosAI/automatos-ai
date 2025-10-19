/**
 * System Settings API Client
 * ==========================
 * 
 * API client for managing system-wide configuration settings.
 * Replaces .env file with database-backed settings management.
 */

import { apiClient } from '../api-client'

export interface SystemSetting {
  id: number
  category: string
  key: string
  value: string | null
  value_type: string
  description: string | null
  is_sensitive: boolean
  is_required: boolean
  default_value: string | null
  validation_rules: Record<string, any> | null
  created_at: string
  updated_at: string
  created_by: string
}

export interface SystemSettingUpdate {
  value: string
  description?: string
}

export interface SystemSettingCreate {
  category: string
  key: string
  value: string
  value_type?: string
  description?: string
  is_sensitive?: boolean
  is_required?: boolean
  default_value?: string
  validation_rules?: Record<string, any>
}

export interface SystemSettingsByCategory {
  category: string
  settings: SystemSetting[]
  total_count: number
}

export interface SystemSettingsStats {
  total_settings: number
  categories: Record<string, number>
  sensitive_settings: number
  required_settings: number
}

export interface BulkUpdateRequest {
  id: number
  value: string
}

/**
 * List all system settings
 */
export async function listSystemSettings(category?: string): Promise<SystemSetting[]> {
  const params = category ? { category } : {}
  const response = await apiClient.request('/api/system-settings/', {
    method: 'GET',
    params
  })
  return response.data || response
}

/**
 * List all setting categories
 */
export async function listSettingCategories(): Promise<string[]> {
  const response = await apiClient.request('/api/system-settings/categories')
  return response.data || response
}

/**
 * Get settings grouped by category
 */
export async function getSettingsByCategory(): Promise<SystemSettingsByCategory[]> {
  const response = await apiClient.request('/api/system-settings/by-category')
  return response.data || response
}

/**
 * Get system settings statistics
 */
export async function getSettingsStats(): Promise<SystemSettingsStats> {
  const response = await apiClient.request('/api/system-settings/stats')
  return response.data || response
}

/**
 * Get a specific system setting
 */
export async function getSystemSetting(settingId: number): Promise<SystemSetting> {
  const response = await apiClient.request(`/api/system-settings/${settingId}`)
  return response.data || response
}

/**
 * Update a system setting
 */
export async function updateSystemSetting(
  settingId: number,
  data: SystemSettingUpdate
): Promise<SystemSetting> {
  const response = await apiClient.request(`/api/system-settings/${settingId}`, {
    method: 'PUT',
    body: data
  })
  return response.data || response
}

/**
 * Create a new system setting
 */
export async function createSystemSetting(data: SystemSettingCreate): Promise<SystemSetting> {
  const response = await apiClient.request('/api/system-settings/', {
    method: 'POST',
    body: data
  })
  return response.data || response
}

/**
 * Delete a system setting
 */
export async function deleteSystemSetting(settingId: number): Promise<void> {
  await apiClient.request(`/api/system-settings/${settingId}`, {
    method: 'DELETE'
  })
}

/**
 * Bulk update multiple settings
 */
export async function bulkUpdateSettings(updates: BulkUpdateRequest[]): Promise<{ message: string }> {
  const response = await apiClient.request('/api/system-settings/bulk-update', {
    method: 'POST',
    body: updates
  })
  return response.data || response
}

/**
 * Reset settings to default values
 */
export async function resetSettingsToDefaults(category?: string): Promise<{ message: string }> {
  const params = category ? { category } : {}
  const response = await apiClient.request('/api/system-settings/reset-to-defaults', {
    method: 'POST',
    params
  })
  return response.data || response
}

/**
 * Get settings for a specific category
 */
export async function getSettingsForCategory(category: string): Promise<SystemSetting[]> {
  return listSystemSettings(category)
}

/**
 * Update multiple settings in a category
 */
export async function updateCategorySettings(
  category: string,
  updates: Record<string, string>
): Promise<{ message: string }> {
  // First get all settings for the category
  const settings = await getSettingsForCategory(category)
  
  // Create bulk update request
  const bulkUpdates: BulkUpdateRequest[] = []
  
  for (const setting of settings) {
    if (updates[setting.key] !== undefined) {
      bulkUpdates.push({
        id: setting.id,
        value: updates[setting.key]
      })
    }
  }
  
  if (bulkUpdates.length > 0) {
    return bulkUpdateSettings(bulkUpdates)
  }
  
  return { message: 'No updates needed' }
}
