/**
 * Credentials API Client
 * ======================
 * 
 * Client functions for credential management API
 */

import { apiClient } from '../api-client'

export interface CredentialType {
  id: number
  name: string
  display_name: string
  category: string | null
  icon: string | null
  logo?: string | null  // Logo file path
  description: string | null
  schema_definition: CredentialFieldDefinition[]
  test_endpoint: any | null
  documentation_url: string | null
  is_system: boolean
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface CredentialFieldDefinition {
  displayName: string
  name: string
  type: 'string' | 'password' | 'number' | 'boolean' | 'options' | 'hidden'
  required?: boolean
  default?: any
  description?: string
  placeholder?: string
  typeOptions?: Record<string, any>
  displayOptions?: Record<string, any>
}

export interface Credential {
  id: number
  name: string
  credential_type_id: number
  credential_type_name: string
  credential_type_display_name: string
  environment: string
  description: string | null
  tags: string[]
  is_active: boolean
  expires_at: string | null
  last_tested: string | null
  test_status: string | null
  test_message: string | null
  created_by: string | null
  created_at: string
  updated_at: string
  has_credentials: boolean
  field_names: string[]
}

export interface CredentialCreateRequest {
  name: string
  credential_type_id: number
  credential_data: Record<string, any>
  environment?: string
  description?: string
  tags?: string[]
  expires_at?: string
}

export interface CredentialUpdateRequest {
  name?: string
  credential_data?: Record<string, any>
  description?: string
  tags?: string[]
  is_active?: boolean
  expires_at?: string
}

export interface CredentialTestResponse {
  success: boolean
  message: string
  details?: Record<string, any>
  tested_at: string
}

export interface CredentialAuditLog {
  id: number
  credential_id: string
  tool_id: string | null
  action: string
  user_id: string | null
  ip_address: string | null
  success: boolean | null
  error_message: string | null
  metadata: Record<string, any> | null
  timestamp: string
  created_at?: string // Backward compatibility
}

export interface CredentialStats {
  credential_types: {
    total: number
    active: number
  }
  credentials: {
    total: number
    active: number
    by_environment: Record<string, number>
    by_type: Record<string, number>
  }
  timestamp: string
}

/**
 * List all credential types
 */
export async function listCredentialTypes(params?: {
  category?: string
  active_only?: boolean
}): Promise<CredentialType[]> {
  const response = await apiClient.request('/api/credentials/types', {
    method: 'GET',
    query: params
  })
  return response.data || response
}

/**
 * Get a specific credential type by ID
 */
export async function getCredentialType(typeId: number): Promise<CredentialType> {
  const response = await apiClient.request(`/api/credentials/types/${typeId}`)
  return response.data || response
}

/**
 * Get credential type by name
 */
export async function getCredentialTypeByName(typeName: string): Promise<CredentialType> {
  const response = await apiClient.request(`/api/credentials/types/by-name/${typeName}`)
  return response.data || response
}

/**
 * Get all credential categories
 */
export async function getCredentialCategories(): Promise<string[]> {
  const response = await apiClient.request('/api/credentials/types/categories')
  return response.data || response
}

/**
 * Create a new credential
 */
export async function createCredential(data: CredentialCreateRequest): Promise<Credential> {
  const response = await apiClient.request('/api/credentials/', {
    method: 'POST',
    body: data
  })
  return response.data || response
}

/**
 * List all credentials (values NOT included)
 */
export async function listCredentials(params?: {
  credential_type_id?: number
  environment?: string
  active_only?: boolean
  tags?: string
  skip?: number
  limit?: number
  search?: string
}): Promise<Credential[] | { items: Credential[], total: number }> {
  // Build query params into URL (apiClient doesn't handle 'query' option)
  const queryParams = new URLSearchParams()
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        queryParams.append(key, String(value))
      }
    })
  }
  const url = queryParams.toString() ? `/api/credentials/?${queryParams}` : '/api/credentials/'
  const response = await apiClient.request(url, {
    method: 'GET'
  })
  return response.data || response
}

/**
 * Get a specific credential
 */
export async function getCredential(credentialId: number): Promise<Credential> {
  const response = await apiClient.request(`/api/credentials/${credentialId}`)
  return response.data || response
}

/**
 * Update a credential
 */
export async function updateCredential(
  credentialId: number,
  data: CredentialUpdateRequest
): Promise<Credential> {
  const response = await apiClient.request(`/api/credentials/${credentialId}`, {
    method: 'PUT',
    body: data
  })
  return response.data || response
}

/**
 * Delete a credential
 */
export async function deleteCredential(credentialId: number): Promise<void> {
  await apiClient.request(`/api/credentials/${credentialId}`, {
    method: 'DELETE'
  })
}

/**
 * Test a credential connection
 */
export async function testCredential(credentialId: number): Promise<CredentialTestResponse> {
  const response = await apiClient.request(`/api/credentials/${credentialId}/test`, {
    method: 'POST'
  })
  return response.data || response
}

/**
 * Get audit logs
 */
export async function getCredentialAuditLogs(params?: {
  credential_id?: number
  action?: string
  user_id?: string
  limit?: number
}): Promise<CredentialAuditLog[]> {
  const response = await apiClient.request('/api/credentials/audit/logs', {
    method: 'GET',
    query: params
  })
  return response.data || response
}

/**
 * Get credential statistics
 */
export async function getCredentialStats(): Promise<CredentialStats> {
  const response = await apiClient.request('/api/credentials/stats')
  return response.data || response
}

/**
 * Clear credential cache
 */
export async function clearCredentialCache(credentialName?: string): Promise<void> {
  await apiClient.request('/api/credentials/cache/clear', {
    method: 'POST',
    query: credentialName ? { credential_name: credentialName } : undefined
  })
}

