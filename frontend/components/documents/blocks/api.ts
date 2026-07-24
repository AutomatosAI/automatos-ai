// API helpers for the document-template block editor (PRD-167 S3/S4/S5).
import { apiClient } from '@/lib/api-client'
import type { BlockDocument, BrandKit, VariablesResponse } from './types'

export interface PreviewBlocksResult {
  html: string
  unresolved: string[]
  unknown: string[]
}

export const templateBlocksApi = {
  // Variable catalog with resolved sample values (drives the chip picker).
  getVariables: () => apiClient.get<VariablesResponse>('/api/documents/variables'),

  // Brand kit (defaults merged in).
  getBrandKit: () => apiClient.get<BrandKit>('/api/documents/brand-kit'),

  updateBrandKit: (patch: Partial<BrandKit>) =>
    apiClient.put<BrandKit>('/api/documents/brand-kit', patch),

  // Live preview: render a block tree to HTML without persisting.
  previewBlocks: (blocks: BlockDocument, data: Record<string, any> = {}) =>
    apiClient.post<PreviewBlocksResult>('/api/documents/preview-blocks', { blocks, data }),
}
