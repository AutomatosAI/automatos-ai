// API helpers for the document-template studio (PRD-167 S3/S4/S5 → PRD-242).
import { apiClient } from '@/lib/api-client'
import type {
  BlockDocument,
  BrandFontFile,
  BrandKit,
  BrandSuggestions,
  GenerateDocumentResult,
  TemplateDetail,
  TemplatePreset,
  TemplateSummary,
  TemplateWriteBody,
  VariablesResponse,
} from './types'

export interface PreviewBlocksResult {
  html: string
  unresolved: string[]
  unknown: string[]
}

export const BRAND_LOGO_PATH = '/api/documents/brand-kit/logo'
// PRD-251 D5: the square logo mark and the font files, stored like the logo.
export const BRAND_LOGO_MARK_PATH = '/api/documents/brand-kit/logo-mark'
export const BRAND_FONTS_PATH = '/api/documents/brand-kit/fonts'

// The face an uploaded font file provides.
export interface BrandFontFace {
  family: string
  weight: number
  style: BrandFontFile['style']
}

export const templateBlocksApi = {
  // Variable catalog with resolved sample values (drives the chip picker).
  getVariables: () => apiClient.get<VariablesResponse>('/api/documents/variables'),

  // Templates
  listTemplates: () => apiClient.get<TemplateSummary[]>('/api/documents/templates'),
  // The layout each category starts from (PRD-243).
  listPresets: () => apiClient.get<TemplatePreset[]>('/api/documents/templates/presets'),
  getTemplate: (id: string) => apiClient.get<TemplateDetail>(`/api/documents/templates/${id}`),
  createTemplate: (body: TemplateWriteBody) =>
    apiClient.post<{ id: string; name: string }>('/api/documents/templates', body),
  updateTemplate: (id: string, body: TemplateWriteBody) =>
    apiClient.put<{ id: string; updated: boolean }>(`/api/documents/templates/${id}`, body),
  deleteTemplate: (id: string) => apiClient.delete<{ deleted: boolean }>(`/api/documents/templates/${id}`),

  // Render a real document from a template — it lands in Deliverables (PRD-242 S4).
  generateDocument: (input: { title: string; format: string; template_id: string; data: Record<string, any> }) =>
    apiClient.post<GenerateDocumentResult>('/api/documents/generate', input),

  // Brand kit (defaults merged in).
  getBrandKit: () => apiClient.get<BrandKit>('/api/documents/brand-kit'),
  updateBrandKit: (patch: Partial<BrandKit>) => apiClient.put<BrandKit>('/api/documents/brand-kit', patch),
  getBrandSuggestions: () =>
    apiClient.get<{ suggestions: BrandSuggestions }>('/api/documents/brand-kit/suggestions'),

  // Logo upload/removal (PRD-242 S3). FormData: the client drops the JSON content-type.
  uploadLogo: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiClient.post<BrandKit & { logo_route: string }>(BRAND_LOGO_PATH, form)
  },
  deleteLogo: () => apiClient.delete<BrandKit>(BRAND_LOGO_PATH),
  uploadLogoMark: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return apiClient.post<BrandKit & { logo_mark_route: string }>(BRAND_LOGO_MARK_PATH, form)
  },
  deleteLogoMark: () => apiClient.delete<BrandKit>(BRAND_LOGO_MARK_PATH),

  // Font files (PRD-251 D5): a woff2 and the face it provides. The same face again replaces it.
  uploadFont: (file: File, face: BrandFontFace) => {
    const form = new FormData()
    form.append('file', file)
    form.append('family', face.family)
    form.append('weight', String(face.weight))
    form.append('style', face.style)
    return apiClient.post<BrandKit>(BRAND_FONTS_PATH, form)
  },
  deleteFont: (id: string) => apiClient.delete<BrandKit>(`${BRAND_FONTS_PATH}/${id}`),

  // A stored brand file (the logo, the mark, a font) needs auth headers a plain
  // <img src> or @font-face cannot send (SaaS), so fetch it and hand back an
  // object URL (the FilePreview pattern). null = none.
  fetchBrandFileObjectUrl: async (path: string): Promise<string | null> => {
    const headers = await apiClient.getAuthHeaders()
    const resp = await fetch(`${apiClient.getBaseUrl()}${path}`, { headers })
    if (!resp.ok) return null
    return URL.createObjectURL(await resp.blob())
  },

  // Live preview: render a block tree to HTML without persisting.
  previewBlocks: (blocks: BlockDocument, data: Record<string, any> = {}) =>
    apiClient.post<PreviewBlocksResult>('/api/documents/preview-blocks', { blocks, data }),
}

// Generated files sit behind an authenticated route; a plain <a href> cannot send the
// Authorization header in SaaS. Fetch with auth and hand the bytes to the browser.
export async function downloadGeneratedFile(downloadUrl: string, filename: string): Promise<void> {
  const headers = await apiClient.getAuthHeaders()
  const resp = await fetch(`${apiClient.getBaseUrl()}${downloadUrl}`, { headers })
  if (!resp.ok) throw new Error(`Download failed (HTTP ${resp.status})`)
  const objectUrl = URL.createObjectURL(await resp.blob())
  const anchor = document.createElement('a')
  anchor.href = objectUrl
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(objectUrl)
}
