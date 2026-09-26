// Canonical block schema (PRD-167 S2) — the TS mirror of the backend Pydantic models
// in orchestrator/modules/documents/blocks/schema.py. This is the storage + render
// contract; the editor is a surface over it.

export type Mark = 'bold' | 'italic' | 'underline' | 'strike' | 'code'

export interface TextRun {
  type: 'text'
  text: string
  marks?: Mark[]
}

export interface VariableRun {
  type: 'variable'
  path: string
  fallback?: string | null
}

export type Inline = TextRun | VariableRun

export interface HeadingBlock {
  type: 'heading'
  id: string
  level: number // 1..6
  content: Inline[]
}

export interface TextBlock {
  type: 'text'
  id: string
  content: Inline[]
}

export interface TableBlock {
  type: 'table'
  id: string
  header: boolean
  rows: Inline[][][] // rows -> cells -> inline content
}

export interface ImageBlock {
  type: 'image'
  id: string
  source: 'url' | 'upload' | 'brand_logo'
  src?: string | null
  alt: string
  width_mm?: number | null
}

export interface VariableBlock {
  type: 'variable'
  id: string
  path: string
  fallback?: string | null
}

// PRD-243: a table whose rows come from a data.* list supplied at generation time.
export interface DataTableColumn {
  key: string
  label: string
  align?: 'left' | 'right' | 'center'
}

export interface DataTableBlock {
  type: 'data_table'
  id: string
  path: string // must be data.<field>
  columns: DataTableColumn[]
  empty_text?: string | null
}

export interface PageBreakBlock {
  type: 'page_break'
  id: string
}

export interface SectionBlock {
  type: 'section'
  id: string
  title?: string | null
  children: Block[]
}

export type Block =
  | HeadingBlock
  | TextBlock
  | TableBlock
  | ImageBlock
  | VariableBlock
  | DataTableBlock
  | PageBreakBlock
  | SectionBlock

export interface BlockDocument {
  version: number
  blocks: Block[]
}

export const SCHEMA_VERSION = 1

export type BlockType = Block['type']

// Variable catalog entry (from GET /api/documents/variables)
export interface VariableEntry {
  path: string
  category: string
  label: string
  sample: string
  value?: string | null
  resolved?: boolean
}

export interface VariablesResponse {
  variables: VariableEntry[]
  by_category: Record<string, VariableEntry[]>
}

export interface BrandKit {
  name: string
  tagline: string
  logo_url: string
  // PRD-242 S3: storage path of an UPLOADED logo (server-managed; set by the
  // upload route). When present the renderers inline it; the UI streams it
  // from /api/documents/brand-kit/logo.
  logo_path: string
  primary_color: string
  secondary_color: string
  accent_color: string
  text_color: string
  // The body font (PRD-251 D5's body_font): every renderer reads this key.
  font_family: string
  company: {
    name: string
    address: string
    email: string
    phone: string
    website: string
  }
  // PRD-251 D5 (S1.3), what a social render reads.
  // The headings' font; empty means the body font.
  heading_font: string
  // Uploaded woff2 files (server-managed: the font routes write them).
  font_files: BrandFontFile[]
  // A square mark, separate from the wordmark: a public URL, or an upload at
  // logo_mark_path (server-managed, streamed from /api/documents/brand-kit/logo-mark).
  logo_mark_url: string
  logo_mark_path: string
  // The brand's handle per Composio toolkit (linkedin, twitter, ...), without the "@".
  social_handles: Record<string, string>
  voice: BrandVoice
}

// One uploaded font file and the face it provides (modules/documents/brand_kit.py BrandFontFile).
export interface BrandFontFile {
  id: string
  family: string
  weight: number
  style: 'normal' | 'italic'
  path: string
  file_name: string
  bytes: number
}

// How the brand sounds: three to five tone words (or none) and the phrases it never uses.
export interface BrandVoice {
  tone: string[]
  banned_phrases: string[]
}

// GET /api/documents/brand-kit/suggestions — prefill candidates with provenance.
export type BrandSuggestionSource = 'business_profile' | 'workspace' | 'user'
export interface BrandSuggestion {
  value: string
  source: BrandSuggestionSource
}
export type BrandSuggestions = Partial<
  Record<'name' | 'company_name' | 'website' | 'logo_url' | 'tagline' | 'email', BrandSuggestion>
>

export type TemplateFormat = 'pdf' | 'docx' | 'xlsx'

export interface ListField {
  field: string
  columns: string[]
}

// GET /api/documents/templates/presets — the layout a category starts from (PRD-243).
export interface TemplatePreset {
  category: string
  name: string
  description: string
  format: TemplateFormat | string
  includes: string[]
  variable_paths: string[]
  data_fields: string[]
  list_fields: ListField[]
  blocks: BlockDocument
  sample_data: Record<string, any>
}

// GET /api/documents/templates entry (PRD-242 S2 — modules/documents/template_summary.py)
export interface TemplateSummary {
  id: string
  name: string
  description?: string | null
  format: TemplateFormat | string
  category: string
  tags: string[]
  version: number
  sample_data?: Record<string, any> | null
  data_schema?: Record<string, any> | null
  // Block-editable (vs a legacy Jinja/uploaded-DOCX template that can only be copied or rendered).
  has_blocks: boolean
  // Seeded by the platform — copy-on-customise.
  is_starter: boolean
  // Every variable chip the template references, and the data.* names an agent must supply.
  variable_paths: string[]
  data_fields: string[]
  // The data.* LIST fields (data_table blocks) with their column keys (PRD-243).
  list_fields: ListField[]
  created_at?: string | null
  updated_at?: string | null
}

export interface TemplateDetail extends TemplateSummary {
  blocks: BlockDocument | null
  template_content?: string | null
  template_file_path?: string | null
}

export interface TemplateWriteBody {
  name: string
  description: string
  category: string
  format: string
  blocks: BlockDocument
  sample_data?: Record<string, any>
}

// POST /api/documents/generate
export interface GenerateDocumentResult {
  status: string
  filename: string
  format: string
  download_url: string
  size_kb: number
  deliverable_id?: string | null
  app_url?: string | null
  share_url?: string | null
  template_id?: string | null
  template_name?: string | null
}

// 422 from the finalisation gate (P2-09 S3): which chips did not resolve.
export interface UnresolvedDetail {
  message: string
  unresolved?: string[]
  unknown?: string[]
}
