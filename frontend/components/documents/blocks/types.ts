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
  primary_color: string
  secondary_color: string
  accent_color: string
  text_color: string
  font_family: string
  company: {
    name: string
    address: string
    email: string
    phone: string
    website: string
  }
}
