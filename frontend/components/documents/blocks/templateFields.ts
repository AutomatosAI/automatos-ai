// Pure helpers over the block tree (PRD-242 S2): which chips a template uses,
// which data.* fields an agent (or a person) must supply, and an immutable
// nested get/set for dotted field names. Mirrors
// orchestrator/modules/documents/template_summary.py.
import type { Block, Inline, ListField, VariableEntry } from './types'

export const DATA_PREFIX = 'data.'

function inlinePaths(content: Inline[] | undefined): string[] {
  return (content || []).flatMap((run) => (run.type === 'variable' ? [run.path] : []))
}

function blockPaths(block: Block): string[] {
  switch (block.type) {
    case 'heading':
    case 'text':
      return inlinePaths(block.content)
    case 'table':
      return block.rows.flatMap((row) => row.flatMap((cell) => inlinePaths(cell)))
    case 'variable':
    case 'data_table':
      return [block.path]
    case 'image':
      return block.source === 'brand_logo' ? ['brand.logo_url'] : []
    case 'section':
      return block.children.flatMap(blockPaths)
    case 'page_break':
      return []
  }
}

// Sorted, de-duplicated variable paths referenced anywhere in the tree.
export function collectVariablePaths(blocks: Block[]): string[] {
  return Array.from(new Set(blocks.flatMap(blockPaths))).sort()
}

// The data.* names (prefix stripped) the template expects at generation time.
export function collectDataFields(blocks: Block[]): string[] {
  return collectVariablePaths(blocks)
    .filter((p) => p.startsWith(DATA_PREFIX) && p.length > DATA_PREFIX.length)
    .map((p) => p.slice(DATA_PREFIX.length))
}

// The data.* LIST fields (data_table blocks) with their column keys, in document order.
export function collectListFields(blocks: Block[]): ListField[] {
  const out: ListField[] = []
  const seen = new Set<string>()
  const walk = (block: Block) => {
    if (block.type === 'data_table') {
      const field = block.path.startsWith(DATA_PREFIX) ? block.path.slice(DATA_PREFIX.length) : block.path
      if (!seen.has(field)) {
        seen.add(field)
        out.push({ field, columns: block.columns.map((c) => c.key) })
      }
    } else if (block.type === 'section') {
      block.children.forEach(walk)
    }
  }
  blocks.forEach(walk)
  return out
}

// Catalog chips the template uses that have NO value on file for this workspace/user
// (e.g. company.address before the brand kit is filled). data.* is excluded — those are
// supplied per generation, not "missing".
export function collectMissingOnFile(paths: string[], variables: VariableEntry[]): string[] {
  const resolved = new Set(variables.filter((v) => v.resolved && v.value).map((v) => v.path))
  const known = new Set(variables.map((v) => v.path))
  return paths.filter((p) => !p.startsWith(DATA_PREFIX) && known.has(p) && !resolved.has(p))
}

export function getDataField(data: Record<string, any>, field: string): string {
  const value = field.split('.').reduce<any>((cur, part) => (cur && typeof cur === 'object' ? cur[part] : undefined), data)
  if (value === undefined || value === null) return ''
  return typeof value === 'string' ? value : JSON.stringify(value)
}

// Immutable nested set: setDataField({}, 'client.name', 'Acme') → { client: { name: 'Acme' } }
export function setDataField(data: Record<string, any>, field: string, value: string): Record<string, any> {
  const [head, ...rest] = field.split('.')
  if (!head) return data
  if (rest.length === 0) return { ...data, [head]: value }
  const child = data[head] && typeof data[head] === 'object' && !Array.isArray(data[head]) ? data[head] : {}
  return { ...data, [head]: setDataField(child, rest.join('.'), value) }
}

// Label a dotted field for a form: 'client.name' → 'Client name', 'summary' → 'Summary'.
export function fieldLabel(field: string): string {
  const words = field.replace(/[._]+/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2').trim()
  return words.charAt(0).toUpperCase() + words.slice(1).toLowerCase()
}
