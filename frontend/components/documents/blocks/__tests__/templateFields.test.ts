import { describe, it, expect } from 'vitest'
import {
  collectDataFields,
  collectMissingOnFile,
  collectVariablePaths,
  fieldLabel,
  getDataField,
  setDataField,
} from '../templateFields'
import type { Block, VariableEntry } from '../types'

const blocks: Block[] = [
  { type: 'image', id: 'logo', source: 'brand_logo', alt: 'Logo', width_mm: 40 },
  { type: 'heading', id: 'h', level: 1, content: [{ type: 'variable', path: 'data.title' }] },
  {
    type: 'section',
    id: 's',
    title: 'Summary',
    children: [
      { type: 'text', id: 't', content: [{ type: 'text', text: 'By ' }, { type: 'variable', path: 'user.name' }] },
      { type: 'variable', id: 'v', path: 'data.summary' },
    ],
  },
  { type: 'table', id: 'tb', header: true, rows: [[[{ type: 'variable', path: 'data.client.name' }], []]] },
  { type: 'page_break', id: 'pb' },
  { type: 'text', id: 'dup', content: [{ type: 'variable', path: 'data.title' }] },
]

describe('collectVariablePaths / collectDataFields', () => {
  it('walks every block type, de-duplicates and sorts', () => {
    expect(collectVariablePaths(blocks)).toEqual([
      'brand.logo_url',
      'data.client.name',
      'data.summary',
      'data.title',
      'user.name',
    ])
  })
  it('strips the data. prefix for the fill-in fields', () => {
    expect(collectDataFields(blocks)).toEqual(['client.name', 'summary', 'title'])
  })
  it('handles an empty tree', () => {
    expect(collectVariablePaths([])).toEqual([])
    expect(collectDataFields([])).toEqual([])
  })
})

describe('collectMissingOnFile', () => {
  const catalog: VariableEntry[] = [
    { path: 'user.name', category: 'user', label: 'Name', sample: 'Jane', value: 'Gerard', resolved: true },
    { path: 'brand.logo_url', category: 'brand', label: 'Logo', sample: '/l.png', value: null, resolved: false },
    { path: 'company.address', category: 'company', label: 'Address', sample: '1 St', value: '', resolved: false },
  ]
  it('reports known catalog chips with no value, never data.* chips', () => {
    expect(collectMissingOnFile(collectVariablePaths(blocks), catalog)).toEqual(['brand.logo_url'])
    expect(collectMissingOnFile(['company.address', 'data.x', 'not.in.catalog'], catalog)).toEqual(['company.address'])
  })
})

describe('get/setDataField', () => {
  it('sets nested dotted fields immutably', () => {
    const base = { title: 'T', client: { name: 'Old' } }
    const next = setDataField(base, 'client.name', 'Acme')
    expect(next).toEqual({ title: 'T', client: { name: 'Acme' } })
    expect(base.client.name).toBe('Old')
    expect(setDataField({}, 'summary', 'S')).toEqual({ summary: 'S' })
    expect(setDataField({ client: 'not-an-object' }, 'client.name', 'A')).toEqual({ client: { name: 'A' } })
  })
  it('reads nested fields and stringifies non-strings', () => {
    expect(getDataField({ client: { name: 'Acme' } }, 'client.name')).toBe('Acme')
    expect(getDataField({ n: 3 }, 'n')).toBe('3')
    expect(getDataField({}, 'missing.deep')).toBe('')
  })
})

describe('fieldLabel', () => {
  it('humanises dotted / snake / camel names', () => {
    expect(fieldLabel('client.name')).toBe('Client name')
    expect(fieldLabel('recipient_name')).toBe('Recipient name')
    expect(fieldLabel('executiveSummary')).toBe('Executive summary')
  })
})
