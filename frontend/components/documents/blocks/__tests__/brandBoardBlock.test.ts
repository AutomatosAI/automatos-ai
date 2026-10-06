// PRD-255 US-009: the Brand Board starter's parts are `brand` blocks, drawn from the kit
// itself, so the Studio asks for no field and lists no chip for them.
import { describe, it, expect } from 'vitest'
import { collectDataFields, collectListFields, collectVariablePaths } from '../templateFields'
import type { Block } from '../types'

const board: Block[] = [
  { type: 'heading', id: 'title', level: 1, content: [{ type: 'text', text: 'Brand board' }] },
  { type: 'brand', id: 'board-logo', part: 'logo' },
  { type: 'brand', id: 'board-colours', part: 'colours' },
  {
    type: 'section',
    id: 'board-row-1',
    title: null,
    children: [
      { type: 'brand', id: 'board-variants', part: 'variants' },
      { type: 'brand', id: 'board-voice', part: 'voice' },
    ],
  },
  { type: 'brand', id: 'board-applications', part: 'applications' },
]

describe('the brand board blocks', () => {
  it('reference no chip and ask for no data field', () => {
    expect(collectVariablePaths(board)).toEqual([])
    expect(collectDataFields(board)).toEqual([])
    expect(collectListFields(board)).toEqual([])
  })
})
