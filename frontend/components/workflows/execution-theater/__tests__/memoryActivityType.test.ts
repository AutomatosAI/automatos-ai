/**
 * memoryActivityType — CodeQL js/identity-replacement #2.
 *
 * The inline chain used to end in .replace('_', '_'), replacing '_' with
 * itself (dead code; a no-op). Extracted and tested directly so the real
 * transform — stripping only the "_memory" suffix — is proven for every
 * MemoryActivity['type'] value the UI switches on.
 */
import { describe, expect, it } from 'vitest'
import { memoryActivityType } from '../enhanced-orchestrator-view'

describe('memoryActivityType', () => {
  it('strips the _memory suffix', () => {
    expect(memoryActivityType('working_memory')).toBe('working')
  })

  it('keeps the underscore in a multi-word kind with no _memory suffix', () => {
    expect(memoryActivityType('short_term')).toBe('short_term')
    expect(memoryActivityType('long_term')).toBe('long_term')
  })

  it('strips _memory from collective_memory too', () => {
    expect(memoryActivityType('collective_memory')).toBe('collective')
  })
})
