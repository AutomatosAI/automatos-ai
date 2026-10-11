import { describe, expect, it } from 'vitest'
import { getStatusVariant } from '../design-utils'

describe('getStatusVariant', () => {
  it.each([
    ['Completed', 'success'],
    ['  In Progress ', 'active'],
    ['in-progress', 'active'],
    ['canceled', 'neutral'],
    ['cancelled', 'neutral'],
    ['pending', 'warning'],
    ['failed', 'error'],
    ['banana', 'neutral'],
  ] as const)('maps %j to %s', (status, expected) => {
    expect(getStatusVariant(status)).toBe(expected)
  })
})
