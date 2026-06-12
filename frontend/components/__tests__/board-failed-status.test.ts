import { describe, it, expect } from 'vitest'
import { BOARD_COLUMNS, STATUS_CONFIG } from '@/types/board'

// PRD-161 S3 deterministic proxy for the "failed column" browser AC. The visual
// confirmation is deferred to a morning browser check; this guards the data the
// board renders from. tsc's Record<BoardStatus> exhaustiveness covers COLUMN_META
// + STATUS_CONFIG having a 'failed' entry at all.
describe('PRD-161 S3: board failed column', () => {
  it('exposes a Failed column in the board order', () => {
    const statuses = BOARD_COLUMNS.map((c) => c.status)
    expect(statuses).toContain('failed')
  })

  it('has a data-driven config entry for failed (destructive tone, no hardcoded hex)', () => {
    expect(STATUS_CONFIG.failed).toBeDefined()
    expect(STATUS_CONFIG.failed.label).toBe('Failed')
    expect(STATUS_CONFIG.failed.cssVar).toBe('--destructive')
  })
})
