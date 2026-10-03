/**
 * PRD-252 R5 — the activity feed uses the board's status words. The feed's own
 * status said "failed" ("burnt" in the classic feed) for a Blocked ticket and
 * "pending" for one in Review; every feed view now shows the board's word.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
import { boardStatusWord } from '../board-status-word'

const read = (rel: string) => readFileSync(path.resolve(__dirname, '..', '..', '..', rel), 'utf8')

describe('boardStatusWord', () => {
  it("names a ticket's stage as the board does", () => {
    expect(boardStatusWord({ board_status: 'blocked' })).toBe('Blocked')
    expect(boardStatusWord({ board_status: 'review' })).toBe('Review')
    expect(boardStatusWord({ board_status: 'in_progress' })).toBe('In Progress')
    expect(boardStatusWord({ board_status: 'done' })).toBe('Done')
  })

  it('leaves anything that is not a ticket to its own words', () => {
    expect(boardStatusWord({})).toBeNull()
    expect(boardStatusWord({ board_status: null })).toBeNull()
    expect(boardStatusWord({ board_status: 'not-a-status' })).toBeNull()
  })

  it('every feed view uses it', () => {
    for (const view of [
      'components/command-center/activity-tab.tsx',
      'components/activity/activity-feed.tsx',
      'components/activity/command-center-history.tsx',
      'components/activity/widgets/activity-widget.tsx',
    ]) {
      expect(read(view), view).toContain('boardStatusWord(')
    }
  })
})
