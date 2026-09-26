/**
 * PRD-251 S4.4 — the Analytics page's "Spend by Lane" names the media lane
 * "Media": renders (seconds at $0) and footage, stills and voice paid on the
 * workspace's own Composio toolkit. The label map moved from the costs tab
 * into lib/analytics-usage.ts; every lane keeps the label it had.
 */
import { describe, expect, it } from 'vitest'

import { LANE_LABELS, laneLabel } from '@/lib/analytics-usage'

describe('the media lane on the Analytics page', () => {
  it('labels the media lane Media', () => {
    expect(laneLabel('media')).toBe('Media')
    expect(LANE_LABELS.media).toBe('Media')
  })

  it('keeps the other lanes, and an unknown lane keeps its name', () => {
    expect(laneLabel('rerank')).toBe('Rerank')
    expect(laneLabel('recipe')).toBe('Playbooks')
    expect(laneLabel('board_task')).toBe('Board tickets')
    expect(laneLabel('some_new_lane')).toBe('some new lane')
  })
})
