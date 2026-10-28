/**
 * PRD-264 EUAI-11a — where the AI label's check hit-tests a line of the words (lib/ai-label/visibility.ts
 * samplePoints): SAMPLE_POINTS_PER_LINE points evenly across it, ends included, at its middle height, and
 * JITTERED_POINTS_PER_LINE more at random across it and inside its middle half, so a striped overlay laid
 * between known points can't stay unseen. The rest of the check is proven against the real labels in
 * lib/theme-host/__tests__/ai-label.test.ts and ai-label-widget.test.tsx.
 */
import { describe, expect, it } from 'vitest'

import { JITTERED_POINTS_PER_LINE, SAMPLE_POINTS_PER_LINE, samplePoints } from '../visibility'

const LINE = { left: 100, top: 40, right: 210, bottom: 56, width: 110, height: 16 }
const INSET = 2

describe('samplePoints (PRD-264 EUAI-11a)', () => {
  it('takes evenly spaced points across the line, ends included, and a bounded number at random', () => {
    const points = samplePoints(LINE, () => 0.5)
    expect(points).toHaveLength(SAMPLE_POINTS_PER_LINE + JITTERED_POINTS_PER_LINE)
    const even = points.slice(0, SAMPLE_POINTS_PER_LINE)
    expect(even[0]).toEqual([LINE.left + INSET, LINE.top + LINE.height / 2])
    expect(even[even.length - 1]).toEqual([LINE.right - INSET, LINE.top + LINE.height / 2])
    const gaps = even.slice(1).map(([x], index) => x - even[index][0])
    gaps.forEach((gap) => expect(gap).toBeCloseTo((LINE.width - 2 * INSET) / (SAMPLE_POINTS_PER_LINE - 1), 6))
  })

  it('puts the random points where the random numbers say, inside the line and its middle half', () => {
    const draws = [0, 0, 1, 1]
    const points = samplePoints(LINE, () => draws.shift() ?? 0.5).slice(SAMPLE_POINTS_PER_LINE)
    expect(points).toEqual([
      [LINE.left + INSET, LINE.top + LINE.height * 0.25],
      [LINE.right - INSET, LINE.top + LINE.height * 0.75],
    ])
  })

  it('moves the random points from one check to the next', () => {
    const first = samplePoints(LINE).slice(SAMPLE_POINTS_PER_LINE)
    const second = samplePoints(LINE).slice(SAMPLE_POINTS_PER_LINE)
    expect(first).not.toEqual(second)
    for (const [x, y] of [...first, ...second]) {
      expect(x).toBeGreaterThanOrEqual(LINE.left)
      expect(x).toBeLessThanOrEqual(LINE.right)
      expect(y).toBeGreaterThanOrEqual(LINE.top)
      expect(y).toBeLessThanOrEqual(LINE.bottom)
    }
  })
})
