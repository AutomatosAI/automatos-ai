/**
 * PRD-252 R5 — the kinds behind the one Needs-you number, in words:
 * "2 in review · 1 question · 1 failed". Empty kinds are left out.
 */
import type { NeedsYouCounts } from '@/hooks/use-needs-you'

const WORDS: [keyof NeedsYouCounts, string, string][] = [
  ['review', 'in review', 'in review'],
  ['question', 'question', 'questions'],
  ['approval', 'approval', 'approvals'],
  ['stuck', 'stuck', 'stuck'],
  ['failed', 'failed', 'failed'],
]

export function needsYouBreakdown(counts: NeedsYouCounts | undefined): string {
  if (!counts) return ''
  return WORDS.filter(([kind]) => counts[kind] > 0)
    .map(([kind, one, many]) => `${counts[kind]} ${counts[kind] === 1 ? one : many}`)
    .join(' · ')
}
