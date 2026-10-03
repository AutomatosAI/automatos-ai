/**
 * PRD-252 R5 — the activity feed names a ticket's stage in the board's words.
 *
 * The feed's own status vocabulary (running, pending, completed, failed) drives
 * its filters and tones, so a ticket in Blocked read "failed" ("burnt" in the
 * classic feed) and one in Review read "pending". Feed items for tickets carry
 * the board's status as `board_status`; this is its label from the board.
 */
import type { ActivityFeedItem } from '@/hooks/use-activity-api'
import { STATUS_CONFIG, type BoardStatus } from '@/types/board'

export function boardStatusWord(item: Pick<ActivityFeedItem, 'board_status'>): string | null {
  const status = item.board_status as BoardStatus | null | undefined
  return (status && STATUS_CONFIG[status]?.label) || null
}
