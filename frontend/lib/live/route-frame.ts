/**
 * PRD-180 S1, PRD-205 S7, PRD-228, PRD-268 US-W01b — where each frame of the board stream goes.
 *
 * The stream (`GET /api/v1/tasks/stream`, hooks/use-board-event-stream.ts) sends three named frames, and each is
 * told to the page as a window event, so a view that holds no stream of its own can follow it:
 * - `board_changed`: the board's cache is asked again (the Command Centre refetches once), then
 *   `automatos:board-changed` (the Agents fleet tab, the suggestions list);
 * - `chat_changed`: `automatos:chat-changed` (the open conversation merges a background message in);
 * - `suggestion_changed`: `automatos:suggestion-changed` (the open suggestions are asked again, so a card started,
 *   hidden or expired elsewhere changes in place). Its data names ids and the status only.
 * Each event's `detail` is the frame's data; any other frame is ignored.
 */

/** One parsed SSE frame: its event name and its JSON-decoded data, if any. */
export interface StreamFrame {
  readonly event: string
  readonly data: unknown
}

/** The window event each named frame is told as. */
export const FRAME_EVENTS: Readonly<Record<string, string>> = Object.freeze({
  board_changed: 'automatos:board-changed',
  chat_changed: 'automatos:chat-changed',
  suggestion_changed: 'automatos:suggestion-changed',
})

const BOARD_CHANGED = 'board_changed'

/** What routing a frame touches: the board's cache, and the page's events. */
export interface FrameSinks {
  readonly invalidateBoard: () => void
  readonly dispatch: (event: Event) => void
}

/** Send one frame where it goes; a frame with no event of its own does nothing. */
export function routeFrame(frame: StreamFrame, sinks: FrameSinks): void {
  if (!Object.prototype.hasOwnProperty.call(FRAME_EVENTS, frame.event)) return
  if (frame.event === BOARD_CHANGED) sinks.invalidateBoard()
  sinks.dispatch(new CustomEvent(FRAME_EVENTS[frame.event], { detail: frame.data }))
}
