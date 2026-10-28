/**
 * PRD-264 EUAI-11a — the record of what was done to the AI label's protected copy (protected-caption.tsx),
 * and the warnings it leaves: one the first time each kind of tampering is seen, then, while tampering
 * goes on, a count at most every TAMPER_SUMMARY_MS, so a theme that keeps at it shows in the console for
 * as long as it does, not once.
 */

const LOG_PREFIX = '[theme-host]'
/** While tampering goes on, a summary at most this often. */
export const TAMPER_SUMMARY_MS = 10_000
const MS_PER_SECOND = 1000

export type CaptionTamper = 'removed' | 'changed' | 'hidden' | 'failed'

const TAMPER_VERB: Readonly<Record<CaptionTamper, string>> = {
  removed: 'removed',
  changed: 'changed',
  hidden: 'hid',
  failed: 'broke the keeping of',
}

/** The warning the first tampering of each kind leaves in the console. */
export function aiLabelCopyTamperedWarning(kind: CaptionTamper): string {
  return `something in the page ${TAMPER_VERB[kind]} the AI label's protected copy; the host put it back`
}

/** The warning that sums up tampering that went on: `count` times in about `seconds` seconds. */
export function aiLabelCopyTamperSummary(count: number, seconds: number): string {
  return `something in the page has tampered with the AI label's protected copy ${count} times in the last ${seconds} s; the host keeps putting it back`
}

export class TamperLog {
  private warned: readonly CaptionTamper[] = []
  private windowStart = 0
  private lastAt = 0
  private count = 0

  /** Notes one tampering at `now` (ms), and warns as above; `detail` (an error, say) goes with a first warning. */
  note(kind: CaptionTamper, now: number, detail?: unknown): void {
    if (!this.warned.includes(kind)) {
      console.warn(LOG_PREFIX, aiLabelCopyTamperedWarning(kind), ...(detail === undefined ? [] : [detail]))
      this.warned = [...this.warned, kind]
    }
    // a run of tampering ends after a quiet TAMPER_SUMMARY_MS; the next one starts a new count
    if (this.count > 0 && now - this.lastAt > TAMPER_SUMMARY_MS) this.count = 0
    if (this.count === 0) this.windowStart = now
    this.count += 1
    this.lastAt = now
    if (now - this.windowStart < TAMPER_SUMMARY_MS) return
    console.warn(LOG_PREFIX, aiLabelCopyTamperSummary(this.count, Math.round((now - this.windowStart) / MS_PER_SECOND)))
    this.count = 0
  }
}
