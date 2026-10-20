'use client'

/**
 * PRD-261 C-03b — the isolated review job's note (C-02b), on the hosted edition.
 *
 * The reviewer runs the kit's "Theme review" workflow (no network, no secrets) with the version's
 * package and the digest shown here, then pastes the job summary with its outcome. A `pass` note
 * must be the kit's summary for THIS version: the server stores it only when the kit's header holds
 * `Package digest:  <digest>` and `Outcome: pass` (`review.note_names_this_version`), and approve
 * needs the latest note to pass. The form disables "Save" for a pass note without the digest line;
 * the server's check is the one that counts.
 *
 * The local edition has no isolated stage: there the person's approval is the review, and the
 * form is not shown (`LOCAL_REVIEW_NOTE`).
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { ThemeReportNote, ThemeReportNoteOutcome, ThemeReviewReport } from '@/lib/api-client'
import { formatRelative } from '@/lib/format-relative'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { LOCAL_REVIEW_NOTE, NOTE_NEEDS_DIGEST, noteDigestLine, noteNamesDigest } from './review-words'
import { useAddReportNote } from './use-theme-review'

export const SAVE_NOTE = 'Save review note'
const SUMMARY_MAX = 20000
const OUTCOMES: readonly ThemeReportNoteOutcome[] = ['pass', 'fail']

function LatestNote({ note }: { note: ThemeReportNote | null }) {
  if (!note) return <p className="text-xs text-muted-foreground">No isolated review note yet.</p>
  return (
    <div className="space-y-1 text-xs" data-testid="theme-report-note">
      <p>
        Latest note: <span className="font-medium">{note.outcome === 'pass' ? 'Pass' : 'Fail'}</span>
        {note.at && <span className="text-muted-foreground"> · {formatRelative(note.at)}</span>}
      </p>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted p-2">{note.summary}</pre>
    </div>
  )
}

function OutcomeChoice({ value, onChange }: { value: ThemeReportNoteOutcome; onChange: (next: ThemeReportNoteOutcome) => void }) {
  return (
    <fieldset className="flex gap-4 text-xs">
      <legend className="sr-only">Outcome</legend>
      {OUTCOMES.map((outcome) => (
        <label key={outcome} className="flex items-center gap-1">
          <input type="radio" name="theme-note-outcome" checked={value === outcome} onChange={() => onChange(outcome)} />
          {outcome === 'pass' ? 'Pass' : 'Fail'}
        </label>
      ))}
    </fieldset>
  )
}

function NoteForm({ report }: { report: ThemeReviewReport }) {
  const addNote = useAddReportNote()
  const [summary, setSummary] = useState('')
  const [outcome, setOutcome] = useState<ThemeReportNoteOutcome>('pass')
  const digestLine = noteDigestLine(report.digest)
  const missingDigest = outcome === 'pass' && !noteNamesDigest(summary, report.digest)
  const blocked = summary.trim() === '' || missingDigest || addNote.isLoading
  return (
    <form
      className="space-y-2"
      onSubmit={(event) => {
        event.preventDefault()
        if (!blocked) addNote.mutate({ packageId: report.id, summary, outcome }, { onSuccess: () => setSummary('') })
      }}
    >
      <label className="block text-xs font-medium" htmlFor="theme-note-summary">
        Paste the Theme review job summary
      </label>
      <Textarea
        id="theme-note-summary"
        value={summary}
        maxLength={SUMMARY_MAX}
        rows={6}
        onChange={(event) => setSummary(event.target.value)}
      />
      <OutcomeChoice value={outcome} onChange={setOutcome} />
      {missingDigest && (
        <p className="text-xs text-muted-foreground" data-testid="theme-note-digest-hint">
          {NOTE_NEEDS_DIGEST} <code className="break-all">{digestLine}</code>
        </p>
      )}
      <Button type="submit" variant="outline" disabled={blocked}>
        {SAVE_NOTE}
      </Button>
      {addNote.error ? (
        <p role="alert" className="text-xs text-destructive">
          {themeErrorMessage(addNote.error)}
        </p>
      ) : null}
    </form>
  )
}

export function ReportNoteSection({ report, hosted }: { report: ThemeReviewReport; hosted: boolean }) {
  const pending = report.status === 'pending'
  return (
    <section aria-label="Isolated review" className="space-y-2 rounded-lg border border-border/50 p-3">
      <p className="text-sm font-medium">Isolated review (kit job)</p>
      {hosted ? (
        <p className="text-xs text-muted-foreground">
          Run the kit&apos;s Theme review workflow with this version&apos;s package and the expected digest{' '}
          <code className="break-all">{report.digest}</code>, then paste its summary.
        </p>
      ) : (
        <p className="text-xs text-muted-foreground" data-testid="theme-local-review">
          {LOCAL_REVIEW_NOTE}
        </p>
      )}
      <LatestNote note={report.reportNote} />
      {hosted && pending && <NoteForm report={report} />}
    </section>
  )
}
