'use client'

/**
 * PRD-264 EUAI-10 — the /verify form: pick a file, see whether Automatos AI made it.
 *
 * The file is fingerprinted (SHA-256) on this device and only the fingerprint is sent
 * (lib/provenance.ts). The answer is yes, with the media kind and the date, or no.
 */
import { useRef, useState, type ChangeEvent } from 'react'
import {
  CONTENT_CREDENTIALS_VERIFY_URL,
  ProvenanceCheckError,
  lookupProvenance,
  sha256Hex,
  type ProvenanceAnswer,
  type ProvenanceKind,
} from '@/lib/provenance'

const RATE_LIMITED_STATUS = 429
const KIND_LABELS: Readonly<Record<ProvenanceKind, string>> = {
  video: 'Video',
  image: 'Image',
  audio: 'Audio',
  document: 'Document',
}
const RATE_LIMITED_MESSAGE = 'Too many checks from your address. Wait a minute and try again.'
const FAILED_MESSAGE = 'The check did not go through. Try again in a moment.'

type CheckState =
  | { readonly step: 'idle' }
  | { readonly step: 'hashing'; readonly fileName: string }
  | { readonly step: 'checking'; readonly fileName: string; readonly hash: string }
  | { readonly step: 'answered'; readonly fileName: string; readonly hash: string; readonly answer: ProvenanceAnswer }
  | { readonly step: 'failed'; readonly fileName: string; readonly message: string }

function failureMessage(error: unknown): string {
  if (error instanceof ProvenanceCheckError) return error.message
  const status = (error as { status?: unknown } | null)?.status
  return status === RATE_LIMITED_STATUS ? RATE_LIMITED_MESSAGE : FAILED_MESSAGE
}

function AnswerPanel({ answer }: { answer: ProvenanceAnswer }) {
  if (answer.made_by_automatos_ai) {
    return (
      <div data-testid="provenance-answer" data-made-by-automatos-ai="true" className="space-y-1">
        <p className="text-lg font-semibold text-foreground">Made by Automatos AI: yes</p>
        <p className="text-sm text-muted-foreground">
          {KIND_LABELS[answer.kind]} · made on {answer.date}
        </p>
      </div>
    )
  }
  return (
    <div data-testid="provenance-answer" data-made-by-automatos-ai="false" className="space-y-1">
      <p className="text-lg font-semibold text-foreground">Made by Automatos AI: no</p>
      <p className="text-sm text-muted-foreground">
        Automatos has no record of this exact file. A copy that was edited, cropped or re-encoded has a
        different fingerprint, so it reads no even when Automatos made the original.
      </p>
    </div>
  )
}

function StatePanel({ state }: { state: CheckState }) {
  if (state.step === 'idle') return null
  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-4 text-card-foreground">
      <p className="text-sm text-muted-foreground">
        File: <span className="text-foreground">{state.fileName}</span>
      </p>
      {'hash' in state && (
        <p className="text-xs text-muted-foreground">
          SHA-256: <code data-testid="provenance-hash" className="break-all font-mono text-foreground">{state.hash}</code>
        </p>
      )}
      {state.step === 'hashing' && <p className="text-sm text-muted-foreground">Fingerprinting the file on this device…</p>}
      {state.step === 'checking' && <p className="text-sm text-muted-foreground">Checking the fingerprint…</p>}
      {state.step === 'answered' && <AnswerPanel answer={state.answer} />}
      {state.step === 'failed' && <p role="alert" className="text-sm text-destructive">{state.message}</p>}
    </div>
  )
}

/** The whole check: file picker, progress, the answer and the Content Credentials link. */
export function ProvenanceCheck() {
  const [state, setState] = useState<CheckState>({ step: 'idle' })
  const latest = useRef(0)

  const onFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const run = latest.current + 1
    latest.current = run
    const fileName = file.name
    setState({ step: 'hashing', fileName })
    try {
      const hash = await sha256Hex(file)
      if (latest.current !== run) return
      setState({ step: 'checking', fileName, hash })
      const answer = await lookupProvenance(hash)
      if (latest.current !== run) return
      setState({ step: 'answered', fileName, hash, answer })
    } catch (error) {
      if (latest.current !== run) return
      setState({ step: 'failed', fileName, message: failureMessage(error) })
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <label htmlFor="provenance-file" className="block text-sm font-medium text-foreground">
          Choose a file to check
        </label>
        <input
          id="provenance-file"
          type="file"
          onChange={onFile}
          className="block w-full text-sm text-muted-foreground file:mr-4 file:rounded-md file:border file:border-border file:bg-secondary file:px-3 file:py-2 file:text-sm file:text-foreground"
        />
        <p className="text-xs text-muted-foreground">
          The file stays on your device. Only its SHA-256 fingerprint is sent to Automatos.
        </p>
      </div>
      <div aria-live="polite">
        <StatePanel state={state} />
      </div>
      <p className="text-sm text-muted-foreground">
        Files that carry Content Credentials (C2PA), from Automatos or any other tool, can also be checked at{' '}
        <a
          href={CONTENT_CREDENTIALS_VERIFY_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline underline-offset-2"
        >
          contentcredentials.org/verify
        </a>
        .
      </p>
    </div>
  )
}
