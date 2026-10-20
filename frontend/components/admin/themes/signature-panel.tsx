'use client'

/**
 * PRD-261 C-03b — signing an approved version: request, then paste back (D3-c, 19 Oct).
 *
 * The orchestrator never holds a signing key (`services/themes/signing.py`). Approve records a
 * signature request; a person runs the kit's "Theme sign" workflow with the WHOLE request JSON
 * shown here (its `request` input), and pastes the base64 Ed25519 signature it prints. The server
 * checks the shape and that it answers this version's open request, and stores it only when an
 * active Automatos signing key verifies it (C-05b; a 400 otherwise). The report checks the stored
 * signature again on every read (`signatureVerified`, `keyId`): the panel names the active key that
 * verifies it, or says no active key does.
 */
import { useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { ThemeReviewReport, ThemeSignatureRequest } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { storedSignatureWords } from './review-words'
import { useStoreThemeSignature } from './use-theme-review'

export const SAVE_SIGNATURE = 'Save signature'
export const COPY_REQUEST = 'Copy request'
// signing.py SIGNATURE_TEXT_MAX: base64 of 64 bytes is 88 characters, with room for padding variants.
const SIGNATURE_MAX = 200
const JSON_INDENT = 2

function requestText(request: ThemeSignatureRequest): string {
  return JSON.stringify(request, null, JSON_INDENT)
}

export const COPY_FAILED = "Couldn't copy: select the request and copy it by hand."

async function copyText(value: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(value)
    toast.success('Request copied.')
  } catch {
    toast.error(COPY_FAILED)
  }
}

function SignatureForm({ packageId, request }: { packageId: string; request: ThemeSignatureRequest }) {
  const store = useStoreThemeSignature()
  const [signature, setSignature] = useState('')
  const blank = signature.trim() === ''
  return (
    <form
      className="space-y-2"
      onSubmit={(event) => {
        event.preventDefault()
        if (!blank) store.mutate({ packageId, requestId: request.requestId, signature: signature.trim() })
      }}
    >
      <label className="block text-xs font-medium" htmlFor="theme-signature">
        2. Paste the signature the workflow printed
      </label>
      <Input
        id="theme-signature"
        value={signature}
        maxLength={SIGNATURE_MAX}
        autoComplete="off"
        spellCheck={false}
        onChange={(event) => setSignature(event.target.value)}
      />
      <Button type="submit" variant="outline" disabled={blank || store.isLoading}>
        {SAVE_SIGNATURE}
      </Button>
      {store.error ? (
        <p role="alert" className="text-xs text-destructive">
          {themeErrorMessage(store.error)}
        </p>
      ) : null}
    </form>
  )
}

function OpenRequest({ packageId, request }: { packageId: string; request: ThemeSignatureRequest }) {
  const text = requestText(request)
  return (
    <div className="space-y-2" data-testid="theme-signature-request">
      <p className="text-xs">
        1. Run the kit&apos;s Theme sign workflow with this whole request as its <code>request</code> input.
      </p>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-muted p-2 text-xs">{text}</pre>
      <Button type="button" size="sm" variant="ghost" onClick={() => void copyText(text)}>
        {COPY_REQUEST}
      </Button>
      <SignatureForm packageId={packageId} request={request} />
    </div>
  )
}

export function SignaturePanel({ report }: { report: ThemeReviewReport }) {
  if (report.status !== 'approved' && !report.signed) return null
  return (
    <section aria-label="Signature" className="space-y-2 rounded-lg border border-border/50 p-3">
      <p className="text-sm font-medium">Signature</p>
      {report.signed ? (
        <p className={report.signatureVerified ? 'text-xs' : 'text-xs text-destructive'} data-testid="theme-signed">
          Signed. {storedSignatureWords(report.signatureVerified, report.keyId)}
        </p>
      ) : report.signatureRequest ? (
        <OpenRequest packageId={report.id} request={report.signatureRequest} />
      ) : (
        <p className="text-xs text-muted-foreground">This version has no open signature request.</p>
      )}
    </section>
  )
}
