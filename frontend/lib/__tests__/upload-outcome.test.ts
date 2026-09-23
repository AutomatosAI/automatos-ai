import { describe, expect, it } from 'vitest'
import { UPLOAD_OK_FALLBACK, uploadOutcome } from '../upload-outcome'

describe('uploadOutcome', () => {
  it('says a same-name upload replaced the earlier document', () => {
    const replaced = 'Replaced christmas-box-2026.csv (version 2) — agents now read the new copy; ' +
      "the previous one is kept in the document's history."
    expect(uploadOutcome({ status: 'completed', message: replaced })).toEqual({ ok: true, text: replaced })
  })

  it('reports a failed pipeline as a failure, in the server words', () => {
    expect(uploadOutcome({ status: 'failed', message: 'Uploaded, but processing failed: no embedding provider' }))
      .toEqual({ ok: false, text: 'Uploaded, but processing failed: no embedding provider' })
  })

  it('keeps the plain success text when the server sends none', () => {
    expect(uploadOutcome({ status: 'completed' })).toEqual({ ok: true, text: UPLOAD_OK_FALLBACK })
    expect(uploadOutcome(undefined)).toEqual({ ok: true, text: UPLOAD_OK_FALLBACK })
  })
})
