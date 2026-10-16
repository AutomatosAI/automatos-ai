/**
 * P260-FX-12 (M49) — Download gets the file, not the API's JSON envelope. The
 * backend names a text Deliverable (.md, .csv, .py…) by its content route
 * (/api/workspaces/{ws}/files/content?path=…), which answers JSON for the code
 * viewer; Download fetches the same file's raw route (the core's
 * apiClient.getWorkspaceFileRawUrl) instead. Any other route (a raw file, a
 * generated document) is handed over as it is, and what a page shows is still
 * the route the detail answered with (deliverableFileUrl).
 */
import { describe, expect, it, vi } from 'vitest'

import { deliverableDownloadUrl, deliverableFileUrl, fileDownloadUrl } from '../download'

const CONTENT_MD = '/api/workspaces/ws-1/files/content?path=outputs%2Fa.md'
const RAW_MD = '/api/workspaces/ws-1/files/raw?path=outputs%2Fa.md'
const GENERATED_PDF = '/api/documents/generated/x.pdf'

describe('fileDownloadUrl', () => {
  it("swaps a workspace text file's content route for its raw route", () => {
    expect(fileDownloadUrl(CONTENT_MD)).toBe(RAW_MD)
  })

  it('decodes the path once, so the raw route carries it encoded once (the backend quotes it with / kept)', () => {
    expect(fileDownloadUrl('/api/workspaces/ws-1/files/content?path=reports/q3%20plan.csv')).toBe(
      '/api/workspaces/ws-1/files/raw?path=reports%2Fq3%20plan.csv',
    )
  })

  it('hands over any other route unchanged', () => {
    expect(fileDownloadUrl(GENERATED_PDF)).toBe(GENERATED_PDF)
    expect(fileDownloadUrl(RAW_MD)).toBe(RAW_MD)
    // Only the API's own relative route: the same words elsewhere in a URL are not it.
    const elsewhere = 'https://cdn.example/api/workspaces/ws-1/files/content?path=a.md'
    expect(fileDownloadUrl(elsewhere)).toBe(elsewhere)
  })

  it('keeps a content route whose path is not valid percent-encoding, and says so in the log', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const broken = '/api/workspaces/ws-1/files/content?path=outputs%2Fa%E0%A4%A.md'
    expect(fileDownloadUrl(broken)).toBe(broken)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('[deliverables]'), expect.any(URIError))
    warn.mockRestore()
  })
})

describe('deliverableDownloadUrl', () => {
  it("is the raw route of the Deliverable's file: content_url, else preview_url", () => {
    expect(deliverableDownloadUrl({ content_url: CONTENT_MD, preview_url: null })).toBe(RAW_MD)
    expect(deliverableDownloadUrl({ content_url: null, preview_url: CONTENT_MD })).toBe(RAW_MD)
    expect(deliverableDownloadUrl({ content_url: GENERATED_PDF, preview_url: CONTENT_MD })).toBe(GENERATED_PDF)
  })

  it('is null when the Deliverable has no file', () => {
    expect(deliverableDownloadUrl({ content_url: null, preview_url: null })).toBeNull()
    expect(deliverableDownloadUrl({ content_url: '', preview_url: null })).toBeNull()
  })

  it('leaves what a page shows as the detail answered it', () => {
    expect(deliverableFileUrl({ content_url: CONTENT_MD, preview_url: null })).toBe(CONTENT_MD)
  })
})
