/**
 * F353 (issue #947) — a document card shows its first page once the picture
 * loads, and keeps its type artwork while it loads, when it fails, and when the
 * Deliverable has no picture (then nothing is fetched).
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const blob = vi.hoisted(() => ({
  src: null as string | null,
  asked: [] as Array<string | undefined>,
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  useAuthenticatedBlobUrl: (url: string | undefined) => {
    blob.asked.push(url)
    return { src: url ? blob.src : null, error: null }
  },
}))
vi.mock('@/components/deliverables/deliverable-artwork', () => ({
  DeliverableArtwork: ({ type }: { type: string }) => <div data-testid="artwork" data-type={type} />,
}))

import { DocumentThumbnail } from '@/components/workspace/gallery-view/document-thumbnail'

const URL_OF_PICTURE = '/api/deliverables/5f0c2a8e-2b7d-4c55-9a51-0d6f1e2b2980/thumbnail'

afterEach(() => {
  cleanup()
  blob.src = null
  blob.asked = []
})

describe('DocumentThumbnail', () => {
  it('shows the first page once its picture has loaded', () => {
    blob.src = 'blob:first-page'
    render(<DocumentThumbnail type="document" thumbnailUrl={URL_OF_PICTURE} title="Price list" />)
    const img = screen.getByRole('img', { name: 'Price list' })
    expect(img.getAttribute('src')).toBe('blob:first-page')
    expect(blob.asked).toContain(URL_OF_PICTURE)
    expect(screen.queryByTestId('artwork')).toBeNull()
  })

  it('keeps the type artwork while the picture loads or when it fails', () => {
    render(<DocumentThumbnail type="spreadsheet" thumbnailUrl={URL_OF_PICTURE} title="Stock" />)
    expect(screen.getByTestId('artwork').dataset.type).toBe('spreadsheet')
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('fetches nothing for a Deliverable without a picture', () => {
    render(<DocumentThumbnail type="report" thumbnailUrl={null} title="Weekly report" />)
    expect(screen.getByTestId('artwork').dataset.type).toBe('report')
    expect(blob.asked.every((url) => url === undefined)).toBe(true)
  })
})
