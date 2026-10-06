/**
 * An email attachment's Download opens its link only when the mail tool gave an http(s) URL
 * (Outlook's contentLocation); a script URL gets no Download button and opens nothing.
 */
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { AttachmentCard } from '@/components/widgets/EmailWidget/EmailViewer'

afterEach(() => vi.restoreAllMocks())

const ATTACHMENT = { id: 'a1', filename: 'quote.pdf', mimeType: 'application/pdf', size: 2048 }

describe('AttachmentCard', () => {
  it('opens an http(s) download link in a new tab with no opener', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadUrl: 'https://files.example/quote.pdf' }} />)

    fireEvent.click(screen.getByTitle('Download'))

    expect(open).toHaveBeenCalledTimes(1)
    expect(open).toHaveBeenCalledWith('https://files.example/quote.pdf', '_blank', 'noopener,noreferrer')
  })

  it('offers no download for a script link, and a click opens nothing', () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<AttachmentCard attachment={{ ...ATTACHMENT, downloadUrl: 'javascript:alert(document.cookie)' }} />)

    expect(screen.queryByTitle('Download')).toBeNull()
    fireEvent.click(screen.getByText('quote.pdf'))
    expect(open).not.toHaveBeenCalled()
  })

  it('offers no download when the mail tool gave no link (Gmail)', () => {
    render(<AttachmentCard attachment={ATTACHMENT} />)
    expect(screen.queryByTitle('Download')).toBeNull()
  })
})
