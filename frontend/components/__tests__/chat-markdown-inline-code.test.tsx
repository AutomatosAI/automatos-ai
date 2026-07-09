/**
 * Chat markdown rendering — inline code must stay inline.
 *
 * react-markdown v9 removed the `inline` prop from the `code` renderer; the
 * old chat components destructured it (always undefined), so EVERY backtick
 * span — including single values inside table cells — rendered as the full
 * CodeBlock card with a header and Copy button, shredding tables and prose.
 *
 * The fix routes fenced blocks through the `pre` renderer (which owns
 * CodeBlock) and makes the `code` renderer inline-only. These tests pin that
 * split: inline spans never grow a Copy button; fenced blocks (with or
 * without a language tag) always do.
 */
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { chatMarkdownComponents } from '@/components/chatbot/markdown-components'

function renderMarkdown(md: string) {
  return render(
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={chatMarkdownComponents}>
      {md}
    </ReactMarkdown>
  )
}

describe('chat markdown inline vs block code', () => {
  it('renders inline code as a plain styled span — no CodeBlock chrome', () => {
    const { container } = renderMarkdown('Missing required param `run_id` on the call.')

    const code = container.querySelector('code')
    expect(code).not.toBeNull()
    expect(code!.textContent).toBe('run_id')
    // No CodeBlock card: no copy button, no pre wrapper
    expect(screen.queryByTitle('Copy code')).toBeNull()
    expect(container.querySelector('pre')).toBeNull()
    // Inline code sits inside the paragraph, not adjacent to it
    expect(code!.closest('p')).not.toBeNull()
  })

  it('renders a language-tagged fenced block via CodeBlock (header + copy)', () => {
    renderMarkdown('```python\nprint("hello")\n```')

    expect(screen.getByTitle('Copy code')).toBeInTheDocument()
    expect(screen.getByText('Python')).toBeInTheDocument()
  })

  it('renders an untagged fenced block via CodeBlock too', () => {
    renderMarkdown('```\nplain block\n```')

    expect(screen.getByTitle('Copy code')).toBeInTheDocument()
    expect(screen.getByText('plain block')).toBeInTheDocument()
  })

  it('keeps inline code inside table cells — tables do not explode into cards', () => {
    const { container } = renderMarkdown(
      '| Field | Value |\n|---|---|\n| Agent | `337` |\n| Cost | `4.205629` |'
    )

    const table = container.querySelector('table')
    expect(table).not.toBeNull()
    const cellCodes = table!.querySelectorAll('td code')
    expect(cellCodes.length).toBe(2)
    expect(cellCodes[0].textContent).toBe('337')
    // No CodeBlock chrome anywhere in the table
    expect(screen.queryByTitle('Copy code')).toBeNull()
    expect(table!.querySelector('pre')).toBeNull()
  })

  it('renders inline and fenced code together with the right treatment each', () => {
    const { container } = renderMarkdown(
      'Check `status` first:\n\n```bash\necho ok\n```'
    )

    expect(screen.getByTitle('Copy code')).toBeInTheDocument()
    expect(screen.getByText('Bash')).toBeInTheDocument()
    const inline = container.querySelector('p code')
    expect(inline).not.toBeNull()
    expect(inline!.textContent).toBe('status')
  })
})
