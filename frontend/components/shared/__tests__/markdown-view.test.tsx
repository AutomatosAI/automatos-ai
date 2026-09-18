/**
 * PRD-244 review batch 3 — the one markdown renderer behind every file preview.
 * The app has no @tailwindcss/typography, so `prose` classes were inert and
 * preflight had flattened documents; these assert the structure the `.md-view`
 * CSS then styles, and that fenced code still gets the copy-button card.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@/components/chatbot/code-block', () => ({
  CodeBlock: ({ code, language }: { code: string; language: string }) => (
    <div data-testid="code-block" data-language={language}>{code}</div>
  ),
}))

import { MarkdownView } from '@/components/shared/markdown-view'

const DOC = [
  '# Task Report',
  '',
  'Intro with **bold** and `inline code`.',
  '',
  '## Result',
  '',
  '- first finding',
  '- second finding',
  '',
  '| Check | State |',
  '| --- | --- |',
  '| build | green |',
  '',
  '```js',
  'const x = 1',
  '```',
  '',
  '> a quoted line',
].join('\n')

afterEach(cleanup)

describe('MarkdownView', () => {
  it('renders a document structure: headings, lists, a table and a quote', () => {
    const { container } = render(<MarkdownView>{DOC}</MarkdownView>)
    expect(container.firstElementChild).toHaveClass('md-view')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Task Report')
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Result')
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'green' })).toBeInTheDocument()
    expect(container.querySelector('blockquote')).toHaveTextContent('a quoted line')
    expect(container.querySelector('strong')).toHaveTextContent('bold')
  })

  it('sends fenced code to the copy-button block and leaves inline code inline', () => {
    const { container } = render(<MarkdownView>{DOC}</MarkdownView>)
    const block = screen.getByTestId('code-block')
    expect(block).toHaveAttribute('data-language', 'js')
    expect(block).toHaveTextContent('const x = 1')
    const inline = [...container.querySelectorAll('code')].filter((c) => c.closest('[data-testid="code-block"]') === null)
    expect(inline).toHaveLength(1)
    expect(inline[0]).toHaveTextContent('inline code')
  })

  it('compact density is a modifier on the same view', () => {
    const { container } = render(<MarkdownView density="compact">hello</MarkdownView>)
    expect(container.firstElementChild).toHaveClass('md-view', 'md-view-compact')
  })

  it('a surface can override one element without forking the renderer', () => {
    render(
      <MarkdownView components={{ a: ({ children }: any) => <span data-testid="flat-link">{children}</span> }}>
        {'[report](sandbox://x)'}
      </MarkdownView>,
    )
    expect(screen.getByTestId('flat-link')).toHaveTextContent('report')
  })

  it('drops disallowed elements but keeps their text, and survives empty content', () => {
    const { container } = render(<MarkdownView disallowedElements={['img']}>{'text ![alt](x.png)'}</MarkdownView>)
    expect(container.querySelector('img')).toBeNull()
    expect(container).toHaveTextContent('text')
    const empty = render(<MarkdownView>{null}</MarkdownView>)
    expect(empty.container.firstElementChild).toHaveClass('md-view')
  })

  it('renders GFM task lists as read-only markers', () => {
    const { container } = render(<MarkdownView>{'- [x] done\n- [ ] open'}</MarkdownView>)
    const boxes = container.querySelectorAll('input[type="checkbox"]')
    expect(boxes).toHaveLength(2)
    expect(boxes[0]).toBeChecked()
    expect(boxes[0]).toBeDisabled()
  })
})
