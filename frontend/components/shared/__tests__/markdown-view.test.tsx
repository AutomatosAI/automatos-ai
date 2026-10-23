/**
 * PRD-244 review batch 3 — the one markdown renderer behind every file preview.
 * The app has no @tailwindcss/typography, so `prose` classes were inert and
 * preflight had flattened documents; these assert the structure the `.md-view`
 * CSS then styles, and that fenced code still gets the copy-button card.
 *
 * PRD-261 W-A8: `skipHtml` (which the SDK's cards.MarkdownView hands through for model output) drops
 * raw HTML in the text; without it, as on the core's own surfaces, react-markdown prints the tags
 * as literal text. Neither ever renders them as elements. Builtin Play's markdown test held the
 * dropping until it left core; only this renderer can prove it.
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

/** Model output with raw HTML in it: a script, a bold run, an image with a handler. */
const WITH_HTML = 'Aim for <script>alert(1)</script> **16 GB** <img src=x onerror=alert(1)>'
const RAW_TAG = /<\/?(?:script|img)\b/i

describe('MarkdownView and raw HTML in the text', () => {
  it('with skipHtml drops it: no script or img element, no literal tag, the words and the bold kept', () => {
    const { container } = render(<MarkdownView skipHtml>{WITH_HTML}</MarkdownView>)
    expect(container.querySelector('script, img')).toBeNull()
    expect(container.textContent).not.toMatch(RAW_TAG)
    expect(container.textContent).not.toContain('onerror')
    expect(container.querySelector('strong')).toHaveTextContent('16 GB')
    expect(container).toHaveTextContent('Aim for alert(1) 16 GB')
  })

  it('without it (the default) prints the tags as literal text, never as elements', () => {
    const { container } = render(<MarkdownView>{WITH_HTML}</MarkdownView>)
    expect(container.querySelector('script, img')).toBeNull()
    expect(container.textContent).toMatch(RAW_TAG)
    expect(container).toHaveTextContent('Aim for <script>alert(1)</script> 16 GB <img src=x onerror=alert(1)>')
    expect(container.querySelector('strong')).toHaveTextContent('16 GB')
  })
})
