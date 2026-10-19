/**
 * PRD-260 P260-FX-13 — Auto's reply as markdown (PlayMarkdown) on Automatos
 * Play's rules for model output: a fenced block prints in Play's own <pre>, not
 * the core's dark CodeBlock card with its Copy header (M36), and raw HTML is
 * dropped, never printed as literal tags (M39). The Deliverable sheet shares
 * these components (play-deliverable.test.tsx). Roles, attributes and text only
 * (CSS modules are proxies under css: false).
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { PlayMarkdown } from '../play-markdown'

afterEach(cleanup)

describe("PlayMarkdown — Auto's reply", () => {
  it("prints a fenced block in Play's own <pre>, not the core's CodeBlock card with Copy", () => {
    const { container } = render(<PlayMarkdown text={'Run this:\n\n```py\nprint("hi")\n```'} />)
    const pre = container.querySelector('pre')
    expect(pre).toHaveTextContent('print("hi")')
    expect(pre?.querySelector('code')).toHaveTextContent('print("hi")')
    expect(pre).not.toHaveAttribute('class')
    expect(screen.queryByRole('button', { name: /copy/i })).toBeNull()
  })

  it('drops raw HTML instead of printing its tags, and keeps the words around them', () => {
    const { container } = render(<PlayMarkdown text={'a<br>b and <script>alert(1)</script> done'} />)
    expect(container.textContent).not.toMatch(/<\/?(?:br|script)\b/i)
    expect(container.querySelector('br, script')).toBeNull()
    expect(container).toHaveTextContent('ab and alert(1) done')
  })

  it('still renders markdown, opens links in a new tab and shows an image as its alt text', () => {
    const { container } = render(<PlayMarkdown text={'**Bold** [site](https://example.com) ![a chart](https://evil.example/c.png)'} />)
    expect(screen.getByText('Bold').tagName).toBe('STRONG')
    expect(screen.getByRole('link', { name: 'site' })).toHaveAttribute('rel', 'noopener noreferrer')
    expect(screen.getByText('a chart')).toBeInTheDocument()
    expect(container.querySelector('img')).toBeNull()
  })
})
