/**
 * sanitizeHtml — CodeQL js/bad-tag-filter #4, js/incomplete-multi-character-sanitization #3.
 *
 * A widget's README is third-party marketplace-submitted content, rendered
 * via dangerouslySetInnerHTML. The old sanitizer was a hand-rolled regex
 * blocklist (the file's own comment said "for production you'd want
 * DOMPurify"); it's bypassable by nested/overlapping tags that a browser's
 * HTML parser still recognizes. sanitizeHtml now delegates to DOMPurify,
 * the same library already used for email HTML
 * (components/widgets/EmailWidget/EmailViewer.tsx).
 */
import { describe, expect, it } from 'vitest'
import { sanitizeHtml } from '../page'

describe('sanitizeHtml', () => {
  it('strips an ordinary <script> block', () => {
    const out = sanitizeHtml('<p>Hi</p><script>alert(1)</script><p>Bye</p>')
    expect(out).not.toContain('<script')
    expect(out).not.toContain('alert')
    expect(out).toContain('Hi')
    expect(out).toContain('Bye')
  })

  it('strips a script closed with trailing junk before the > (the old regex missed this)', () => {
    const out = sanitizeHtml('<script>alert(1)</script foo><p>Bye</p>')
    expect(out).not.toContain('alert')
    expect(out).toContain('Bye')
  })

  it('leaves no real <script> element standing, even inside malformed markup', () => {
    // A browser's tokenizer parses "<scr<script>" as ONE bogus "<scr ...>"
    // tag (a "<" inside a tag name does not start a new tag) — there is no
    // real nested <script> element here, so DOMPurify (using a real parser)
    // correctly treats the rest as inert text, the same as a browser would.
    const out = sanitizeHtml('<scr<script>ipt>alert(1)</script>')
    expect(out.toLowerCase()).not.toContain('<script')
  })

  it('strips inline event handlers', () => {
    const out = sanitizeHtml('<img src="x.png" onerror="alert(1)">')
    expect(out).not.toContain('onerror')
    expect(out).not.toContain('alert')
  })

  it('strips a javascript: href', () => {
    const out = sanitizeHtml('<a href="javascript:alert(1)">click</a>')
    expect(out).not.toContain('javascript:')
  })

  it('keeps the README-ish tags a widget page needs', () => {
    const out = sanitizeHtml(
      '<h1>Title</h1><p>Some <strong>bold</strong> text.</p>' +
        '<ul><li>one</li></ul><pre><code>const x = 1</code></pre>' +
        '<a href="https://example.com">link</a><img src="https://example.com/a.png" alt="a">',
    )
    expect(out).toContain('<h1>Title</h1>')
    expect(out).toContain('<strong>bold</strong>')
    expect(out).toContain('<li>one</li>')
    expect(out).toContain('<code>const x = 1</code>')
    expect(out).toContain('href="https://example.com"')
    expect(out).toContain('<img')
  })

  it('strips iframe/object/embed/form', () => {
    const out = sanitizeHtml('<iframe src="https://evil.example"></iframe><form action="x"></form>')
    expect(out).not.toContain('<iframe')
    expect(out).not.toContain('<form')
  })
})
