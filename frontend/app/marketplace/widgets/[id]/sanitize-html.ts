import DOMPurify, { type Config as DOMPurifyConfig } from 'dompurify'

// A widget's README is third-party marketplace-submitted content, rendered
// verbatim via dangerouslySetInnerHTML — it must go through a real HTML
// parser, not a hand-rolled regex (CodeQL js/bad-tag-filter,
// js/incomplete-multi-character-sanitization: a regex blocklist like the
// old one here is bypassable by nested/overlapping tags such as
// "<scr<script>ipt>" or doubled attribute prefixes). DOMPurify is already a
// dependency (components/widgets/EmailWidget/EmailViewer.tsx), with the same
// allowlist shape a README needs: headings, lists, links, code, tables,
// images.
const README_SANITIZE_CONFIG: DOMPurifyConfig = {
  ALLOWED_TAGS: [
    'p', 'br', 'hr', 'div', 'span', 'a', 'b', 'i', 'u', 'strong', 'em', 'del',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'img',
  ],
  ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class'],
  ALLOW_DATA_ATTR: false,
  FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button', 'style'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onmouseout', 'onfocus', 'onblur', 'style'],
}

export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, README_SANITIZE_CONFIG)
}
