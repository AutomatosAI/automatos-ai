/**
 * GitHub-style renderers for a document's markdown in the chat's Document widget.
 * Moved out of DocumentWidget unchanged (F358) so the component stays within its size limit.
 */
import { cn } from '@/lib/utils'

export const DOCUMENT_MARKDOWN_COMPONENTS = {
  // Links
  a: ({ href, children, ...props }: any) => (
    <a
      {...props}
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-[#58a6ff] hover:underline"
    >
      {children}
    </a>
  ),
  // react-markdown v9 has no `inline` prop: inline code is any code node
  // NOT wrapped in a pre, so the pre wrapper below re-styles its child and
  // this renderer only ever handles inline spans.
  code: ({ className, children, ...props }: any) => {
    if (/language-/.test(className || '')) {
      // Fenced block child (rendered inside the pre wrapper below)
      return (
        <code className={cn("text-[13px] font-mono text-[#e6edf3]", className)} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code className="rounded-md bg-[#343942] px-1.5 py-0.5 text-[13px] font-mono text-[#e6edf3]" {...props}>
        {children}
      </code>
    )
  },
  // Code blocks wrapper — normalizes the child so even untagged fenced
  // blocks lose the inline pill styling.
  pre: ({ children }: any) => (
    <pre className="rounded-md bg-[#161b22] border border-[#30363d] p-4 overflow-x-auto my-4 [&_code]:rounded-none [&_code]:bg-transparent [&_code]:p-0 [&_code]:border-0">
      {children}
    </pre>
  ),
  // Headings - GitHub style with bottom border
  h1: ({ children }: any) => (
    <h1 className="text-[2em] font-semibold text-[#e6edf3] border-b border-[#30363d] pb-2 mt-6 mb-4">{children}</h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-[1.5em] font-semibold text-[#e6edf3] border-b border-[#30363d] pb-2 mt-6 mb-4">{children}</h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-[1.25em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h3>
  ),
  h4: ({ children }: any) => (
    <h4 className="text-[1em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h4>
  ),
  h5: ({ children }: any) => (
    <h5 className="text-[0.875em] font-semibold text-[#e6edf3] mt-6 mb-4">{children}</h5>
  ),
  h6: ({ children }: any) => (
    <h6 className="text-[0.85em] font-semibold text-[#8b949e] mt-6 mb-4">{children}</h6>
  ),
  // Paragraphs
  p: ({ children }: any) => (
    <p className="text-[#e6edf3] leading-[1.6] mb-4">{children}</p>
  ),
  // Lists
  ul: ({ children }: any) => (
    <ul className="list-disc pl-8 mb-4 space-y-1 text-[#e6edf3]">{children}</ul>
  ),
  ol: ({ children }: any) => (
    <ol className="list-decimal pl-8 mb-4 space-y-1 text-[#e6edf3]">{children}</ol>
  ),
  li: ({ children }: any) => (
    <li className="text-[#e6edf3] leading-[1.6]">{children}</li>
  ),
  // Blockquote - GitHub style with left border
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-4 border-[#30363d] pl-4 my-4 text-[#8b949e]">{children}</blockquote>
  ),
  // Horizontal rule
  hr: () => (
    <hr className="border-t border-[#30363d] my-6" />
  ),
  // Tables - GitHub style
  table: ({ children }: any) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border-collapse border border-[#30363d] text-sm">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => (
    <thead className="bg-[#161b22]">{children}</thead>
  ),
  tbody: ({ children }: any) => (
    <tbody className="divide-y divide-[#30363d]">{children}</tbody>
  ),
  tr: ({ children }: any) => (
    <tr className="even:bg-[#161b22]/50">{children}</tr>
  ),
  th: ({ children }: any) => (
    <th className="px-4 py-3 text-left text-[#e6edf3] font-semibold border border-[#30363d]">{children}</th>
  ),
  td: ({ children }: any) => (
    <td className="px-4 py-3 text-[#e6edf3] border border-[#30363d]">{children}</td>
  ),
  // Strong/bold
  strong: ({ children }: any) => (
    <strong className="font-semibold text-[#e6edf3]">{children}</strong>
  ),
  // Emphasis/italic
  em: ({ children }: any) => (
    <em className="italic text-[#e6edf3]">{children}</em>
  ),
  // Images
  img: ({ src, alt, ...props }: any) => (
    <img src={src} alt={alt} className="max-w-full rounded-md border border-[#30363d] my-4" {...props} />
  ),
}
