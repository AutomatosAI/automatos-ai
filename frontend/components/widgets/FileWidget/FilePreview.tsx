'use client'

/**
 * FilePreview — Unified file renderer
 *
 * Single source of truth for rendering files across the app:
 *   - Chat FileWidget
 *   - Workspace > Outputs (DeliverablePreview)
 *   - Workspace > Explorer (Preview/Source toggle)
 *
 * Supported formats:
 *   text-based (via `content`):
 *     - html        — sandboxed iframe + source toggle
 *     - markdown    — react-markdown + source toggle
 *     - code        — Prism syntax highlighting
 *     - json        — Prism (JSON grammar)
 *     - csv / tsv   — parsed to HTML table (papaparse)
 *     - text        — monospace <pre>
 *
 *   binary (via `url`):
 *     - pdf         — browser-native iframe
 *     - image       — <img>
 *     - video       — <video controls>
 *     - audio       — <audio controls>
 *     - docx        — mammoth.js → HTML (lazy-loaded)
 *     - xlsx        — SheetJS → HTML tables (lazy-loaded)
 *
 *   fallback:
 *     - binary      — "no preview" message with download hint
 */

import { useEffect, useMemo, useState } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { FileQuestion, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-sql'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-markup'
import 'prismjs/components/prism-go'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-docker'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { apiClient } from '@/lib/api-client'

/**
 * Fetch a URL with Automatos Bearer auth. Relative URLs are prefixed with
 * the API base URL. Used by DocxPreview/XlsxPreview for binary files served
 * by the authenticated /files/raw endpoint. Returns the response arrayBuffer.
 */
async function authenticatedArrayBufferFetch(url: string): Promise<ArrayBuffer> {
  const isAbsolute = /^https?:\/\//i.test(url)
  const fullUrl = isAbsolute ? url : `${apiClient.getBaseUrl()}${url}`
  const headers = isAbsolute ? {} : await apiClient.getAuthHeaders()
  const resp = await fetch(fullUrl, { headers, credentials: isAbsolute ? 'include' : 'omit' })
  if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`)
  return resp.arrayBuffer()
}

/**
 * Turn an authenticated relative URL into a short-lived blob: URL so
 * <img>, <video>, <audio>, and <iframe src=…> can render bytes that live
 * behind a Bearer-auth API. For already-absolute URLs (presigned S3 etc.)
 * the URL is returned unchanged. The blob URL is revoked on cleanup.
 */
function useAuthenticatedBlobUrl(url: string | undefined): {
  src: string | null
  error: string | null
} {
  const [src, setSrc] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!url) {
      setSrc(null)
      return
    }
    const isAbsolute = /^https?:\/\//i.test(url)
    if (isAbsolute) {
      setSrc(url)
      setError(null)
      return
    }

    let cancelled = false
    let createdUrl: string | null = null
    ;(async () => {
      try {
        const fullUrl = `${apiClient.getBaseUrl()}${url}`
        const headers = await apiClient.getAuthHeaders()
        const resp = await fetch(fullUrl, { headers })
        if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`)
        const blob = await resp.blob()
        if (cancelled) return
        createdUrl = URL.createObjectURL(blob)
        setSrc(createdUrl)
        setError(null)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Fetch failed')
      }
    })()

    return () => {
      cancelled = true
      if (createdUrl) URL.revokeObjectURL(createdUrl)
    }
  }, [url])

  return { src, error }
}

export type FilePreviewType =
  | 'text'
  | 'image'
  | 'pdf'
  | 'code'
  | 'binary'
  | 'html'
  | 'markdown'
  | 'csv'
  | 'video'
  | 'audio'
  | 'docx'
  | 'xlsx'
  | 'json'

interface FilePreviewProps {
  /** Text-based content (html source, markdown, code, csv, json, plain text) */
  content?: string
  /** URL to binary content (pdf, image, video, audio, docx, xlsx) */
  url?: string
  /** Explicit type — when omitted, inferred from filename extension */
  previewType?: FilePreviewType
  /** Used for syntax language detection and <img>/<video> alt/title */
  filename?: string
  className?: string
}

/**
 * Map filename extension → Prism language id.
 */
function getLanguageFromFilename(filename: string): string {
  const base = filename.split('/').pop() || ''
  if (base.toLowerCase() === 'dockerfile') return 'docker'
  const ext = base.split('.').pop()?.toLowerCase() || ''
  const langMap: Record<string, string> = {
    js: 'javascript',
    jsx: 'javascript',
    ts: 'typescript',
    tsx: 'typescript',
    py: 'python',
    rb: 'ruby',
    go: 'go',
    rs: 'rust',
    java: 'java',
    c: 'c',
    cpp: 'cpp',
    h: 'c',
    cs: 'csharp',
    php: 'php',
    swift: 'swift',
    kt: 'kotlin',
    sql: 'sql',
    sh: 'bash',
    bash: 'bash',
    zsh: 'bash',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    xml: 'markup',
    html: 'markup',
    htm: 'markup',
    css: 'css',
    scss: 'scss',
    less: 'less',
    md: 'markdown',
    markdown: 'markdown',
  }
  return langMap[ext] || 'text'
}

/**
 * Prism.highlight encodes HTML entities during tokenization — output is safe
 * for innerHTML rendering.
 */
function highlightCode(code: string, language: string): string | null {
  const grammar = Prism.languages[language]
  if (!grammar) return null
  return Prism.highlight(code, grammar, language)
}

// ---------------------------------------------------------------------------
// Unavailable / loading helpers
// ---------------------------------------------------------------------------

function NotAvailable({ message, className }: { message: string; className?: string }) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center h-full text-muted-foreground',
        className,
      )}
    >
      <FileQuestion className="h-12 w-12 mb-2 opacity-50" />
      <p className="text-sm">{message}</p>
    </div>
  )
}

function LoadingPane({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
      <Loader2 className="h-6 w-6 animate-spin" />
      <p className="text-sm">{message}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// CSV/TSV parsing (tiny, no deps — runtime imports papaparse lazily)
// ---------------------------------------------------------------------------

interface CsvTable {
  columns: string[]
  rows: string[][]
  truncated: boolean
}

const MAX_CSV_ROWS = 500

function useCsvTable(content: string | undefined, delimiter: string): CsvTable | null {
  const [table, setTable] = useState<CsvTable | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!content) {
      setTable(null)
      return
    }
    ;(async () => {
      const Papa = (await import('papaparse')).default
      const parsed = Papa.parse<string[]>(content, {
        delimiter,
        skipEmptyLines: true,
      })
      if (cancelled) return
      const rows = parsed.data as string[][]
      if (rows.length === 0) {
        setTable({ columns: [], rows: [], truncated: false })
        return
      }
      const [header, ...dataRows] = rows
      const truncated = dataRows.length > MAX_CSV_ROWS
      setTable({
        columns: header,
        rows: truncated ? dataRows.slice(0, MAX_CSV_ROWS) : dataRows,
        truncated,
      })
    })()
    return () => {
      cancelled = true
    }
  }, [content, delimiter])

  return table
}

function CsvPreview({ content, className }: { content?: string; className?: string }) {
  const table = useCsvTable(content, ',')
  if (!content) return <NotAvailable message="No content" className={className} />
  if (!table) return <LoadingPane message="Parsing…" />
  return (
    <ScrollArea className={cn('h-full', className)}>
      <div className="p-4">
        <table className="w-full text-xs border-collapse">
          <thead className="sticky top-0 bg-background">
            <tr>
              {table.columns.map((col, i) => (
                <th
                  key={i}
                  className="text-left px-2 py-1.5 font-semibold border-b border-border/60 whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r} className="hover:bg-muted/30">
                {row.map((cell, c) => (
                  <td key={c} className="px-2 py-1 border-b border-border/30 font-mono">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {table.truncated && (
          <p className="text-xs text-muted-foreground mt-3">
            Showing first {MAX_CSV_ROWS} rows — download for full file.
          </p>
        )}
      </div>
    </ScrollArea>
  )
}

// ---------------------------------------------------------------------------
// Bearer-authenticated binary renderers (image, pdf, video, audio)
// ---------------------------------------------------------------------------

function ImageContent({
  url,
  filename,
  className,
}: {
  url?: string
  filename?: string
  className?: string
}) {
  const { src, error } = useAuthenticatedBlobUrl(url)
  if (!url) return <NotAvailable message="No image URL" className={className} />
  if (error) return <NotAvailable message={`Image load failed: ${error}`} className={className} />
  if (!src) return <LoadingPane message="Loading image…" />
  return (
    <div className={cn('flex items-center justify-center h-full p-4', className)}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={filename || 'Image preview'}
        style={{ maxWidth: '100%', maxHeight: '100%' }}
        className="object-contain rounded-md"
      />
    </div>
  )
}

function PdfContent({
  url,
  filename,
  className,
}: {
  url?: string
  filename?: string
  className?: string
}) {
  const { src, error } = useAuthenticatedBlobUrl(url)
  if (!url) return <NotAvailable message="No PDF URL" className={className} />
  if (error) return <NotAvailable message={`PDF load failed: ${error}`} className={className} />
  if (!src) return <LoadingPane message="Loading PDF…" />
  return (
    <iframe
      src={src}
      title={filename || 'PDF preview'}
      className={cn('w-full h-full border-0 bg-white', className)}
    />
  )
}

function VideoContent({ url, className }: { url?: string; className?: string }) {
  const { src, error } = useAuthenticatedBlobUrl(url)
  if (!url) return <NotAvailable message="No video URL" className={className} />
  if (error) return <NotAvailable message={`Video load failed: ${error}`} className={className} />
  if (!src) return <LoadingPane message="Loading video…" />
  return (
    <div className={cn('flex items-center justify-center h-full p-4 bg-black', className)}>
      <video src={src} controls style={{ maxWidth: '100%', maxHeight: '100%' }}>
        <track kind="captions" />
      </video>
    </div>
  )
}

function AudioContent({ url, className }: { url?: string; className?: string }) {
  const { src, error } = useAuthenticatedBlobUrl(url)
  if (!url) return <NotAvailable message="No audio URL" className={className} />
  if (error) return <NotAvailable message={`Audio load failed: ${error}`} className={className} />
  if (!src) return <LoadingPane message="Loading audio…" />
  return (
    <div className={cn('flex items-center justify-center h-full p-6', className)}>
      <audio src={src} controls className="w-full max-w-md">
        <track kind="captions" />
      </audio>
    </div>
  )
}

// ---------------------------------------------------------------------------
// DOCX preview — lazy load mammoth, convert arrayBuffer → HTML
// ---------------------------------------------------------------------------

function DocxPreview({ url, className }: { url?: string; className?: string }) {
  const [html, setHtml] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!url) return
    ;(async () => {
      try {
        const mammoth = await import('mammoth')
        const buffer = await authenticatedArrayBufferFetch(url)
        const result = await mammoth.convertToHtml({ arrayBuffer: buffer })
        if (!cancelled) setHtml(result.value)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Conversion failed')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [url])

  if (!url) return <NotAvailable message="No document URL" className={className} />
  if (error) return <NotAvailable message={`DOCX preview failed: ${error}`} className={className} />
  if (html === null) return <LoadingPane message="Rendering document…" />

  return (
    <ScrollArea className={cn('h-full', className)}>
      <div
        className="prose prose-sm dark:prose-invert max-w-none p-6"
        // Safe: mammoth produces sanitized semantic HTML from a docx file,
        // not arbitrary user input. Same trust model as react-markdown.
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </ScrollArea>
  )
}

// ---------------------------------------------------------------------------
// XLSX preview — lazy load SheetJS, render each sheet as a table
// ---------------------------------------------------------------------------

interface XlsxSheet {
  name: string
  html: string
}

function XlsxPreview({ url, className }: { url?: string; className?: string }) {
  const [sheets, setSheets] = useState<XlsxSheet[] | null>(null)
  const [activeSheet, setActiveSheet] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!url) return
    ;(async () => {
      try {
        const XLSX = await import('xlsx')
        const buffer = await authenticatedArrayBufferFetch(url)
        const wb = XLSX.read(buffer, { type: 'array' })
        const parsed: XlsxSheet[] = wb.SheetNames.map((name) => ({
          name,
          html: XLSX.utils.sheet_to_html(wb.Sheets[name]),
        }))
        if (!cancelled) setSheets(parsed)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Conversion failed')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [url])

  if (!url) return <NotAvailable message="No spreadsheet URL" className={className} />
  if (error) return <NotAvailable message={`XLSX preview failed: ${error}`} className={className} />
  if (!sheets) return <LoadingPane message="Rendering spreadsheet…" />

  const current = sheets[activeSheet]

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {sheets.length > 1 && (
        <div className="flex items-center gap-1 border-b bg-muted/30 px-2 py-1 overflow-x-auto">
          {sheets.map((s, i) => (
            <button
              key={s.name}
              type="button"
              onClick={() => setActiveSheet(i)}
              className={cn(
                'rounded px-2 py-1 text-xs font-medium transition-colors whitespace-nowrap',
                i === activeSheet
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {s.name}
            </button>
          ))}
        </div>
      )}
      <ScrollArea className="flex-1">
        <div
          className="xlsx-preview p-2 text-xs"
          // Safe: SheetJS sheet_to_html produces static HTML from the workbook bytes.
          dangerouslySetInnerHTML={{ __html: current.html }}
        />
      </ScrollArea>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function FilePreview({
  content,
  url,
  previewType,
  filename,
  className,
}: FilePreviewProps) {
  // HTML and Markdown both support a Preview / Source toggle
  const [view, setView] = useState<'preview' | 'source'>('preview')

  // Compute syntax-highlighted HTML for code/json previews
  const highlightedHtml = useMemo(() => {
    if (previewType !== 'code' && previewType !== 'json') return null
    if (!content) return null
    const language =
      previewType === 'json' ? 'json' : filename ? getLanguageFromFilename(filename) : 'text'
    return highlightCode(content, language)
  }, [content, previewType, filename])

  // Nothing at all
  if (!content && !url) {
    return <NotAvailable message="No preview available" className={className} />
  }

  // ---------- Binary / URL-backed ----------

  if (previewType === 'binary') {
    return <NotAvailable message="Binary file — preview not available" className={className} />
  }

  if (previewType === 'image') {
    return <ImageContent url={url || content} filename={filename} className={className} />
  }

  if (previewType === 'pdf') {
    return <PdfContent url={url || content} filename={filename} className={className} />
  }

  if (previewType === 'video') {
    return <VideoContent url={url || content} className={className} />
  }

  if (previewType === 'audio') {
    return <AudioContent url={url || content} className={className} />
  }

  if (previewType === 'docx') {
    return <DocxPreview url={url} className={className} />
  }

  if (previewType === 'xlsx') {
    return <XlsxPreview url={url} className={className} />
  }

  // ---------- Text-based ----------

  if (previewType === 'html') {
    if (!content) return <NotAvailable message="No HTML content" className={className} />
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex items-center gap-1 border-b bg-muted/30 px-2 py-1">
          <button
            type="button"
            onClick={() => setView('preview')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              view === 'preview'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Preview
          </button>
          <button
            type="button"
            onClick={() => setView('source')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              view === 'source'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Source
          </button>
        </div>
        {view === 'preview' ? (
          <iframe
            // Sandboxed: allow-scripts only. No allow-same-origin → iframe
            // cannot access parent cookies, storage, or DOM. Agent-generated
            // HTML is untrusted input.
            sandbox="allow-scripts"
            srcDoc={content}
            title={filename || 'HTML preview'}
            className="flex-1 w-full border-0 bg-white"
          />
        ) : (
          <ScrollArea className="flex-1">
            <pre className="p-4 text-sm font-mono whitespace-pre-wrap text-gray-200 bg-[#2d2d2d] min-h-full">
              <code>{content}</code>
            </pre>
          </ScrollArea>
        )}
      </div>
    )
  }

  if (previewType === 'markdown') {
    if (!content) return <NotAvailable message="No markdown content" className={className} />
    return (
      <div className={cn('flex flex-col h-full', className)}>
        <div className="flex items-center gap-1 border-b bg-muted/30 px-2 py-1">
          <button
            type="button"
            onClick={() => setView('preview')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              view === 'preview'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Preview
          </button>
          <button
            type="button"
            onClick={() => setView('source')}
            className={cn(
              'rounded px-2 py-1 text-xs font-medium transition-colors',
              view === 'source'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
            )}
          >
            Source
          </button>
        </div>
        {view === 'preview' ? (
          <ScrollArea className="flex-1">
            <div className="prose prose-sm dark:prose-invert max-w-none p-6">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          </ScrollArea>
        ) : (
          <ScrollArea className="flex-1">
            <pre className="p-4 text-sm font-mono whitespace-pre-wrap text-gray-200 bg-[#2d2d2d] min-h-full">
              <code>{content}</code>
            </pre>
          </ScrollArea>
        )}
      </div>
    )
  }

  if (previewType === 'csv') {
    return <CsvPreview content={content} className={className} />
  }

  if ((previewType === 'code' || previewType === 'json') && content) {
    const language =
      previewType === 'json' ? 'json' : filename ? getLanguageFromFilename(filename) : 'text'
    return (
      <ScrollArea className={cn('h-full', className)}>
        <div className="bg-[#2d2d2d] p-4 min-h-full">
          {highlightedHtml ? (
            <pre className="text-sm font-mono leading-relaxed">
              {/* Safe: Prism.highlight encodes HTML entities during tokenization */}
              <code
                className={`language-${language}`}
                dangerouslySetInnerHTML={{ __html: highlightedHtml }}
              />
            </pre>
          ) : (
            <pre className="text-sm font-mono leading-relaxed text-gray-200">
              <code>{content}</code>
            </pre>
          )}
        </div>
      </ScrollArea>
    )
  }

  // Default text preview with font-mono
  return (
    <ScrollArea className={cn('h-full', className)}>
      <pre className="p-4 text-sm font-mono whitespace-pre-wrap">{content}</pre>
    </ScrollArea>
  )
}

// ---------------------------------------------------------------------------
// Shared helper — infer preview type from filename / mime. Exported so other
// surfaces (Outputs, Explorer) can reuse the same detection logic.
// ---------------------------------------------------------------------------

export function inferPreviewType(
  filename?: string,
  mimeType?: string,
): FilePreviewType {
  const mime = (mimeType || '').toLowerCase()
  const base = (filename || '').split('/').pop() || ''
  const ext = base.split('.').pop()?.toLowerCase() || ''

  // MIME-based (most specific)
  if (mime === 'text/html' || mime === 'application/xhtml+xml') return 'html'
  if (mime === 'text/markdown' || mime === 'text/x-markdown') return 'markdown'
  if (mime === 'application/pdf') return 'pdf'
  if (mime.startsWith('image/')) return 'image'
  if (mime.startsWith('video/')) return 'video'
  if (mime.startsWith('audio/')) return 'audio'
  if (mime === 'text/csv' || mime === 'text/tab-separated-values') return 'csv'
  if (mime === 'application/json') return 'json'
  if (
    mime === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  )
    return 'docx'
  if (
    mime === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
    mime === 'application/vnd.ms-excel'
  )
    return 'xlsx'

  // Extension-based fallback
  switch (ext) {
    case 'html':
    case 'htm':
      return 'html'
    case 'md':
    case 'markdown':
      return 'markdown'
    case 'pdf':
      return 'pdf'
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'webp':
    case 'svg':
    case 'bmp':
    case 'ico':
      return 'image'
    case 'mp4':
    case 'webm':
    case 'mov':
    case 'm4v':
    case 'avi':
      return 'video'
    case 'mp3':
    case 'wav':
    case 'ogg':
    case 'flac':
    case 'm4a':
      return 'audio'
    case 'csv':
    case 'tsv':
      return 'csv'
    case 'json':
      return 'json'
    case 'docx':
      return 'docx'
    case 'xlsx':
    case 'xls':
      return 'xlsx'
    case 'txt':
    case 'log':
      return 'text'
    case '':
      return 'text'
    default:
      // Treat common code extensions as code, else text.
      return [
        'js', 'jsx', 'ts', 'tsx', 'py', 'rb', 'go', 'rs', 'java', 'c', 'cpp',
        'h', 'cs', 'php', 'swift', 'kt', 'sql', 'sh', 'bash', 'zsh', 'yaml',
        'yml', 'xml', 'css', 'scss', 'less',
      ].includes(ext)
        ? 'code'
        : 'text'
  }
}
