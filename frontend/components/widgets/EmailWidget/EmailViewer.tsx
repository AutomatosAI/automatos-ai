'use client'

/**
 * EmailViewer Component for PRD-38.2 Extended Widgets
 *
 * Full email display with HTML support and XSS prevention via DOMPurify.
 * SECURITY: All HTML content is sanitized before rendering.
 */

import { useMemo } from 'react'
import DOMPurify from 'dompurify'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  ArrowLeft,
  Paperclip,
  Download,
  ExternalLink,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { format } from 'date-fns'
import { EmailActions } from './EmailActions'
import type { EmailFull, EmailAddress, EmailAttachment } from '../types'

/**
 * DOMPurify configuration for email HTML sanitization
 * Strict whitelist to prevent XSS attacks
 */
const DOMPURIFY_CONFIG: DOMPurify.Config = {
  ALLOWED_TAGS: [
    'p', 'br', 'div', 'span', 'a', 'b', 'i', 'u', 'strong', 'em',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'blockquote', 'pre', 'code',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'img',  // Images handled via proxy
  ],
  ALLOWED_ATTR: ['href', 'src', 'alt', 'title', 'class', 'style', 'target'],
  ALLOW_DATA_ATTR: false,
  FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'input', 'button'],
  FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onmouseout', 'onfocus', 'onblur'],
}

/**
 * Sanitize HTML content to prevent XSS attacks
 */
function sanitizeEmailHtml(html: string): string {
  return DOMPurify.sanitize(html, DOMPURIFY_CONFIG)
}

/**
 * Proxy external images to prevent tracking pixels
 * In production, this would route through /api/image-proxy
 */
const IMAGE_PROXY_URL = '/api/image-proxy'

function proxyExternalImages(html: string): string {
  return html.replace(
    /src=["']((https?:)?\/\/[^"']+)["']/gi,
    (match, url) => {
      // Skip already-proxied images
      if (url.startsWith(IMAGE_PROXY_URL)) return match
      // Skip data: URLs
      if (url.startsWith('data:')) return match
      // Proxy external images
      const proxiedUrl = `${IMAGE_PROXY_URL}?url=${encodeURIComponent(url)}`
      return `src="${proxiedUrl}"`
    }
  )
}

/**
 * Format email address for display
 */
function formatAddress(address: EmailAddress): string {
  return address.name ? `${address.name} <${address.email}>` : address.email
}

/**
 * Format file size for display
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface EmailViewerProps {
  email: EmailFull
  onBack?: () => void
  onReply?: (email: EmailFull) => void
  onForward?: (email: EmailFull) => void
  onArchive?: (emailId: string) => void
  onDelete?: (emailId: string) => void
}

export function EmailViewer({
  email,
  onBack,
  onReply,
  onForward,
  onArchive,
  onDelete,
}: EmailViewerProps) {
  // Sanitize HTML content
  const sanitizedHtml = useMemo(() => {
    if (!email.bodyHtml) return null
    const sanitized = sanitizeEmailHtml(email.bodyHtml)
    return proxyExternalImages(sanitized)
  }, [email.bodyHtml])

  // Format date
  const formattedDate = useMemo(() => {
    try {
      return format(new Date(email.date), 'MMM d, yyyy h:mm a')
    } catch {
      return email.date
    }
  }, [email.date])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/30">
        {onBack && (
          <Button
            variant="ghost"
            size="sm"
            className="h-8 -ml-2"
            onClick={onBack}
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
        )}

        <EmailActions
          email={email}
          onReply={onReply}
          onForward={onForward}
          onArchive={onArchive}
          onDelete={onDelete}
          compact
        />
      </div>

      {/* Email content */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          {/* Subject */}
          <h2 className="text-lg font-semibold mb-2">{email.subject}</h2>

          {/* Sender info */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="font-medium">
                {email.from.name || email.from.email}
              </div>
              <div className="text-sm text-muted-foreground">
                To: {email.to.map(formatAddress).join(', ')}
              </div>
              {email.cc && email.cc.length > 0 && (
                <div className="text-sm text-muted-foreground">
                  Cc: {email.cc.map(formatAddress).join(', ')}
                </div>
              )}
            </div>
            <div className="text-sm text-muted-foreground">
              {formattedDate}
            </div>
          </div>

          {/* Labels */}
          {email.labels && email.labels.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-4">
              {email.labels.map((label) => (
                <Badge key={label} variant="secondary" className="text-xs">
                  {label}
                </Badge>
              ))}
            </div>
          )}

          <Separator className="my-4" />

          {/* Body content */}
          <div className="email-body">
            {sanitizedHtml ? (
              <div
                className="prose prose-sm max-w-none dark:prose-invert"
                dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
              />
            ) : (
              <pre className="whitespace-pre-wrap font-sans text-sm">
                {email.body}
              </pre>
            )}
          </div>

          {/* Attachments */}
          {email.attachments && email.attachments.length > 0 && (
            <div className="mt-6">
              <Separator className="mb-4" />
              <div className="flex items-center gap-2 mb-3">
                <Paperclip className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">
                  {email.attachments.length} attachment
                  {email.attachments.length !== 1 ? 's' : ''}
                </span>
              </div>

              <div className="grid gap-2">
                {email.attachments.map((attachment) => (
                  <AttachmentItem key={attachment.id} attachment={attachment} />
                ))}
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

/**
 * Single attachment display
 */
function AttachmentItem({ attachment }: { attachment: EmailAttachment }) {
  const handleDownload = () => {
    if (attachment.downloadUrl) {
      window.open(attachment.downloadUrl, '_blank')
    }
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-3 py-2 rounded-md',
        'bg-muted/30 hover:bg-muted/50 transition-colors'
      )}
    >
      <Paperclip className="h-4 w-4 text-muted-foreground flex-shrink-0" />

      <div className="flex-1 min-w-0">
        <div className="text-sm truncate">{attachment.filename}</div>
        <div className="text-xs text-muted-foreground">
          {attachment.mimeType} - {formatFileSize(attachment.size)}
        </div>
      </div>

      {attachment.downloadUrl && (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 flex-shrink-0"
          onClick={handleDownload}
          title="Download"
        >
          <Download className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
