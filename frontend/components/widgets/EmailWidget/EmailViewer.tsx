'use client'

/**
 * EmailViewer Component for PRD-38.2 Extended Widgets
 *
 * Full email display with HTML support and XSS prevention via DOMPurify.
 * Modern, clean design inspired by premium email clients.
 * SECURITY: All HTML content is sanitized before rendering.
 */

import { useMemo } from 'react'
import DOMPurify, { type Config as DOMPurifyConfig } from 'dompurify'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  ArrowLeft,
  Paperclip,
  Download,
  ExternalLink,
  Calendar,
  User,
  Users,
  Mail,
  Reply,
  Forward,
  MoreHorizontal,
  FileText,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { format, isToday, isYesterday } from 'date-fns'
import { EmailActions } from './EmailActions'
import type { EmailFull, EmailAddress, EmailAttachment } from '../types'

/**
 * DOMPurify configuration for email HTML sanitization
 * Strict whitelist to prevent XSS attacks
 */
const DOMPURIFY_CONFIG: DOMPurifyConfig = {
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

/**
 * Get sender initials for avatar
 */
function getSenderInitials(from: EmailAddress): string {
  const name = from.name || from.email
  const parts = name.split(/[\s@]/)
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase()
  }
  return name.substring(0, 2).toUpperCase()
}

/**
 * Generate consistent color for sender avatar
 */
function getAvatarColor(email: string): string {
  const colors = [
    'bg-blue-500', 'bg-green-500', 'bg-agent',
    'bg-orange-500', 'bg-pink-500', 'bg-teal-500',
    'bg-indigo-500', 'bg-rose-500', 'bg-cyan-500'
  ]
  let hash = 0
  for (let i = 0; i < email.length; i++) {
    hash = email.charCodeAt(i) + ((hash << 5) - hash)
  }
  return colors[Math.abs(hash) % colors.length]
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

  // Format date - smart formatting
  const formattedDate = useMemo(() => {
    try {
      const date = new Date(email.date)
      if (isToday(date)) {
        return `Today at ${format(date, 'h:mm a')}`
      } else if (isYesterday(date)) {
        return `Yesterday at ${format(date, 'h:mm a')}`
      }
      return format(date, "EEEE, MMMM d, yyyy 'at' h:mm a")
    } catch {
      return email.date
    }
  }, [email.date])

  return (
    <div className="flex flex-col h-full">
      {/* Header with navigation and actions */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/40 bg-muted/20">
        <div className="flex items-center gap-2">
          {onBack && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-2 -ml-2"
              onClick={onBack}
            >
              <ArrowLeft className="h-4 w-4 mr-1" />
              <span className="text-sm">Back</span>
            </Button>
          )}
        </div>

        <div className="flex items-center gap-1">
          {onReply && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-3"
              onClick={() => onReply(email)}
            >
              <Reply className="h-4 w-4 mr-1.5" />
              Reply
            </Button>
          )}
          {onForward && (
            <Button
              variant="ghost"
              size="sm"
              className="h-8 px-3"
              onClick={() => onForward(email)}
            >
              <Forward className="h-4 w-4 mr-1.5" />
              Forward
            </Button>
          )}
          <EmailActions
            email={email}
            onArchive={onArchive}
            onDelete={onDelete}
            compact
          />
        </div>
      </div>

      {/* Email content */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Subject */}
          <div>
            <h1 className="text-xl font-semibold leading-tight text-foreground">
              {email.subject || '(No Subject)'}
            </h1>
            {email.labels && email.labels.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {email.labels.map((label) => (
                  <Badge
                    key={label}
                    variant="secondary"
                    className="text-xs px-2 py-0.5 bg-muted/80"
                  >
                    {label}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Sender card */}
          <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30 border border-border/40">
            {/* Avatar */}
            <div className={cn(
              "flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-white text-sm font-medium",
              getAvatarColor(email.from.email)
            )}>
              {getSenderInitials(email.from)}
            </div>

            {/* Sender info */}
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="font-semibold text-sm">
                    {email.from.name || email.from.email}
                  </div>
                  {email.from.name && (
                    <div className="text-xs text-muted-foreground">
                      {email.from.email}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Calendar className="h-3 w-3" />
                  {formattedDate}
                </div>
              </div>

              {/* Recipients */}
              <div className="flex flex-col gap-0.5 text-xs text-muted-foreground">
                <div className="flex items-start gap-1.5">
                  <span className="text-muted-foreground/70 min-w-[24px]">To:</span>
                  <span className="flex-1 break-all">
                    {email.to.map(formatAddress).join(', ')}
                  </span>
                </div>
                {email.cc && email.cc.length > 0 && (
                  <div className="flex items-start gap-1.5">
                    <span className="text-muted-foreground/70 min-w-[24px]">Cc:</span>
                    <span className="flex-1 break-all">
                      {email.cc.map(formatAddress).join(', ')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-border/30" />

          {/* Body content */}
          <div className="email-body min-h-[100px]">
            {sanitizedHtml ? (
              <div
                className={cn(
                  "prose prose-sm max-w-none dark:prose-invert",
                  "prose-headings:font-semibold prose-headings:text-foreground",
                  "prose-p:text-foreground/90 prose-p:leading-relaxed",
                  "prose-a:text-primary prose-a:no-underline hover:prose-a:underline",
                  "prose-blockquote:border-l-primary/50 prose-blockquote:text-muted-foreground",
                  "prose-code:bg-muted prose-code:px-1 prose-code:rounded",
                  "prose-pre:bg-muted prose-pre:border prose-pre:border-border/50"
                )}
                dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
              />
            ) : email.body ? (
              <pre className="whitespace-pre-wrap font-sans text-sm text-foreground/90 leading-relaxed m-0">
                {email.body}
              </pre>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <FileText className="h-12 w-12 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground">
                  No email content available
                </p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  The email body could not be loaded
                </p>
              </div>
            )}
          </div>

          {/* Attachments */}
          {email.attachments && email.attachments.length > 0 && (
            <div className="pt-2">
              <div className="border-t border-border/30 pt-4" />
              <div className="flex items-center gap-2 mb-3">
                <Paperclip className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">
                  {email.attachments.length} attachment
                  {email.attachments.length !== 1 ? 's' : ''}
                </span>
              </div>

              <div className="grid gap-2 sm:grid-cols-2">
                {email.attachments.map((attachment) => (
                  <AttachmentCard key={attachment.id} attachment={attachment} />
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
 * Attachment card component
 */
function AttachmentCard({ attachment }: { attachment: EmailAttachment }) {
  const handleDownload = () => {
    if (attachment.downloadUrl) {
      window.open(attachment.downloadUrl, '_blank')
    }
  }

  // Get file icon based on mime type
  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) return '🖼️'
    if (mimeType.startsWith('video/')) return '🎬'
    if (mimeType.startsWith('audio/')) return '🎵'
    if (mimeType.includes('pdf')) return '📄'
    if (mimeType.includes('spreadsheet') || mimeType.includes('excel')) return '📊'
    if (mimeType.includes('presentation') || mimeType.includes('powerpoint')) return '📽️'
    if (mimeType.includes('document') || mimeType.includes('word')) return '📝'
    if (mimeType.includes('zip') || mimeType.includes('compressed')) return '📦'
    return '📎'
  }

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-3 py-2.5 rounded-lg',
        'bg-muted/40 border border-border/40',
        'hover:bg-muted/60 hover:border-border/60 transition-colors',
        'group cursor-pointer'
      )}
      onClick={handleDownload}
      role="button"
      tabIndex={0}
    >
      <div className="text-xl flex-shrink-0">
        {getFileIcon(attachment.mimeType)}
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate text-foreground/90">
          {attachment.filename}
        </div>
        <div className="text-xs text-muted-foreground">
          {formatFileSize(attachment.size)}
        </div>
      </div>

      {attachment.downloadUrl && (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={(e) => {
            e.stopPropagation()
            handleDownload()
          }}
          title="Download"
        >
          <Download className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
