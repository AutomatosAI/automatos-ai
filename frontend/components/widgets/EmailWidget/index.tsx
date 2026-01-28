'use client'

/**
 * EmailWidget Component for PRD-38.2 Extended Widgets
 *
 * Displays, composes, and manages emails from Composio integrations
 * (Gmail, Outlook, etc.)
 */

import { useState, useCallback } from 'react'
import { Mail, Pencil, RefreshCw } from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import { EmailList } from './EmailList'
import { EmailViewer } from './EmailViewer'
import { EmailComposer } from './EmailComposer'
import type {
  WidgetBaseProps,
  EmailWidgetData,
  EmailSummary,
  EmailFull,
  EmailDraft,
  WidgetDefinition,
} from '../types'
import { toast } from 'sonner'

type ViewMode = 'list' | 'view' | 'compose'

export function EmailWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
  onRefresh,
}: WidgetBaseProps<EmailWidgetData>) {
  // View state
  const [viewMode, setViewMode] = useState<ViewMode>(data.mode || 'list')
  const [selectedEmail, setSelectedEmail] = useState<EmailFull | undefined>(
    data.email
  )
  const [replyToEmail, setReplyToEmail] = useState<EmailFull | undefined>(
    data.replyTo
  )

  // Handle email selection from list
  const handleEmailSelect = useCallback((email: EmailSummary) => {
    // In a real implementation, we'd fetch the full email content
    // For now, we'll use the summary data
    const fullEmail: EmailFull = {
      ...email,
      body: email.snippet,
    }
    setSelectedEmail(fullEmail)
    setViewMode('view')
  }, [])

  // Handle back to list
  const handleBack = useCallback(() => {
    setSelectedEmail(undefined)
    setViewMode('list')
  }, [])

  // Handle compose new email
  const handleCompose = useCallback(() => {
    setReplyToEmail(undefined)
    setViewMode('compose')
  }, [])

  // Handle reply to email
  const handleReply = useCallback((email: EmailFull) => {
    setReplyToEmail(email)
    setViewMode('compose')
  }, [])

  // Handle forward email
  const handleForward = useCallback((email: EmailFull) => {
    // For forward, we don't set replyTo, but we could pass content differently
    setReplyToEmail(undefined)
    setViewMode('compose')
    toast.info('Forward functionality will be implemented with backend integration')
  }, [])

  // Handle send email
  const handleSend = useCallback(async (draft: EmailDraft) => {
    // In a real implementation, this would call the Composio email API
    toast.success('Email sent (simulated)')
    setViewMode('list')
    setReplyToEmail(undefined)
  }, [])

  // Handle discard draft
  const handleDiscard = useCallback(() => {
    setViewMode(selectedEmail ? 'view' : 'list')
    setReplyToEmail(undefined)
  }, [selectedEmail])

  // Handle archive
  const handleArchive = useCallback((emailId: string) => {
    toast.info('Archive functionality will be implemented with backend integration')
  }, [])

  // Handle delete
  const handleDelete = useCallback((emailId: string) => {
    toast.info('Delete functionality will be implemented with backend integration')
  }, [])

  // Get title based on mode and content
  const displayTitle = viewMode === 'compose'
    ? (replyToEmail ? 'Reply' : 'New Email')
    : (selectedEmail?.subject || title || 'Email')

  return (
    <WidgetBase
      title={displayTitle}
      icon={<Mail className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onRefresh={viewMode === 'list' ? onRefresh : undefined}
      customActions={
        viewMode === 'list'
          ? [
              {
                label: 'Compose',
                icon: <Pencil className="h-4 w-4 mr-2" />,
                onClick: handleCompose,
              },
            ]
          : undefined
      }
    >
      <div className="flex flex-col h-full">
        {/* List mode */}
        {viewMode === 'list' && (
          <EmailList
            emails={data.emails || []}
            selectedId={selectedEmail?.id}
            onSelect={handleEmailSelect}
            onRefresh={onRefresh}
            isLoading={isLoading}
          />
        )}

        {/* View mode */}
        {viewMode === 'view' && selectedEmail && (
          <EmailViewer
            email={selectedEmail}
            onBack={handleBack}
            onReply={handleReply}
            onForward={handleForward}
            onArchive={handleArchive}
            onDelete={handleDelete}
          />
        )}

        {/* Compose mode */}
        {viewMode === 'compose' && (
          <EmailComposer
            replyTo={replyToEmail}
            onSend={handleSend}
            onDiscard={handleDiscard}
          />
        )}
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const EmailWidgetDef: WidgetDefinition<EmailWidgetData> = {
  type: 'email',
  displayName: 'Email',
  description: 'View, compose, and manage emails',
  icon: Mail,
  component: EmailWidget,
  defaultSize: { width: 6, height: 5 },
  minSize: { width: 4, height: 3 },
  capabilities: ['refreshable', 'fullscreen'],
}

// Register the widget
registerWidget(EmailWidgetDef)
