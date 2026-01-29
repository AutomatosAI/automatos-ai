'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import { Bot, ArrowDown, Database, GitBranch, Wrench } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChat } from '@/lib/chat/hooks'
import { Message } from './message'
import { MultimodalInput } from './multimodal-input'
import { ArtifactViewer } from './artifact-viewer'
import { generateTitle } from '@/lib/utils'
import { updateChatTitle } from '@/lib/chat/api'
import type { ChatMessage, VisibilityType, Artifact, AppUsage, CodeSnippet, DocumentReference, DatabaseResult } from '@/types'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

// Widget Architecture (PRD-38.1)
import { useWorkspaceStore } from '@/stores/workspace-store'
import { Canvas } from '@/components/workspace'
import type { CodeWidgetData, DataWidgetData, DocumentWidgetData } from '@/components/widgets/types'

// PRD-40: Dynamic Tool Suggestions
import { ToolSuggestionBar } from '@/components/suggestions/ToolSuggestionBar'
import type { SuggestionResponse } from '@/components/suggestions/types'
import { analytics } from '@/lib/analytics'

export interface ChatProps {
  id: string
  initialMessages?: ChatMessage[]
  initialChatModel?: string
  initialVisibilityType?: VisibilityType
  isReadonly?: boolean
  autoResume?: boolean
  initialLastContext?: AppUsage
}

export function Chat({
  id,
  initialMessages = [],
  initialChatModel = 'gpt-4',
  initialVisibilityType = 'private',
  isReadonly = false,
  autoResume = false,
  initialLastContext,
}: ChatProps) {
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null)
  const [isArtifactViewerVisible, setIsArtifactViewerVisible] = useState(false)
  const [currentModelId, setCurrentModelId] = useState(initialChatModel)

  // Widget Architecture (PRD-38.1) - workspace store
  const widgetIds = useWorkspaceStore((s) => s.widgetIds)
  const addWidget = useWorkspaceStore((s) => s.addWidget)
  const clearWidgets = useWorkspaceStore((s) => s.clearWidgets)
  const hasWidgets = widgetIds.length > 0

  // Handler to close canvas and reset all overlay states
  const handleCloseCanvas = useCallback(() => {
    clearWidgets()
    setIsArtifactViewerVisible(false)
    setSelectedArtifact(null)
  }, [clearWidgets])

  const [canvasWidth, setCanvasWidth] = useState(800)
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)
  const [visibilityType, setVisibilityType] = useState<VisibilityType>(initialVisibilityType)
  const [usage, setUsage] = useState<AppUsage | undefined>(initialLastContext)
  const [hasGeneratedTitle, setHasGeneratedTitle] = useState(false)

  // PRD-40: Dynamic Tool Suggestions state
  const [activeTool, setActiveTool] = useState<string | null>(null)
  const [toolSuggestions, setToolSuggestions] = useState<string[]>([])
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false)

  const messagesContainerRef = useRef<HTMLDivElement | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1920)

  // Track window width
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const [activeChatId, setActiveChatId] = useState(id)

  const { messages, setMessages, sendMessage, status, stop, reload } = useChat({
    id: activeChatId,
    initialMessages,
    selectedModelId: currentModelId,
    selectedAgentId,
    onData: (dataPart) => {
      if (dataPart.type === 'data-usage') {
        setUsage(dataPart.data)
      }

      // PRD-38.1: Auto-create widgets when tool-data arrives
      if (dataPart.type === 'tool-data' && dataPart.data) {
        const toolData = dataPart.data

        // Create widgets for database results
        if (toolData.database_results && Array.isArray(toolData.database_results)) {
          toolData.database_results.forEach((dbResult: any) => {
            const columns = dbResult.columns && dbResult.columns.length > 0
              ? dbResult.columns
              : (dbResult.data && dbResult.data.length > 0 ? Object.keys(dbResult.data[0]) : [])

            const charts = dbResult.pandas_ai?.charts?.map((chart: any) => ({
              filename: chart.filename || 'chart.png',
              mimeType: chart.mime_type || 'image/png',
              base64: chart.base64,
            }))

            addWidget({
              type: 'data',
              title: `${dbResult.database || 'Query'} Result`,
              data: {
                columns,
                rows: dbResult.data || [],
                sql: dbResult.sql,
                database: dbResult.database,
                rowCount: dbResult.row_count || dbResult.data?.length || 0,
                executionTime: dbResult.execution_time_ms,
                charts,
                pandasAiSummary: dbResult.pandas_ai?.summary,
                explanation: dbResult.explanation,
                rephrased_query: dbResult.rephrased_query,
                follow_up_questions: dbResult.follow_up_questions,
              },
              metadata: {
                source: { type: 'tool', name: 'smart_query_database', provider: 'nl2sql' },
                createdAt: new Date(),
                conversationId: id,
              },
              state: 'ready',
              createdAt: new Date().toISOString(),
            } as any)
          })
        }

        // Create widgets for documents
        if (toolData.documents && Array.isArray(toolData.documents)) {
          toolData.documents.forEach((doc: any) => {
            const initialContent = doc.full_content || doc.excerpt || doc.preview || doc.content || ''
            const chunks = doc.chunks?.map((chunk: any) => ({
              content: chunk.content || chunk.excerpt || '',
              excerpt: chunk.excerpt,
              similarity: chunk.similarity,
              chunkIndex: chunk.chunk_index,
            }))

            addWidget({
              type: 'document',
              title: doc.filename || doc.title || 'Document',
              data: {
                content: initialContent,
                format: 'markdown',
                filename: doc.filename,
                filePath: doc.file_path,
                similarity: doc.similarity,
                relevance: doc.relevance,
                chunkCount: doc.chunk_count,
                chunks,
                downloadUrl: doc.download_url,
                hasFullContent: doc.has_full_content ?? false,
              },
              metadata: {
                source: { type: 'tool', name: 'search_knowledge', provider: 'rag' },
                createdAt: new Date(),
                conversationId: id,
              },
              state: doc.has_full_content ? 'ready' : 'ready',
              createdAt: new Date().toISOString(),
            } as any)
          })
        }

        // Create widgets for code snippets
        if (toolData.code_snippets && Array.isArray(toolData.code_snippets)) {
          toolData.code_snippets.forEach((snippet: any) => {
            addWidget({
              type: 'code',
              title: snippet.symbol_name || snippet.file_path || 'Code Snippet',
              data: {
                code: snippet.code,
                language: snippet.language || 'python',
                filePath: snippet.file_path,
                lineNumber: snippet.line_number,
                explanation: snippet.explanation,
                symbolName: snippet.symbol_name,
              },
              metadata: {
                source: { type: 'tool', name: 'search_codebase', provider: 'codegraph' },
                createdAt: new Date(),
                conversationId: id,
              },
              state: 'ready',
              createdAt: new Date().toISOString(),
            } as any)
          })
        }

        // Create widget for emails (Composio Gmail/Outlook)
        // EmailWidget expects data.emails array for list mode
        console.log('[Widget] tool-data received:', Object.keys(toolData))
        if (toolData.emails && Array.isArray(toolData.emails) && toolData.emails.length > 0) {
          console.log('[Widget] Creating email widget with', toolData.emails.length, 'emails')

          // Helper to parse email address string like "John Doe <john@example.com>" into { name, email }
          const parseEmailAddress = (addr: any): { email: string; name?: string } => {
            if (!addr) return { email: 'unknown' }
            if (typeof addr === 'object' && addr.email) return addr
            if (typeof addr !== 'string') return { email: String(addr) }

            // Match "Name <email>" or just "email"
            const match = addr.match(/^(.+?)\s*<([^>]+)>$/)
            if (match) {
              return { name: match[1].trim(), email: match[2].trim() }
            }
            return { email: addr.trim() }
          }

          // Helper to parse array of email addresses
          const parseEmailAddresses = (addrs: any): { email: string; name?: string }[] => {
            if (!addrs) return []
            if (typeof addrs === 'string') return [parseEmailAddress(addrs)]
            if (Array.isArray(addrs)) return addrs.map(parseEmailAddress)
            return [parseEmailAddress(addrs)]
          }

          // Transform emails to the format EmailWidget expects
          const emailList = toolData.emails.map((email: any) => ({
            id: email.id || email.messageId || crypto.randomUUID(),
            threadId: email.threadId,
            subject: email.subject || '(No Subject)',
            from: parseEmailAddress(email.from || email.sender),
            to: parseEmailAddresses(email.to || email.recipients),
            date: email.date || email.receivedAt || new Date().toISOString(),
            snippet: email.snippet || email.body?.substring(0, 200) || '',
            body: email.body || email.content || email.snippet || '',
            bodyHtml: email.bodyHtml,  // HTML content for rich rendering
            isRead: email.isRead ?? true,
            hasAttachments: email.hasAttachments || (email.attachments?.length > 0),
            attachments: email.attachments,
            labels: email.labels || email.tags,
          }))

          addWidget({
            type: 'email',
            title: `Emails (${emailList.length})`,
            data: {
              mode: 'list',
              emails: emailList,
            },
            metadata: {
              source: { type: 'tool', name: 'composio_execute', provider: 'gmail' },
              createdAt: new Date(),
              conversationId: id,
            },
            state: 'ready',
            createdAt: new Date().toISOString(),
          } as any)
        }

        // Create widget for terminal/command output
        if (toolData.terminal_output) {
          const term = toolData.terminal_output
          console.log('[Widget] Creating terminal widget for command:', term.command)
          addWidget({
            type: 'terminal',
            title: term.command ? `$ ${term.command.substring(0, 30)}${term.command.length > 30 ? '...' : ''}` : 'Terminal',
            data: {
              command: term.command || '',
              output: term.output || '',
              stderr: term.stderr || '',
              exitCode: term.exitCode ?? 0,
              executionTime: term.executionTime,
              workingDirectory: term.workingDirectory,
            },
            metadata: {
              source: { type: 'tool', name: 'execute_command', provider: 'shell' },
              createdAt: new Date(),
              conversationId: id,
            },
            state: 'ready',
            createdAt: new Date().toISOString(),
          } as any)
        }
      }
    },
    onChatIdUpdate: (newChatId) => {
      setActiveChatId(newChatId)
    },
  })

  const regenerate = () => reload()

  // Track scroll - ensure listener is always attached to the current container
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    const checkScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container
      setIsAtBottom(scrollHeight - scrollTop - clientHeight < 50)
    }

    container.addEventListener('scroll', checkScroll)
    // Run once to initialize state
    checkScroll()

    return () => {
      container.removeEventListener('scroll', checkScroll)
    }
    // Re-attach when the artifact viewer visibility toggles, since the DOM node can change
  }, [isArtifactViewerVisible])

  // Auto-scroll
  useEffect(() => {
    if (status === 'streaming') {
      requestAnimationFrame(() => {
        messagesContainerRef.current?.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: 'smooth',
        })
      })
    }
  }, [status])

  // Generate title
  useEffect(() => {
    if (!hasGeneratedTitle && messages.length >= 2 && activeChatId) {
      const firstUserMessage = messages.find(m => m.role === 'user')
      if (firstUserMessage && firstUserMessage.parts) {
        const textPart = firstUserMessage.parts.find(p => p.type === 'text')
        if (textPart && 'text' in textPart) {
          const title = generateTitle(textPart.text)
          updateChatTitle(activeChatId, title).catch(console.error)
          setHasGeneratedTitle(true)
        }
      }
    }
  }, [messages, activeChatId, hasGeneratedTitle])

  // PRD-40: Tool icon click handler
  const handleToolIconClick = useCallback(async (appName: string) => {
    // Track tool icon click
    analytics.track('tool_icon_clicked', {
      app: appName,
      location: 'chat',
    })

    // Toggle off if clicking same tool
    if (activeTool === appName) {
      setActiveTool(null)
      setToolSuggestions([])
      return
    }

    setActiveTool(appName)
    setIsLoadingSuggestions(true)

    try {
      const response = await fetch(`/api/tools/${appName}/suggestions`)
      if (!response.ok) {
        throw new Error(`Failed to fetch suggestions: ${response.statusText}`)
      }
      const data: SuggestionResponse = await response.json()
      setToolSuggestions(data.suggestions || [])

      // Track suggestions loaded
      analytics.track('suggestions_loaded', {
        app: appName,
        source: data.source,
        count: data.suggestions.length,
      })
    } catch (error) {
      console.error('Failed to fetch tool suggestions:', error)
      toast.error(`Failed to load suggestions for ${appName}`)
      setToolSuggestions([])
      setActiveTool(null)
    } finally {
      setIsLoadingSuggestions(false)
    }
  }, [activeTool])

  // PRD-40: Suggestion click handler
  const handleSuggestionClick = useCallback((suggestion: string) => {
    // Track suggestion click
    analytics.track('suggestion_clicked', {
      app: activeTool || 'unknown',
      suggestion: suggestion,
      location: 'chat',
    })

    // Send the suggestion as a message
    sendMessage(suggestion)
    // Close suggestions after use
    setActiveTool(null)
    setToolSuggestions([])
  }, [sendMessage, activeTool])

  // Handle selections
  const handleArtifactSelect = useCallback((artifact: Artifact) => {
    setSelectedArtifact(artifact)
    setIsArtifactViewerVisible(true)
  }, [])

  const handleCodeSelect = useCallback((code: CodeSnippet) => {
    // PRD-38.1: Create widget instead of artifact
    const codeTitle = code.symbol_name || code.file_path || 'Code Snippet'

    // Check if a widget for this code already exists - if so, just activate it
    const existingWidgets = useWorkspaceStore.getState().widgets
    const existingWidget = Object.values(existingWidgets).find(
      w => w.type === 'code' && (
        (w.data as any)?.filePath === code.file_path &&
        (w.data as any)?.symbolName === code.symbol_name
      )
    )
    if (existingWidget) {
      useWorkspaceStore.getState().setActiveWidget(existingWidget.id)
      return
    }

    const widgetData: Omit<import('@/components/widgets/types').Widget<CodeWidgetData>, 'id'> = {
      type: 'code',
      title: codeTitle,
      data: {
        code: code.code,
        language: code.language || 'python',
        filePath: code.file_path,
        lineNumber: code.line_number,
        explanation: code.explanation,
        symbolName: code.symbol_name,
      },
      metadata: {
        source: { type: 'tool', name: 'search_codebase', provider: 'codegraph' },
        createdAt: new Date(),
        conversationId: id,
      },
      state: 'ready',
      createdAt: new Date().toISOString(),
    }
    addWidget(widgetData)
  }, [id, addWidget])

  const handleDocumentSelect = useCallback((doc: DocumentReference) => {
    // PRD-38.1: Create widget instead of artifact
    const initialContent = doc.full_content || doc.excerpt || doc.preview || doc.content || ''
    const docFilename = doc.filename || doc.title || 'Document'

    // Check if a widget for this document already exists - if so, just activate it
    const existingWidgets = useWorkspaceStore.getState().widgets
    const existingWidget = Object.values(existingWidgets).find(
      w => w.type === 'document' && (w.data as any)?.filename === docFilename
    )
    if (existingWidget) {
      useWorkspaceStore.getState().setActiveWidget(existingWidget.id)
      return
    }

    // Transform chunks to widget format
    const chunks = doc.chunks?.map((chunk: any) => ({
      content: chunk.content || chunk.excerpt || '',
      excerpt: chunk.excerpt,
      similarity: chunk.similarity,
      chunkIndex: chunk.chunk_index,
    }))

    const widgetData: Omit<import('@/components/widgets/types').Widget<DocumentWidgetData>, 'id'> = {
      type: 'document',
      title: docFilename,
      data: {
        content: initialContent,
        format: 'markdown',
        filename: docFilename,
        filePath: doc.file_path,
        similarity: doc.similarity,
        relevance: doc.relevance,
        chunkCount: doc.chunk_count,
        chunks,
        downloadUrl: doc.download_url,
        hasFullContent: doc.has_full_content ?? false,
      },
      metadata: {
        source: { type: 'tool', name: 'search_knowledge', provider: 'rag' },
        createdAt: new Date(),
        conversationId: id,
      },
      state: doc.has_full_content ? 'ready' : 'loading',
      createdAt: new Date().toISOString(),
    }

    const widgetId = addWidget(widgetData)

    // Fetch full content if needed
    if (doc.id && !doc.has_full_content) {
      apiClient.request(`/api/documents/${doc.id}/content`)
        .then((data: any) => {
          const fullContent = Array.isArray(data?.chunks)
            ? data.chunks.map((chunk: any) => chunk?.content ?? '').filter(Boolean).join('\n\n')
            : initialContent

          // Update widget with full content
          useWorkspaceStore.getState().updateWidget(widgetId, {
            data: {
              ...widgetData.data,
              content: fullContent || initialContent,
              chunkCount: data?.chunk_count ?? doc.chunk_count,
              hasFullContent: true,
            },
            state: 'ready',
          })
        })
        .catch((error) => {
          console.error('Failed to load document content', error)
          toast.error('Failed to load full document content')
          useWorkspaceStore.getState().updateWidget(widgetId, {
            state: 'error',
            error: { message: 'Failed to load full document' },
          })
        })
    }
  }, [id, addWidget])

  const handleDatabaseSelect = useCallback((db: DatabaseResult) => {
    // PRD-38.1: Create widget instead of artifact
    const columns = db.columns && db.columns.length > 0
      ? db.columns
      : (db.data && db.data.length > 0 ? Object.keys(db.data[0]) : [])

    // Transform pandas_ai charts to widget format
    const charts = db.pandas_ai?.charts?.map((chart: any) => ({
      filename: chart.filename || 'chart.png',
      mimeType: chart.mime_type || 'image/png',
      base64: chart.base64,
    }))

    const widgetData: Omit<import('@/components/widgets/types').Widget<DataWidgetData>, 'id'> = {
      type: 'data',
      title: `${db.database || 'Query'} Result`,
      data: {
        columns,
        rows: db.data || [],
        sql: db.sql,
        database: db.database,
        rowCount: db.row_count || db.data?.length || 0,
        executionTime: db.execution_time_ms,
        charts,
        pandasAiSummary: db.pandas_ai?.summary,
        explanation: db.explanation,
        rephrased_query: db.rephrased_query,
        follow_up_questions: db.follow_up_questions,
      },
      metadata: {
        source: { type: 'tool', name: 'smart_query_database', provider: 'nl2sql' },
        createdAt: new Date(),
        conversationId: id,
      },
      state: 'ready',
      createdAt: new Date().toISOString(),
    }

    addWidget(widgetData)
  }, [id, addWidget])

  const isTyping = status === 'streaming'
  const hasSentMessage = messages.length > 0

  const suggestedActions = [
    "Using the workflow_executions table, break down workflow completions and failures by day over the last 14 days and plot the trend",
    "From the document_usage table, summarize daily document search volume and average response time for the past 7 days with a chart",
    "Plot hourly CPU and memory utilization over the last 24 hours using the system_metrics table",
    "Highlight the most frequent CodeGraph queries this week by counting entries in codegraph_query_logs and visualize their counts",
  ]

  const quickLinks = [
    { label: 'Create an Agent', href: '/agents', icon: Bot },
    { label: 'Knowledge Base', href: '/documents', icon: Database },
    { label: 'Create a Workflow', href: '/workflows', icon: GitBranch },
    { label: 'Edit Tools', href: '/tools', icon: Wrench },
  ] as const

  const showWelcomeCard = !hasSentMessage && !isTyping

  return (
    <>
      {/* PRD-38.1: Widget Canvas Layout - shows when widgets exist */}
      {hasWidgets && (
        <div className="fixed top-0 left-0 z-50 flex h-screen w-screen flex-row bg-background">
          {/* Chat Column - LEFT 400px */}
          <div className="relative h-screen w-[400px] shrink-0 bg-muted dark:bg-background border-r border-border/50">
            <div className="flex h-full flex-col items-center justify-between">
              {/* Messages */}
              <div
                ref={messagesContainerRef}
                className="flex-1 w-full overflow-y-scroll overscroll-contain"
                style={{ overflowAnchor: 'none' }}
              >
                <div className="mx-auto flex min-w-0 flex-col gap-4 px-4 py-4 md:gap-6">
                  {messages.map((message, index) => (
                    <Message
                      key={message.id}
                      chatId={id}
                      message={message}
                      isLoading={isTyping && index === messages.length - 1}
                      setMessages={setMessages}
                      regenerate={regenerate}
                      isReadonly={isReadonly}
                      onArtifactSelect={handleArtifactSelect}
                      onCodeSelect={handleCodeSelect}
                      onDocumentSelect={handleDocumentSelect}
                      onDatabaseSelect={handleDatabaseSelect}
                    />
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              {/* Input at bottom of chat column */}
              {!isReadonly && (
                <div className="relative flex w-full flex-row items-end gap-2 px-4 pb-4">
                  <MultimodalInput
                    chatId={id}
                    status={status}
                    stop={stop}
                    sendMessage={sendMessage}
                    selectedModelId={currentModelId}
                    onModelChange={setCurrentModelId}
                    selectedAgentId={selectedAgentId}
                    onAgentChange={setSelectedAgentId}
                    selectedVisibilityType={visibilityType}
                    usage={usage}
                  />
                </div>
              )}
            </div>
          </div>

          {/* PRD-38.1: Widget Canvas - RIGHT side */}
          <div className="relative z-10 flex h-full flex-1 flex-col bg-background">
            <div className="flex-1 overflow-hidden">
              <Canvas width={canvasWidth} onClose={handleCloseCanvas} />
            </div>
          </div>
        </div>
      )}

      {/* Legacy Artifact Viewer overlay - for backward compatibility */}
      <AnimatePresence>
        {isArtifactViewerVisible && selectedArtifact && !hasWidgets && (
          <motion.div
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { delay: 0.4 } }}
            className="fixed top-0 left-0 z-50 flex h-screen w-screen flex-row bg-transparent"
          >
            <motion.div className="fixed h-screen bg-background w-full" />
            <div className="relative h-screen w-[400px] shrink-0 bg-muted dark:bg-background">
              <div className="flex h-full flex-col">
                <div className="flex-1 w-full overflow-y-scroll">
                  <div className="flex flex-col gap-4 px-4 py-4">
                    {messages.map((message, index) => (
                      <Message
                        key={message.id}
                        chatId={id}
                        message={message}
                        isLoading={isTyping && index === messages.length - 1}
                        setMessages={setMessages}
                        regenerate={regenerate}
                        isReadonly={isReadonly}
                        onArtifactSelect={handleArtifactSelect}
                        onCodeSelect={handleCodeSelect}
                        onDocumentSelect={handleDocumentSelect}
                        onDatabaseSelect={handleDatabaseSelect}
                      />
                    ))}
                  </div>
                </div>
                {!isReadonly && (
                  <div className="px-4 pb-4">
                    <MultimodalInput
                      chatId={id}
                      status={status}
                      stop={stop}
                      sendMessage={sendMessage}
                      selectedModelId={currentModelId}
                      onModelChange={setCurrentModelId}
                      selectedAgentId={selectedAgentId}
                      onAgentChange={setSelectedAgentId}
                      selectedVisibilityType={visibilityType}
                      usage={usage}
                    />
                  </div>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-scroll bg-background">
              <ArtifactViewer
                artifact={selectedArtifact}
                onClose={() => {
                  setIsArtifactViewerVisible(false)
                  setSelectedArtifact(null)
                }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Normal chat view - NO widgets */}
      {!hasWidgets && !isArtifactViewerVisible && (
        <div className="relative flex flex-col bg-transparent" style={{ height: '100%', width: '100%', minHeight: 0 }}>
          {/* Incredible-style centered welcome card (empty state) */}
          {showWelcomeCard && (
            <div className="flex flex-1 flex-col items-center justify-center px-4 py-10 md:py-16">
              {/* Hero header (dark-mode version of marketing header) */}
              <div className="w-full max-w-5xl text-center mb-8 md:mb-10">
                <div className="text-xs uppercase tracking-[0.35em] text-muted-foreground dark:text-orange-200/70">
                  Future-Ready AI Agency
                </div>
                <h1 className="mt-3 text-4xl md:text-6xl font-semibold tracking-tight text-foreground dark:text-white leading-[1.05]">
                  <span className="block">AI Services That</span>
                  <span className="block mt-2">
                    <span className="gradient-text">[Elevate]</span>{' '}
                    <span className="inline-flex align-middle px-1">
                      <span
                        className={[
                          'inline-flex h-10 w-10 md:h-14 md:w-14 items-center justify-center',
                          'rounded-2xl bg-white/95 ring-2 ring-orange-500/60',
                          'shadow-[0_0_0_1px_rgba(249,115,22,0.25),0_18px_45px_rgba(0,0,0,0.35),0_0_40px_rgba(249,115,22,0.25)]',
                          '-rotate-12',
                        ].join(' ')}
                        aria-hidden="true"
                      >
                        <img
                          src="/brand/automatos-mark-hi.png"
                          alt=""
                          className="h-6 w-6 md:h-8 md:w-8 object-contain drop-shadow-[0_0_10px_rgba(249,115,22,0.35)]"
                          draggable={false}
                        />
                      </span>
                    </span>
                    <span className="whitespace-nowrap">Your Workflow</span>
                  </span>
                </h1>
                <p className="mt-4 text-base md:text-lg text-muted-foreground max-w-3xl mx-auto">
                  Transforming ideas into intelligent solutions—explore documents, databases, workflows, and tools in one place.
                </p>
              </div>
              <div
                className={[
                  'relative w-full max-w-3xl md:max-w-4xl overflow-hidden rounded-3xl',
                  'border border-orange-500/12 bg-background/40 backdrop-blur-xl',
                  'shadow-[0_0_64px_rgba(249,115,22,0.10)]',
                ].join(' ')}
              >
                {/* soft glow layer */}
                <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-orange-500/10 via-transparent to-transparent" />

                <div className="relative space-y-6 p-6 md:p-8">

                  <div className="grid gap-2 sm:grid-cols-2">
                    {suggestedActions.map((suggestion, index) => (
                      <motion.div
                        key={suggestion}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.12 + 0.03 * index }}
                      >
                        <Button
                          variant="outline"
                          className={[
                            'w-full rounded-2xl border-orange-500/25 bg-transparent',
                            'px-4 py-3 text-left text-sm font-medium leading-snug',
                            'hover:border-orange-500/45 hover:bg-orange-500/5',
                            'shadow-[0_0_0_1px_rgba(249,115,22,0.10)]',
                          ].join(' ')}
                          onClick={() => sendMessage(suggestion)}
                          title={suggestion}
                        >
                          <span className="block line-clamp-2 text-left text-sm text-foreground/90">
                            {suggestion}
                          </span>
                        </Button>
                      </motion.div>
                    ))}
                  </div>

                  <div className="space-y-2">
                    <MultimodalInput
                      chatId={id}
                      status={status}
                      stop={stop}
                      sendMessage={sendMessage}
                      selectedModelId={currentModelId}
                      onModelChange={setCurrentModelId}
                      selectedAgentId={selectedAgentId}
                      onAgentChange={setSelectedAgentId}
                      selectedVisibilityType={visibilityType}
                      usage={usage}
                    />
                    <div className="flex flex-wrap justify-center gap-2 pt-1">
                      {quickLinks.map((item) => {
                        const Icon = item.icon
                        return (
                          <Link
                            key={item.href}
                            href={item.href}
                            className={[
                              'inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium',
                              'bg-black/10 backdrop-blur text-foreground/90',
                              'hover:bg-orange-500/10 transition-colors',
                              'shadow-[0_0_18px_rgba(249,115,22,0.10)]',
                            ].join(' ')}
                          >
                            <Icon className="h-3.5 w-3.5 text-orange-400" />
                            <span>{item.label}</span>
                          </Link>
                        )
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {!showWelcomeCard && (
            <div
              ref={messagesContainerRef}
              className="flex-1 overflow-y-scroll overscroll-contain"
              style={{ overflowAnchor: 'none' }}
            >
              <div className="mx-auto flex min-w-0 max-w-4xl flex-col gap-4 px-4 py-4 md:gap-6 md:px-8">

                {/* Messages */}
                <AnimatePresence>
                  {messages.map((message, index) => (
                    <Message
                      key={message.id}
                      chatId={id}
                      message={message}
                      isLoading={isTyping && index === messages.length - 1}
                      setMessages={setMessages}
                      regenerate={regenerate}
                      isReadonly={isReadonly}
                      onArtifactSelect={handleArtifactSelect}
                      onCodeSelect={handleCodeSelect}
                      onDocumentSelect={handleDocumentSelect}
                      onDatabaseSelect={handleDatabaseSelect}
                    />
                  ))}
                </AnimatePresence>

                {/* Typing indicator removed: we show "Thinking…" on the streaming message */}

                <div ref={messagesEndRef} />
              </div>
            </div>
          )}

          {/* Scroll to Bottom */}
          {!isAtBottom && hasSentMessage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="absolute bottom-24 left-1/2 -translate-x-1/2 z-10"
            >
              <Button
                variant="outline"
                size="icon"
                className="rounded-full shadow-lg"
                onClick={() => {
                  messagesContainerRef.current?.scrollTo({
                    top: messagesContainerRef.current.scrollHeight,
                    behavior: 'smooth',
                  })
                }}
              >
                <ArrowDown className="h-4 w-4" />
              </Button>
            </motion.div>
          )}

          {/* Input Area */}
          {!isReadonly && !showWelcomeCard && (
            <div className="sticky bottom-0 z-10 bg-transparent backdrop-blur-none supports-[backdrop-filter]:bg-transparent border-0">
              <div className="mx-auto max-w-4xl px-4 py-4 md:px-8 space-y-3">
                {/* PRD-40: Tool Suggestion Bar */}
                <ToolSuggestionBar
                  suggestions={toolSuggestions}
                  activeTool={activeTool}
                  onSuggestionClick={handleSuggestionClick}
                  onClose={() => {
                    setActiveTool(null)
                    setToolSuggestions([])
                  }}
                  isLoading={isLoadingSuggestions}
                />

                <MultimodalInput
                  chatId={id}
                  status={status}
                  stop={stop}
                  sendMessage={sendMessage}
                  selectedModelId={currentModelId}
                  onModelChange={setCurrentModelId}
                  selectedAgentId={selectedAgentId}
                  onAgentChange={setSelectedAgentId}
                  selectedVisibilityType={visibilityType}
                  usage={usage}
                  onToolIconClick={handleToolIconClick}
                />
                <div className="flex flex-wrap justify-center gap-2">
                  {quickLinks.map((item) => {
                    const Icon = item.icon
                    return (
                      <Link
                        key={item.href}
                        href={item.href}
                        className={[
                          'inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium',
                          'bg-black/10 backdrop-blur text-foreground/90',
                          'hover:bg-orange-500/10 transition-colors',
                          'shadow-[0_0_18px_rgba(249,115,22,0.10)]',
                        ].join(' ')}
                      >
                        <Icon className="h-3.5 w-3.5 text-orange-400" />
                        <span>{item.label}</span>
                      </Link>
                    )
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </>
  )
}
