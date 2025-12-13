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
  const [visibilityType, setVisibilityType] = useState<VisibilityType>(initialVisibilityType)
  const [usage, setUsage] = useState<AppUsage | undefined>(initialLastContext)
  const [hasGeneratedTitle, setHasGeneratedTitle] = useState(false)

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
    onData: (dataPart) => {
      if (dataPart.type === 'data-usage') {
        setUsage(dataPart.data)
      }
    },
    onChatIdUpdate: (newChatId) => {
      // Update local state when backend creates/returns chat ID
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
    if (!hasGeneratedTitle && messages.length >= 2) {
      const firstUserMessage = messages.find(m => m.role === 'user')
      if (firstUserMessage && firstUserMessage.parts) {
        const textPart = firstUserMessage.parts.find(p => p.type === 'text')
        if (textPart && 'text' in textPart) {
          const title = generateTitle(textPart.text)
          updateChatTitle(id, title).catch(console.error)
          setHasGeneratedTitle(true)
        }
      }
    }
  }, [messages, id, hasGeneratedTitle])

  // Handle selections
  const handleArtifactSelect = useCallback((artifact: Artifact) => {
    setSelectedArtifact(artifact)
    setIsArtifactViewerVisible(true)
  }, [])
  
  const handleCodeSelect = useCallback((code: CodeSnippet) => {
    setSelectedArtifact({
      id: `code-${Date.now()}`,
      kind: 'code',
      title: code.symbol_name || 'Code Snippet',
      content: code.code,
      language: code.language,
      metadata: code,
    })
    setIsArtifactViewerVisible(true)
  }, [])
  
  const handleDocumentSelect = useCallback((doc: DocumentReference) => {
    const artifactId = `doc-${Date.now()}`
    const initialContent = doc.full_content || doc.excerpt || doc.preview || doc.content || 'Loading document...'

    const curatedMetadata: Record<string, any> = {
      document_id: doc.id,
      filename: doc.filename,
      similarity: doc.similarity,
      chunk_count: doc.chunk_count ?? undefined,
      preview_chunk_start: doc.preview_chunk_start ?? undefined,
      preview_chunk_end: doc.preview_chunk_end ?? undefined,
      has_full_content: doc.has_full_content ?? false,
      preview_excerpt: doc.excerpt,
      is_loading_full_content: !doc.has_full_content,
      download_url: doc.download_url,
      file_path: doc.file_path,
      chunks: doc.chunks,
    }

    setSelectedArtifact({
      id: artifactId,
      kind: 'text',
      title: doc.filename,
      content: initialContent,
      metadata: curatedMetadata,
    })
    setIsArtifactViewerVisible(true)

    if (!doc.id) {
      return
    }

    apiClient.request(`/api/documents/${doc.id}/content`)
      .then((data: any) => {
        const fullContent = Array.isArray(data?.chunks)
          ? data.chunks.map((chunk: any) => chunk?.content ?? '').filter(Boolean).join('\n\n')
          : initialContent

        setSelectedArtifact((current) => {
          if (!current || current.id !== artifactId) {
            return current
          }
          return {
            ...current,
            content: fullContent || initialContent,
            metadata: {
              ...(current.metadata || {}),
              chunk_count: data?.chunk_count ?? current.metadata?.chunk_count,
              document_id: data?.document_id ?? current.metadata?.document_id ?? doc.id,
              is_loading_full_content: false,
            },
          }
        })
      })
      .catch((error) => {
        console.error('Failed to load document content', error)
        toast.error('Failed to load full document content')
        setSelectedArtifact((current) => {
          if (!current || current.id !== artifactId) {
            return current
          }
          return {
            ...current,
            metadata: {
              ...(current.metadata || {}),
              is_loading_full_content: false,
              load_error: 'Unable to load full document',
            },
          }
        })
      })
  }, [])
  
  const handleDatabaseSelect = useCallback((db: DatabaseResult) => {
    const idBase = `db-${Date.now()}`
    const columns = db.columns && db.columns.length > 0
      ? db.columns
      : (db.data && db.data.length > 0 ? Object.keys(db.data[0]) : [])

    const summaryContentLines = [
      `# ${db.database} Insight`,
      `**Rows returned:** ${db.row_count}`,
      `**Execution time:** ${db.execution_time_ms?.toFixed(0)} ms`,
    ]
 
    if (db.pandas_ai?.summary) {
      summaryContentLines.push('', db.pandas_ai.summary)
    }
 
    if (db.data && db.data.length > 0) {
      const numericColumns = columns.filter((col) =>
        db.data!.some((row) => typeof row[col] === 'number' && !Number.isNaN(row[col]))
      )

      if (numericColumns.length > 0) {
        summaryContentLines.push('', '## Key Metrics')
        numericColumns.slice(0, 3).forEach((col) => {
          const values = db.data!
            .map((row) => (typeof row[col] === 'number' ? (row[col] as number) : null))
            .filter((value): value is number => value !== null && !Number.isNaN(value))

          if (values.length > 0) {
            const max = Math.max(...values)
            const min = Math.min(...values)
            const avg = values.reduce((acc, val) => acc + val, 0) / values.length

            summaryContentLines.push(
              `- **${col}** → peak ${max.toFixed(2)}, low ${min.toFixed(2)}, avg ${avg.toFixed(2)}`
            )
          }
        })
      }
    }

    if (columns.length > 0 && db.data && db.data.length > 0) {
      summaryContentLines.push('', '## Sample Data (first 5 rows)', '')
      const headerRow = `| ${columns.join(' | ')} |`
      const separatorRow = `| ${columns.map(() => '---').join(' | ')} |`
      summaryContentLines.push(headerRow, separatorRow)
      db.data.slice(0, 5).forEach((row) => {
        const rowValues = columns.map((col) => {
          const value = row[col]
          if (value === null || value === undefined) return '-'
          const str = String(value)
          return str.length > 60 ? `${str.slice(0, 57)}…` : str
        })
        summaryContentLines.push(`| ${rowValues.join(' | ')} |`)
      })
    }

    summaryContentLines.push('', '<details><summary>View SQL query</summary>', '', '```sql', db.sql, '```', '', '</details>')

    const artifact = {
      id: idBase,
      kind: 'sheet' as const,
      title: `${db.database} Insight`,
      content: summaryContentLines.join('\n'),
      metadata: {
        database: db.database,
        status: db.status,
        sql: db.sql,
        row_count: db.row_count,
        execution_time_ms: db.execution_time_ms,
        columns,
        data: db.data,
        pandas_ai: db.pandas_ai,
        explanation: db.explanation,
        rephrased_query: db.rephrased_query,
        visualization: db.visualization,
        follow_up_questions: db.follow_up_questions,
        clarifications: db.clarifications,
        message: db.message,
      },
    }

    setSelectedArtifact(artifact)
    setIsArtifactViewerVisible(true)
  }, [])

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
      {/* Full-screen artifact overlay - like ai-chatbot */}
      <AnimatePresence>
        {isArtifactViewerVisible && selectedArtifact && (
          <motion.div
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { delay: 0.4 } }}
            className="fixed top-0 left-0 z-50 flex h-screen w-screen flex-row bg-transparent"
          >
            {/* Background */}
            <motion.div
              initial={{ width: '100%', right: 0 }}
              animate={{ width: '100%', right: 0 }}
              exit={{ width: '100%', right: 0 }}
              className="fixed h-screen bg-background"
            />

            {/* Chat Column - LEFT 400px */}
            <motion.div
              initial={{ opacity: 0, x: 10, scale: 1 }}
              animate={{
                opacity: 1,
                x: 0,
                scale: 1,
                transition: {
                  delay: 0.1,
                  type: 'spring',
                  stiffness: 300,
                  damping: 30,
                },
              }}
              exit={{
                opacity: 0,
                x: 0,
                scale: 1,
                transition: { duration: 0 },
              }}
              className="relative h-screen w-[400px] shrink-0 bg-muted dark:bg-background"
            >
              <div className="flex h-full flex-col items-center justify-between">
                {/* Messages */}
                <div
                  ref={messagesContainerRef}
                  className="flex-1 w-full overflow-y-scroll overscroll-contain"
                  style={{ overflowAnchor: 'none' }}
                >
                  <div className="mx-auto flex min-w-0 flex-col gap-4 px-4 py-4 md:gap-6">
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
                      selectedVisibilityType={visibilityType}
                      usage={usage}
                    />
                  </div>
                )}
              </div>
            </motion.div>

            {/* Artifact Viewer - RIGHT side, rest of space */}
            <motion.div
              initial={{ opacity: 0, x: windowWidth, scale: 0.98 }}
              animate={{
                opacity: 1,
                x: 0,
                scale: 1,
                transition: {
                  delay: 0.2,
                  type: 'spring',
                  stiffness: 400,
                  damping: 40,
                },
              }}
              exit={{
                opacity: 0,
                x: 400,
                scale: 0.98,
                transition: {
                  delay: 0,
                  type: 'spring',
                  stiffness: 400,
                  damping: 40,
                },
              }}
              className="relative z-10 flex h-full flex-1 flex-col overflow-y-scroll bg-background"
            >
              <ArtifactViewer
                artifact={selectedArtifact}
                onClose={() => {
                  setIsArtifactViewerVisible(false)
                  setSelectedArtifact(null)
                }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Normal chat view - NO artifact */}
      {!isArtifactViewerVisible && (
        <div className="relative flex h-full w-full flex-col bg-transparent">
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
                <MultimodalInput
                  chatId={id}
                  status={status}
                  stop={stop}
                  sendMessage={sendMessage}
                  selectedModelId={currentModelId}
                  onModelChange={setCurrentModelId}
                  selectedVisibilityType={visibilityType}
                  usage={usage}
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
