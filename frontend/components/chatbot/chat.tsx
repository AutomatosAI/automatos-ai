'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bot, ArrowDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChat } from '@/lib/chat/hooks-simple'
import { Message } from './message'
import { MultimodalInput } from './multimodal-input'
import { ArtifactViewer } from './artifact-viewer'
import { generateTitle } from '@/lib/utils'
import { updateChatTitle } from '@/lib/chat/api'
import type { ChatMessage, VisibilityType, Artifact, AppUsage, CodeSnippet, DocumentReference, DatabaseResult } from '@/types'

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

  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1920)

  // Track window width
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const { messages, setMessages, sendMessage, status, stop, reload } = useChat({
    id,
    initialMessages,
    initialChatModel,
    initialVisibilityType,
    onData: (dataPart) => {
      if (dataPart.type === 'usage') {
        setUsage(dataPart.data)
      } else if (dataPart.type === 'artifact') {
        setSelectedArtifact(dataPart.data)
        setIsArtifactViewerVisible(true)
      }
    },
  })

  const regenerate = () => reload()

  // Track scroll
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return
    const checkScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container
      setIsAtBottom(scrollHeight - scrollTop - clientHeight < 50)
    }
    container.addEventListener('scroll', checkScroll)
    return () => container.removeEventListener('scroll', checkScroll)
  }, [])

  // Auto-scroll
  useEffect(() => {
    if (status === 'submitted') {
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
    setSelectedArtifact({
      id: `doc-${Date.now()}`,
      kind: 'text',
      title: doc.filename,
      content: doc.excerpt,
      metadata: doc,
    })
    setIsArtifactViewerVisible(true)
  }, [])
  
  const handleDatabaseSelect = useCallback((db: DatabaseResult) => {
    setSelectedArtifact({
      id: `db-${Date.now()}`,
      kind: 'text',
      title: db.database,
      content: `SQL Query:\n${db.sql}\n\nRows: ${db.row_count}\nExecution time: ${db.execution_time_ms}ms\n\nResults:\n${JSON.stringify(db.data.slice(0, 5), null, 2)}${db.data.length > 5 ? '\n\n... and ' + (db.data.length - 5) + ' more rows' : ''}`,
      metadata: db,
    })
    setIsArtifactViewerVisible(true)
  }, [])

  const isTyping = status === 'streaming' || status === 'awaiting_message'
  const hasSentMessage = messages.length > 0

  const suggestedActions = [
    "How does authentication work?",
    "Search documents about deployment",
    "Show me the top 10 customers",
    "Explain the agent coordination system",
  ]

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
          {/* Messages */}
          <div
            ref={messagesContainerRef}
            className="flex-1 overflow-y-scroll overscroll-contain"
            style={{ overflowAnchor: 'none' }}
          >
            <div className="mx-auto flex min-w-0 max-w-4xl flex-col gap-4 px-4 py-4 md:gap-6 md:px-8">
              {/* Greeting */}
              {!hasSentMessage && (
                <div className="mx-auto mt-4 flex size-full max-w-3xl flex-col justify-center md:mt-16">
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="font-semibold text-xl md:text-2xl"
                  >
                    Hello <span className="text-orange-500">there!</span>
                  </motion.div>
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    className="text-xl text-muted-foreground md:text-2xl"
                  >
                    How can I help you today?
                  </motion.div>
                </div>
              )}

            {/* Suggested Actions - will be positioned above input */}

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

              {/* Typing Indicator */}
              {isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex items-start gap-3"
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-orange-500 to-red-500">
                    <Bot className="h-5 w-5 text-white" />
                  </div>
                  <div className="rounded-lg bg-muted/50 px-4 py-3">
                    <div className="flex space-x-1">
                      {[0, 1, 2].map((i) => (
                        <motion.div
                          key={i}
                          className="h-2 w-2 rounded-full bg-muted-foreground/50"
                          animate={{ y: [0, -8, 0] }}
                          transition={{ repeat: Infinity, duration: 0.8, delay: i * 0.2 }}
                        />
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

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
          {!isReadonly && (
            <div className="sticky bottom-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-t border-border">
              <div className="mx-auto max-w-4xl px-4 py-4 md:px-8 space-y-3">
                {/* Suggested Actions - above input */}
                {!hasSentMessage && (
                  <div className="grid w-full gap-2 sm:grid-cols-2">
                    {suggestedActions.map((suggestion, index) => (
                      <motion.div
                        key={suggestion}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ delay: 0.05 * index }}
                      >
                        <Button
                          variant="outline"
                          className="h-auto w-full whitespace-normal p-3 text-left rounded-xl border-orange-500/30 hover:border-orange-500/50 hover:bg-orange-500/5"
                          onClick={() => {
                            sendMessage({
                              role: 'user',
                              parts: [{ type: 'text', text: suggestion }],
                            })
                          }}
                        >
                          {suggestion}
                        </Button>
                      </motion.div>
                    ))}
                  </div>
                )}
                
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
            </div>
          )}
        </div>
      )}
    </>
  )
}
