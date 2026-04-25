'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowDown, Target, X } from 'lucide-react'
import { LLM_DEFAULTS } from '@/lib/llm-defaults'
import { Button } from '@/components/ui/button'
import { useChat } from '@/lib/chat/hooks'
import { Message } from './message'
import { MultimodalInput } from './multimodal-input'
import { ArtifactViewer } from './artifact-viewer'
import { generateTitle } from '@/lib/utils'
import { updateChatTitle } from '@/lib/chat/api'
import type { ChatMessage, VisibilityType, Artifact, AppUsage, CodeSnippet, DocumentReference, DatabaseResult, RoutingInfo } from '@/types'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'
import { useUser } from '@clerk/nextjs'
import { useRouter } from 'next/navigation'

// Widget Architecture (PRD-38.1)
import { useWorkspaceStore } from '@/stores/workspace-store'
import { Canvas } from '@/components/workspace'
import type { Widget, CodeWidgetData, DataWidgetData, DocumentWidgetData, CodingCanvasWidgetData } from '@/components/widgets/types'
import { useWorkspace } from '@/components/workspace-provider'

// Resizable panels for chat + widget split
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable'

// PRD-40: Dynamic Tool Suggestions
import { ToolSuggestionBar } from '@/components/suggestions/ToolSuggestionBar'
import type { SuggestionResponse } from '@/components/suggestions/types'
import { analytics } from '@/lib/analytics'

// PRD-82A: Mission mode
import { useMissionStore } from '@/stores/mission-store'
import { useCreateMission } from '@/hooks/use-missions-api'

// Chat Mode Bar (ralph/chat-mode-bar)
import { ChatModeBar } from '@/components/chatbot/chat-mode-bar'
import { PinAgentPicker } from '@/components/chatbot/pin-agent-picker'
import { usePinnedAgents } from '@/hooks/use-pinned-agents'
import { useAgents } from '@/hooks/use-agent-api'
import { MissionCreatedCard } from '@/components/chatbot/mission-created-card'
import { MissionSuggestionCard } from '@/components/chatbot/mission-suggestion-card'
import { CreateMissionModal } from '@/components/missions/create-mission-modal'

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
  initialChatModel = LLM_DEFAULTS.model_id,
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

  // US-015: SSE event dispatchers for memory & workflow widgets
  const dispatchMemoryInjected = useWorkspaceStore((s) => s.dispatchMemoryInjected)
  const dispatchMemoryStored = useWorkspaceStore((s) => s.dispatchMemoryStored)
  const dispatchWorkflowUpdate = useWorkspaceStore((s) => s.dispatchWorkflowUpdate)

  // PRD-82A: Mission mode + Plan mode
  const isMissionMode = useMissionStore((s) => s.isMissionMode)
  const setMissionMode = useMissionStore((s) => s.setMissionMode)
  const isPlanMode = useMissionStore((s) => s.isPlanMode)
  const setPlanMode = useMissionStore((s) => s.setPlanMode)
  const setActivePlanningMissionId = useMissionStore((s) => s.setActivePlanningMissionId)
  const activePlanningMissionId = useMissionStore((s) => s.activePlanningMissionId)
  const createMission = useCreateMission()
  const router = useRouter()

  // PRD-66: Workspace context for Code Canvas
  const { workspace } = useWorkspace()

  // Chat Mode Bar: pinned agents + agent roster
  const { pinnedIds, pin, unpin } = usePinnedAgents(workspace?.id?.toString() ?? '')
  const { data: agentsData } = useAgents()
  const agents = agentsData ?? []

  const handleOpenCodeCanvas = useCallback(() => {
    if (!workspace?.id) {
      toast.error('No workspace selected')
      return
    }

    // Check if a coding canvas widget is already open for this workspace
    const allWidgets = useWorkspaceStore.getState().widgets
    const existing = Object.values(allWidgets).find(
      (w: Widget) => w.type === 'coding_canvas' && (w.data as CodingCanvasWidgetData).workspaceId === workspace.id
    )
    if (existing) {
      useWorkspaceStore.getState().setActiveWidget(existing.id)
      return
    }

    const widgetData: CodingCanvasWidgetData = { workspaceId: workspace.id }
    addWidget({
      type: 'coding_canvas',
      title: 'Code Canvas',
      data: widgetData,
      metadata: {
        source: { type: 'user', name: 'code_canvas' },
        createdAt: new Date(),
      },
      state: 'ready',
      createdAt: new Date().toISOString(),
    })
  }, [workspace?.id, addWidget])

  // Handler to close canvas and reset all overlay states
  const handleCloseCanvas = useCallback(() => {
    clearWidgets()
    setIsArtifactViewerVisible(false)
    setSelectedArtifact(null)
  }, [clearWidgets])

  // Plan → Mission launch state
  const [planMissionModalOpen, setPlanMissionModalOpen] = useState(false)
  const [planMissionContent, setPlanMissionContent] = useState<{ goal: string; description: string } | null>(null)

  // PRD-125 Phase 1: Mission suggestion from AutoBrain complexity detection
  const [missionSuggestion, setMissionSuggestion] = useState<{
    goal: string
    complexity: string
    agentId?: number | string
  } | null>(null)

  const handleLaunchPlanAsMission = useCallback((content: string) => {
    // Extract first 2-3 sentences as goal, full content as description
    const sentences = content.split(/(?<=[.!?])\s+/).filter(Boolean)
    const goal = sentences.slice(0, 3).join(' ').slice(0, 200)
    setPlanMissionContent({ goal, description: content })
    setPlanMissionModalOpen(true)
  }, [])

  const [canvasWidth, setCanvasWidth] = useState(800)
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)
  const [visibilityType, setVisibilityType] = useState<VisibilityType>(initialVisibilityType)
  const [usage, setUsage] = useState<AppUsage | undefined>(initialLastContext)
  const [hasGeneratedTitle, setHasGeneratedTitle] = useState(false)

  // PRD-50: Agent name cache for routing indicators + last routing decision for corrections
  const agentNameCache = useRef<Record<number, string>>({})
  const lastRoutingDecision = useRef<{ requestId?: string; agentId?: number } | null>(null)

  // PRD-40/41: Dynamic Tool Suggestions state
  const [activeTool, setActiveTool] = useState<string | null>(null)
  const [toolSuggestions, setToolSuggestions] = useState<string[]>([])
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false)
  const [hasContextSuggestions, setHasContextSuggestions] = useState(false)

  // PRD-41: Get user for context-aware suggestions
  const { user } = useUser()

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
    missionMode: isMissionMode,
    planMode: isPlanMode,
    onData: (dataPart) => {
      if (dataPart.type === 'data-usage') {
        setUsage(dataPart.data)
      }

      // PRD-38.1: Auto-create widgets when tool-data arrives
      if (dataPart.type === 'tool-data' && dataPart.data) {
        const toolData = dataPart.data
        console.log('[WIDGET-DEBUG] tool-data keys:', Object.keys(toolData), 'has generated_document:', !!toolData.generated_document)

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

        // PRD-63: Create widget for generated documents
        if (toolData.generated_document) {
          const genDoc = toolData.generated_document
          console.log('[Widget] Creating generated document widget:', genDoc.filename)
          addWidget({
            type: 'document',
            title: genDoc.title || genDoc.filename || 'Generated Document',
            data: {
              content: genDoc.content || `*Document generated: ${genDoc.filename}*`,
              format: 'markdown',
              filename: genDoc.filename,
              downloadUrl: genDoc.download_url,
              hasFullContent: true,
            },
            metadata: {
              source: { type: 'tool', name: 'generate_document', provider: 'document_generation' },
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

      // US-015: Dispatch memory & workflow SSE events to workspace store
      if (dataPart.type === 'memory-injected' && dataPart.data) {
        dispatchMemoryInjected(dataPart.data)
      }
      if (dataPart.type === 'memory-stored' && dataPart.data) {
        dispatchMemoryStored(dataPart.data)
      }
      if (dataPart.type === 'workflow-update' && dataPart.data) {
        dispatchWorkflowUpdate(dataPart.data)
      }

      // PRD-125 Phase 1: Mission suggestion from AutoBrain
      if (dataPart.type === 'mission-suggestion' && dataPart.data) {
        setMissionSuggestion({
          goal: dataPart.data.goal,
          complexity: dataPart.data.complexity,
          agentId: dataPart.data.agent_id,
        })
      }
    },
    onChatIdUpdate: (newChatId) => {
      setActiveChatId(newChatId)
    },
    onRoutingDecision: (info: RoutingInfo) => {
      // Store for correction API
      lastRoutingDecision.current = { agentId: info.agentId }

      // Resolve agent name asynchronously if not cached
      if (!info.agentName && info.agentId && !agentNameCache.current[info.agentId]) {
        apiClient.request(`/api/agents/${info.agentId}`)
          .then((agent: any) => {
            const name = agent?.name || `Agent #${info.agentId}`
            agentNameCache.current[info.agentId] = name
            // Update all messages with this agent's routing info to include name
            setMessages(prev =>
              prev.map(m =>
                m.routingInfo?.agentId === info.agentId && !m.routingInfo.agentName
                  ? { ...m, routingInfo: { ...m.routingInfo, agentName: name } }
                  : m
              )
            )
          })
          .catch(() => {
            // Silently fail — indicator will show "Agent #id" instead
          })
      } else if (agentNameCache.current[info.agentId]) {
        // Already cached — update immediately
        setMessages(prev =>
          prev.map(m =>
            m.routingInfo?.agentId === info.agentId && !m.routingInfo.agentName
              ? { ...m, routingInfo: { ...m.routingInfo, agentName: agentNameCache.current[info.agentId] } }
              : m
          )
        )
      }
    },
  })

  const regenerate = () => reload()

  // PRD-82A US-008: Handle /mission slash command + US-003: Mission mode intercept
  const handleSendMessage = useCallback(
    (message: any) => {
      // PRD-125: Clear stale mission suggestion on new message
      setMissionSuggestion(null)

      const text = typeof message === 'string' ? message : message?.content
      const trimmedText = text?.trim() ?? ''

      // US-008: /mission slash command takes priority over mission mode
      if (trimmedText.toLowerCase().startsWith('/mission ')) {
        const goal = trimmedText.slice(9).trim()
        if (!goal) return

        createMission.mutate(
          { goal },
          {
            onSuccess: (mission) => {
              setActivePlanningMissionId(mission.id)
              setMissionMode(false)
              toast.success('Mission created — plan is being generated')
            },
            onError: (err) => {
              toast.error(err.message || 'Failed to create mission')
            },
          }
        )
        return
      }

      // US-003: Mission mode — route through normal chat with missionMode flag
      // Auto will converse about the mission, then create via /mission command
      sendMessage(message)
    },
    [sendMessage, createMission, setActivePlanningMissionId, setMissionMode]
  )

  // PRD-50: Handle agent change — fire correction API when overriding auto-routed agent
  const handleAgentChange = useCallback((newAgentId: number | null) => {
    const prev = lastRoutingDecision.current
    setSelectedAgentId(newAgentId)

    // If user selects a specific agent after an auto-route, record the correction
    if (newAgentId && prev?.agentId && newAgentId !== prev.agentId) {
      // Find the most recent message with a routing request ID
      const routedMessage = [...messages].reverse().find(m => m.routingInfo?.requestId)
      const requestId = routedMessage?.routingInfo?.requestId
      if (requestId) {
        apiClient.post('/api/routing/corrections', {
          request_id: requestId,
          correct_agent_id: newAgentId,
        }).catch((err) => {
          console.warn('[Routing] Failed to record correction:', err)
        })
      }
    }
  }, [messages])

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

  // PRD-40/41: Tool icon click handler (Phase 1 + Phase 2: Context-Aware)
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
      // PRD-41: Include user_id and session_id for context-aware suggestions
      const userId = user?.id
      const sessionId = activeChatId || id

      // Build URL with query params if we have user context
      let url = `/api/tools/${appName}/suggestions`
      if (userId && sessionId) {
        const params = new URLSearchParams({
          user_id: userId,
          session_id: sessionId,
        })
        url = `${url}?${params.toString()}`
      }

      const data = await apiClient.request<SuggestionResponse>(url)
      setToolSuggestions(data.suggestions || [])
      setHasContextSuggestions(data.has_context || false)

      // Track suggestions loaded (including context flag)
      analytics.track('suggestions_loaded', {
        app: appName,
        source: data.source,
        count: data.suggestions.length,
        has_context: data.has_context || false,
      })
    } catch (error) {
      console.error('Failed to fetch tool suggestions:', error)
      toast.error(`Failed to load suggestions for ${appName}`)
      setToolSuggestions([])
      setActiveTool(null)
    } finally {
      setIsLoadingSuggestions(false)
    }
  }, [activeTool, user, activeChatId, id])

  // PRD-40: Suggestion click handler
  const handleSuggestionClick = useCallback((suggestion: string) => {
    // Track suggestion click
    analytics.track('suggestion_clicked', {
      app: activeTool || 'unknown',
      suggestion: suggestion,
      location: 'chat',
    })

    // Send the suggestion as a message
    handleSendMessage(suggestion)
    // Close suggestions after use
    setActiveTool(null)
    setToolSuggestions([])
  }, [handleSendMessage, activeTool])

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

  const showWelcomeCard = !hasSentMessage && !isTyping

  return (
    <>
      {/* PRD-38.1: Widget Canvas Layout - shows when widgets exist */}
      {hasWidgets && (
        <div className="fixed top-0 left-0 z-50 h-screen w-screen bg-background">
          <ResizablePanelGroup direction="horizontal" className="h-full">
            {/* Chat Column - LEFT (resizable, default 35%, min 20%) */}
            <ResizablePanel defaultSize={35} minSize={20} maxSize={60}>
              <div className="relative h-full bg-muted dark:bg-background border-r border-border/50 overflow-hidden">
                <div className="flex h-full flex-col items-center justify-between min-w-0">
                  {/* Messages */}
                  <div
                    ref={messagesContainerRef}
                    className="flex-1 w-full overflow-y-scroll overflow-x-hidden overscroll-contain"
                    style={{ overflowAnchor: 'none' }}
                  >
                    <div className="mx-auto flex min-w-0 flex-col gap-4 px-4 py-4 md:gap-6 break-words">
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
                          onLaunchAsMission={handleLaunchPlanAsMission}
                          onCodeSelect={handleCodeSelect}
                          onDocumentSelect={handleDocumentSelect}
                          onDatabaseSelect={handleDatabaseSelect}
                        />
                      ))}

                      {/* PRD-125 Phase 1: Mission suggestion card */}
                      {missionSuggestion && !activePlanningMissionId && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <MissionSuggestionCard
                            goal={missionSuggestion.goal}
                            complexity={missionSuggestion.complexity}
                            agentId={missionSuggestion.agentId}
                            chatId={id}
                            recentMessages={messages.slice(-5).map(m => ({ role: m.role, content: m.content }))}
                          />
                        </motion.div>
                      )}

                      {/* PRD-82A: Inline mission card after creation */}
                      {activePlanningMissionId && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <MissionCreatedCard missionId={activePlanningMissionId} />
                        </motion.div>
                      )}

                      <div ref={messagesEndRef} />
                    </div>
                  </div>

                  {/* Input at bottom of chat column */}
                  {!isReadonly && (
                    <div className="relative flex w-full flex-col gap-3 px-4 pb-4">
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
                        hasContext={hasContextSuggestions}
                      />
                      {/* PRD-82A US-002: Mission mode indicator banner */}
                      <AnimatePresence>
                        {isMissionMode && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="flex items-center gap-2 rounded-lg bg-orange-500/10 border border-orange-500/30 px-3 py-2 text-sm text-orange-400"
                          >
                            <Target className="h-4 w-4 shrink-0" />
                            <span className="flex-1">Mission Mode — Describe your goal and an AI team will execute it</span>
                            <button
                              type="button"
                              onClick={() => setMissionMode(false)}
                              className="shrink-0 rounded p-0.5 hover:bg-orange-500/20 transition-colors"
                            >
                              <X className="h-3.5 w-3.5" />
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                      <MultimodalInput
                        chatId={id}
                        status={status}
                        stop={stop}
                        sendMessage={handleSendMessage}
                        selectedModelId={currentModelId}
                        onModelChange={setCurrentModelId}
                        selectedAgentId={selectedAgentId}
                        onAgentChange={handleAgentChange}
                        selectedVisibilityType={visibilityType}
                        usage={usage}
                        onToolIconClick={handleToolIconClick}
                      />
                    </div>
                  )}
                </div>
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            {/* PRD-38.1: Widget Canvas - RIGHT side (resizable) */}
            <ResizablePanel defaultSize={65} minSize={30}>
              <div className="relative z-10 flex h-full flex-col bg-background">
                <div className="flex-1 overflow-hidden">
                  <Canvas width={canvasWidth} onClose={handleCloseCanvas} />
                </div>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      )}

      {/* Legacy Artifact Viewer overlay - for backward compatibility */}
      <AnimatePresence>
        {isArtifactViewerVisible && selectedArtifact && !hasWidgets && (
          <motion.div
            initial={{ opacity: 1 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { delay: 0.4 } }}
            className="fixed top-0 left-0 z-50 h-screen w-screen bg-background"
          >
            <ResizablePanelGroup direction="horizontal" className="h-full">
              <ResizablePanel defaultSize={35} minSize={20} maxSize={60}>
                <div className="relative h-full bg-muted dark:bg-background overflow-hidden">
                  <div className="flex h-full flex-col min-w-0">
                    <div className="flex-1 w-full overflow-y-scroll overflow-x-hidden">
                      <div className="flex flex-col gap-4 px-4 py-4 min-w-0 break-words">
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
                          onLaunchAsMission={handleLaunchPlanAsMission}
                            onCodeSelect={handleCodeSelect}
                            onDocumentSelect={handleDocumentSelect}
                            onDatabaseSelect={handleDatabaseSelect}
                          />
                        ))}
                      </div>
                    </div>
                    {!isReadonly && (
                      <div className="px-4 pb-4 space-y-3">
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
                        {/* PRD-82A US-002: Mission mode indicator banner */}
                        <AnimatePresence>
                          {isMissionMode && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="flex items-center gap-2 rounded-lg bg-orange-500/10 border border-orange-500/30 px-3 py-2 text-sm text-orange-400"
                            >
                              <Target className="h-4 w-4 shrink-0" />
                              <span className="flex-1">Mission Mode — Describe your goal and an AI team will execute it</span>
                              <button
                                type="button"
                                onClick={() => setMissionMode(false)}
                                className="shrink-0 rounded p-0.5 hover:bg-orange-500/20 transition-colors"
                              >
                                <X className="h-3.5 w-3.5" />
                              </button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                        <MultimodalInput
                          chatId={id}
                          status={status}
                          stop={stop}
                          sendMessage={handleSendMessage}
                          selectedModelId={currentModelId}
                          onModelChange={setCurrentModelId}
                          selectedAgentId={selectedAgentId}
                          onAgentChange={handleAgentChange}
                          selectedVisibilityType={visibilityType}
                          usage={usage}
                          onToolIconClick={handleToolIconClick}
                        />
                      </div>
                    )}
                  </div>
                </div>
              </ResizablePanel>

              <ResizableHandle withHandle />

              <ResizablePanel defaultSize={65} minSize={30}>
                <div className="flex-1 h-full overflow-y-scroll bg-background">
                  <ArtifactViewer
                    artifact={selectedArtifact}
                    onClose={() => {
                      setIsArtifactViewerVisible(false)
                      setSelectedArtifact(null)
                    }}
                  />
                </div>
              </ResizablePanel>
            </ResizablePanelGroup>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Normal chat view - NO widgets */}
      {!hasWidgets && !isArtifactViewerVisible && (
        <div className="relative flex flex-col bg-transparent" style={{ height: '100%', width: '100%', minHeight: 0 }}>
          {/* Clean welcome state — greeting + chat input */}
          {showWelcomeCard && (
            <div className="flex flex-1 flex-col items-center justify-center px-4 py-10 md:py-16">
              {/* Personal greeting */}
              <motion.div
                className="w-full max-w-3xl md:max-w-4xl text-center mb-8"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground/90">
                  Hey{user?.firstName ? <> <span className="gradient-text">{user.firstName}</span></> : ''}, what can I do for you today?
                </h1>
              </motion.div>

              {/* Chat input + quick links — no outer box */}
              <div className="w-full max-w-3xl md:max-w-4xl space-y-3">
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
                {/* PRD-82A US-002: Mission mode indicator banner */}
                <AnimatePresence>
                  {isMissionMode && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="flex items-center gap-2 rounded-lg bg-orange-500/10 border border-orange-500/30 px-3 py-2 text-sm text-orange-400"
                    >
                      <Target className="h-4 w-4 shrink-0" />
                      <span className="flex-1">Mission Mode — Describe your goal and an AI team will execute it</span>
                      <button
                        type="button"
                        onClick={() => setMissionMode(false)}
                        className="shrink-0 rounded p-0.5 hover:bg-orange-500/20 transition-colors"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
                <MultimodalInput
                  chatId={id}
                  status={status}
                  stop={stop}
                  sendMessage={handleSendMessage}
                  setMessages={setMessages}
                  selectedModelId={currentModelId}
                  onModelChange={setCurrentModelId}
                  selectedAgentId={selectedAgentId}
                  onAgentChange={handleAgentChange}
                  selectedVisibilityType={visibilityType}
                  usage={usage}
                  onToolIconClick={handleToolIconClick}
                />
                <div className="flex flex-wrap justify-center gap-3 md:gap-2 pt-1">
                  <ChatModeBar
                    isCodeActive={hasWidgets}
                    isPlanActive={isPlanMode}
                    isMissionActive={isMissionMode}
                    onCodeClick={handleOpenCodeCanvas}
                    onPlanClick={() => setPlanMode(!isPlanMode)}
                    onMissionClick={() => router.push('/assignments?tab=missions')}
                    pinnedAgentIds={pinnedIds}
                    agents={agents}
                    selectedAgentId={selectedAgentId}
                    onAgentSelect={handleAgentChange}
                  />
                  <PinAgentPicker
                    agents={agents}
                    pinnedIds={pinnedIds}
                    onPin={pin}
                    onUnpin={unpin}
                  />
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

                {/* PRD-125 Phase 1: Mission suggestion card */}
                {missionSuggestion && !activePlanningMissionId && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <MissionSuggestionCard
                      goal={missionSuggestion.goal}
                      complexity={missionSuggestion.complexity}
                      agentId={missionSuggestion.agentId}
                      chatId={id}
                      recentMessages={messages.slice(-5).map(m => ({ role: m.role, content: m.content }))}
                    />
                  </motion.div>
                )}

                {/* PRD-82A: Inline mission card after creation */}
                {activePlanningMissionId && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <MissionCreatedCard missionId={activePlanningMissionId} />
                  </motion.div>
                )}

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

                {/* PRD-82A US-002: Mission mode indicator banner */}
                <AnimatePresence>
                  {isMissionMode && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="flex items-center gap-2 rounded-lg bg-orange-500/10 border border-orange-500/30 px-3 py-2 text-sm text-orange-400"
                    >
                      <Target className="h-4 w-4 shrink-0" />
                      <span className="flex-1">Mission Mode — Describe your goal and an AI team will execute it</span>
                      <button
                        type="button"
                        onClick={() => setMissionMode(false)}
                        className="shrink-0 rounded p-0.5 hover:bg-orange-500/20 transition-colors"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>

                <MultimodalInput
                  chatId={id}
                  status={status}
                  stop={stop}
                  sendMessage={handleSendMessage}
                  setMessages={setMessages}
                  selectedModelId={currentModelId}
                  onModelChange={setCurrentModelId}
                  selectedAgentId={selectedAgentId}
                  onAgentChange={handleAgentChange}
                  selectedVisibilityType={visibilityType}
                  usage={usage}
                  onToolIconClick={handleToolIconClick}
                />
                <div className="flex flex-wrap justify-center gap-3 md:gap-2">
                  <ChatModeBar
                    isCodeActive={hasWidgets}
                    isPlanActive={isPlanMode}
                    isMissionActive={isMissionMode}
                    onCodeClick={handleOpenCodeCanvas}
                    onPlanClick={() => setPlanMode(!isPlanMode)}
                    onMissionClick={() => router.push('/assignments?tab=missions')}
                    pinnedAgentIds={pinnedIds}
                    agents={agents}
                    selectedAgentId={selectedAgentId}
                    onAgentSelect={handleAgentChange}
                  />
                  <PinAgentPicker
                    agents={agents}
                    pinnedIds={pinnedIds}
                    onPin={pin}
                    onUnpin={unpin}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      {/* Plan → Mission modal */}
      <CreateMissionModal
        open={planMissionModalOpen}
        onOpenChange={setPlanMissionModalOpen}
        initialGoal={planMissionContent?.goal}
        initialDescription={planMissionContent?.description}
      />
    </>
  )
}
