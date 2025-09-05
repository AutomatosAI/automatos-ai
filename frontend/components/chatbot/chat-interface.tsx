
'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  X, 
  Send, 
  Bot, 
  User, 
  Copy, 
  ThumbsUp, 
  ThumbsDown,
  Paperclip,
  Download,
  ExternalLink,
  Minimize2,
  MoreVertical,
  Sparkles
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'

interface ChatMessage {
  id: string
  type: 'user' | 'bot' | 'system'
  content: string
  timestamp: string
  actions?: ActionButton[]
  context?: any
  metadata?: {
    intent: string
    confidence: number
    processing_time: number
  }
}

interface ActionButton {
  type: string
  label: string
  data: any
}

interface ChatInterfaceProps {
  onClose: () => void
  context: {
    currentPage: string
    selectedItems: any[]
    userRole: string
    recentActions: any[]
  }
  initialQuery?: string
}

export function ChatInterface({ onClose, context, initialQuery }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState(initialQuery || '')
  const [isTyping, setIsTyping] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Generate session ID
    setSessionId(`session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
    
    // Add welcome message
    const welcomeMessage: ChatMessage = {
      id: 'welcome',
      type: 'bot',
      content: `👋 Hi! I'm your Automatos AI Assistant.\n\nI can help you with:\n• 📄 **Documents** - Search, analyze, and manage your files\n• 🤖 **Agents** - Create, assign, and monitor AI agents\n• 🔄 **Workflows** - Automate processes and track executions\n• 📊 **Analytics** - Performance insights and cost analysis\n• 🖥️ **System** - Health monitoring and troubleshooting\n\nWhat would you like to explore?`,
      timestamp: new Date().toISOString(),
      actions: [
        { type: 'help_documents', label: '📄 Document Help', data: {} },
        { type: 'help_agents', label: '🤖 Agent Help', data: {} },
        { type: 'help_workflows', label: '🔄 Workflow Help', data: {} },
        { type: 'help_system', label: '🖥️ System Help', data: {} }
      ]
    }
    
    setMessages([welcomeMessage])
    
    // Focus input
    setTimeout(() => {
      inputRef.current?.focus()
    }, 500)
    
    // If there's an initial query, send it
    if (initialQuery) {
      setTimeout(() => {
        handleSendMessage()
      }, 1000)
    }
  }, [initialQuery])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSendMessage = async (message?: string) => {
    const messageText = message || inputMessage.trim()
    if (!messageText) return

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: messageText,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsTyping(true)

    try {
      const response = await fetch('/api/chatbot/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: messageText,
          context: context,
          session_id: sessionId
        })
      })

      if (response.ok) {
        const data = await response.json()
        
        const botMessage: ChatMessage = {
          id: data.message.id,
          type: 'bot',
          content: data.message.content,
          timestamp: data.message.timestamp,
          actions: data.message.actions,
          context: data.message.context,
          metadata: data.message.metadata
        }

        setMessages(prev => [...prev, botMessage])
      } else {
        // Error handling
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          type: 'system',
          content: '⚠️ Sorry, I encountered an error processing your request. Please try again or rephrase your question.',
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (error) {
      console.error('Chat error:', error)
      
      // Fallback response for demo
      const fallbackMessage: ChatMessage = {
        id: `bot-${Date.now()}`,
        type: 'bot',
        content: `I understand you're asking about "${messageText}". I'm working on processing your request and will provide more detailed responses as the backend services are fully connected.\n\nFor now, here's what I can help with:\n• System monitoring and health checks\n• Agent performance insights\n• Document management guidance\n• Workflow automation support`,
        timestamp: new Date().toISOString(),
        actions: [
          { type: 'system_health', label: '🟢 Check System Health', data: {} },
          { type: 'agent_metrics', label: '📊 View Agent Metrics', data: {} },
          { type: 'document_search', label: '🔍 Search Documents', data: {} }
        ]
      }
      setMessages(prev => [...prev, fallbackMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleActionClick = async (action: ActionButton) => {
    try {
      const response = await fetch('/api/chatbot/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          action: action.type,
          parameters: action.data,
          session_id: sessionId
        })
      })

      if (response.ok) {
        const result = await response.json()
        
        if (result.success) {
          const actionMessage: ChatMessage = {
            id: `action-${Date.now()}`,
            type: 'system',
            content: `✅ Action "${action.label}" completed successfully.`,
            timestamp: new Date().toISOString()
          }
          setMessages(prev => [...prev, actionMessage])
        }
      }
    } catch (error) {
      console.error('Action execution error:', error)
    }
  }

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  const formatMessageContent = (content: string) => {
    return content.split('\n').map((line, index) => (
      <div key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </div>
    ))
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0, y: 20 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 20 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
        className="w-full max-w-4xl h-[90vh] bg-gray-900/95 backdrop-blur-sm border border-gray-800/50 rounded-2xl shadow-2xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-800/50">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Automatos AI Assistant</h2>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span>Online</span>
                <Separator orientation="vertical" className="h-4 bg-gray-700" />
                <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                  {context.currentPage}
                </Badge>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white">
                  <MoreVertical className="w-4 h-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-gray-800 border-gray-700">
                <DropdownMenuItem className="text-gray-300 hover:text-white hover:bg-gray-700">
                  <Download className="w-4 h-4 mr-2" />
                  Export Chat
                </DropdownMenuItem>
                <DropdownMenuItem className="text-gray-300 hover:text-white hover:bg-gray-700">
                  <Copy className="w-4 h-4 mr-2" />
                  Copy Session ID
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="text-gray-400 hover:text-white hover:bg-gray-800"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-6">
          <div className="space-y-6">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                    <div className={`flex items-start space-x-3 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                      {/* Avatar */}
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.type === 'user' 
                          ? 'bg-blue-600' 
                          : message.type === 'system'
                          ? 'bg-yellow-600'
                          : 'bg-gradient-to-br from-orange-500 to-red-500'
                      }`}>
                        {message.type === 'user' ? (
                          <User className="w-5 h-5 text-white" />
                        ) : (
                          <Bot className="w-5 h-5 text-white" />
                        )}
                      </div>

                      {/* Message content */}
                      <div className="flex-1 space-y-2">
                        <Card className={`${
                          message.type === 'user' 
                            ? 'bg-blue-600/20 border-blue-500/30' 
                            : message.type === 'system'
                            ? 'bg-yellow-600/20 border-yellow-500/30'
                            : 'bg-gray-800/50 border-gray-700/50'
                        }`}>
                          <CardContent className="p-4">
                            <div className={`text-sm ${
                              message.type === 'user' ? 'text-blue-100' : 'text-gray-100'
                            }`}>
                              {formatMessageContent(message.content)}
                            </div>
                            
                            {/* Message metadata */}
                            {message.metadata && (
                              <div className="mt-3 pt-3 border-t border-gray-700/50">
                                <div className="flex items-center space-x-4 text-xs text-gray-400">
                                  <span>Intent: {message.metadata.intent}</span>
                                  <span>Confidence: {(message.metadata.confidence * 100).toFixed(0)}%</span>
                                  <span>Response: {message.metadata.processing_time}s</span>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>

                        {/* Action buttons */}
                        {message.actions && message.actions.length > 0 && (
                          <div className="flex flex-wrap gap-2">
                            {message.actions.map((action, actionIndex) => (
                              <Button
                                key={actionIndex}
                                variant="outline"
                                size="sm"
                                onClick={() => handleActionClick(action)}
                                className="bg-gray-800/50 border-gray-700 hover:bg-gray-700 text-gray-300 hover:text-white text-xs"
                              >
                                {action.label}
                              </Button>
                            ))}
                          </div>
                        )}

                        {/* Message actions */}
                        <div className="flex items-center space-x-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(message.content)}
                            className="text-gray-500 hover:text-gray-300 p-1 h-auto"
                          >
                            <Copy className="w-3 h-3" />
                          </Button>
                          
                          {message.type === 'bot' && (
                            <>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-gray-500 hover:text-green-400 p-1 h-auto"
                              >
                                <ThumbsUp className="w-3 h-3" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-gray-500 hover:text-red-400 p-1 h-auto"
                              >
                                <ThumbsDown className="w-3 h-3" />
                              </Button>
                            </>
                          )}
                          
                          <span className="text-xs text-gray-500">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Typing indicator */}
            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start"
              >
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
                    <div className="flex space-x-1">
                      <motion.div
                        className="w-2 h-2 bg-gray-400 rounded-full"
                        animate={{ y: [0, -8, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8, delay: 0 }}
                      />
                      <motion.div
                        className="w-2 h-2 bg-gray-400 rounded-full"
                        animate={{ y: [0, -8, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8, delay: 0.2 }}
                      />
                      <motion.div
                        className="w-2 h-2 bg-gray-400 rounded-full"
                        animate={{ y: [0, -8, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8, delay: 0.4 }}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="p-6 border-t border-gray-800/50">
          <form 
            onSubmit={(e) => {
              e.preventDefault()
              handleSendMessage()
            }}
            className="flex space-x-3"
          >
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-gray-400 hover:text-white hover:bg-gray-800"
            >
              <Paperclip className="w-5 h-5" />
            </Button>
            
            <Input
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask me anything about your system..."
              className="flex-1 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
              disabled={isTyping}
            />
            
            <Button 
              type="submit" 
              disabled={!inputMessage.trim() || isTyping}
              className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 px-6"
            >
              {isTyping ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1 }}
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </Button>
          </form>
          
          <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4" />
              <span>AI-powered assistance for your Automatos platform</span>
            </div>
            <div>Session: {sessionId.slice(-8)}</div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
