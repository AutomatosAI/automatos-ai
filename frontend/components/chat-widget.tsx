
'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageCircle, 
  X, 
  Send, 
  Maximize2, 
  Bot,
  Sparkles,
  ArrowUp
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ChatInterface } from './chat-interface'

// Configuration to disable chatbot UI
const CHATBOT_DISABLED = true

interface ChatWidgetProps {
  position?: 'bottom-right' | 'bottom-left'
  context?: {
    currentPage: string
    selectedItems: any[]
    userRole: string
    recentActions: any[]
  }
}

interface ChatSuggestion {
  text: string
  category: string
  action?: string
}

export function ChatWidget({ 
  position = 'bottom-right', 
  context = {
    currentPage: 'dashboard',
    selectedItems: [],
    userRole: 'user',
    recentActions: []
  }
}: ChatWidgetProps) {
  const [isMinimized, setIsMinimized] = useState(true)
  const [isExpanded, setIsExpanded] = useState(false)
  const [quickMessage, setQuickMessage] = useState('')
  const [suggestions, setSuggestions] = useState<ChatSuggestion[]>([])
  const [hasUnreadMessages, setHasUnreadMessages] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  
  // Load contextual suggestions based on current page
  useEffect(() => {
    const loadSuggestions = async () => {
      try {
        const response = await fetch(`/api/chatbot/suggestions?page=${context.currentPage}&user_role=${context.userRole}`)
        if (response.ok) {
          const data = await response.json()
          setSuggestions(data.slice(0, 3)) // Show top 3 suggestions
        } else {
          // Fallback suggestions based on current page
          const fallbackSuggestions = getFallbackSuggestions(context.currentPage)
          setSuggestions(fallbackSuggestions)
        }
      } catch (error) {
        console.error('Error loading suggestions:', error)
        const fallbackSuggestions = getFallbackSuggestions(context.currentPage)
        setSuggestions(fallbackSuggestions)
      }
    }
    
    loadSuggestions()
  }, [context.currentPage, context.userRole])

  const getFallbackSuggestions = (page: string): ChatSuggestion[] => {
    const suggestionMap: Record<string, ChatSuggestion[]> = {
      dashboard: [
        { text: "What's the current system health?", category: "system" },
        { text: "Show me agent performance", category: "agent" },
        { text: "How many workflows are running?", category: "workflow" }
      ],
      agents: [
        { text: "Which agent performs best?", category: "agent" },
        { text: "Create a new agent", category: "agent" },
        { text: "Assign a task to an agent", category: "agent" }
      ],
      documents: [
        { text: "What documents were uploaded today?", category: "document" },
        { text: "Summarize a document", category: "document" },
        { text: "Search documents about...", category: "document" }
      ],
      workflows: [
        { text: "Show active workflows", category: "workflow" },
        { text: "Create a new workflow", category: "workflow" },
        { text: "Why did a workflow fail?", category: "workflow" }
      ],
      analytics: [
        { text: "Show performance trends", category: "analytics" },
        { text: "What are peak usage hours?", category: "analytics" },
        { text: "Generate a report", category: "analytics" }
      ]
    }
    
    return suggestionMap[page] || suggestionMap.dashboard
  }

  const handleQuickMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!quickMessage.trim()) return

    // Show typing indicator
    setIsTyping(true)
    
    try {
      // Send message to chatbot API
      const response = await fetch('/api/chatbot/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: quickMessage,
          context: context
        })
      })

      if (response.ok) {
        // Open full chat interface with the response
        setIsExpanded(true)
        setIsMinimized(false)
      }
    } catch (error) {
      console.error('Error sending message:', error)
    } finally {
      setIsTyping(false)
      setQuickMessage('')
    }
  }

  const handleSuggestionClick = (suggestion: ChatSuggestion) => {
    setQuickMessage(suggestion.text)
  }

  const positionClasses = {
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4'
  }

  // If chatbot is disabled, don't render anything
  if (CHATBOT_DISABLED) {
    return null
  }

  return (
    <>
      <div className={`fixed ${positionClasses[position]} z-50`}>
        <AnimatePresence mode="wait">
          {isMinimized ? (
            // Minimized chat bubble
            <motion.div
              key="minimized"
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
            >
              <Button
                onClick={() => setIsMinimized(false)}
                className="w-16 h-16 rounded-full bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 shadow-xl hover:shadow-2xl transition-all duration-300 group relative"
                size="lg"
              >
                <MessageCircle className="w-8 h-8 text-white group-hover:scale-110 transition-transform" />
                {hasUnreadMessages && (
                  <motion.div
                    className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full flex items-center justify-center"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                  >
                    <span className="text-xs font-bold text-gray-900">!</span>
                  </motion.div>
                )}
                <motion.div
                  className="absolute inset-0 rounded-full bg-gradient-to-br from-orange-400/50 to-red-400/50"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 3 }}
                />
              </Button>
            </motion.div>
          ) : !isExpanded ? (
            // Quick preview widget
            <motion.div
              key="preview"
              initial={{ scale: 0, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0, opacity: 0, y: 20 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="w-80"
            >
              <Card className="bg-gray-900/95 backdrop-blur-sm border-gray-800/50 shadow-2xl">
                <CardHeader className="flex flex-row items-center justify-between pb-3">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">AI Assistant</h3>
                      <p className="text-xs text-gray-400">Ready to help!</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsExpanded(true)}
                      className="text-gray-400 hover:text-white hover:bg-gray-800"
                    >
                      <Maximize2 className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsMinimized(true)}
                      className="text-gray-400 hover:text-white hover:bg-gray-800"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  <div className="flex items-center space-x-2 text-sm text-gray-300">
                    <Sparkles className="w-4 h-4 text-orange-400" />
                    <span>Need help? Try asking:</span>
                  </div>
                  
                  <div className="space-y-2">
                    {suggestions.map((suggestion, index) => (
                      <motion.button
                        key={index}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="w-full text-left p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition-colors text-sm text-gray-300 hover:text-white"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        • {suggestion.text}
                      </motion.button>
                    ))}
                  </div>

                  <form onSubmit={handleQuickMessage} className="flex space-x-2">
                    <Input
                      value={quickMessage}
                      onChange={(e) => setQuickMessage(e.target.value)}
                      placeholder="Type your question..."
                      className="bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 text-sm"
                      disabled={isTyping}
                    />
                    <Button 
                      type="submit" 
                      size="sm" 
                      disabled={!quickMessage.trim() || isTyping}
                      className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600"
                    >
                      {isTyping ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ repeat: Infinity, duration: 1 }}
                          className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                        />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </Button>
                  </form>

                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span>AI online</span>
                    </div>
                    <Badge 
                      variant="outline" 
                      className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-xs"
                    >
                      {context.currentPage}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      {/* Full-screen chat interface */}
      <AnimatePresence>
        {isExpanded && (
          <ChatInterface
            onClose={() => {
              setIsExpanded(false)
              setIsMinimized(true)
            }}
            context={context}
            initialQuery={quickMessage}
          />
        )}
      </AnimatePresence>
    </>
  )
}
