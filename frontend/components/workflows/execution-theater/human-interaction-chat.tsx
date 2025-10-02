'use client'

import { useState, useRef, useEffect } from 'react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  MessageCircle,
  Send,
  AlertCircle,
  User,
  Bot,
  CheckCircle
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface HumanInteractionChatProps {
  workflowId: number
  isExecuting: boolean
}

interface Message {
  id: string
  timestamp: string
  from: 'agent' | 'human' | 'orchestrator'
  agentName?: string
  message: string
  type: 'question' | 'answer' | 'notification' | 'blocker'
  priority?: 'low' | 'medium' | 'high' | 'urgent'
  status?: 'pending' | 'answered' | 'resolved'
}

export function HumanInteractionChat({ workflowId, isExecuting }: HumanInteractionChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  // TODO: Real-time message polling/WebSocket integration
  // For now, show empty state until backend supports human-in-the-loop
  useEffect(() => {
    // Clear messages when not executing
    if (!isExecuting) {
      setMessages([])
    }
    
    // TODO: Poll for agent questions from backend
    // fetch(`/api/workflows/${workflowId}/agent-questions`)
  }, [isExecuting, workflowId])

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  const handleSendMessage = () => {
    if (!inputValue.trim()) return

    const newMessage: Message = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleTimeString(),
      from: 'human',
      message: inputValue,
      type: 'answer',
      status: 'resolved'
    }

    setMessages(prev => [...prev, newMessage])
    setInputValue('')

    // TODO: Send to backend API
    // apiClient.sendHumanResponse(workflowId, inputValue)
  }

  const getPriorityColor = (priority?: string) => {
    switch (priority) {
      case 'urgent': return 'text-red-400 border-red-400/50'
      case 'high': return 'text-orange-400 border-orange-400/50'
      case 'medium': return 'text-yellow-400 border-yellow-400/50'
      default: return 'text-blue-400 border-blue-400/50'
    }
  }

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'answered': return <CheckCircle className="w-3 h-3 text-green-400" />
      case 'pending': return <AlertCircle className="w-3 h-3 text-yellow-400 animate-pulse" />
      default: return null
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-border/30">
        <div className="flex items-center space-x-2">
          <MessageCircle className="w-5 h-5 text-primary" />
          <h3 className="font-semibold">Human-in-the-Loop</h3>
          {messages.filter(m => m.status === 'pending').length > 0 && (
            <Badge variant="destructive" className="animate-pulse">
              {messages.filter(m => m.status === 'pending').length} Pending
            </Badge>
          )}
        </div>
        <Badge variant={isExecuting ? 'default' : 'secondary'}>
          {isExecuting ? 'Active' : 'Idle'}
        </Badge>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <MessageCircle className="w-12 h-12 mb-3 opacity-50" />
              <p className="text-sm">No agent questions yet</p>
              <p className="text-xs mt-1">Agents will ask here if they need guidance</p>
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className={`flex ${msg.from === 'human' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      msg.from === 'human'
                        ? 'bg-primary text-primary-foreground'
                        : msg.type === 'blocker'
                        ? 'bg-red-500/20 border-2 border-red-500/50'
                        : msg.type === 'question'
                        ? 'bg-blue-500/20 border border-blue-500/50'
                        : 'bg-background/50 border border-border/50'
                    }`}
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {msg.from === 'agent' && <Bot className="w-4 h-4 text-blue-400" />}
                        {msg.from === 'human' && <User className="w-4 h-4" />}
                        <span className="text-xs font-medium">
                          {msg.from === 'agent' ? msg.agentName : msg.from === 'human' ? 'You' : 'Orchestrator'}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        {msg.priority && (
                          <Badge variant="outline" className={`text-xs ${getPriorityColor(msg.priority)}`}>
                            {msg.priority.toUpperCase()}
                          </Badge>
                        )}
                        {getStatusIcon(msg.status)}
                      </div>
                    </div>

                    {/* Message */}
                    <p className="text-sm">{msg.message}</p>

                    {/* Footer */}
                    <div className="flex items-center justify-between mt-2 pt-2 border-t border-border/30">
                      <span className="text-xs text-muted-foreground font-mono">{msg.timestamp}</span>
                      {msg.type === 'blocker' && (
                        <Badge variant="destructive" className="text-xs">BLOCKER</Badge>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="p-3 border-t border-border/30">
        <div className="flex space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder={
              messages.filter(m => m.status === 'pending').length > 0
                ? 'Respond to agent question...'
                : 'Type a message to agents...'
            }
            className="flex-1"
            disabled={!isExecuting}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || !isExecuting}
            size="icon"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        {messages.filter(m => m.status === 'pending').length > 0 && (
          <p className="text-xs text-yellow-400 mt-2 flex items-center">
            <AlertCircle className="w-3 h-3 mr-1" />
            {messages.filter(m => m.status === 'pending').length} agent(s) waiting for response
          </p>
        )}
      </div>
    </div>
  )
}

