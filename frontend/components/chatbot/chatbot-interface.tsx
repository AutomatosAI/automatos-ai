'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Bot, User, Send, Code, FileText, Sparkles, 
  Copy, ThumbsUp, ThumbsDown, Download, 
  ChevronRight, Search, Terminal, BookOpen,
  Eye, EyeOff, RotateCw, MessageCircle, Database
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import { apiClient } from '@/lib/api-client'
import type { ChatMessage, CodeSnippet, DocumentReference, DatabaseResult, PandasAIChart } from '@/types'

export function ChatbotInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const [selectedCode, setSelectedCode] = useState<CodeSnippet | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<DocumentReference | null>(null)
  const [isViewerVisible, setIsViewerVisible] = useState(true)
  const [activeTab, setActiveTab] = useState<'code' | 'docs'>('code')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [availableModels, setAvailableModels] = useState<Record<string, any[]>>({})
  const [defaultProvider, setDefaultProvider] = useState<string>('openai')
  const [defaultModel, setDefaultModel] = useState<string>('gpt-4')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setSessionId(`session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
    
    // Load available models
    const loadModels = async () => {
      try {
        const data = await apiClient.request<any>('/api/chatbot/models')
        setAvailableModels(data?.providers || {})
        setDefaultProvider(data?.default_provider || 'openai')
        setDefaultModel(data?.default_model || 'gpt-4')
        setSelectedProvider(data?.default_provider || 'openai')
        setSelectedModel(data?.default_model || 'gpt-4')
      } catch (error) {
        console.error('Failed to load models:', error)
      }
    }
    loadModels()
    
    const welcomeMessage: ChatMessage = {
      id: 'welcome',
      type: 'bot',
      content: `👋 **Welcome to Automatos AI Assistant!**

I can help you with:
• 💻 **Code Questions** - Search your codebase, understand functions & classes
• 📄 **Document Search** - Find and analyze your documents
• 🗄️ **Database Queries** - Ask questions about your data in natural language
• 🤖 **Agent Help** - Manage and optimize your AI agents
• 🔄 **Workflow Assistance** - Debug and improve workflows
• 📊 **System Insights** - Performance metrics and health monitoring

Try asking: 
*"How does authentication work?"* (CodeGraph)
*"Find documents about deployment"* (RAG)
*"Show me the top 10 customers by revenue"* (Database)`,
      timestamp: new Date().toISOString(),
      metadata: {
        intent: 'welcome',
        confidence: 1.0,
        source: 'llm',
        processing_time: 0
      }
    }
    
    setMessages([welcomeMessage])
    setTimeout(() => inputRef.current?.focus(), 500)
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (selectedCode) {
      setTimeout(() => Prism.highlightAll(), 100)
    }
  }, [selectedCode])

  const handleSendMessage = async () => {
    const messageText = inputMessage.trim()
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
      console.log('[Chatbot] Sending query:', messageText)
      const data = await apiClient.sendChatbotQuery({
        query: messageText,
        context: {
          currentPage: 'chat',
          selectedItems: [],
          userRole: 'user',
          recentActions: []
        },
        sessionId,
        provider: selectedProvider || undefined,
        model: selectedModel || undefined
      })

      console.log('[Chatbot] Response data:', data)
      
      const botMessage: ChatMessage = {
        id: data.message.id,
        type: 'bot',
        content: data.message.content,
        timestamp: data.message.timestamp,
        codeSnippets: data.message.code_snippets,
        documents: data.message.documents,
        database_results: data.message.database_results,
        metadata: data.message.metadata
      }

      setMessages(prev => [...prev, botMessage])
      
      if (data.message.code_snippets && data.message.code_snippets.length > 0) {
        setSelectedCode(data.message.code_snippets[0])
        setActiveTab('code')
        setIsViewerVisible(true)
      } else if (data.message.documents && data.message.documents.length > 0) {
        setSelectedDocument(data.message.documents[0])
        setActiveTab('docs')
        setIsViewerVisible(true)
      }
    } catch (error) {
      console.error('Chat error:', error)
      
      const fallbackMessage: ChatMessage = {
        id: `bot-${Date.now()}`,
        type: 'bot',
        content: `I understand you're asking about **"${messageText}"**. 

The chatbot backend is in development mode. Once fully connected, I'll be able to:
• Search your codebase using **CodeGraph** 
• Find relevant documents using **RAG**
• Provide intelligent answers using **LLM**

Here's a sample of what I can show you:`,
        timestamp: new Date().toISOString(),
        codeSnippets: [{
          language: 'python',
          code: `# Example: CodeGraph Service
async def search_symbols(query: str, limit: int = 10):
    """Search for code symbols by name"""
    results = await db.execute(
        text("""
            SELECT name, symbol_type, file_path, line_number
            FROM codegraph_symbols
            WHERE name ILIKE :query
            LIMIT :limit
        """),
        {"query": f"%{query}%", "limit": limit}
    )
    return results.fetchall()`,
          file_path: 'orchestrator/services/codegraph_service.py',
          line_number: 615,
          symbol_name: 'search_symbols',
          explanation: 'This function performs symbol search in the codebase'
        }],
        metadata: {
          intent: 'code_search',
          confidence: 0.85,
          source: 'llm',
          processing_time: 0.5
        }
      }
      
      setMessages(prev => [...prev, fallbackMessage])
      setSelectedCode(fallbackMessage.codeSnippets![0])
      setIsViewerVisible(true)
    } finally {
      setIsTyping(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const renderMessage = (message: ChatMessage) => {
    const parts = message.content.split(/(\*\*[^*]+\*\*)/g)
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="text-white">{part.slice(2, -2)}</strong>
      }
      return <span key={i}>{part}</span>
    })
  }

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">AI Assistant</h1>
          <p className="text-muted-foreground mt-1">
            Powered by CodeGraph, RAG & Semantic Search
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2" />
            AI Online
          </Badge>
          <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
            Session: {sessionId.slice(-8)}
          </Badge>
          
          {/* Model Selector */}
          <div className="flex items-center space-x-2">
            <Select
              value={selectedProvider}
              onValueChange={(value) => {
                setSelectedProvider(value)
                // Reset model when provider changes
                const providerModels = availableModels[value] || []
                if (providerModels.length > 0) {
                  setSelectedModel(providerModels[0].model_id)
                } else {
                  setSelectedModel('')
                }
              }}
            >
              <SelectTrigger className="w-32 bg-gray-800/50 border-gray-700 text-white">
                <SelectValue placeholder="Provider" />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700">
                {Object.keys(availableModels).map(provider => (
                  <SelectItem key={provider} value={provider} className="text-white">
                    {provider.charAt(0).toUpperCase() + provider.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select
              value={selectedModel}
              onValueChange={setSelectedModel}
            >
              <SelectTrigger className="w-48 bg-gray-800/50 border-gray-700 text-white">
                <SelectValue placeholder="Model" />
              </SelectTrigger>
              <SelectContent className="bg-gray-900 border-gray-700">
                {(availableModels[selectedProvider] || []).map((model: any) => (
                  <SelectItem key={model.model_id} value={model.model_id} className="text-white">
                    {model.display_name || model.model_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => setMessages([messages[0]])}
            className="border-gray-700 hover:bg-gray-800"
          >
            <RotateCw className="w-4 h-4 mr-2" />
            Clear
          </Button>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className={`grid gap-6 transition-all duration-300 ${
        isViewerVisible ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'
      }`}>
        {/* Chat Panel */}
        <Card className="glass-card flex flex-col" style={{ height: 'calc(100vh - 280px)' }}>
          <CardContent className="flex flex-col h-full p-0">
            {/* Messages */}
            <ScrollArea className="flex-1 p-6">
              <div className="space-y-6">
                <AnimatePresence>
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`flex items-start space-x-3 max-w-[85%] ${
                        message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                      }`}>
                        {/* Avatar */}
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                          message.type === 'user' 
                            ? 'bg-blue-600' 
                            : 'bg-gradient-to-br from-orange-500 to-red-500'
                        }`}>
                          {message.type === 'user' ? (
                            <User className="w-5 h-5 text-white" />
                          ) : (
                            <Bot className="w-5 h-5 text-white" />
                          )}
                        </div>

                        {/* Message Content */}
                        <div className="flex-1 space-y-3">
                          <div className={`p-4 rounded-lg ${
                            message.type === 'user'
                              ? 'bg-blue-600/20 border border-blue-500/30'
                              : 'bg-gray-800/50 border border-gray-700/50'
                          }`}>
                            <div className="text-sm text-gray-100 whitespace-pre-wrap leading-relaxed">
                              {renderMessage(message)}
                            </div>
                            
                            {/* Metadata */}
                            {message.metadata && message.type === 'bot' && (
                              <div className="mt-3 pt-3 border-t border-gray-700/50 flex items-center justify-between text-xs">
                                <div className="flex items-center space-x-3 text-gray-400">
                                  {message.metadata.source && (
                                    <Badge variant="outline" className={`
                                      ${message.metadata.source === 'codegraph' ? 'bg-purple-500/10 border-purple-500/20 text-purple-400' : ''}
                                      ${message.metadata.source === 'rag' ? 'bg-blue-500/10 border-blue-500/20 text-blue-400' : ''}
                                      ${message.metadata.source === 'semantic' ? 'bg-green-500/10 border-green-500/20 text-green-400' : ''}
                                      ${message.metadata.source === 'database' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : ''}
                                      ${message.metadata.source === 'llm' ? 'bg-orange-500/10 border-orange-500/20 text-orange-400' : ''}
                                    `}>
                                      {message.metadata.source.toUpperCase()}
                                    </Badge>
                                  )}
                                  {message.metadata.processing_time !== undefined && (
                                    <span>{(message.metadata.processing_time).toFixed(2)}s</span>
                                  )}
                                  {message.metadata.confidence !== undefined && (
                                    <span>{(message.metadata.confidence * 100).toFixed(0)}% confidence</span>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Code Snippets */}
                          {message.codeSnippets && message.codeSnippets.length > 0 && (
                            <div className="space-y-2">
                              {message.codeSnippets.map((snippet, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => {
                                    setSelectedCode(snippet)
                                    setActiveTab('code')
                                    setIsViewerVisible(true)
                                  }}
                                  className="w-full text-left p-3 rounded-lg bg-gray-800/30 border border-gray-700/50 hover:bg-gray-800/50 hover:border-gray-600 transition-all group"
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2">
                                      <Code className="w-4 h-4 text-purple-400" />
                                      <span className="text-sm text-gray-300 font-mono">{snippet.symbol_name || 'Code'}</span>
                                      <span className="text-xs text-gray-500">{snippet.file_path}</span>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-gray-300" />
                                  </div>
                                </button>
                              ))}
                            </div>
                          )}

                          {/* Documents */}
                          {message.documents && message.documents.length > 0 && (
                            <div className="space-y-2">
                              {message.documents.map((doc, idx) => (
                                <button
                                  key={idx}
                                  onClick={() => {
                                    setSelectedDocument(doc)
                                    setActiveTab('docs')
                                    setIsViewerVisible(true)
                                  }}
                                  className="w-full text-left p-3 rounded-lg bg-gray-800/30 border border-gray-700/50 hover:bg-gray-800/50 hover:border-gray-600 transition-all group"
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-2">
                                      <FileText className="w-4 h-4 text-blue-400" />
                                      <span className="text-sm text-gray-300">{doc.filename}</span>
                                      <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                                        {(doc.similarity * 100).toFixed(0)}%
                                      </Badge>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-gray-300" />
                                  </div>
                                </button>
                              ))}
                            </div>
                          )}

                          {/* Database Results */}
                          {message.database_results && message.database_results.length > 0 && (
                            <div className="space-y-3">
                              {message.database_results.map((dbResult, idx) => (
                                <div
                                  key={idx}
                                  className="p-4 rounded-lg bg-gradient-to-br from-green-900/20 to-emerald-900/20 border border-green-500/30"
                                >
                                  <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center space-x-2">
                                      <Database className="w-4 h-4 text-green-400" />
                                      <span className="text-sm font-medium text-green-300">{dbResult.database}</span>
                                      <Badge variant="outline" className="bg-green-500/10 border-green-500/20 text-green-400 text-xs">
                                        {dbResult.row_count} rows • {dbResult.execution_time_ms?.toFixed(0)}ms
                                      </Badge>
                                    </div>
                                  </div>
                                  
                                  {/* SQL Query */}
                                  <div className="mt-2 p-2 bg-gray-900/50 rounded text-xs font-mono text-gray-300 overflow-x-auto">
                                    {dbResult.sql}
                                  </div>
                                  
                                  {/* Data Preview */}
                                  {dbResult.data && dbResult.data.length > 0 && (
                                    <div className="mt-3 overflow-x-auto">
                                      <table className="w-full text-xs">
                                        <thead>
                                          <tr className="border-b border-green-500/20">
                                            {dbResult.columns?.map(col => (
                                              <th key={col} className="text-left p-1.5 text-green-300 font-medium">
                                                {col}
                                              </th>
                                            ))}
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {dbResult.data.slice(0, 3).map((row, rowIdx) => (
                                            <tr key={rowIdx} className="border-b border-gray-700/30">
                                              {dbResult.columns?.map(col => (
                                                <td key={col} className="p-1.5 text-gray-300">
                                                  {String(row[col]).substring(0, 30)}
                                                </td>
                                              ))}
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                      {dbResult.row_count > 3 && (
                                        <p className="text-xs text-gray-500 mt-2 text-center">
                                          +{dbResult.row_count - 3} more rows
                                        </p>
                                      )}
                                    </div>
                                  )}

                                  {/* PandasAI Insight */}
                                  {dbResult.pandas_ai && (
                                    <div className="mt-4 space-y-3">
                                      {dbResult.pandas_ai.summary && (
                                        <div className="p-3 rounded bg-gray-900/40 border border-gray-800/60 text-sm text-gray-200">
                                          {dbResult.pandas_ai.summary}
                                        </div>
                                      )}
                                      {dbResult.pandas_ai.charts && dbResult.pandas_ai.charts.length > 0 && (
                                        <div className="grid gap-3 md:grid-cols-2">
                                          {dbResult.pandas_ai.charts.map((chart, chartIdx) => (
                                            <div
                                              key={`${chart.filename}-${chartIdx}`}
                                              className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-3 flex flex-col items-center"
                                            >
                                              <img
                                                src={`data:${chart.mime_type};base64,${chart.base64}`}
                                                alt={chart.filename}
                                                className="rounded-md border border-gray-800/40 max-h-64 object-contain"
                                              />
                                              <span className="mt-2 text-xs text-gray-500">
                                                {chart.filename}
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Actions */}
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
                                <Button variant="ghost" size="sm" className="text-gray-500 hover:text-green-400 p-1 h-auto">
                                  <ThumbsUp className="w-3 h-3" />
                                </Button>
                                <Button variant="ghost" size="sm" className="text-gray-500 hover:text-red-400 p-1 h-auto">
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
                    </motion.div>
                  ))}
                </AnimatePresence>

                {/* Typing Indicator */}
                {isTyping && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-start space-x-3"
                  >
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                    <div className="bg-gray-800/50 border border-gray-700/50 rounded-lg p-4">
                      <div className="flex space-x-1">
                        {[0, 1, 2].map((i) => (
                          <motion.div
                            key={i}
                            className="w-2 h-2 bg-gray-400 rounded-full"
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
                <Input
                  ref={inputRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask about code, documents, agents, workflows..."
                  className="flex-1 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
                  disabled={isTyping}
                />
                <Button 
                  type="submit" 
                  disabled={!inputMessage.trim() || isTyping}
                  className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 px-6"
                >
                  <Send className="w-5 h-5" />
                </Button>
              </form>
              <div className="mt-2 text-xs text-gray-500 flex items-center justify-center space-x-2">
                <Sparkles className="w-3 h-3" />
                <span>AI-powered with CodeGraph, RAG & Semantic Search</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Code/Docs Viewer */}
        {isViewerVisible && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
          >
            <Card className="glass-card flex flex-col" style={{ height: 'calc(100vh - 280px)' }}>
              <CardContent className="flex flex-col h-full p-0">
                <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'code' | 'docs')} className="flex flex-col h-full">
                  <div className="p-4 border-b border-gray-800/50 flex items-center justify-between">
                    <TabsList className="bg-gray-800/50">
                      <TabsTrigger value="code" className="data-[state=active]:bg-purple-500/20">
                        <Code className="w-4 h-4 mr-2" />
                        Code
                      </TabsTrigger>
                      <TabsTrigger value="docs" className="data-[state=active]:bg-blue-500/20">
                        <BookOpen className="w-4 h-4 mr-2" />
                        Documents
                      </TabsTrigger>
                    </TabsList>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsViewerVisible(false)}
                      className="text-gray-400 hover:text-white"
                    >
                      <EyeOff className="w-4 h-4" />
                    </Button>
                  </div>

                  <TabsContent value="code" className="flex-1 m-0 min-h-0">
                    <ScrollArea className="h-full">
                      {selectedCode ? (
                        <div className="p-6 space-y-4">
                          {/* Code Info */}
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <div>
                                <h3 className="text-lg font-semibold text-white font-mono">{selectedCode.symbol_name}</h3>
                                <p className="text-sm text-gray-400 font-mono">{selectedCode.file_path}</p>
                              </div>
                              <Badge variant="outline" className="bg-purple-500/10 border-purple-500/20 text-purple-400">
                                {selectedCode.language}
                              </Badge>
                            </div>
                            {selectedCode.line_number && (
                              <p className="text-xs text-gray-500">Line {selectedCode.line_number}</p>
                            )}
                            {selectedCode.explanation && (
                              <p className="text-sm text-gray-300 italic">{selectedCode.explanation}</p>
                            )}
                          </div>

                          {/* Code Block */}
                          <div className="relative">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => copyToClipboard(selectedCode.code)}
                              className="absolute top-2 right-2 text-gray-400 hover:text-white z-10"
                            >
                              <Copy className="w-4 h-4" />
                            </Button>
                            <pre className="rounded-lg overflow-x-auto bg-[#2d2d2d] p-4 text-sm">
                              <code className={`language-${selectedCode.language}`}>
                                {selectedCode.code}
                              </code>
                            </pre>
                          </div>
                        </div>
                      ) : (
                        <div className="h-full flex items-center justify-center text-gray-500">
                          <div className="text-center space-y-3">
                            <Terminal className="w-16 h-16 mx-auto opacity-50" />
                            <p>No code selected</p>
                            <p className="text-sm">Ask a code question to see results here</p>
                          </div>
                        </div>
                      )}
                    </ScrollArea>
                  </TabsContent>

                  <TabsContent value="docs" className="flex-1 m-0 min-h-0">
                    <ScrollArea className="h-full">
                      {selectedDocument ? (
                        <div className="p-6 space-y-4">
                          {/* Document Info */}
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <div>
                                <h3 className="text-lg font-semibold text-white">{selectedDocument.filename}</h3>
                                <p className="text-sm text-gray-400">Document ID: {selectedDocument.id}</p>
                              </div>
                              <Badge variant="outline" className="bg-blue-500/10 border-blue-500/20 text-blue-400">
                                {(selectedDocument.similarity * 100).toFixed(1)}% Match
                              </Badge>
                            </div>
                          </div>

                          {/* Document Content */}
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Content Excerpt</h4>
                            <div className="rounded-lg bg-gray-800/50 p-4 border border-gray-700/50">
                              <div className="prose prose-invert prose-sm max-w-none">
                                <p className="text-gray-300 whitespace-pre-wrap">{selectedDocument.excerpt}</p>
                              </div>
                            </div>
                          </div>

                          {/* Actions */}
                          <div className="flex gap-2 pt-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => copyToClipboard(selectedDocument.excerpt)}
                              className="bg-gray-800/50 border-gray-700 hover:bg-gray-800 hover:border-gray-600"
                            >
                              <Copy className="w-4 h-4 mr-2" />
                              Copy Excerpt
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                // Open Documents page - user can find their document there
                                window.open('/documents', '_blank')
                              }}
                              className="bg-gray-800/50 border-gray-700 hover:bg-gray-800 hover:border-gray-600"
                            >
                              <BookOpen className="w-4 h-4 mr-2" />
                              View in Documents
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="h-full flex items-center justify-center text-gray-500">
                          <div className="text-center space-y-3">
                            <BookOpen className="w-16 h-16 mx-auto opacity-50" />
                            <p>No document selected</p>
                            <p className="text-sm">Ask about documents to see results here</p>
                          </div>
                        </div>
                      )}
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Show Viewer Button */}
        {!isViewerVisible && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed right-6 top-1/2 transform -translate-y-1/2"
          >
            <Button
              onClick={() => setIsViewerVisible(true)}
              className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600"
              size="sm"
            >
              <Eye className="w-4 h-4 mr-2" />
              Show Viewer
            </Button>
          </motion.div>
        )}
      </div>
    </div>
  )
}

