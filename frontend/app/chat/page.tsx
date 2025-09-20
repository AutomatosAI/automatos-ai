
'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { ChatInterface } from '@/components/chat-interface'
import { Bot, MessageCircle, Sparkles, ArrowRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

export default function ChatPage() {
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [initialQuery, setInitialQuery] = useState('')
  const pathname = usePathname()

  const chatContext = {
    currentPage: 'chat',
    selectedItems: [],
    userRole: 'user',
    recentActions: []
  }

  const quickStartQuestions = [
    {
      category: 'System Health',
      icon: '🟢',
      questions: [
        "What's the current system health?",
        "Show me system performance metrics",
        "Are there any active alerts?"
      ]
    },
    {
      category: 'Agent Management',
      icon: '🤖',
      questions: [
        "Which agents are performing best?",
        "Show me agent utilization rates",
        "How can I optimize agent performance?"
      ]
    },
    {
      category: 'Document Processing',
      icon: '📄',
      questions: [
        "What documents were processed today?",
        "Find documents with processing errors",
        "Analyze document processing trends"
      ]
    },
    {
      category: 'Workflow Automation',
      icon: '🔄',
      questions: [
        "Show active workflow status",
        "Why did my workflow fail?",
        "Help me create a new workflow"
      ]
    }
  ]

  const handleQuestionClick = (question: string) => {
    setInitialQuery(question)
    setIsChatOpen(true)
  }

  if (isChatOpen) {
    return (
      <ChatInterface
        onClose={() => setIsChatOpen(false)}
        context={chatContext}
        initialQuery={initialQuery}
      />
    )
  }

  return (
    <div className="min-h-screen p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-6xl mx-auto space-y-8"
      >
        {/* Header */}
        <div className="text-center space-y-4">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="w-20 h-20 mx-auto rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center"
          >
            <Bot className="w-10 h-10 text-white" />
          </motion.div>
          
          <div>
            <h1 className="text-4xl font-bold gradient-text mb-2">
              Automatos AI Assistant
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Your intelligent companion for managing agents, workflows, documents, and system operations
            </p>
          </div>
          
          <div className="flex items-center justify-center space-x-4">
            <Badge variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-2" />
              AI Online
            </Badge>
            <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
              Orchestrator Connected
            </Badge>
          </div>
        </div>

        {/* Quick Start Button */}
        <div className="text-center">
          <Button
            onClick={() => setIsChatOpen(true)}
            className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 px-8 py-6 text-lg font-semibold shadow-lg hover:shadow-xl transition-all"
            size="lg"
          >
            <MessageCircle className="w-6 h-6 mr-2" />
            Start Conversation
            <ArrowRight className="w-6 h-6 ml-2" />
          </Button>
        </div>

        {/* Quick Start Questions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {quickStartQuestions.map((category, index) => (
            <motion.div
              key={category.category}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card className="glass-card hover:bg-secondary/50 transition-all duration-300 hover:scale-105">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center space-x-2 text-lg">
                    <span className="text-2xl">{category.icon}</span>
                    <span>{category.category}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {category.questions.map((question, questionIndex) => (
                    <button
                      key={questionIndex}
                      onClick={() => handleQuestionClick(question)}
                      className="w-full text-left p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors text-sm text-muted-foreground hover:text-foreground group"
                    >
                      <div className="flex items-center justify-between">
                        <span>• {question}</span>
                        <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </button>
                  ))}
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Features */}
        <div className="text-center space-y-6">
          <h2 className="text-2xl font-semibold gradient-text">
            What I Can Help You With
          </h2>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { icon: '📊', label: 'System Monitoring' },
              { icon: '🤖', label: 'Agent Management' },
              { icon: '📄', label: 'Document Analysis' },
              { icon: '🔄', label: 'Workflow Automation' },
              { icon: '📈', label: 'Performance Analytics' },
              { icon: '🛠️', label: 'Troubleshooting' },
              { icon: '💡', label: 'Optimization Tips' },
              { icon: '📚', label: 'Best Practices' }
            ].map((feature, index) => (
              <motion.div
                key={feature.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="p-4 rounded-lg bg-secondary/20 hover:bg-secondary/30 transition-colors"
              >
                <div className="text-2xl mb-2">{feature.icon}</div>
                <div className="text-sm font-medium">{feature.label}</div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-muted-foreground">
          <div className="flex items-center justify-center space-x-2 mb-2">
            <Sparkles className="w-4 h-4" />
            <span>Powered by advanced AI orchestration</span>
          </div>
          <p>Ask me anything about your Automatos platform!</p>
        </div>
      </motion.div>
    </div>
  )
}
