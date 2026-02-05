
'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bug,
  X,
  HelpCircle,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  ImageIcon,
  Trash2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useUser } from '@clerk/nextjs'
import { useSubmitBugReport, type BugReportRequest } from '@/hooks/use-bug-report-api'

// =============================================================================
// US-006: Per-page help content
// =============================================================================

interface HelpItem {
  title: string
  description: string
  link?: string
}

const PAGE_HELP_CONTENT: Record<string, HelpItem[]> = {
  dashboard: [
    { title: 'System Overview', description: 'View real-time health, active agents, and recent workflow activity.' },
    { title: 'Quick Actions', description: 'Use the cards to jump to agents, workflows, or documents.' },
    { title: 'Performance Metrics', description: 'Monitor response times, throughput, and error rates from the analytics panel.' },
  ],
  agents: [
    { title: 'Create an Agent', description: 'Click "New Agent" to configure a custom AI agent with specific skills and tools.' },
    { title: 'Assign Tools', description: 'Connect Composio integrations (Slack, Gmail, Jira, etc.) to extend agent capabilities.', link: '/tools' },
    { title: 'Monitor Performance', description: 'Track each agent\'s success rate, latency, and task throughput on their detail page.' },
  ],
  documents: [
    { title: 'Upload Documents', description: 'Drag and drop files to add them to the RAG knowledge base for agent retrieval.' },
    { title: 'Cloud Sync', description: 'Connect Google Drive or S3 to automatically sync documents.', link: '/documents?tab=cloud' },
    { title: 'Search & Query', description: 'Use natural language to search across all uploaded documents.' },
  ],
  workflows: [
    { title: 'Build a Workflow', description: 'Use the visual editor to chain agents, tools, and conditions into automated pipelines.' },
    { title: 'Templates', description: 'Start from a pre-built template to accelerate workflow creation.' },
    { title: 'Execution History', description: 'Review past runs, inspect step-by-step logs, and debug failures.' },
  ],
  tools: [
    { title: 'Browse Integrations', description: 'Explore 500+ Composio-powered tools across productivity, dev, and communication apps.' },
    { title: 'Connect an App', description: 'Click "Connect" on any tool to authenticate via OAuth and enable it for your agents.' },
    { title: 'Manage Permissions', description: 'Control which agents can access specific tools and actions.' },
  ],
  analytics: [
    { title: 'Performance Trends', description: 'Track agent and workflow performance over time with interactive charts.' },
    { title: 'Usage Insights', description: 'See which agents and tools are used most frequently.' },
    { title: 'Export Reports', description: 'Download analytics data as CSV or PDF for offline analysis.' },
  ],
  context: [
    { title: 'Context Engineering', description: 'Fine-tune how agents retrieve and prioritize information from your knowledge base.' },
    { title: 'Similarity Settings', description: 'Adjust RAG thresholds to control retrieval precision vs. recall.' },
    { title: 'Memory Management', description: 'View and manage agent memory across short-term and long-term stores.' },
  ],
  playbooks: [
    { title: 'Create Playbooks', description: 'Define step-by-step procedures that agents follow for recurring tasks.' },
    { title: 'Version Control', description: 'Track changes to playbooks and roll back to previous versions.' },
    { title: 'Share & Reuse', description: 'Publish playbooks for your team or import community playbooks.' },
  ],
  chat: [
    { title: 'Chat with Agents', description: 'Send messages to any configured agent and get streaming AI responses.' },
    { title: 'Artifacts', description: 'Agents can generate code, tables, and rich content inline.' },
    { title: 'History', description: 'All conversations are saved — pick up where you left off anytime.' },
  ],
}

// =============================================================================
// Types
// =============================================================================

interface ChatWidgetProps {
  position?: 'bottom-right' | 'bottom-left'
  context?: {
    currentPage: string
    selectedItems: any[]
    userRole: string
    recentActions: any[]
  }
}

type FormState = 'form' | 'loading' | 'success' | 'error'

// =============================================================================
// Component
// =============================================================================

export function PilotHelperWidget({
  position = 'bottom-right',
  context = {
    currentPage: 'dashboard',
    selectedItems: [],
    userRole: 'user',
    recentActions: [],
  },
}: ChatWidgetProps) {
  const { user } = useUser()
  const submitBugReport = useSubmitBugReport()

  // Widget state
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('help')

  // Form state
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [severity, setSeverity] = useState('Minor')
  const [category, setCategory] = useState('Other')
  const [screenshot, setScreenshot] = useState<string | null>(null)
  const [formState, setFormState] = useState<FormState>('form')
  const [errorMessage, setErrorMessage] = useState('')
  const [successKey, setSuccessKey] = useState('')
  const [successUrl, setSuccessUrl] = useState('')

  // Console error capture
  const consoleErrorsRef = useRef<string[]>([])

  useEffect(() => {
    const originalError = console.error
    const capture = (...args: any[]) => {
      const msg = args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')
      consoleErrorsRef.current = [...consoleErrorsRef.current.slice(-19), msg]
      originalError.apply(console, args)
    }
    console.error = capture
    return () => {
      console.error = originalError
    }
  }, [])

  // Escape key closes widget
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Screenshot paste handler
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (!file) continue
        if (file.size > 5 * 1024 * 1024) {
          setErrorMessage('Screenshot must be under 5 MB')
          return
        }
        const reader = new FileReader()
        reader.onload = () => {
          setScreenshot(reader.result as string)
        }
        reader.readAsDataURL(file)
        break
      }
    }
  }, [])

  const resetForm = () => {
    setTitle('')
    setDescription('')
    setSeverity('Minor')
    setCategory('Other')
    setScreenshot(null)
    setFormState('form')
    setErrorMessage('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !description.trim()) return

    setFormState('loading')
    setErrorMessage('')

    const payload: BugReportRequest = {
      title: title.trim(),
      description: description.trim(),
      severity,
      category,
      screenshot_base64: screenshot || undefined,
      context: {
        url: window.location.href,
        page: context.currentPage,
        user_agent: navigator.userAgent,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
        user_email: user?.primaryEmailAddress?.emailAddress,
        user_name: user?.fullName || undefined,
        console_errors: consoleErrorsRef.current.slice(-20),
        timestamp: new Date().toISOString(),
      },
    }

    submitBugReport.mutate(payload, {
      onSuccess: (data) => {
        if (data.success) {
          setFormState('success')
          setSuccessKey(data.jira_key || '')
          setSuccessUrl(data.jira_url || '')
        } else {
          setFormState('error')
          setErrorMessage(data.message || 'Failed to create bug report')
        }
      },
      onError: (err: any) => {
        setFormState('error')
        setErrorMessage(err?.message || 'An unexpected error occurred')
      },
    })
  }

  const positionClasses = {
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
  }

  const helpItems = PAGE_HELP_CONTENT[context.currentPage] || PAGE_HELP_CONTENT.dashboard

  return (
    <div className={`fixed ${positionClasses[position]} z-[60]`}>
      <AnimatePresence mode="wait">
        {isOpen && (
          <motion.div
            key="popup"
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
            className="mb-4 w-96 max-h-[600px] overflow-y-auto glass-card card-glow rounded-2xl shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 pt-4 pb-2">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                  <Bug className="w-4 h-4 text-white" />
                </div>
                <h3 className="font-semibold text-foreground">Pilot Helper</h3>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="px-4 pb-4">
              <TabsList className="w-full mb-3">
                <TabsTrigger value="help" className="flex-1">Help</TabsTrigger>
                <TabsTrigger value="bug" className="flex-1">Report Bug</TabsTrigger>
              </TabsList>

              {/* ---- Help Tab ---- */}
              <TabsContent value="help" className="space-y-3 mt-0">
                {helpItems.map((item, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-secondary/40 hover:bg-secondary/60 transition-colors"
                  >
                    <div className="flex items-start gap-2">
                      {item.link ? (
                        <BookOpen className="w-4 h-4 mt-0.5 shrink-0 text-orange-400" />
                      ) : (
                        <HelpCircle className="w-4 h-4 mt-0.5 shrink-0 text-orange-400" />
                      )}
                      <div>
                        <p className="text-sm font-medium text-foreground">{item.title}</p>
                        <p className="text-xs text-muted-foreground mt-0.5">{item.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </TabsContent>

              {/* ---- Bug Report Tab ---- */}
              <TabsContent value="bug" className="mt-0">
                {formState === 'form' || formState === 'loading' ? (
                  <form onSubmit={handleSubmit} className="space-y-3" onPaste={handlePaste}>
                    {/* Title */}
                    <div className="space-y-1">
                      <Label htmlFor="bug-title" className="text-xs">Title</Label>
                      <Input
                        id="bug-title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="Brief summary of the issue"
                        required
                        disabled={formState === 'loading'}
                        className="text-sm"
                      />
                    </div>

                    {/* Description */}
                    <div className="space-y-1">
                      <Label htmlFor="bug-desc" className="text-xs">Description</Label>
                      <Textarea
                        id="bug-desc"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="What happened? Steps to reproduce..."
                        required
                        disabled={formState === 'loading'}
                        className="text-sm min-h-[80px] resize-none"
                      />
                    </div>

                    {/* Severity + Category row */}
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <Label className="text-xs">Severity</Label>
                        <Select value={severity} onValueChange={setSeverity} disabled={formState === 'loading'}>
                          <SelectTrigger className="text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Critical">Critical</SelectItem>
                            <SelectItem value="Major">Major</SelectItem>
                            <SelectItem value="Minor">Minor</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Category</Label>
                        <Select value={category} onValueChange={setCategory} disabled={formState === 'loading'}>
                          <SelectTrigger className="text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="UI Bug">UI Bug</SelectItem>
                            <SelectItem value="Data Issue">Data Issue</SelectItem>
                            <SelectItem value="Performance">Performance</SelectItem>
                            <SelectItem value="Other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {/* Screenshot paste area */}
                    <div className="space-y-1">
                      <Label className="text-xs">Screenshot</Label>
                      {screenshot ? (
                        <div className="relative rounded-lg overflow-hidden border border-border">
                          <img src={screenshot} alt="Screenshot preview" className="w-full max-h-32 object-contain bg-black/5" />
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setScreenshot(null)}
                            className="absolute top-1 right-1 h-6 w-6 p-0 bg-black/50 hover:bg-black/70 text-white rounded-full"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 p-3 rounded-lg border border-dashed border-border text-muted-foreground text-xs">
                          <ImageIcon className="w-4 h-4 shrink-0" />
                          <span>Paste a screenshot (Ctrl+V / Cmd+V)</span>
                        </div>
                      )}
                    </div>

                    {/* Error message inline */}
                    {errorMessage && formState === 'form' && (
                      <p className="text-xs text-red-400">{errorMessage}</p>
                    )}

                    {/* Submit */}
                    <Button
                      type="submit"
                      disabled={formState === 'loading' || !title.trim() || !description.trim()}
                      className="w-full bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white"
                    >
                      {formState === 'loading' ? (
                        <span className="flex items-center gap-2">
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                            className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                          />
                          Submitting...
                        </span>
                      ) : (
                        'Submit Bug Report'
                      )}
                    </Button>
                  </form>
                ) : formState === 'success' ? (
                  <div className="flex flex-col items-center py-6 space-y-3">
                    <CheckCircle2 className="w-12 h-12 text-green-500" />
                    <p className="font-medium text-foreground">Bug report created!</p>
                    {successKey && (
                      successUrl ? (
                        <a
                          href={successUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-orange-400 hover:text-orange-300 underline"
                        >
                          {successKey}
                        </a>
                      ) : (
                        <span className="text-sm text-muted-foreground">{successKey}</span>
                      )
                    )}
                    <Button variant="outline" size="sm" onClick={resetForm}>
                      Submit another
                    </Button>
                  </div>
                ) : (
                  /* error state */
                  <div className="flex flex-col items-center py-6 space-y-3">
                    <AlertCircle className="w-12 h-12 text-red-500" />
                    <p className="font-medium text-foreground">Something went wrong</p>
                    <p className="text-xs text-muted-foreground text-center">{errorMessage}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setFormState('form')}
                    >
                      Try again
                    </Button>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating button */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      >
        <Button
          onClick={() => setIsOpen((v) => !v)}
          className="w-16 h-16 rounded-full bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 shadow-xl hover:shadow-2xl transition-all duration-300 group relative"
          size="lg"
        >
          <HelpCircle className="w-8 h-8 text-white group-hover:scale-110 transition-transform" />
          <motion.div
            className="absolute inset-0 rounded-full bg-gradient-to-br from-orange-400/50 to-red-400/50"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 3 }}
          />
        </Button>
      </motion.div>
    </div>
  )
}

// Backward compatibility export
export const ChatWidget = PilotHelperWidget
