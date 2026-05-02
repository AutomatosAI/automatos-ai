'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bot,
  ChevronDown,
  ChevronRight,
  Clock,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Loader2,
  Wrench,
  MessageSquare,
  FileText,
  Copy,
  Check,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import Prism from 'prismjs'
import 'prismjs/components/prism-json'

// --- Types ---

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  name?: string
  tool_call_id?: string
}

export interface ToolCall {
  id?: string
  app_name: string
  action_name: string
  params?: Record<string, unknown>
  result?: unknown
  status?: 'success' | 'error' | 'pending'
  duration_ms?: number
}

export interface StepExecutionData {
  step_id: string
  order: number
  agent_id?: number
  agent_name?: string
  agent_model?: string
  agent_tool_count?: number
  prompt_template?: string
  status: 'pending' | 'running' | 'success' | 'failed'
  started_at?: string
  completed_at?: string
  duration_ms?: number
  tokens_used?: number
  messages?: LLMMessage[]
  tool_calls?: ToolCall[]
  result?: unknown
  error?: string
  retries?: number
}

export interface TheaterStepExecutionProps {
  step: StepExecutionData
  index: number
}

// --- Helpers ---

const formatDuration = (ms?: number) => (ms ? `${(ms / 1000).toFixed(1)}s` : '—')

const formatTimestamp = (ts?: string) => {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return '—'
  }
}

// SECURITY: Escape HTML entities to prevent XSS when Prism highlighting fails (OWASP A03:2021 - Injection)
function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function highlightJson(value: unknown): string {
  try {
    const json = typeof value === 'string' ? value : JSON.stringify(value, null, 2)
    return Prism.highlight(json, Prism.languages.json, 'json')
  } catch {
    // Fallback: escape HTML to prevent XSS when used with dangerouslySetInnerHTML
    return escapeHtml(String(value))
  }
}

// --- Sub-components ---

function StatusBadge({ status }: { status: StepExecutionData['status'] }) {
  const config = {
    pending: { label: 'Pending', className: 'text-muted-foreground border-border/40', icon: null },
    running: {
      label: 'Running',
      className: 'text-orange-400 border-orange-400/40',
      icon: <Loader2 className="w-3 h-3 animate-spin" />,
    },
    success: {
      label: 'Success',
      className: 'text-success border-success/40',
      icon: <CheckCircle className="w-3 h-3" />,
    },
    failed: {
      label: 'Failed',
      className: 'text-destructive border-destructive/40',
      icon: <AlertCircle className="w-3 h-3" />,
    },
  }[status]

  return (
    <Badge variant="outline" className={cn('text-[11px] gap-1', config.className)}>
      {config.icon}
      {config.label}
    </Badge>
  )
}

function CollapsibleSection({
  title,
  icon,
  count,
  defaultOpen = false,
  children,
}: {
  title: string
  icon: React.ReactNode
  count?: number
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-border/30 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 p-3 text-left hover:bg-white/5 transition-colors"
      >
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        )}
        <span className="text-muted-foreground flex-shrink-0">{icon}</span>
        <span className="text-sm font-medium">{title}</span>
        {count !== undefined && (
          <Badge variant="outline" className="text-[10px] h-5 ml-auto">
            {count}
          </Badge>
        )}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="p-3 pt-0 space-y-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function LLMMessagesSection({ messages }: { messages: LLMMessage[] }) {
  const getRoleStyle = (role: LLMMessage['role']) => {
    switch (role) {
      case 'system':
        return 'text-agent bg-agent/10 border-agent/20'
      case 'user':
        return 'text-info bg-info/10 border-info/20'
      case 'assistant':
        return 'text-success bg-success/10 border-success/20'
      case 'tool':
        return 'text-orange-400 bg-orange-400/10 border-orange-400/20'
      default:
        return 'text-muted-foreground bg-secondary/50 border-border/20'
    }
  }

  return (
    <div className="space-y-2">
      {messages.map((msg, i) => (
        <div key={i} className={cn('rounded-lg border p-3', getRoleStyle(msg.role))}>
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="outline" className="text-[10px] h-5">
              {msg.role}
            </Badge>
            {msg.name && (
              <span className="text-[10px] text-muted-foreground">{msg.name}</span>
            )}
          </div>
          <pre className="text-sm whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto theater-scrollbar">
            {msg.content}
          </pre>
        </div>
      ))}
    </div>
  )
}

function ToolCallsSection({ toolCalls }: { toolCalls: ToolCall[] }) {
  const [expandedTools, setExpandedTools] = useState<Set<number>>(new Set())

  const toggleTool = (idx: number) => {
    setExpandedTools((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const getToolStatusIcon = (status?: ToolCall['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-3.5 h-3.5 text-success" />
      case 'error':
        return <AlertCircle className="w-3.5 h-3.5 text-destructive" />
      case 'pending':
        return <Loader2 className="w-3.5 h-3.5 text-orange-400 animate-spin" />
      default:
        return <CheckCircle className="w-3.5 h-3.5 text-success" />
    }
  }

  return (
    <div className="space-y-2">
      {toolCalls.map((tool, i) => {
        const isExpanded = expandedTools.has(i)
        return (
          <div key={i} className="rounded-lg border border-border/30 bg-black/20 overflow-hidden">
            <button
              onClick={() => toggleTool(i)}
              className="w-full flex items-center gap-2 p-2.5 text-left hover:bg-white/5 transition-colors"
            >
              {getToolStatusIcon(tool.status)}
              <Badge
                variant="outline"
                className="text-[10px] h-5 text-orange-400 border-orange-400/30"
              >
                {tool.app_name}
              </Badge>
              <span className="text-sm font-medium truncate">{tool.action_name}</span>
              {tool.duration_ms !== undefined && (
                <span className="text-[10px] text-muted-foreground ml-auto flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatDuration(tool.duration_ms)}
                </span>
              )}
              {isExpanded ? (
                <ChevronDown className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
              )}
            </button>
            <AnimatePresence>
              {isExpanded && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.15 }}
                  className="overflow-hidden"
                >
                  <div className="p-2.5 pt-0 space-y-2">
                    {tool.params && Object.keys(tool.params).length > 0 && (
                      <div>
                        <div className="text-[10px] uppercase text-muted-foreground mb-1">
                          Parameters
                        </div>
                        <pre
                          className="text-xs bg-black/30 rounded-lg p-2 overflow-x-auto max-h-[150px] overflow-y-auto theater-scrollbar"
                          dangerouslySetInnerHTML={{
                            __html: highlightJson(tool.params),
                          }}
                        />
                      </div>
                    )}
                    {tool.result !== undefined && (
                      <div>
                        <div className="text-[10px] uppercase text-muted-foreground mb-1">
                          Result
                        </div>
                        <pre
                          className="text-xs bg-black/30 rounded-lg p-2 overflow-x-auto max-h-[200px] overflow-y-auto theater-scrollbar"
                          dangerouslySetInnerHTML={{
                            __html: highlightJson(tool.result),
                          }}
                        />
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}

function StepResultSection({ result }: { result: unknown }) {
  const [copied, setCopied] = useState(false)

  const jsonString = typeof result === 'string' ? result : JSON.stringify(result, null, 2)

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1.5 rounded-lg bg-white/5 hover:bg-white/10 transition-colors z-10"
        title="Copy result"
      >
        {copied ? (
          <Check className="w-3.5 h-3.5 text-success" />
        ) : (
          <Copy className="w-3.5 h-3.5 text-muted-foreground" />
        )}
      </button>
      <pre
        className="text-sm bg-black/30 border border-border/30 rounded-xl p-4 whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto theater-scrollbar"
        dangerouslySetInnerHTML={{ __html: highlightJson(result) }}
      />
    </div>
  )
}

// --- Main Component ---

export function TheaterStepExecution({ step, index }: TheaterStepExecutionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.3 }}
      className={cn(
        'glass-card rounded-2xl p-4 transition-all',
        step.status === 'running' && 'ring-1 ring-orange-400/30',
        step.status === 'failed' && 'ring-1 ring-destructive/30'
      )}
    >
      {/* Header: Step name, agent, status */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          {/* Step number */}
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center text-sm font-bold text-primary flex-shrink-0">
            {step.order}
          </div>

          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-semibold truncate">
                Step {step.order}: {step.step_id}
              </span>
              <StatusBadge status={step.status} />
            </div>

            {/* Agent info */}
            {step.agent_name && (
              <div className="flex items-center gap-2 mt-1">
                <Bot className="w-3.5 h-3.5 text-primary" />
                <span className="text-xs text-muted-foreground">
                  {step.agent_name}
                  {step.agent_model && (
                    <span className="text-primary/60"> ({step.agent_model})</span>
                  )}
                  {step.agent_tool_count !== undefined && (
                    <span className="text-primary/60"> &bull; {step.agent_tool_count} tools</span>
                  )}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Metrics */}
        <div className="flex items-center gap-3 text-xs text-muted-foreground flex-shrink-0">
          {step.tokens_used !== undefined && step.tokens_used > 0 && (
            <span className="flex items-center gap-1">
              <Sparkles className="w-3 h-3 text-primary" />
              {step.tokens_used.toLocaleString()}
            </span>
          )}
          {step.duration_ms !== undefined && step.duration_ms > 0 && (
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-success" />
              {formatDuration(step.duration_ms)}
            </span>
          )}
          {step.retries !== undefined && step.retries > 0 && (
            <Badge variant="outline" className="text-[10px] text-warning border-warning/30">
              {step.retries} retries
            </Badge>
          )}
        </div>
      </div>

      {/* Timestamps */}
      {(step.started_at || step.completed_at) && (
        <div className="flex items-center gap-4 mt-2 text-[10px] text-muted-foreground">
          {step.started_at && <span>Started: {formatTimestamp(step.started_at)}</span>}
          {step.completed_at && <span>Completed: {formatTimestamp(step.completed_at)}</span>}
        </div>
      )}

      {/* Error message */}
      {step.error && (
        <div className="mt-3 p-3 rounded-xl border border-destructive/30 bg-destructive/10 text-destructive/70 text-sm flex items-start gap-2">
          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold text-destructive/80">Error</div>
            <p className="mt-0.5 break-words">{step.error}</p>
          </div>
        </div>
      )}

      {/* Collapsible sections */}
      <div className="mt-3 space-y-2">
        {/* LLM Messages */}
        {step.messages && step.messages.length > 0 && (
          <CollapsibleSection
            title="LLM Messages"
            icon={<MessageSquare className="w-4 h-4" />}
            count={step.messages.length}
            defaultOpen={false}
          >
            <LLMMessagesSection messages={step.messages} />
          </CollapsibleSection>
        )}

        {/* Tool Calls */}
        {step.tool_calls && step.tool_calls.length > 0 && (
          <CollapsibleSection
            title="Tool Calls"
            icon={<Wrench className="w-4 h-4" />}
            count={step.tool_calls.length}
            defaultOpen={false}
          >
            <ToolCallsSection toolCalls={step.tool_calls} />
          </CollapsibleSection>
        )}

        {/* Step Result */}
        {step.result !== undefined && step.result !== null && (
          <CollapsibleSection
            title="Step Result"
            icon={<FileText className="w-4 h-4" />}
            defaultOpen={step.status === 'success'}
          >
            <StepResultSection result={step.result} />
          </CollapsibleSection>
        )}
      </div>
    </motion.div>
  )
}

export type { LLMMessage as TheaterLLMMessage, ToolCall as TheaterToolCall }
