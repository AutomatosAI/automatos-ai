'use client'

/**
 * The chat composer's action row, inside the bottom of the box (multimodal-input.tsx): on the left the attach button
 * and the agent picker, on the right the picked agent's tools (PRD-40: a click asks for that tool's suggestions), the
 * voice button (composer-voice.tsx: hold to talk, the quick ask) and Send, or Stop while a reply streams.
 *
 * PRD-264 EUAI-11a: every control here but the voice button sits in the AI label's fence, so a theme that hides the
 * page's label pauses them. The voice button stays out of it: its own caption and the spoken notice say Auto is an AI.
 */
import { Paperclip, Send, StopCircle } from 'lucide-react'

import { AiLabelFence } from '@/components/ai-label/ai-label-fence'
import { Button } from '@/components/ui/button'
import { ToolLogo } from '@/components/ui/tool-logo'

import { AgentSelector, type Agent } from './agent-selector'
import { ComposerVoice } from './composer-voice'

/** How many of the agent's tools the row shows as logos; the rest are a count. */
const SHOWN_TOOLS = 4

type ToolIconClick = (appName: string) => void

export interface ComposerActionsProps {
  readonly isStreaming: boolean
  /** Files are uploading: the attach button waits for them. */
  readonly uploading: boolean
  /** Opens the file picker. */
  readonly onAttach: () => void
  readonly selectedAgentId?: number | null
  readonly onAgentChange?: (agentId: number | null) => void
  /** The picker hands back the picked agent, whose tools the row shows. */
  readonly onAgentData: (agent: Agent | null) => void
  readonly activeAgent: Agent | null
  readonly onToolIconClick?: ToolIconClick
  /** Something is typed, so Send can send. */
  readonly canSend: boolean
  readonly stop: () => void
}

interface AgentToolIconsProps {
  readonly agent: Agent | null
  readonly onToolIconClick?: ToolIconClick
}

/** The picked agent's tools: the first SHOWN_TOOLS as logos, then how many more. Nothing when it has none. */
function AgentToolIcons({ agent, onToolIconClick }: AgentToolIconsProps) {
  const tools: NonNullable<Agent['tools']> = agent?.tools ?? []
  if (tools.length === 0) return null
  return (
    <div className="flex gap-2 items-center animate-in fade-in zoom-in-50 duration-300">
      {tools.slice(0, SHOWN_TOOLS).map((tool) => (
        <button
          key={tool.id}
          type="button"
          onClick={() => onToolIconClick?.(tool.name)}
          className="relative hover:scale-110 transition-all duration-220 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-lg"
          title={`Click for ${tool.name} suggestions`}
          aria-label={`Show suggestions for ${tool.name}`}
        >
          <ToolLogo
            name={tool.name}
            logo={tool.icon}
            size={28}
            showBackground={true}
            className="bg-background/80 ring-1 ring-primary/30 border border-primary/20 rounded-lg"
          />
        </button>
      ))}
      {tools.length > SHOWN_TOOLS && (
        <div className="w-8 h-8 rounded-lg bg-secondary/80 ring-1 ring-primary/30 flex items-center justify-center text-[10px] font-bold text-muted-foreground border border-primary/20 relative z-10">
          +{tools.length - SHOWN_TOOLS}
        </div>
      )}
    </div>
  )
}

interface SendOrStopProps {
  readonly isStreaming: boolean
  readonly canSend: boolean
  readonly stop: () => void
}

/** Send (the form's submit), or Stop while a reply streams. */
function SendOrStop({ isStreaming, canSend, stop }: SendOrStopProps) {
  if (isStreaming) {
    return (
      <Button
        type="button"
        aria-label="Stop generating"
        onClick={stop}
        size="icon"
        className="bg-destructive hover:bg-destructive/80 h-9 w-9 rounded-xl shadow-sm"
      >
        <StopCircle aria-hidden="true" className="w-4 h-4" />
      </Button>
    )
  }
  return (
    <Button
      type="submit"
      aria-label="Send message"
      disabled={!canSend}
      size="icon"
      className="h-10 w-10 rounded-2xl disabled:opacity-50 bg-primary hover:bg-primary/90 transition-colors"
    >
      <Send aria-hidden="true" className="w-4 h-4" />
    </Button>
  )
}

/** The composer's bottom toolbar, inside the box. */
export function ComposerActions({
  isStreaming,
  uploading,
  onAttach,
  selectedAgentId,
  onAgentChange,
  onAgentData,
  activeAgent,
  onToolIconClick,
  canSend,
  stop,
}: ComposerActionsProps) {
  return (
    <div className="flex items-center justify-between px-3 py-2 border-t border-transparent absolute bottom-0 left-0 right-0">
      {/* Left side: Attachment + Agent Selector */}
      <AiLabelFence>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
            disabled={isStreaming || uploading}
            onClick={onAttach}
          >
            <Paperclip className="w-4 h-4" />
          </Button>

          {/* PRD: Unified Agent-Chat System — the Agent Selector is the real
              routing control. PRD-180 S3 (F035): the placebo model picker
              fallback was removed (the backend never read the chosen model —
              a control that does nothing corrodes trust in the real ones). */}
          {onAgentChange && (
            <AgentSelector selectedAgentId={selectedAgentId} onAgentChange={onAgentChange} onAgentData={onAgentData} />
          )}
        </div>
      </AiLabelFence>

      {/* Active Agent Tools, the voice button and Send */}
      <div className="flex items-center gap-3 ml-auto mr-2 mt-1">
        <AiLabelFence>
          <AgentToolIcons agent={activeAgent} onToolIconClick={onToolIconClick} />
        </AiLabelFence>
        <ComposerVoice />
        <AiLabelFence>
          <SendOrStop isStreaming={isStreaming} canSend={canSend} stop={stop} />
        </AiLabelFence>
      </div>
    </div>
  )
}
