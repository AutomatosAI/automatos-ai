'use client'

import { FileText } from 'lucide-react'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

interface CitationBadgeProps {
  index: number
  title: string
  similarity?: number
  onClick?: () => void
}

export function CitationBadge({ index, title, similarity, onClick }: CitationBadgeProps) {
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            onClick={onClick}
            className="inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30 hover:border-blue-400/50 transition-all cursor-pointer align-super ml-0.5"
          >
            {index}
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="flex items-center gap-2">
            <FileText className="w-3 h-3 text-blue-400 flex-shrink-0" />
            <span className="text-xs font-medium truncate">{title}</span>
            {similarity != null && (
              <span className="text-xs text-muted-foreground ml-1">
                {(similarity * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export interface CitationRef {
  index: number
  title: string
  documentId?: number
  similarity?: number
}

/**
 * Parse citation references like [1], [2] from text.
 * Returns segments alternating between text and citation markers.
 */
export function parseCitations(
  text: string,
  refs: CitationRef[]
): Array<{ type: 'text'; value: string } | { type: 'citation'; ref: CitationRef }> {
  const pattern = /\[(\d+)\]/g
  const segments: Array<{ type: 'text'; value: string } | { type: 'citation'; ref: CitationRef }> = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = pattern.exec(text)) !== null) {
    const idx = parseInt(match[1], 10)
    const ref = refs.find(r => r.index === idx)

    if (match.index > lastIndex) {
      segments.push({ type: 'text', value: text.slice(lastIndex, match.index) })
    }

    if (ref) {
      segments.push({ type: 'citation', ref })
    } else {
      segments.push({ type: 'text', value: match[0] })
    }

    lastIndex = pattern.lastIndex
  }

  if (lastIndex < text.length) {
    segments.push({ type: 'text', value: text.slice(lastIndex) })
  }

  return segments
}
