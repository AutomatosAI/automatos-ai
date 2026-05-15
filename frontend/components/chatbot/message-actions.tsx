'use client'

import { useState, useEffect } from 'react'
import { Copy, ThumbsUp, ThumbsDown, RotateCw, Check, Rocket } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { voteMessage } from '@/lib/chat/api'
import { copyToClipboard } from '@/lib/utils'
import { toast } from 'sonner'
import { useMissionStore } from '@/stores/mission-store'

interface MessageActionsProps {
  chatId: string
  messageId: string
  role: 'user' | 'assistant'
  content: string
  isReadonly: boolean
  regenerate: () => void
  createdAt?: string
  onLaunchAsMission?: (content: string) => void
}

function relativeTime(dateStr?: string): string {
  if (!dateStr) return ''
  const diff = Date.now() - new Date(dateStr).getTime()
  const seconds = Math.floor(diff / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

export function MessageActions({
  chatId,
  messageId,
  role,
  content,
  isReadonly,
  regenerate,
  createdAt,
  onLaunchAsMission,
}: MessageActionsProps) {
  const isPlanMode = useMissionStore((s) => s.isPlanMode)
  const [isUpvoted, setIsUpvoted] = useState<boolean | undefined>()
  const [copied, setCopied] = useState(false)
  const [timeLabel, setTimeLabel] = useState(() => relativeTime(createdAt))

  useEffect(() => {
    setTimeLabel(relativeTime(createdAt))
    const interval = setInterval(() => setTimeLabel(relativeTime(createdAt)), 30_000)
    return () => clearInterval(interval)
  }, [createdAt])

  const handleCopy = async () => {
    if (await copyToClipboard(content)) {
      setCopied(true)
      toast.success('Copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleVote = async (upvote: boolean) => {
    try {
      await voteMessage(chatId, messageId, upvote)
      setIsUpvoted(upvote)
      toast.success(upvote ? 'Upvoted' : 'Downvoted')
    } catch {
      toast.error('Failed to vote')
    }
  }

  return (
    <div className="flex items-center space-x-1 opacity-0 group-hover/msg:opacity-100 transition-opacity duration-220">
      <Button
        variant="ghost"
        size="sm"
        onClick={handleCopy}
        className="text-muted-foreground hover:text-foreground/90 p-1 h-auto"
      >
        {copied ? <Check className="w-3 h-3 text-success" /> : <Copy className="w-3 h-3" />}
      </Button>

      {role === 'assistant' && !isReadonly && (
        <>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleVote(true)}
            className={`p-1 h-auto ${
              isUpvoted === true ? 'text-success' : 'text-muted-foreground hover:text-success'
            }`}
          >
            <ThumbsUp className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleVote(false)}
            className={`p-1 h-auto ${
              isUpvoted === false ? 'text-destructive' : 'text-muted-foreground hover:text-destructive'
            }`}
          >
            <ThumbsDown className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={regenerate}
            className="text-muted-foreground hover:text-foreground/90 p-1 h-auto"
          >
            <RotateCw className="w-3 h-3" />
          </Button>
          {isPlanMode && onLaunchAsMission && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onLaunchAsMission(content)}
              className="text-primary hover:text-primary/80 hover:bg-primary/10 px-2 h-auto gap-1"
            >
              <Rocket className="w-3 h-3" />
              <span className="text-[11px] font-medium">Launch as Mission</span>
            </Button>
          )}
        </>
      )}

      {timeLabel && (
        <span className="text-[11px] text-muted-foreground pl-1">{timeLabel}</span>
      )}
    </div>
  )
}
