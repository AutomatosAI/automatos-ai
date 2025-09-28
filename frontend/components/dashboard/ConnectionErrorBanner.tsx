'use client'

import React, { useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'

interface ConnectionErrorBannerProps {
  message?: string
  onRetry?: () => void
  onDismiss?: () => void
}

export function ConnectionErrorBanner({ 
  message = 'Unable to connect to the API. Please check your network connection.',
  onRetry,
  onDismiss
}: ConnectionErrorBannerProps) {
  const [dismissed, setDismissed] = useState(false)
  
  if (dismissed) return null
  
  return (
    <div className=fixed bottom-4 right-4 z-50 max-w-md bg-destructive/90 text-destructive-foreground rounded-lg shadow-lg>
      <div className=flex items-start p-4 gap-3>
        <AlertTriangle className=h-6 w-6 flex-shrink-0 />
        <div className=flex-1 space-y-1>
          <h3 className=font-medium>Connection Error</h3>
          <p className=text-sm opacity-90>{message}</p>
          <div className=flex gap-2 pt-1>
            {onRetry && (
              <button 
                onClick={onRetry}
                className=text-xs px-3 py-1.5 bg-destructive-foreground/10 hover:bg-destructive-foreground/20 text-destructive-foreground rounded-md
              >
                Try Again
              </button>
            )}
            <button 
              onClick={() => {
                setDismissed(true)
                onDismiss?.()
              }}
              className=text-xs px-3 py-1.5 bg-destructive-foreground/10 hover:bg-destructive-foreground/20 text-destructive-foreground rounded-md
            >
              Dismiss
            </button>
          </div>
        </div>
        <button
          onClick={() => {
            setDismissed(true)
            onDismiss?.()
          }}
          className=text-destructive-foreground/70 hover:text-destructive-foreground
        >
          <X className=h-5 w-5 />
        </button>
      </div>
    </div>
  )
}
