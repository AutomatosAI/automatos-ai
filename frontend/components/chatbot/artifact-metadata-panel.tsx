'use client'

/** The artifact viewer's footer: the artifact's metadata fields, shown on request. */
import { useState } from 'react'
import { Eye, EyeOff } from 'lucide-react'

import { Button } from '@/components/ui/button'

export function ArtifactMetadataPanel({ metadata }: { metadata: Record<string, unknown> }) {
  const [showMetadata, setShowMetadata] = useState(false)
  const fieldCount = Object.keys(metadata).length
  if (fieldCount === 0) return null

  return (
    <div className="p-4 border-t border-border">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h4 className="text-sm font-semibold text-muted-foreground">Metadata</h4>
          <p className="text-xs text-muted-foreground">
            {fieldCount} field{fieldCount === 1 ? '' : 's'}
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground dark:text-muted-foreground dark:hover:text-white"
          onClick={() => setShowMetadata((prev) => !prev)}
        >
          {showMetadata ? (
            <>
              <EyeOff className="w-4 h-4 mr-2" />
              Hide
            </>
          ) : (
            <>
              <Eye className="w-4 h-4 mr-2" />
              Show
            </>
          )}
        </Button>
      </div>
      {showMetadata && (
        <div className="space-y-1 text-xs text-muted-foreground">
          {Object.entries(metadata).map(([key, value]) => (
            <div key={key} className="flex justify-between gap-4">
              <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
              <span className="text-foreground dark:text-foreground/90 text-right">{String(value)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
