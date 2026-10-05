'use client'

import { useMemo, useState } from 'react'
import { X, Copy, Download, Maximize2 } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { copyToClipboard } from '@/lib/utils'
import { CodeArtifact } from './code-artifact'
import { TextArtifact } from './text-artifact'
import { SheetArtifact } from './sheet-artifact'
import { DocumentArtifact } from './document-artifact'
import { ArtifactMetadataPanel } from './artifact-metadata-panel'
import type { Artifact } from '@/types'
import { toast } from 'sonner'

export interface ArtifactViewerProps {
  artifact: Artifact
  onClose: () => void
}

export function ArtifactViewer({ artifact, onClose }: ArtifactViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  const cleanedMetadata = useMemo(() => {
    if (!artifact.metadata) return {}
    return Object.fromEntries(
      Object.entries(artifact.metadata).filter(([_, value]) => value !== undefined && value !== null && value !== '')
    )
  }, [artifact.metadata])

  const handleCopy = async () => {
    if (await copyToClipboard(artifact.content)) {
      toast.success('Copied to clipboard')
    }
  }

  const handleDownload = () => {
    const blob = new Blob([artifact.content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${artifact.title}.${artifact.kind === 'code' ? artifact.language || 'txt' : 'txt'}`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Downloaded')
  }

  return (
    <div className="flex h-full w-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted">
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-foreground dark:text-white truncate">{artifact.title}</h3>
          <div className="flex items-center space-x-2 mt-1">
            <Badge variant="outline" className="bg-agent/10 border-agent/20 text-agent text-xs">
              {artifact.kind}
            </Badge>
            {artifact.language && (
              <Badge variant="outline" className="bg-info/10 border-info/20 text-info text-xs">
                {artifact.language}
              </Badge>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-muted-foreground hover:text-foreground dark:text-muted-foreground dark:hover:text-white"
          >
            <Copy className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            className="text-muted-foreground hover:text-foreground dark:text-muted-foreground dark:hover:text-white"
          >
            <Download className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="text-muted-foreground hover:text-foreground dark:text-muted-foreground dark:hover:text-white"
          >
            <Maximize2 className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground dark:text-muted-foreground dark:hover:text-white"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-scroll">
        <div className="px-6 py-8">
          {artifact.kind === 'code' && (
            <CodeArtifact
              content={artifact.content}
              language={artifact.language || 'javascript'}
              metadata={artifact.metadata}
            />
          )}
          {artifact.kind === 'text' && (
            <TextArtifact
              content={artifact.content}
              metadata={artifact.metadata}
            />
          )}
          {artifact.kind === 'sheet' && (
            <SheetArtifact
              content={artifact.content}
              metadata={artifact.metadata}
            />
          )}
          {artifact.kind === 'image' && (
            <img src={artifact.content} alt={artifact.title} className="max-w-full h-auto" />
          )}
          {artifact.kind === 'document' && <DocumentArtifact artifact={artifact} />}
        </div>
      </div>

      {/* Metadata */}
      <ArtifactMetadataPanel metadata={cleanedMetadata} />
    </div>
  )
}

