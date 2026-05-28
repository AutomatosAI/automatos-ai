'use client'

import { useMemo, useState } from 'react'
import { X, Copy, Download, Maximize2, Eye, EyeOff, FileText } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { copyToClipboard } from '@/lib/utils'
import { CodeArtifact } from './code-artifact'
import { TextArtifact } from './text-artifact'
import { SheetArtifact } from './sheet-artifact'
import type { Artifact } from '@/types'
import { toast } from 'sonner'

export interface ArtifactViewerProps {
  artifact: Artifact
  onClose: () => void
}

export function ArtifactViewer({ artifact, onClose }: ArtifactViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showMetadata, setShowMetadata] = useState(false)

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
          {artifact.kind === 'document' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <FileText className="h-8 w-8 text-primary" />
                <div>
                  <h3 className="font-semibold">{artifact.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {artifact.metadata?.format?.toUpperCase()} {artifact.metadata?.size_kb ? `• ${artifact.metadata.size_kb}KB` : ''}
                  </p>
                </div>
              </div>
              {artifact.metadata?.format === 'pdf' && artifact.metadata?.preview_url && (
                <iframe
                  src={artifact.metadata.preview_url}
                  className="w-full h-[600px] rounded-lg border"
                  title={artifact.title}
                />
              )}
              <div className="flex gap-2">
                {artifact.metadata?.download_url && (
                  <Button
                    onClick={() => window.open(artifact.metadata!.download_url, '_blank')}
                    className="bg-primary hover:bg-primary/90 text-primary-foreground"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download {artifact.metadata?.format?.toUpperCase()}
                  </Button>
                )}
                {artifact.metadata?.format && artifact.metadata.format !== 'pdf' && (
                  <Button variant="outline" disabled>
                    <FileText className="h-4 w-4 mr-2" />
                    Convert to PDF
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Metadata */}
      {cleanedMetadata && Object.keys(cleanedMetadata).length > 0 && (
        <div className="p-4 border-t border-border">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h4 className="text-sm font-semibold text-muted-foreground">Metadata</h4>
              <p className="text-xs text-muted-foreground">
                {Object.keys(cleanedMetadata).length} field{Object.keys(cleanedMetadata).length === 1 ? '' : 's'}
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
              {Object.entries(cleanedMetadata).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-4">
                  <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                  <span className="text-foreground dark:text-foreground/90 text-right">{String(value)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

