'use client'

import { useState } from 'react'
import { X, Copy, Download, Maximize2 } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { copyToClipboard } from '@/lib/utils'
import { CodeArtifact } from './code-artifact'
import { TextArtifact } from './text-artifact'
import type { Artifact } from '@/types'
import { toast } from 'sonner'

export interface ArtifactViewerProps {
  artifact: Artifact
  onClose: () => void
}

export function ArtifactViewer({ artifact, onClose }: ArtifactViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)

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
          <h3 className="text-lg font-semibold text-white truncate">{artifact.title}</h3>
          <div className="flex items-center space-x-2 mt-1">
            <Badge variant="outline" className="bg-purple-500/10 border-purple-500/20 text-purple-400 text-xs">
              {artifact.kind}
            </Badge>
            {artifact.language && (
              <Badge variant="outline" className="bg-blue-500/10 border-blue-500/20 text-blue-400 text-xs">
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
            className="text-gray-400 hover:text-white"
          >
            <Copy className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            className="text-gray-400 hover:text-white"
          >
            <Download className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="text-gray-400 hover:text-white"
          >
            <Maximize2 className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-gray-400 hover:text-white"
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
          {artifact.kind === 'image' && (
            <img src={artifact.content} alt={artifact.title} className="max-w-full h-auto" />
          )}
        </div>
      </div>

      {/* Metadata */}
      {artifact.metadata && Object.keys(artifact.metadata).length > 0 && (
        <div className="p-4 border-t border-gray-800">
          <h4 className="text-sm font-semibold text-gray-400 mb-2">Metadata</h4>
          <div className="space-y-1 text-xs text-gray-500">
            {Object.entries(artifact.metadata).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                <span className="text-gray-400">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

