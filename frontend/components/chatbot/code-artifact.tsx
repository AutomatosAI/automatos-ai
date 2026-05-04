'use client'

import { useEffect, useRef } from 'react'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'

export interface CodeArtifactProps {
  content: string
  language: string
  metadata?: Record<string, any>
}

export function CodeArtifact({ content, language, metadata }: CodeArtifactProps) {
  const codeRef = useRef<HTMLElement>(null)

  useEffect(() => {
    if (codeRef.current) {
      Prism.highlightElement(codeRef.current)
    }
  }, [content, language])

  return (
    <div className="space-y-4">
      {metadata && (metadata.file_path || metadata.line_number) && (
        <div className="text-sm text-muted-foreground">
          {metadata.file_path && (
            <div className="font-mono">{metadata.file_path}</div>
          )}
          {metadata.line_number && (
            <div>Line {metadata.line_number}</div>
          )}
          {metadata.explanation && (
            <div className="mt-2 text-foreground/90 italic">{metadata.explanation}</div>
          )}
        </div>
      )}
      
      <pre className="rounded-lg overflow-x-auto bg-[#2d2d2d] p-4 text-sm !m-0">
        <code ref={codeRef} className={`language-${language}`}>
          {content}
        </code>
      </pre>
    </div>
  )
}

