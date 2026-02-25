'use client'

/**
 * CodeEditor — Monaco editor wrapper with dynamic import
 * PRD-66 Phase 1: Code Viewer Widget (read-only)
 */

import dynamic from 'next/dynamic'
import { Loader2 } from 'lucide-react'
import type { OpenFileTab } from '../types'

// Dynamic import — Monaco doesn't work with SSR
const Editor = dynamic(() => import('@monaco-editor/react').then((mod) => mod.default), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
    </div>
  ),
})

interface CodeEditorProps {
  file: OpenFileTab | null
}

export function CodeEditor({ file }: CodeEditorProps) {
  if (!file) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        Select a file to view
      </div>
    )
  }

  if (file.isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <Editor
      height="100%"
      language={file.language}
      value={file.content ?? ''}
      theme="vs-dark"
      options={{
        readOnly: true,
        minimap: { enabled: true },
        fontSize: 13,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        wordWrap: 'off',
        automaticLayout: true,
        folding: true,
        renderWhitespace: 'selection',
        bracketPairColorization: { enabled: true },
        padding: { top: 8, bottom: 8 },
      }}
    />
  )
}
