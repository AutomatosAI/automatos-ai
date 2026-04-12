'use client'

/**
 * CodeEditor — Monaco editor wrapper with edit + save support
 * PRD-66 Phase 1 → now editable with Ctrl+S save
 */

import { useCallback, useEffect, useRef } from 'react'
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
  onContentChange?: (path: string, content: string) => void
  onSave?: (path: string, content: string) => void
}

export function CodeEditor({ file, onContentChange, onSave }: CodeEditorProps) {
  const editorRef = useRef<any>(null)

  // Register Ctrl+S handler on mount
  const handleEditorMount = useCallback(
    (editor: any) => {
      editorRef.current = editor

      // Ctrl+S / Cmd+S to save
      editor.addCommand(
        // Monaco keybinding: CtrlCmd + S
        2097 /* KeyMod.CtrlCmd | KeyCode.KeyS */,
        () => {
          if (!file || !onSave) return
          const currentContent = editor.getValue()
          onSave(file.path, currentContent)
        },
      )
    },
    [file, onSave],
  )

  // Update save handler when file changes (the closure captures file.path)
  useEffect(() => {
    if (!editorRef.current || !file || !onSave) return
    // Re-register the save command with the new file reference
    editorRef.current.addCommand(2097, () => {
      const currentContent = editorRef.current.getValue()
      onSave(file.path, currentContent)
    })
  }, [file?.path, onSave])

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
      onMount={handleEditorMount}
      onChange={(value) => {
        if (onContentChange && file) {
          onContentChange(file.path, value ?? '')
        }
      }}
      loading={
        <div className="flex items-center justify-center h-full">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      }
      options={{
        readOnly: false,
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
