'use client'

/**
 * CodeEditor — Monaco editor wrapper with edit + save support
 * PRD-66 Phase 1 → now editable with Ctrl+S save
 */

import { useCallback, useRef } from 'react'
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
  // Store latest callbacks + file path in refs so the Monaco command
  // always sees current values without needing to re-register.
  const onSaveRef = useRef(onSave)
  onSaveRef.current = onSave
  const filePathRef = useRef(file?.path)
  filePathRef.current = file?.path
  const editorRef = useRef<any>(null)

  const handleEditorMount = useCallback(
    (editor: any, monaco: any) => {
      editorRef.current = editor

      // Ctrl+S / Cmd+S to save — uses refs so it always sees latest file/callback
      editor.addAction({
        id: 'save-file',
        label: 'Save File',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
        run: () => {
          const path = filePathRef.current
          const save = onSaveRef.current
          if (path && save) {
            save(path, editor.getValue())
          }
        },
      })
    },
    [],
  )

  const handleChange = useCallback(
    (value: string | undefined) => {
      if (file && onContentChange) {
        onContentChange(file.path, value ?? '')
      }
    },
    [file?.path, onContentChange],
  )

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
      onChange={handleChange}
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
