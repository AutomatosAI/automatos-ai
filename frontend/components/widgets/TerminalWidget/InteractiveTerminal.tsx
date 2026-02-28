'use client'

/**
 * InteractiveTerminal — VS Code-style interactive shell panel
 *
 * Features:
 * - Command input with $ prompt, submits on Enter
 * - ANSI color output via ansi-to-react
 * - Command history with up/down arrow
 * - Auto-scroll to bottom
 * - Working directory display
 * - Spinner while command is running
 * - Clear button
 */

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import Ansi from 'ansi-to-react'
import { Loader2, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { cn } from '@/lib/utils'

interface HistoryEntry {
  command: string
  output: string
  exitCode: number
  duration?: number
  cwd: string
}

interface ExecResult {
  stdout: string
  stderr: string
  exit_code: number
  duration_ms?: number
}

interface InteractiveTerminalProps {
  workspaceId: string
  className?: string
}

export function InteractiveTerminal({ workspaceId, className }: InteractiveTerminalProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [input, setInput] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [cwd, setCwd] = useState('.')
  const [historyIndex, setHistoryIndex] = useState(-1)

  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when history changes
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [history, isRunning])

  // Focus input when terminal is clicked
  const handleContainerClick = useCallback(() => {
    inputRef.current?.focus()
  }, [])

  const executeCommand = useCallback(async (command: string) => {
    if (!command.trim()) return

    // Handle local-only commands
    if (command.trim() === 'clear') {
      setHistory([])
      return
    }

    setIsRunning(true)
    setHistoryIndex(-1)

    try {
      const result = await apiClient.request(
        `/api/workspaces/${workspaceId}/exec`,
        {
          method: 'POST',
          body: {
            command,
            cwd: cwd === '.' ? undefined : cwd,
            timeout: 120,
          } as any,
        }
      ) as ExecResult

      const entry: HistoryEntry = {
        command,
        output: (result.stdout || '') + (result.stderr ? `\n\x1b[31m${result.stderr}\x1b[0m` : ''),
        exitCode: result.exit_code ?? 0,
        duration: result.duration_ms,
        cwd,
      }

      setHistory(prev => [...prev, entry])

      // Update cwd if the command was cd
      if (command.trim().startsWith('cd ')) {
        const newDir = command.trim().slice(3).trim()
        if (newDir && result.exit_code === 0) {
          // Ask the worker for the actual cwd after cd
          try {
            const pwdResult = await apiClient.request(
              `/api/workspaces/${workspaceId}/exec`,
              {
                method: 'POST',
                body: { command: 'pwd', cwd: newDir } as any,
              }
            ) as ExecResult
            if (pwdResult.exit_code === 0 && pwdResult.stdout) {
              setCwd(pwdResult.stdout.trim())
            }
          } catch {
            // Best effort — keep existing cwd
          }
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Command failed'
      setHistory(prev => [...prev, {
        command,
        output: `\x1b[31mError: ${msg}\x1b[0m`,
        exitCode: 1,
        cwd,
      }])
    } finally {
      setIsRunning(false)
      // Re-focus input after command completes
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [workspaceId, cwd])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isRunning) {
      const cmd = input.trim()
      if (cmd) {
        executeCommand(cmd)
        setInput('')
      }
      return
    }

    // Command history navigation
    const commands = history.map(h => h.command)
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (commands.length === 0) return
      const newIndex = historyIndex === -1
        ? commands.length - 1
        : Math.max(0, historyIndex - 1)
      setHistoryIndex(newIndex)
      setInput(commands[newIndex])
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (historyIndex === -1) return
      const newIndex = historyIndex + 1
      if (newIndex >= commands.length) {
        setHistoryIndex(-1)
        setInput('')
      } else {
        setHistoryIndex(newIndex)
        setInput(commands[newIndex])
      }
    }
  }, [input, isRunning, history, historyIndex, executeCommand])

  const handleClear = useCallback(() => {
    setHistory([])
  }, [])

  // Short display cwd: show last 2 path segments
  const displayCwd = cwd === '.' ? '~' : cwd.split('/').slice(-2).join('/')

  return (
    <div
      className={cn('flex flex-col h-full bg-[#1e1e1e] text-gray-100 font-mono text-sm', className)}
      onClick={handleContainerClick}
    >
      {/* Terminal toolbar */}
      <div className="flex items-center justify-between px-3 py-1 border-b border-[#333] bg-[#252526] text-xs text-gray-400 shrink-0">
        <span>Terminal</span>
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 text-gray-400 hover:text-gray-200"
          onClick={handleClear}
          title="Clear terminal"
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </div>

      {/* Scrollable output area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-auto p-2 min-h-0">
        {history.length === 0 && !isRunning && (
          <div className="text-gray-500 text-xs py-1">
            Type a command and press Enter
          </div>
        )}

        {history.map((entry, i) => (
          <div key={i} className="mb-2">
            {/* Command prompt line */}
            <div className="flex items-center gap-1.5">
              <span className="text-green-400 select-none">$</span>
              <span className="text-gray-200">{entry.command}</span>
              {entry.exitCode !== 0 && (
                <span className="text-red-400 text-xs ml-auto">[{entry.exitCode}]</span>
              )}
              {entry.duration !== undefined && (
                <span className="text-gray-500 text-xs ml-1">
                  {entry.duration < 1000 ? `${entry.duration}ms` : `${(entry.duration / 1000).toFixed(1)}s`}
                </span>
              )}
            </div>
            {/* Output */}
            {entry.output && (
              <div className="pl-4 leading-5 whitespace-pre-wrap break-all">
                <Ansi>{entry.output}</Ansi>
              </div>
            )}
          </div>
        ))}

        {/* Running indicator */}
        {isRunning && (
          <div className="flex items-center gap-2 text-yellow-400 text-xs py-1">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Running...</span>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="flex items-center gap-1.5 px-2 py-1.5 border-t border-[#333] bg-[#1e1e1e] shrink-0">
        <span className="text-gray-500 text-xs select-none">{displayCwd}</span>
        <span className="text-green-400 select-none">$</span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isRunning}
          placeholder={isRunning ? 'Running...' : 'Enter command...'}
          className="flex-1 bg-transparent border-none outline-none text-gray-100 placeholder:text-gray-600 text-sm font-mono disabled:opacity-50"
          autoComplete="off"
          spellCheck={false}
        />
      </div>
    </div>
  )
}
