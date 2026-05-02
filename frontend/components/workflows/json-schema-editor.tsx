'use client'

import * as React from 'react'
import Prism from 'prismjs'
import 'prismjs/components/prism-json'
import { cn } from '@/lib/utils'
import { AlertCircle, Check, WrapText } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface JsonSchemaEditorProps {
  value: string
  onChange: (value: string) => void
  onValidation?: (isValid: boolean, error: string | null) => void
  placeholder?: string
  label?: string
  className?: string
  minHeight?: string
}

export function JsonSchemaEditor({
  value,
  onChange,
  onValidation,
  placeholder,
  label,
  className,
  minHeight = '120px',
}: JsonSchemaEditorProps) {
  const [error, setError] = React.useState<string | null>(null)
  const [isFocused, setIsFocused] = React.useState(false)
  const highlightRef = React.useRef<HTMLElement>(null)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Sync highlighting with value changes
  React.useEffect(() => {
    if (highlightRef.current) {
      try {
        const highlighted = Prism.highlight(
          value || '',
          Prism.languages.json,
          'json'
        )
        highlightRef.current.innerHTML = highlighted
      } catch {
        // SECURITY: textContent is used here (not innerHTML) to prevent XSS (OWASP A03:2021)
        highlightRef.current.textContent = value || ''
      }
    }
  }, [value])

  const validateJson = React.useCallback(
    (text: string) => {
      if (!text.trim() || text.trim() === '{}') {
        setError(null)
        onValidation?.(true, null)
        return
      }
      try {
        JSON.parse(text)
        setError(null)
        onValidation?.(true, null)
      } catch (e) {
        const msg =
          e instanceof SyntaxError ? e.message : 'Invalid JSON'
        setError(msg)
        onValidation?.(false, msg)
      }
    },
    [onValidation]
  )

  // Validate on mount with initial value
  React.useEffect(() => {
    validateJson(value)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  const handleBlur = () => {
    setIsFocused(false)
    validateJson(value)
  }

  const handleFocus = () => {
    setIsFocused(true)
  }

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value)
  }

  const handleFormat = () => {
    try {
      const parsed = JSON.parse(value)
      const formatted = JSON.stringify(parsed, null, 2)
      onChange(formatted)
      setError(null)
      onValidation?.(true, null)
    } catch {
      // Can't format invalid JSON — validation will show error
      validateJson(value)
    }
  }

  // Sync scroll between textarea and highlight overlay
  const handleScroll = () => {
    if (textareaRef.current && containerRef.current) {
      const pre = containerRef.current.querySelector('pre')
      if (pre) {
        pre.scrollTop = textareaRef.current.scrollTop
        pre.scrollLeft = textareaRef.current.scrollLeft
      }
    }
  }

  const isValid = error === null && value.trim().length > 0

  return (
    <div className={cn('space-y-1.5', className)}>
      {label && (
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-foreground">{label}</span>
          <div className="flex items-center gap-2">
            {value.trim().length > 2 && (
              <span
                className={cn(
                  'flex items-center gap-1 text-xs',
                  isValid ? 'text-success' : error ? 'text-destructive' : 'text-muted-foreground'
                )}
              >
                {isValid ? (
                  <>
                    <Check className="w-3 h-3" />
                    Valid JSON
                  </>
                ) : error ? (
                  <>
                    <AlertCircle className="w-3 h-3" />
                    Invalid
                  </>
                ) : null}
              </span>
            )}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleFormat}
              className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
              title="Format JSON"
            >
              <WrapText className="w-3 h-3 mr-1" />
              Format
            </Button>
          </div>
        </div>
      )}

      <div
        ref={containerRef}
        className={cn(
          'relative rounded-2xl overflow-hidden border transition-colors',
          isFocused
            ? 'border-orange-400/50 ring-1 ring-orange-400/20'
            : error
              ? 'border-destructive/50'
              : 'border-input',
          'bg-[#1e1e2e]'
        )}
      >
        {/* Syntax highlighted overlay (non-interactive) */}
        <pre
          className="absolute inset-0 p-3 m-0 overflow-hidden pointer-events-none"
          aria-hidden="true"
        >
          <code
            ref={highlightRef}
            className="language-json font-mono text-xs leading-relaxed block whitespace-pre-wrap break-words"
          />
        </pre>

        {/* Editable textarea (transparent text, visible caret) */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          onBlur={handleBlur}
          onFocus={handleFocus}
          onScroll={handleScroll}
          placeholder={placeholder}
          spellCheck={false}
          className={cn(
            'relative w-full p-3 m-0 bg-transparent font-mono text-xs leading-relaxed resize-y',
            'text-transparent caret-orange-400 placeholder:text-muted-foreground',
            'outline-none border-none ring-0',
            'selection:bg-orange-400/20'
          )}
          style={{ minHeight }}
        />
      </div>

      {error && (
        <p className="flex items-center gap-1 text-xs text-destructive">
          <AlertCircle className="w-3 h-3 flex-shrink-0" />
          {error}
        </p>
      )}
    </div>
  )
}
