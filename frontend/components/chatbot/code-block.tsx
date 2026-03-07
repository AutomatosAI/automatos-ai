'use client'

import { useEffect, useRef, useState } from 'react'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-sql'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-markup'
import 'prismjs/components/prism-go'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-docker'
import { Check, Copy } from 'lucide-react'
import { copyToClipboard } from '@/lib/utils'

interface CodeBlockProps {
  code: string
  language?: string
}

const LANGUAGE_LABELS: Record<string, string> = {
  python: 'Python',
  py: 'Python',
  typescript: 'TypeScript',
  ts: 'TypeScript',
  javascript: 'JavaScript',
  js: 'JavaScript',
  json: 'JSON',
  bash: 'Bash',
  sh: 'Bash',
  shell: 'Bash',
  sql: 'SQL',
  css: 'CSS',
  html: 'HTML',
  markup: 'HTML',
  go: 'Go',
  yaml: 'YAML',
  yml: 'YAML',
  markdown: 'Markdown',
  md: 'Markdown',
  docker: 'Docker',
  dockerfile: 'Docker',
}

// Map aliases to Prism grammar names
const LANGUAGE_ALIAS: Record<string, string> = {
  py: 'python',
  ts: 'typescript',
  js: 'javascript',
  sh: 'bash',
  shell: 'bash',
  yml: 'yaml',
  md: 'markdown',
  dockerfile: 'docker',
  html: 'markup',
}

export function CodeBlock({ code, language = '' }: CodeBlockProps) {
  const codeRef = useRef<HTMLElement>(null)
  const [copied, setCopied] = useState(false)

  const lang = language.toLowerCase()
  const prismLang = LANGUAGE_ALIAS[lang] || lang
  const label = LANGUAGE_LABELS[lang] || (lang ? lang.charAt(0).toUpperCase() + lang.slice(1) : 'Code')

  useEffect(() => {
    if (codeRef.current) {
      Prism.highlightElement(codeRef.current)
    }
  }, [code, prismLang])

  const handleCopy = async () => {
    if (await copyToClipboard(code)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="group/code relative rounded-xl border border-orange-500/10 bg-[#1a1a1a] overflow-hidden my-3">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/30 bg-white/[0.02]">
        <span className="text-[11px] font-medium tracking-wide text-orange-400/80 uppercase select-none">
          {label}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <pre className="overflow-x-auto p-4 text-[13px] leading-relaxed !m-0 !bg-transparent">
        <code ref={codeRef} className={`language-${prismLang}`}>
          {code}
        </code>
      </pre>
    </div>
  )
}
