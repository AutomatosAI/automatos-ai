'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  Search,
  Copy,
  Check,
  Sparkles,
  Zap,
  FileText,
  Settings2,
  BarChart3
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { useRAGRetrievalMutation, type RAGChunk } from '@/hooks/use-rag-api'
import { toast } from 'sonner'

export function RAGContextBuilder() {
  const [query, setQuery] = useState('')
  const [maxChunks, setMaxChunks] = useState(5)
  const [maxTokens, setMaxTokens] = useState(2000)
  const [diversity, setDiversity] = useState(0.3)
  const [copied, setCopied] = useState(false)
  const [showSettings, setShowSettings] = useState(false)

  const ragMutation = useRAGRetrievalMutation()

  const handleSearch = () => {
    if (!query.trim()) {
      toast.error('Please enter a search query')
      return
    }

    console.log('[RAG] Starting retrieval with params:', {
      query: query.trim(),
      max_chunks: maxChunks,
      max_tokens: maxTokens,
      diversity,
    })

    ragMutation.mutate(
      {
        query: query.trim(),
        max_chunks: maxChunks,
        max_tokens: maxTokens,
        diversity,
      },
      {
        onSuccess: (data) => {
          console.log('[RAG] Success! Retrieved data:', data)
          toast.success(`✅ Retrieved ${data.chunks?.length || 0} relevant chunks!`)
        },
        onError: (error: any) => {
          console.error('[RAG] Error occurred:', error)
          const errorMsg = error?.message || 'Unknown error occurred'
          toast.error(`❌ RAG Error: ${errorMsg}`)
        },
      }
    )
  }

  const handleCopyContext = () => {
    if (ragMutation.data?.context) {
      navigator.clipboard.writeText(ragMutation.data.context)
      setCopied(true)
      toast.success('Context copied to clipboard!')
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-success bg-success/10 border-success/20'
    if (similarity >= 0.6) return 'text-warning bg-warning/10 border-warning/20'
    return 'text-muted-foreground bg-muted/30 border-border'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-6 h-6 text-agent" />
            RAG Context Builder
          </h2>
          <p className="text-muted-foreground mt-1">
            Retrieve optimized context for AI agents using semantic search and diversity optimization
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowSettings(!showSettings)}
        >
          <Settings2 className="w-4 h-4 mr-2" />
          {showSettings ? 'Hide' : 'Show'} Settings
        </Button>
      </div>

      {/* Query Input */}
      <Card className="glass-card">
        <CardContent className="p-6">
          <div className="flex gap-3">
            <Input
              placeholder="Enter your query (e.g., 'How to optimize database queries')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1"
              disabled={ragMutation.isLoading}
            />
            <Button
              onClick={handleSearch}
              disabled={ragMutation.isLoading || !query.trim()}
            >
              {ragMutation.isLoading ? (
                <>
                  <Zap className="w-4 h-4 mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Retrieve Context
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="text-lg">Retrieval Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Max Chunks */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">Max Chunks</label>
                    <span className="text-sm text-muted-foreground">{maxChunks}</span>
                  </div>
                  <Slider
                    value={[maxChunks]}
                    onValueChange={(value) => setMaxChunks(value[0])}
                    min={1}
                    max={20}
                    step={1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Number of document chunks to retrieve (1-20)
                  </p>
                </div>

                {/* Max Tokens */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">Token Budget</label>
                    <span className="text-sm text-muted-foreground">{maxTokens}</span>
                  </div>
                  <Slider
                    value={[maxTokens]}
                    onValueChange={(value) => setMaxTokens(value[0])}
                    min={500}
                    max={8000}
                    step={500}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    Maximum tokens for context (500-8000)
                  </p>
                </div>

                {/* Diversity */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium">Diversity</label>
                    <span className="text-sm text-muted-foreground">{diversity.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[diversity]}
                    onValueChange={(value) => setDiversity(value[0])}
                    min={0}
                    max={1}
                    step={0.1}
                    className="w-full"
                  />
                  <p className="text-xs text-muted-foreground">
                    0.0 = max relevance, 1.0 = max diversity
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      {ragMutation.data && ragMutation.data.chunks && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="glass-card">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Chunks</p>
                    <p className="text-2xl font-bold">{ragMutation.data.chunks?.length || 0}</p>
                  </div>
                  <FileText className="w-8 h-8 text-info" />
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Tokens</p>
                    <p className="text-2xl font-bold">{ragMutation.data.total_tokens}</p>
                  </div>
                  <Sparkles className="w-8 h-8 text-agent" />
                </div>
                <Progress 
                  value={(ragMutation.data.total_tokens / maxTokens) * 100} 
                  className="h-1 mt-2" 
                />
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Diversity</p>
                    <p className="text-2xl font-bold">{(ragMutation.data.diversity_score * 100).toFixed(0)}%</p>
                  </div>
                  <BarChart3 className="w-8 h-8 text-success" />
                </div>
              </CardContent>
            </Card>

            <Card className="glass-card">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Time</p>
                    <p className="text-2xl font-bold">{ragMutation.data.execution_time_ms}ms</p>
                  </div>
                  <Zap className="w-8 h-8 text-warning" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Retrieved Chunks */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Retrieved Chunks</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {ragMutation.data.chunks.map((chunk, index) => (
                <motion.div
                  key={chunk.chunk_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="p-4 rounded-lg border border-border/50 bg-secondary/20"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="bg-info/10 text-info">
                        #{index + 1}
                      </Badge>
                      <div>
                        <p className="font-medium">{chunk.source.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          Chunk {chunk.source.chunk_index} • {chunk.tokens} tokens
                          {chunk.truncated && ' • Truncated'}
                        </p>
                      </div>
                    </div>
                    <Badge className={getSimilarityColor(chunk.similarity)}>
                      {(chunk.similarity * 100).toFixed(0)}% Match
                    </Badge>
                  </div>
                  <p className="text-sm leading-relaxed text-foreground/90">
                    {chunk.content}
                  </p>
                </motion.div>
              ))}
            </CardContent>
          </Card>

          {/* Formatted Context */}
          <Card className="glass-card">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Formatted Context (Ready for LLM)</CardTitle>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleCopyContext}
                  disabled={copied}
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4 mr-2" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy Context
                    </>
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <pre className="p-4 rounded-lg bg-black/20 text-sm overflow-x-auto max-h-96 overflow-y-auto">
                {ragMutation.data.context}
              </pre>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Error State */}
      {ragMutation.isError && (
        <Card className="glass-card border-destructive/20">
          <CardContent className="p-6 text-center">
            <p className="text-destructive">
              Error: {(ragMutation.error as Error).message}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Empty State */}
      {!ragMutation.data && !ragMutation.isLoading && !ragMutation.isError && (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Brain className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Ready to Build Context</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Enter a query above to retrieve relevant document chunks optimized for LLM context.
              The system uses MMR (Maximal Marginal Relevance) to balance relevance and diversity.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
