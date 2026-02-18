'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  FileText, 
  Zap, 
  X,
  Filter,
  Sparkles,
  Eye
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { useSemanticSearch, type SearchResult } from '@/hooks/use-semantic-search-api'

interface SemanticSearchProps {
  context: 'documents' | 'chatbot' | 'agent' | 'patterns'
  onResultSelect?: (result: SearchResult) => void
  onResultsChange?: (results: SearchResult[]) => void
  showActions?: boolean
  maxResults?: number
  documentIds?: number[]
  standalone?: boolean
}

export function SemanticSearch({
  context,
  onResultSelect,
  onResultsChange,
  showActions = true,
  maxResults = 10,
  documentIds,
  standalone = false,
}: SemanticSearchProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [minSimilarity, setMinSimilarity] = useState(0.7)
  const [isSearching, setIsSearching] = useState(false)

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery)
      setIsSearching(false)
    }, 500)

    if (searchQuery) {
      setIsSearching(true)
    }

    return () => clearTimeout(timer)
  }, [searchQuery])

  // Perform semantic search
  const { data, isLoading, error } = useSemanticSearch(
    debouncedQuery ? {
      query: debouncedQuery,
      limit: maxResults,
      min_similarity: minSimilarity,
      document_ids: documentIds
    } : null
  )

  // Notify parent of results change
  useEffect(() => {
    if (data?.results && onResultsChange) {
      onResultsChange(data.results)
    }
  }, [data?.results, onResultsChange])

  const handleClearSearch = () => {
    setSearchQuery('')
    setDebouncedQuery('')
  }

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'text-green-400 bg-green-500/10 border-green-500/20'
    if (similarity >= 0.6) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20'
    return 'text-orange-400 bg-orange-500/10 border-orange-500/20'
  }

  const getContextIcon = () => {
    switch (context) {
      case 'documents': return <FileText className="w-5 h-5" />
      case 'chatbot': return <Sparkles className="w-5 h-5" />
      case 'agent': return <Zap className="w-5 h-5" />
      case 'patterns': return <Search className="w-5 h-5" />
    }
  }

  const getContextTitle = () => {
    switch (context) {
      case 'documents': return 'Search Documents'
      case 'chatbot': return 'Find Context'
      case 'agent': return 'Search Knowledge'
      case 'patterns': return 'Find Patterns'
    }
  }

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <Card className="glass-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {getContextIcon()}
              <CardTitle>{getContextTitle()}</CardTitle>
            </div>
            {data && (
              <Badge variant="outline" className="text-blue-400">
                {data.total_results} results in {data.execution_time_ms}ms
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search Input */}
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search by meaning... (e.g., 'vector database optimization')"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-10"
              />
              {searchQuery && (
                <button
                  onClick={handleClearSearch}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Similarity Filter */}
            <Select value={String(minSimilarity)} onValueChange={(v) => setMinSimilarity(parseFloat(v))}>
              <SelectTrigger className="w-40">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.9">Very High (90%)</SelectItem>
                <SelectItem value="0.8">High (80%)</SelectItem>
                <SelectItem value="0.7">Medium (70%)</SelectItem>
                <SelectItem value="0.6">Low (60%)</SelectItem>
                <SelectItem value="0.5">Any (50%)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Search Info */}
          {searchQuery && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Sparkles className="w-4 h-4" />
              <span>
                {isSearching || isLoading
                  ? 'Searching...'
                  : `Semantic search for: "${searchQuery}"`}
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Search Results */}
      {isLoading || isSearching ? (
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="glass-card">
              <CardContent className="p-6">
                <div className="space-y-3">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-5/6" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : error ? (
        <Card className="glass-card border-red-500/20">
          <CardContent className="p-6 text-center">
            <p className="text-red-400">Error: {(error as Error).message}</p>
          </CardContent>
        </Card>
      ) : data && data.results && data.results.length > 0 ? (
        <div className="space-y-4">
          {data.results.map((result, index) => (
            <motion.div
              key={result.document_id || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <Card
                className={`glass-card ${onResultSelect ? 'cursor-pointer hover:shadow-lg transition-shadow' : ''}`}
                onClick={() => onResultSelect && onResultSelect(result)}
              >
                <CardContent className="p-6">
                  {/* Document Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <FileText className="w-5 h-5 text-blue-400" />
                      <div>
                        <h3 className="font-medium text-foreground">
                          {result.source?.filename || 'Document'}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {result.chunk_count ? `${result.chunk_count} chunks` : `Chunk ${(result.chunk_index ?? 0) + 1}`}
                          {result.preview_truncated && ' (preview)'}
                        </p>
                      </div>
                    </div>

                    {/* Similarity Badge */}
                    <Badge className={getSimilarityColor(result.similarity)}>
                      {(result.similarity * 100).toFixed(0)}% Match
                    </Badge>
                  </div>

                  {/* Content Preview */}
                  <div className="bg-muted/30 rounded-lg p-4 mb-3">
                    <p className="text-sm text-foreground line-clamp-4 whitespace-pre-wrap">
                      {result.preview || result.excerpt || result.content || ''}
                    </p>
                  </div>

                  {/* Result Footer */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                      {result.source?.file_type && (
                        <span>{result.source.file_type.toUpperCase()}</span>
                      )}
                      {result.source?.file_size != null && (
                        <>
                          <span>•</span>
                          <span>{(result.source.file_size / 1024).toFixed(1)} KB</span>
                        </>
                      )}
                      {result.source?.upload_date && (
                        <>
                          <span>•</span>
                          <span>{new Date(result.source.upload_date).toLocaleDateString()}</span>
                        </>
                      )}
                      {result.full_content_available && (
                        <>
                          <span>•</span>
                          <span className="text-blue-400">Full content available</span>
                        </>
                      )}
                    </div>

                    {showActions && onResultSelect && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          onResultSelect(result)
                        }}
                      >
                        <Eye className="w-4 h-4 mr-2" />
                        View
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      ) : debouncedQuery ? (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No results found</h3>
            <p className="text-muted-foreground">
              Try a different search query or lower the similarity threshold
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Sparkles className="w-16 h-16 mx-auto text-blue-400 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Semantic Search Ready</h3>
            <p className="text-muted-foreground">
              Search by meaning, not just keywords. Try asking questions or describing concepts.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
