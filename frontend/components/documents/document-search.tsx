
'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { 
  Search, 
  Brain, 
  FileText, 
  Zap, 
  Filter,
  Star,
  Clock,
  Eye
} from 'lucide-react'
import { Switch } from '@/components/ui/switch'
import { Skeleton } from '@/components/ui/skeleton'

// API hooks
import { useSemanticSearch, useDocumentSearch } from '@/hooks/use-document-api'

interface DocumentSearchProps {
  documents: any[]
  onDocumentSelect: (documentId: string) => void
}

export function DocumentSearch({ documents, onDocumentSelect }: DocumentSearchProps) {
  const [query, setQuery] = useState('')
  const [useSemanticSearch, setUseSemanticSearch] = useState(true)
  const [searchExecuted, setSearchExecuted] = useState(false)

  // API hooks
  // NOTE: Semantic search endpoint not implemented yet - will fall back to regular document list
  const { data: searchResults = [], isLoading: searchLoading, refetch: executeSearch } = useSemanticSearch(
    query, 
    { limit: 10 }, 
    false // Don't execute automatically
  )

  const handleSearch = () => {
    if (query.trim()) {
      setSearchExecuted(true)
      executeSearch()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  return (
    <div className="space-y-6">
      {/* Search Interface */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            AI-Powered Document Search
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Search Query</Label>
            <div className="flex gap-2">
              <Input
                placeholder="Ask questions about your documents or search for specific content..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={searchLoading || !query.trim()}>
                {searchLoading ? (
                  <Zap className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Search className="w-4 h-4 mr-2" />
                )}
                Search
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                id="semantic-search"
                checked={useSemanticSearch}
                onCheckedChange={setUseSemanticSearch}
              />
              <Label htmlFor="semantic-search" className="flex items-center gap-2">
                <Brain className="w-4 h-4" />
                Semantic Search
              </Label>
            </div>
            <div className="text-sm text-muted-foreground">
              {useSemanticSearch ? 'AI-powered contextual search' : 'Traditional keyword search'}
            </div>
          </div>

          {/* Search Examples */}
          <div className="border-t pt-4">
            <Label className="text-sm font-medium mb-2 block">Try asking:</Label>
            <div className="flex flex-wrap gap-2">
              {[
                "What are the key findings in the latest reports?",
                "Show me documents about project timelines",
                "Find budget-related information",
                "What contracts were signed this month?"
              ].map((example) => (
                <Button
                  key={example}
                  variant="outline"
                  size="sm"
                  onClick={() => setQuery(example)}
                  className="text-xs"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Search Results */}
      {searchExecuted && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Search Results</span>
              {searchResults.length > 0 && (
                <Badge variant="secondary">
                  {searchResults.length} results
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {searchLoading ? (
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="border rounded-lg p-4">
                    <div className="flex items-center gap-3 mb-3">
                      <Skeleton className="w-10 h-10 rounded" />
                      <div className="flex-1">
                        <Skeleton className="h-5 w-3/4 mb-2" />
                        <Skeleton className="h-4 w-1/2" />
                      </div>
                      <Skeleton className="h-6 w-16" />
                    </div>
                    <Skeleton className="h-4 w-full mb-2" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                ))}
              </div>
            ) : searchResults.length > 0 ? (
              <div className="space-y-4">
                {searchResults.map((result: any, index: number) => (
                  <div
                    key={result.document_id || index}
                    className="border rounded-lg p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                    onClick={() => onDocumentSelect(result.document_id)}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <FileText className="w-10 h-10 text-primary" />
                      <div className="flex-1">
                        <h4 className="font-semibold text-lg">{result.document_name || 'Untitled Document'}</h4>
                        <p className="text-sm text-muted-foreground">
                          {result.document_type || 'Document'} • 
                          {result.relevance_score && ` Relevance: ${(result.relevance_score * 100).toFixed(1)}%`}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        {result.relevance_score && result.relevance_score > 0.8 && (
                          <Star className="w-4 h-4 text-yellow-500" />
                        )}
                        <Badge variant="outline">
                          {result.relevance_score ? `${(result.relevance_score * 100).toFixed(0)}%` : 'Match'}
                        </Badge>
                      </div>
                    </div>

                    {/* Content Preview */}
                    {result.matched_content && (
                      <div className="bg-muted/30 rounded-lg p-3 mb-3">
                        <p className="text-sm">
                          ...{result.matched_content}...
                        </p>
                      </div>
                    )}

                    {/* Metadata */}
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <div className="flex items-center gap-4">
                        {result.page_number && (
                          <span>Page {result.page_number}</span>
                        )}
                        {result.section && (
                          <span>Section: {result.section}</span>
                        )}
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {result.last_updated ? new Date(result.last_updated).toLocaleDateString() : 'Recently updated'}
                        </span>
                      </div>
                      <Button size="sm" variant="ghost" className="h-6 px-2">
                        <Eye className="w-3 h-3 mr-1" />
                        View
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : query && !searchLoading ? (
              <div className="text-center py-12">
                <Search className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No results found</h3>
                <p className="text-muted-foreground mb-4">
                  Try rephrasing your query or using different keywords
                </p>
                <Button variant="outline" onClick={() => setQuery('')}>
                  Clear Search
                </Button>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Search Tips */}
      {!searchExecuted && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Search Tips
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Semantic Search</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Ask natural language questions</li>
                  <li>• Search by concepts and meaning</li>
                  <li>• Find contextually related content</li>
                  <li>• Works across different phrasings</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Keyword Search</h4>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>• Use specific terms and phrases</li>
                  <li>• Search for exact matches</li>
                  <li>• Use quotes for exact phrases</li>
                  <li>• Combine with AND, OR operators</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

