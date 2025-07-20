
"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Search, 
  Settings, 
  FileText, 
  Target,
  BarChart3,
  Zap
} from 'lucide-react';

interface SearchResult {
  chunk_id: number;
  document_id: number;
  filename: string;
  content: string;
  similarity: number;
  metadata: Record<string, any>;
}

interface ContextConfig {
  chunk_size: number;
  chunk_overlap: number;
  similarity_threshold: number;
  max_results: number;
}

const ContextEngineeringPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState<ContextConfig>({
    chunk_size: 1000,
    chunk_overlap: 200,
    similarity_threshold: 0.7,
    max_results: 10
  });

  // Search context
  const searchContext = async () => {
    if (!searchQuery.trim()) return;
    
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8001/api/context/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: config.max_results
        }),
      });
      
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Error searching context:', error);
    } finally {
      setLoading(false);
    }
  };

  // Update configuration
  const updateConfig = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/admin/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        alert('Configuration updated successfully!');
      }
    } catch (error) {
      console.error('Error updating config:', error);
    }
  };

  // Load current configuration
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch('http://localhost:8001/api/admin/config');
        const data = await response.json();
        setConfig(data);
      } catch (error) {
        console.error('Error loading config:', error);
      }
    };
    
    loadConfig();
  }, []);

  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 px-1 rounded">
          {part}
        </mark>
      ) : part
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Context Engineering</h1>
          <p className="text-muted-foreground">
            Search and configure the context retrieval system
          </p>
        </div>
      </div>

      <Tabs defaultValue="search" className="space-y-4">
        <TabsList>
          <TabsTrigger value="search">Context Search</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="preview">Context Preview</TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="space-y-4">
          {/* Search Interface */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Context Search
              </CardTitle>
              <CardDescription>
                Search through processed documents to find relevant context
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter your search query..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchContext()}
                  className="flex-1"
                />
                <Button onClick={searchContext} disabled={loading}>
                  {loading ? 'Searching...' : 'Search'}
                </Button>
              </div>
              
              <div className="flex gap-4 text-sm text-muted-foreground">
                <span>Max Results: {config.max_results}</span>
                <span>Min Similarity: {config.similarity_threshold}</span>
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Search Results ({searchResults.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {searchResults.map((result, index) => (
                    <div key={result.chunk_id} className="border rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">#{index + 1}</Badge>
                          <span className="font-medium">{result.filename}</span>
                          <Badge 
                            variant="secondary"
                            className={
                              result.similarity > 0.8 ? 'bg-green-100 text-green-800' :
                              result.similarity > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }
                          >
                            {(result.similarity * 100).toFixed(1)}% match
                          </Badge>
                        </div>
                        <Button variant="outline" size="sm">
                          <FileText className="h-4 w-4" />
                        </Button>
                      </div>
                      
                      <div className="text-sm text-muted-foreground mb-2">
                        Document ID: {result.document_id} | Chunk ID: {result.chunk_id}
                      </div>
                      
                      <div className="prose prose-sm max-w-none">
                        <p className="text-sm leading-relaxed">
                          {highlightText(result.content, searchQuery)}
                        </p>
                      </div>
                      
                      {Object.keys(result.metadata).length > 0 && (
                        <details className="mt-2">
                          <summary className="text-sm text-muted-foreground cursor-pointer">
                            View Metadata
                          </summary>
                          <pre className="text-xs bg-gray-50 p-2 rounded mt-1 overflow-auto">
                            {JSON.stringify(result.metadata, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {searchQuery && searchResults.length === 0 && !loading && (
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">No results found for "{searchQuery}"</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Try adjusting your search terms or lowering the similarity threshold
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Context Configuration
              </CardTitle>
              <CardDescription>
                Fine-tune the context retrieval parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Chunk Size */}
              <div className="space-y-2">
                <Label>Chunk Size: {config.chunk_size} characters</Label>
                <Slider
                  value={[config.chunk_size]}
                  onValueChange={(value) => setConfig({...config, chunk_size: value[0]})}
                  max={4000}
                  min={100}
                  step={100}
                  className="w-full"
                />
                <p className="text-sm text-muted-foreground">
                  Larger chunks provide more context but may be less precise
                </p>
              </div>

              {/* Chunk Overlap */}
              <div className="space-y-2">
                <Label>Chunk Overlap: {config.chunk_overlap} characters</Label>
                <Slider
                  value={[config.chunk_overlap]}
                  onValueChange={(value) => setConfig({...config, chunk_overlap: value[0]})}
                  max={1000}
                  min={0}
                  step={50}
                  className="w-full"
                />
                <p className="text-sm text-muted-foreground">
                  Overlap helps maintain context continuity between chunks
                </p>
              </div>

              {/* Similarity Threshold */}
              <div className="space-y-2">
                <Label>Similarity Threshold: {config.similarity_threshold}</Label>
                <Slider
                  value={[config.similarity_threshold]}
                  onValueChange={(value) => setConfig({...config, similarity_threshold: value[0]})}
                  max={1.0}
                  min={0.0}
                  step={0.05}
                  className="w-full"
                />
                <p className="text-sm text-muted-foreground">
                  Higher values return more relevant but fewer results
                </p>
              </div>

              {/* Max Results */}
              <div className="space-y-2">
                <Label>Maximum Results: {config.max_results}</Label>
                <Slider
                  value={[config.max_results]}
                  onValueChange={(value) => setConfig({...config, max_results: value[0]})}
                  max={50}
                  min={1}
                  step={1}
                  className="w-full"
                />
                <p className="text-sm text-muted-foreground">
                  Maximum number of results to return per search
                </p>
              </div>

              <Button onClick={updateConfig} className="w-full">
                Update Configuration
              </Button>
            </CardContent>
          </Card>

          {/* Performance Impact */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Performance Impact
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h4 className="font-medium">Current Settings Impact:</h4>
                  <div className="text-sm space-y-1">
                    <div className="flex justify-between">
                      <span>Search Speed:</span>
                      <Badge variant={config.max_results <= 10 ? "default" : "secondary"}>
                        {config.max_results <= 10 ? "Fast" : "Moderate"}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Result Quality:</span>
                      <Badge variant={config.similarity_threshold >= 0.7 ? "default" : "secondary"}>
                        {config.similarity_threshold >= 0.7 ? "High" : "Moderate"}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Context Richness:</span>
                      <Badge variant={config.chunk_size >= 1000 ? "default" : "secondary"}>
                        {config.chunk_size >= 1000 ? "Rich" : "Concise"}
                      </Badge>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-medium">Recommendations:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• For code search: Lower chunk size (500-800)</p>
                    <p>• For documentation: Higher chunk size (1000-2000)</p>
                    <p>• For precise answers: Higher similarity threshold (0.8+)</p>
                    <p>• For broad exploration: Lower threshold (0.6-0.7)</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Context Preview
              </CardTitle>
              <CardDescription>
                Preview how context will be formatted for AI consumption
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter a task description..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="flex-1"
                />
                <Button onClick={searchContext}>
                  Generate Preview
                </Button>
              </div>

              {searchResults.length > 0 && (
                <div className="border rounded-lg p-4 bg-gray-50">
                  <h4 className="font-medium mb-2">Formatted Context for AI:</h4>
                  <div className="text-sm font-mono whitespace-pre-wrap bg-white p-4 rounded border max-h-96 overflow-auto">
                    {`# Relevant Context for: ${searchQuery}

${searchResults.map((result, index) => `
## Source ${index + 1}: ${result.filename}
**File Type:** ${result.metadata.file_type || 'unknown'}
**Relevance:** ${(result.similarity * 100).toFixed(1)}%

**Content:**
${result.content}

---
`).join('')}`}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Context Quality Metrics */}
          {searchResults.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Context Quality Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold">
                      {(searchResults.reduce((acc, r) => acc + r.similarity, 0) / searchResults.length * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground">Average Relevance</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold">
                      {searchResults.reduce((acc, r) => acc + r.content.length, 0)}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Characters</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold">
                      {new Set(searchResults.map(r => r.document_id)).size}
                    </div>
                    <div className="text-sm text-muted-foreground">Unique Sources</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ContextEngineeringPage;
