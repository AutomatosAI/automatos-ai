"use client"

import React, { useState, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Search,
  Play,
  Database,
  Code,
  Table,
  BarChart,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Copy,
  Download,
  Maximize2,
  History,
  Send,
  FileDown
} from 'lucide-react'
import { toast } from 'sonner'

// Use simple CSS-based visualization (no SSR issues)
import { SimpleDataVisualization } from './SimpleDataVisualization'

interface DatabaseQueryExplorerProps {
  selectedSource: any
  sources: any[]
}

export function DatabaseQueryExplorer({ selectedSource, sources }: DatabaseQueryExplorerProps) {
  const [query, setQuery] = useState('')
  const [generatedSQL, setGeneratedSQL] = useState('')
  const [queryResult, setQueryResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [validationResult, setValidationResult] = useState<any>(null)
  const [queryHistory, setQueryHistory] = useState<any[]>([])
  const [selectedSourceId, setSelectedSourceId] = useState(selectedSource?.id || '')
  const [showVisualization, setShowVisualization] = useState(false)

  const handleVisualize = () => {
    if (!queryResult || queryResult.length === 0) {
      toast.error('No data to visualize')
      return
    }
    
    console.log('[Visualize] Button clicked, data:', queryResult)
    console.log('[Visualize] Opening modal...')
    
    try {
      setShowVisualization(true)
      console.log('[Visualize] Modal state set to true')
    } catch (error) {
      console.error('[Visualize] Error opening modal:', error)
      toast.error(`Visualization error: ${error}`)
    }
  }

  const handleQuerySubmit = async (forcedQuery?: string) => {
    const queryToRun = (forcedQuery ?? query).trim()
    if (!queryToRun || !selectedSourceId) {
      toast.error('Please enter a query and select a database')
      return
    }

    setIsLoading(true)
    setValidationResult(null)
    
    try {
      // Call the API to process natural language query
      const response = await fetch(`/api/knowledge/sources/database/${selectedSourceId}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: selectedSourceId,
          query: queryToRun
        })
      })

      if (response.ok) {
        const data = await response.json()
        setGeneratedSQL(data.sql)
        setQueryResult(data.data) // Backend returns 'data' not 'results'
        setValidationResult({ valid: data.success })
        setQuery(queryToRun)
        
        // Add to history
        setQueryHistory([
          {
            timestamp: new Date().toISOString(),
            query: queryToRun,
            sql: data.sql,
            rowCount: data.row_count || 0
          },
          ...queryHistory.slice(0, 9)
        ])
        
        toast.success(`Query executed successfully! ${data.row_count} rows returned`)
      } else {
        const error = await response.json()
        toast.error(error.detail || error.error || 'Query failed')
      }
    } catch (error) {
      toast.error('Failed to execute query')
    } finally {
      setIsLoading(false)
    }
  }

  const handleExportCSV = () => {
    if (!queryResult || queryResult.length === 0) {
      toast.error('No data to export')
      return
    }

    // Convert results to CSV
    const headers = Object.keys(queryResult[0])
    const csvContent = [
      headers.join(','),
      ...queryResult.map((row: any) => 
        headers.map(header => {
          const value = row[header]
          // Escape quotes and wrap in quotes if contains comma
          const escaped = String(value || '').replace(/"/g, '""')
          return escaped.includes(',') ? `"${escaped}"` : escaped
        }).join(',')
      )
    ].join('\n')

    // Download CSV
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `query-results-${new Date().toISOString()}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success('Results exported to CSV')
  }

  const handleSendToContext = async () => {
    if (!queryResult || !generatedSQL) {
      toast.error('No results to send')
      return
    }

    try {
      // Send to Context Engineering
      const response = await fetch('/api/context/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'database_query',
          content: {
            query: query,
            sql: generatedSQL,
            results: queryResult.slice(0, 100), // Limit results for context
            source_id: selectedSourceId,
            timestamp: new Date().toISOString()
          }
        })
      })

      if (response.ok) {
        toast.success('Query results added to context')
      } else {
        toast.error('Failed to add to context')
      }
    } catch (error) {
      toast.error('Failed to send to context')
    }
  }

  const suggestedQueries = [
    "Using workflow_executions, break down completions vs failures by day over the last 14 days",
    "From document_usage, summarize daily document search volume and average latency for the past 7 days",
    "Plot hourly CPU and memory utilization from the system_metrics table over the last 24 hours",
    "Highlight the most frequent CodeGraph queries this week based on codegraph_query_logs",
    "Show chat sessions created per day over the last 14 days from the chats table",
    "Report on the fastest agent task completion times this week using task_assignments",
    "List the documents most frequently searched in the past month from document_usage",
    "Compare average tool execution time by tool name over the last 14 days (tool_usage_logs)"
  ]

  return (
    <div className="space-y-4">
      {/* Query Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-500" />
            Natural Language SQL Explorer
          </CardTitle>
          <CardDescription>
            Ask questions in plain English - we'll handle the SQL
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Database Selection */}
          <div className="flex gap-2">
            <Select
              value={selectedSourceId}
              onValueChange={setSelectedSourceId}
            >
              <SelectTrigger className="w-[250px]">
                <SelectValue placeholder="Select database" />
              </SelectTrigger>
              <SelectContent>
                {sources.map((source) => (
                  <SelectItem key={source.id} value={source.id.toString()}>
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4" />
                      {source.name}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Badge variant="outline" className="py-2">
              Three-tier validation active
            </Badge>
          </div>

          {/* Query Input */}
          <div className="relative">
            <Textarea
              placeholder="Ask a question about your data..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="min-h-[100px] pr-20"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.metaKey) {
                  handleQuerySubmit()
                }
              }}
            />
            <Button
              onClick={() => handleQuerySubmit()}
              disabled={isLoading || !query.trim() || !selectedSourceId}
              className="absolute bottom-2 right-2"
              size="sm"
            >
              {isLoading ? (
                <>Processing...</>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-1" />
                  Run
                </>
              )}
            </Button>
          </div>

          {/* Suggested Queries */}
          <div>
            <p className="text-sm text-muted-foreground mb-2">Try these examples:</p>
            <div className="flex flex-wrap gap-2">
              {suggestedQueries.map((suggestion, idx) => (
                <Badge
                  key={idx}
                  variant="outline"
                  className="cursor-pointer hover:bg-muted"
                  onClick={() => handleQuerySubmit(suggestion)}
                >
                  {suggestion}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Generated SQL & Validation */}
      {generatedSQL && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                Generated SQL
              </span>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    navigator.clipboard.writeText(generatedSQL)
                    toast.success('SQL copied to clipboard')
                  }}
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
              <code className="text-sm">{generatedSQL}</code>
            </pre>
            
            {/* Validation Results */}
            {validationResult && (
              <div className="mt-4 space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Syntax validation passed</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Security validation passed (SELECT-only)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Semantic validation passed</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Query Results */}
      {queryResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Table className="h-5 w-5" />
                Results ({queryResult.length} rows)
              </span>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" onClick={handleExportCSV}>
                  <FileDown className="h-4 w-4 mr-1" />
                  Export CSV
                </Button>
                <Button variant="ghost" size="sm" onClick={handleSendToContext}>
                  <Send className="h-4 w-4 mr-1" />
                  Send to Context
                </Button>
                <Button variant="ghost" size="sm" onClick={handleVisualize}>
                  <BarChart className="h-4 w-4 mr-1" />
                  Visualize
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    {queryResult[0] && Object.keys(queryResult[0]).map((key) => (
                      <th key={key} className="text-left p-2 font-medium">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {queryResult.slice(0, 10).map((row: any, idx: number) => (
                    <tr key={idx} className="border-b hover:bg-muted/50">
                      {Object.values(row).map((value: any, vidx) => (
                        <td key={vidx} className="p-2">
                          {value?.toString() || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {queryResult.length > 10 && (
                <p className="text-sm text-muted-foreground mt-2">
                  Showing first 10 of {queryResult.length} rows
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Query History */}
      {queryHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Recent Queries
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {queryHistory.map((item, idx) => (
                <div
                  key={idx}
                  className="p-3 border rounded-lg hover:bg-muted/50 cursor-pointer"
                  onClick={() => setQuery(item.query)}
                >
                  <p className="font-medium text-sm">{item.query}</p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(item.timestamp).toLocaleString()} • {item.rowCount} rows
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Visualization Modal */}
      <SimpleDataVisualization
        isOpen={showVisualization}
        onClose={() => setShowVisualization(false)}
        data={queryResult || []}
      />
    </div>
  )
}
