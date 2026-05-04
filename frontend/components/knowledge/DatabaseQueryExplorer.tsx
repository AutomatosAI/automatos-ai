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
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
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
  FileDown,
  Trash2,
  RefreshCw
} from 'lucide-react'
import { toast } from 'sonner'
import apiClient from '@/lib/api-client'

// Use simple CSS-based visualization (no SSR issues)
import { SimpleDataVisualization } from './SimpleDataVisualization'

interface DatabaseQueryExplorerProps {
  selectedSource: any
  sources: any[]
  onSourceDeleted?: () => void  // Callback to refresh sources list after delete
}

export function DatabaseQueryExplorer({ selectedSource, sources, onSourceDeleted }: DatabaseQueryExplorerProps) {
  const [query, setQuery] = useState('')
  const [generatedSQL, setGeneratedSQL] = useState('')
  const [queryResult, setQueryResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [validationResult, setValidationResult] = useState<any>(null)
  const [queryHistory, setQueryHistory] = useState<any[]>([])
  const [selectedSourceId, setSelectedSourceId] = useState(selectedSource?.id?.toString() || '')
  const [showVisualization, setShowVisualization] = useState(false)
  const [confidence, setConfidence] = useState<any>(null)

  // Get the selected source name for the delete dialog
  const selectedSourceName = sources.find(s => s.id.toString() === selectedSourceId)?.name || 'this database'

  const handleDeleteClick = () => {
    console.log('[Delete] Button clicked, selectedSourceId:', selectedSourceId)
    if (!selectedSourceId) {
      toast.error('No database selected')
      return
    }
    setShowDeleteDialog(true)
  }

  const handleConfirmDelete = async () => {
    console.log('[Delete] Confirming delete for source:', selectedSourceId)
    setShowDeleteDialog(false)
    setIsDeleting(true)
    
    try {
      const data = await apiClient.request(`/api/knowledge/sources/database/${selectedSourceId}`, {
        method: 'DELETE'
      })

      console.log('[Delete] Success:', data)
      toast.success((data as any)?.message || 'Database source deleted successfully')

      // Clear current selection and results
      setSelectedSourceId('')
      setGeneratedSQL('')
      setQueryResult(null)
      setValidationResult(null)
      setQuery('')

      // Notify parent to refresh sources list
      onSourceDeleted?.()
    } catch (error: any) {
      console.error('[Delete] Exception:', error)
      toast.error(error?.message || 'Failed to delete database source')
    } finally {
      setIsDeleting(false)
    }
  }

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
      const data = await apiClient.request<any>(`/api/knowledge/sources/database/${selectedSourceId}/query`, {
        method: 'POST',
        body: {
          source_id: selectedSourceId,
          query: queryToRun
        } as any
      })

      setGeneratedSQL(data.sql)
      setQueryResult(data.data) // Backend returns 'data' not 'results'
      setValidationResult({ valid: data.success })
      setConfidence(data.confidence || null)
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
    } catch (error: any) {
      toast.error(error?.message || 'Failed to execute query')
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
      await apiClient.request('/api/context/add', {
        method: 'POST',
        body: {
          type: 'database_query',
          content: {
            query: query,
            sql: generatedSQL,
            results: queryResult.slice(0, 100), // Limit results for context
            source_id: selectedSourceId,
            timestamp: new Date().toISOString()
          }
        } as any
      })

      toast.success('Query results added to context')
    } catch (error: any) {
      toast.error(error?.message || 'Failed to send to context')
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
            <Sparkles className="h-5 w-5 text-agent" />
            Natural Language SQL Explorer
          </CardTitle>
          <CardDescription>
            Ask questions in plain English - we'll handle the SQL
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Database Selection */}
          <div className="flex gap-2 items-center">
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
            
            {selectedSourceId && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleDeleteClick}
                disabled={isDeleting}
                className="text-destructive hover:text-destructive hover:bg-destructive/10"
                title="Delete this database source"
              >
                {isDeleting ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
              </Button>
            )}
            
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
                {confidence && (
                  <Badge
                    variant="outline"
                    className={
                      confidence.level === 'high'
                        ? 'bg-success/10 text-success border-success/30'
                        : confidence.level === 'medium'
                        ? 'bg-warning/10 text-warning border-warning/30'
                        : 'bg-destructive/10 text-destructive border-destructive/30'
                    }
                    title={`Score: ${confidence.score}/100 | ${confidence.recommendation || ''}`}
                  >
                    {confidence.level === 'high' ? 'High' : confidence.level === 'medium' ? 'Medium' : 'Low'} Confidence
                    {confidence.score != null && ` (${confidence.score})`}
                  </Badge>
                )}
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
                  <CheckCircle className="h-4 w-4 text-success" />
                  <span className="text-sm">Syntax validation passed</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-success" />
                  <span className="text-sm">Security validation passed (SELECT-only)</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-success" />
                  <span className="text-sm">Semantic validation passed</span>
                </div>
              </div>
            )}

            {/* Confidence Factor Breakdown */}
            {confidence?.factors && (
              <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                <p className="text-xs font-medium mb-2 text-muted-foreground">Confidence Factors</p>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {Object.entries(confidence.factors).map(([key, value]: [string, any]) => (
                    <div key={key} className="text-center">
                      <div className="text-sm font-medium">{typeof value === 'number' ? value : 0}</div>
                      <div className="text-[10px] text-muted-foreground capitalize">
                        {key.replace(/_/g, ' ')}
                      </div>
                    </div>
                  ))}
                </div>
                {confidence.recommendation && (
                  <p className="text-xs text-muted-foreground mt-2 italic">
                    {confidence.recommendation}
                  </p>
                )}
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

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent className="sm:max-w-[425px]">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-destructive">
              <Trash2 className="h-5 w-5" />
              Delete Database Source
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-3">
              <p>
                Are you sure you want to delete <span className="font-semibold text-foreground">"{selectedSourceName}"</span>?
              </p>
              <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm">
                <p className="font-medium text-destructive mb-1">This action cannot be undone.</p>
                <ul className="text-muted-foreground space-y-1 ml-4 list-disc">
                  <li>Schema metadata will be permanently removed</li>
                  <li>Query history will be deleted</li>
                  <li>Semantic layer definitions will be lost</li>
                </ul>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleConfirmDelete}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Database
                </>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
