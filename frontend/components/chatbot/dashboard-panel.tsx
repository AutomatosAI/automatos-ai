'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  BarChart3, LineChart, PieChart, Table2, Download, 
  Maximize2, Minimize2, RefreshCw, Sparkles, X,
  FileDown, Image, Share2, Lightbulb, TrendingUp,
  AlertTriangle, CheckCircle, Info
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { DatabaseResult, PandasAIChart } from '@/types'

interface DashboardPanelProps {
  isOpen: boolean
  onClose: () => void
  databaseResults: DatabaseResult[]
  charts: PandasAIChart[]
  aiInsight?: string
  sql?: string
  onExport?: (format: 'csv' | 'pdf' | 'png') => void
  onRefresh?: () => void
}

export function DashboardPanel({
  isOpen,
  onClose,
  databaseResults,
  charts,
  aiInsight,
  sql,
  onExport,
  onRefresh
}: DashboardPanelProps) {
  const [activeView, setActiveView] = useState<'table' | 'chart' | 'insight'>('table')
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Calculate summary stats from data
  const summaryStats = useMemo(() => {
    if (!databaseResults.length) return null
    
    const firstResult = databaseResults[0]
    const data = firstResult.data || []
    const rowCount = firstResult.row_count || data.length
    const columns = firstResult.columns || Object.keys(data[0] || {})
    
    // Find numeric columns and calculate stats
    const numericStats: Record<string, { sum: number; avg: number; min: number; max: number }> = {}
    
    columns.forEach((col: string) => {
      const values = data.map((row: any) => parseFloat(row[col])).filter((v: number) => !isNaN(v))
      if (values.length > 0) {
        numericStats[col] = {
          sum: values.reduce((a: number, b: number) => a + b, 0),
          avg: values.reduce((a: number, b: number) => a + b, 0) / values.length,
          min: Math.min(...values),
          max: Math.max(...values)
        }
      }
    })
    
    return { rowCount, columns, numericStats }
  }, [databaseResults])

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: 300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 300 }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className={`
          fixed right-0 top-0 h-full bg-gradient-to-b from-gray-900 via-gray-900 to-gray-950
          border-l border-cyan-500/20 shadow-2xl shadow-cyan-500/10
          ${isFullscreen ? 'w-full' : 'w-[600px]'}
          z-50 flex flex-col
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-cyan-500/20 bg-gradient-to-r from-cyan-500/10 to-purple-500/10">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Analytics Dashboard</h2>
              <p className="text-xs text-gray-400">
                {summaryStats?.rowCount || 0} rows • {summaryStats?.columns.length || 0} columns
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={onRefresh}
              className="text-gray-400 hover:text-cyan-400"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="text-gray-400 hover:text-cyan-400"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="text-gray-400 hover:text-red-400"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* AI Insight Banner */}
        {aiInsight && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mx-4 mt-4 p-4 rounded-xl bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30"
          >
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-lg bg-purple-500/20">
                <Sparkles className="w-4 h-4 text-purple-400" />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-purple-300 mb-1">AI Insight</h3>
                <p className="text-sm text-gray-300">{aiInsight}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Quick Stats Cards */}
        {summaryStats?.numericStats && Object.keys(summaryStats.numericStats).length > 0 && (
          <div className="grid grid-cols-3 gap-3 px-4 mt-4">
            {Object.entries(summaryStats.numericStats).slice(0, 3).map(([col, stats]) => (
              <Card key={col} className="bg-gray-800/50 border-gray-700/50">
                <CardContent className="p-3">
                  <p className="text-xs text-gray-500 truncate">{col}</p>
                  <p className="text-lg font-bold text-cyan-400">
                    {stats.sum.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </p>
                  <div className="flex items-center gap-1 mt-1">
                    <TrendingUp className="w-3 h-3 text-green-400" />
                    <span className="text-xs text-gray-400">
                      avg: {stats.avg.toFixed(1)}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* View Tabs */}
        <Tabs value={activeView} onValueChange={(v) => setActiveView(v as any)} className="flex-1 flex flex-col px-4 mt-4">
          <TabsList className="w-full bg-gray-800/50 border border-gray-700/50">
            <TabsTrigger value="table" className="flex-1 data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
              <Table2 className="w-4 h-4 mr-2" />
              Data
            </TabsTrigger>
            <TabsTrigger value="chart" className="flex-1 data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
              <BarChart3 className="w-4 h-4 mr-2" />
              Charts
            </TabsTrigger>
            <TabsTrigger value="insight" className="flex-1 data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400">
              <Lightbulb className="w-4 h-4 mr-2" />
              Analysis
            </TabsTrigger>
          </TabsList>

          {/* Data Table View */}
          <TabsContent value="table" className="flex-1 mt-4">
            <ScrollArea className="h-[calc(100vh-400px)]">
              {databaseResults.map((result, idx) => (
                <div key={idx} className="rounded-lg border border-gray-700/50 overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-800/80">
                      <tr>
                        {(result.columns || Object.keys(result.data?.[0] || {})).map((col: string) => (
                          <th key={col} className="px-3 py-2 text-left text-xs font-semibold text-gray-400 border-b border-gray-700/50">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(result.data || []).map((row: any, rowIdx: number) => (
                        <tr key={rowIdx} className="hover:bg-gray-800/40 transition-colors">
                          {(result.columns || Object.keys(row)).map((col: string) => (
                            <td key={col} className="px-3 py-2 text-gray-300 border-b border-gray-800/50">
                              {typeof row[col] === 'number' 
                                ? row[col].toLocaleString() 
                                : String(row[col] ?? '-')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </ScrollArea>
          </TabsContent>

          {/* Charts View */}
          <TabsContent value="chart" className="flex-1 mt-4">
            <ScrollArea className="h-[calc(100vh-400px)]">
              {charts.length > 0 ? (
                <div className="space-y-4">
                  {charts.map((chart, idx) => (
                    <Card key={idx} className="bg-gray-800/50 border-gray-700/50 overflow-hidden">
                      <CardContent className="p-0">
                        <img
                          src={`data:${chart.mime_type};base64,${chart.base64}`}
                          alt={chart.filename || 'Chart'}
                          className="w-full h-auto"
                        />
                      </CardContent>
                      <div className="px-4 py-2 bg-gray-900/50 border-t border-gray-700/50">
                        <p className="text-xs text-gray-500">{chart.filename}</p>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                  <BarChart3 className="w-12 h-12 mb-4 opacity-50" />
                  <p>No charts generated</p>
                  <p className="text-xs mt-2">Ask for visualizations like "show chart" or "plot graph"</p>
                </div>
              )}
            </ScrollArea>
          </TabsContent>

          {/* Analysis View */}
          <TabsContent value="insight" className="flex-1 mt-4">
            <ScrollArea className="h-[calc(100vh-400px)]">
              <div className="space-y-4">
                {/* SQL Query */}
                {sql && (
                  <Card className="bg-gray-800/50 border-gray-700/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2 text-cyan-400">
                        <Info className="w-4 h-4" />
                        Generated SQL
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="text-xs text-gray-300 bg-gray-900/50 p-3 rounded-lg overflow-x-auto">
                        {sql}
                      </pre>
                    </CardContent>
                  </Card>
                )}

                {/* Data Summary */}
                {summaryStats && (
                  <Card className="bg-gray-800/50 border-gray-700/50">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2 text-amber-400">
                        <TrendingUp className="w-4 h-4" />
                        Data Summary
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Total Rows</span>
                        <span className="text-white font-medium">{summaryStats.rowCount}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Columns</span>
                        <span className="text-white font-medium">{summaryStats.columns.length}</span>
                      </div>
                      {Object.entries(summaryStats.numericStats).map(([col, stats]) => (
                        <div key={col} className="pt-2 border-t border-gray-700/50">
                          <p className="text-xs text-gray-500 mb-1">{col}</p>
                          <div className="grid grid-cols-4 gap-2 text-xs">
                            <div>
                              <span className="text-gray-500">Sum: </span>
                              <span className="text-cyan-400">{stats.sum.toLocaleString()}</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Avg: </span>
                              <span className="text-purple-400">{stats.avg.toFixed(2)}</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Min: </span>
                              <span className="text-green-400">{stats.min.toLocaleString()}</span>
                            </div>
                            <div>
                              <span className="text-gray-500">Max: </span>
                              <span className="text-amber-400">{stats.max.toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                {/* AI Analysis */}
                {aiInsight && (
                  <Card className="bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border-purple-500/30">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center gap-2 text-purple-400">
                        <Sparkles className="w-4 h-4" />
                        AI Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-gray-300">{aiInsight}</p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>

        {/* Export Actions */}
        <div className="px-4 py-4 border-t border-cyan-500/20 bg-gray-900/80">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => onExport?.('csv')}
              className="flex-1 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10"
            >
              <FileDown className="w-4 h-4 mr-2" />
              CSV
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onExport?.('pdf')}
              className="flex-1 border-purple-500/30 text-purple-400 hover:bg-purple-500/10"
            >
              <Download className="w-4 h-4 mr-2" />
              PDF
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onExport?.('png')}
              className="flex-1 border-amber-500/30 text-amber-400 hover:bg-amber-500/10"
            >
              <Image className="w-4 h-4 mr-2" />
              Image
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="border-green-500/30 text-green-400 hover:bg-green-500/10"
            >
              <Share2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

