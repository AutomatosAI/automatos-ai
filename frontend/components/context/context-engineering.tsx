'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import {
  Brain,
  Search,
  Database,
  Network,
  Eye,
  Settings,
  BarChart,
  Zap,
  FileText,
  Target,
  Activity,
  TrendingUp,
  Filter,
  RefreshCw,
  AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { TabsContent } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, BarChart as RechartsBarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { PageHeader, FilterTabs } from '@/components/shared'

// Import all the context hooks
import {
  useContextStats,
  useContextPerformance,
  useContextSources,
  useRecentContextQueries,
  useContextPatterns,
  useTestContextRAG,
  useOptimizeContext,
  useOptimizationRecommendations
} from '@/hooks/use-context-management-api'
import { RAGContextBuilder } from './rag-context-builder'
import { ConfigureRAGModal } from './configure-rag-modal'
import { PatternDetailsModal } from './pattern-details-modal'
import { DatabaseQueryAnalytics } from './DatabaseQueryAnalytics'

const confidenceColors = {
  high: 'text-success',
  medium: 'text-warning',
  low: 'text-destructive'
}

const getConfidenceLevel = (confidence: number) => {
  if (confidence >= 0.9) return 'high'
  if (confidence >= 0.7) return 'medium'
  return 'low'
}

export function ContextEngineering() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [searchTerm, setSearchTerm] = useState('')
  const [configureModalOpen, setConfigureModalOpen] = useState(false)
  const [selectedPattern, setSelectedPattern] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('performance')
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Use real API hooks - they will call API and fallback to mock if needed
  const { data: contextStats, isLoading: statsLoading, refetch: refetchStats } = useContextStats()
  const { data: performanceData, isLoading: perfLoading } = useContextPerformance(selectedTimeRange)
  const { data: contextSources, isLoading: sourcesLoading } = useContextSources()
  const { data: recentQueries, isLoading: queriesLoading } = useRecentContextQueries()
  const { data: contextPatterns, isLoading: patternsLoading } = useContextPatterns()
  const { data: optimizationData, isLoading: optimizationLoading, refetch: refetchOptimization } = useOptimizationRecommendations()
  
  // Mutations
  const testRAGMutation = useTestContextRAG()
  const optimizeMutation = useOptimizeContext()
  
  // Combined loading state
  const loading = statsLoading || perfLoading || sourcesLoading || queriesLoading || patternsLoading

  // Create stats cards from API data ONLY
  const contextStatsCards = useMemo(() => [
    {
      label: 'Context Queries',
      value: (contextStats as any)?.contextQueries?.toString() || '0',
      change: (contextStats as any)?.lastQueryTime ? `Last: ${new Date((contextStats as any).lastQueryTime).toLocaleTimeString()}` : 'No activity',
      icon: Search,
      color: 'text-info'
    },
    {
      label: 'Retrieval Success',
      value: `${((contextStats as any)?.retrievalSuccess || 0).toFixed(1)}%`,
      change: (contextStats as any)?.systemStatus === 'operational' ? 'System operational' : 'No RAG activity',
      icon: Target,
      color: 'text-success'
    },
    {
      label: 'Avg Response Time',
      value: (contextStats as any)?.avgResponseTime || '0.00s',
      change: 'Real-time measurement',
      icon: Zap,
      color: 'text-primary'
    },
    {
      label: 'Vector Embeddings',
      value: (contextStats as any)?.vectorEmbeddings?.toString() || '0',
      change: `${(contextStats as any)?.vectorEmbeddings || 0} total chunks`,
      icon: Database,
      color: 'text-agent'
    }
  ], [contextStats])

  // Process performance data for charts from API ONLY
  const ragPerformanceData = useMemo(() => {
    if (performanceData && Array.isArray(performanceData)) {
      return (performanceData as any[]).map((item: any) => ({
        time: item.time || '00:00',
        queries: item.queries || 0,
        success_rate: item.success_rate || 0,
        avg_latency: item.avg_latency || 0
      }))
    }
    return []
  }, [performanceData])

  // Process context sources from API ONLY
  const contextSourcesData = useMemo(() => {
    // API returns {sources: [...]} so extract the array
    const sources = (contextSources as any)?.sources || contextSources
    
    if (sources && Array.isArray(sources)) {
      // Define colors for different sources
      const colors = ['#60B5FF', '#FF6B9D', '#FFC371', '#4ECDC4', '#95E1D3', '#F38181']
      
      return sources.map((source: any, index: number) => ({
        name: source.name || 'Unknown',
        value: source.count || source.value || 0,  // API uses "count", fallback to "value"
        color: source.color || colors[index % colors.length]
      }))
    }
    return []
  }, [contextSources])

  // Process recent queries from API ONLY
  const recentQueriesData = useMemo(() => {
    if (recentQueries && Array.isArray(recentQueries)) {
      return recentQueries.map((query: any) => ({
        id: query.id || `query-${Date.now()}`,
        query: query.query_text || query.query || 'Unknown query',
        agent: query.agent || 'Unknown',
        confidence: query.confidence || 0,
        sources: query.sources || 0,
        responseTime: `${query.latency || 0}ms`,
        timestamp: query.timestamp || 'Unknown',
        category: query.category || 'General'
      }))
    }
    return []
  }, [recentQueries])

  // Process context patterns from API ONLY
  const contextPatternsData = useMemo(() => {
    if (contextPatterns && Array.isArray(contextPatterns)) {
      return contextPatterns.map((pattern: any) => ({
        id: pattern.id || `pattern-${Date.now()}`,
        name: pattern.name || 'Unknown Pattern',
        description: pattern.description || 'No description available',
        usage: pattern.usage || pattern.count || 0,
        accuracy: pattern.accuracy || 0,
        avgSources: pattern.avgSources || 0,
        category: pattern.category || 'General',
        status: pattern.status || 'unknown'
      }))
    }
    return []
  }, [contextPatterns])

  const handleRefresh = async () => {
    await refetchStats()
  }

  const handleTestRAG = async () => {
    await testRAGMutation.mutateAsync(searchTerm)
  }

  const handleOptimize = async () => {
    try {
      console.log('[Optimization] Starting analysis...')
      const result = await refetchOptimization()
      console.log('[Optimization] Analysis complete:', result)
    } catch (error) {
      console.error('[Optimization] Analysis failed:', error)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <PageHeader
        title="Context"
        titleAccent="Engineering"
        subtitle="Monitor RAG system performance and context retrieval patterns"
        actions={
          <>
            <Button variant="outline" onClick={handleRefresh} disabled={loading}>
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              className="hover:opacity-90"
              onClick={() => setConfigureModalOpen(true)}
            >
              <Settings className="w-4 h-4 mr-2" />
              Configure RAG
            </Button>
          </>
        }
      />

      {/* Stats Overview */}
      <motion.div
        ref={ref}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        {contextStatsCards.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-4 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-10 h-10 rounded-2xl bg-black/20 border border-primary/10 flex items-center justify-center shrink-0">
                  <stat.icon className={`w-5 h-5 ${stat.color}`} />
                </div>
                <div className="min-w-0">
                  <div className="text-2xl font-bold leading-none">{loading ? '…' : stat.value}</div>
                  <div className="text-sm text-muted-foreground truncate">{stat.label}</div>
                </div>
              </div>
              <div className="shrink-0 text-right text-xs text-success">
                {loading ? 'Loading…' : stat.change}
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {/* Context Engineering Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <FilterTabs
          tabs={[
            { value: 'performance', label: 'Performance', icon: BarChart },
            { value: 'queries', label: 'Query Analysis', icon: Search },
            { value: 'database', label: 'Database Analytics', icon: Database },
            { value: 'patterns', label: 'Patterns', icon: Network },
            { value: 'optimization', label: 'Optimization', icon: Brain },
            { value: 'rag', label: 'RAG Context', icon: Zap },
          ]}
          value={activeTab}
          onValueChange={setActiveTab}
        >

          <TabsContent value="performance" className="space-y-6">
            {/* Performance Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* RAG Performance Chart */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>RAG Performance Metrics</span>
                    <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                      <SelectTrigger className="w-24 bg-secondary/50">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1h">1h</SelectItem>
                        <SelectItem value="24h">24h</SelectItem>
                        <SelectItem value="7d">7d</SelectItem>
                        <SelectItem value="30d">30d</SelectItem>
                      </SelectContent>
                    </Select>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    {loading ? (
                      <div className="h-full flex items-center justify-center">
                        <div className="animate-pulse">Loading performance data...</div>
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={ragPerformanceData}>
                          <XAxis 
                            dataKey="time" 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                          />
                          <YAxis 
                            axisLine={false}
                            tickLine={false}
                            tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'hsl(var(--card))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
                              fontSize: '12px'
                            }}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="success_rate" 
                            stroke="#72BF78" 
                            strokeWidth={2}
                            name="Success Rate (%)"
                          />
                          <Line 
                            type="monotone" 
                            dataKey="avg_latency" 
                            stroke="#ff6b35" 
                            strokeWidth={2}
                            name="Avg Latency (s)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Context Sources Distribution */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Context Sources Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    {loading ? (
                      <div className="h-full flex items-center justify-center">
                        <div className="animate-pulse">Loading sources...</div>
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={contextSourcesData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {contextSourcesData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Query Volume Chart */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Query Volume Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  {loading ? (
                    <div className="h-full flex items-center justify-center">
                      <div className="animate-pulse">Loading query trends...</div>
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsBarChart data={ragPerformanceData}>
                        <XAxis 
                          dataKey="time" 
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                        />
                        <YAxis 
                          axisLine={false}
                          tickLine={false}
                          tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--card))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }}
                        />
                        <Bar 
                          dataKey="queries" 
                          fill="#60B5FF" 
                          name="Query Count"
                          radius={[4, 4, 0, 0]}
                        />
                      </RechartsBarChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="queries" className="space-y-6">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search recent queries..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
                />
              </div>
              <Button variant="outline" className="shrink-0">
                <Filter className="w-4 h-4 mr-2" />
                Filters
              </Button>
            </div>

            {/* Recent Queries */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Recent Context Queries</CardTitle>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="space-y-4">
                    {[1, 2, 3, 4].map(i => (
                      <div key={i} className="animate-pulse">
                        <div className="h-20 bg-secondary/50 rounded" />
                      </div>
                    ))}
                  </div>
                ) : recentQueriesData.length === 0 ? (
                  <p className="text-center text-muted-foreground py-8">
                    No recent queries available
                  </p>
                ) : (
                  <div className="space-y-4">
                    {recentQueriesData.map((query, index) => (
                      <motion.div
                        key={query.id}
                        className="p-4 rounded-lg border border-border/50 hover:border-primary/20 transition-all duration-300"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.05 }}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1 min-w-0">
                            <p className="font-medium text-sm mb-1">{query.query}</p>
                            <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                              <span>Agent: {query.agent}</span>
                              <span>Sources: {query.sources}</span>
                              <span>Response: {query.responseTime}</span>
                              <span>{query.timestamp}</span>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Badge variant="outline" className="text-xs">
                              {query.category}
                            </Badge>
                            <div className={`text-xs font-medium ${confidenceColors[getConfidenceLevel(query.confidence)]}`}>
                              {(query.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>
                        <div className="w-full bg-secondary rounded-full h-1">
                          <div 
                            className="bg-gradient-to-r from-warning to-red-500 h-1 rounded-full transition-all duration-300"
                            style={{ width: `${query.confidence * 100}%` }}
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Test RAG System */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Test Context Retrieval
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Input
                    placeholder="Enter a test query..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                  <Button 
                    onClick={handleTestRAG}
                    disabled={(testRAGMutation as any).isPending || !searchTerm}
                    className="w-full"
                  >
                    {(testRAGMutation as any).isPending ? 'Testing...' : 'Test RAG System'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="database" className="space-y-6">
            {/* Real Database Query Analytics - fetched from database_query_audit */}
            <DatabaseQueryAnalytics />
          </TabsContent>

          <TabsContent value="patterns" className="space-y-6">
            {/* Context Patterns */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {loading ? (
                [1, 2, 3].map(i => (
                  <div key={i} className="animate-pulse">
                    <div className="h-48 bg-secondary/50 rounded" />
                  </div>
                ))
              ) : contextPatternsData.length === 0 ? (
                <div className="col-span-full text-center py-8 text-muted-foreground">
                  No context patterns available
                </div>
              ) : (
                contextPatternsData.map((pattern, index) => (
                  <motion.div
                    key={pattern.id}
                    className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="font-semibold text-lg mb-1">{pattern.name}</h3>
                        <p className="text-sm text-muted-foreground">{pattern.description}</p>
                      </div>
                      <Badge 
                        variant={pattern.status === 'active' ? 'default' : 'secondary'}
                        className="text-xs"
                      >
                        {pattern.status}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div>
                        <p className="text-sm font-medium">{pattern.usage}</p>
                        <p className="text-xs text-muted-foreground">Usage</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">{pattern.accuracy}%</p>
                        <p className="text-xs text-muted-foreground">Accuracy</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">{pattern.avgSources}</p>
                        <p className="text-xs text-muted-foreground">Avg Sources</p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {pattern.category}
                      </Badge>
                      <Button 
                        variant="ghost" 
                        size="sm"
                        onClick={() => setSelectedPattern(pattern)}
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        View Details
                      </Button>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="optimization" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4" />
                    RAG System Optimization
                  </div>
                  <Button 
                    size="sm" 
                    onClick={handleOptimize}
                    disabled={optimizationLoading}
                  >
                    {optimizationLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Analyze System
                      </>
                    )}
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* System Health Status */}
                  {optimizationData && (optimizationData as any).system_health && (
                    <div className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          (optimizationData as any).system_health === 'healthy' ? 'bg-success/20' :
                          (optimizationData as any).system_health === 'needs_attention' ? 'bg-warning/20' :
                          'bg-destructive/20'
                        }`}>
                          <Activity className={`w-5 h-5 ${
                            (optimizationData as any).system_health === 'healthy' ? 'text-success' :
                            (optimizationData as any).system_health === 'needs_attention' ? 'text-warning' :
                            'text-destructive'
                          }`} />
                        </div>
                        <div>
                          <p className="font-semibold">System Health</p>
                          <p className="text-sm text-muted-foreground">
                            Last analyzed: {new Date((optimizationData as any).last_analyzed).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge 
                        variant={(optimizationData as any).system_health === 'healthy' ? 'default' : 'destructive'}
                        className="text-lg px-4 py-1"
                      >
                        {(optimizationData as any).system_health === 'healthy' ? '✓ Healthy' :
                         (optimizationData as any).system_health === 'needs_attention' ? '⚠ Needs Attention' :
                         '✗ Critical'}
                      </Badge>
                    </div>
                  )}

                  {/* Recommendations */}
                  {optimizationLoading ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
                      <p>Analyzing RAG system...</p>
                    </div>
                  ) : optimizationData && (optimizationData as any).recommendations?.length > 0 ? (
                    <div className="space-y-3">
                      <h4 className="font-semibold text-sm text-muted-foreground">Optimization Recommendations</h4>
                      {(optimizationData as any).recommendations.map((rec: any, index: number) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className={`p-4 border rounded-lg ${
                            rec.type === 'error' ? 'border-destructive/20 bg-destructive/5' :
                            rec.type === 'warning' ? 'border-warning/20 bg-warning/5' :
                            rec.type === 'success' ? 'border-success/20 bg-success/5' :
                            'border-info/20 bg-info/5'
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                              rec.type === 'error' ? 'bg-destructive/20' :
                              rec.type === 'warning' ? 'bg-warning/20' :
                              rec.type === 'success' ? 'bg-success/20' :
                              'bg-info/20'
                            }`}>
                              {rec.type === 'error' ? '✗' :
                               rec.type === 'warning' ? '⚠' :
                               rec.type === 'success' ? '✓' :
                               'ℹ'}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-2">
                                <h5 className="font-semibold">{rec.title}</h5>
                                <Badge variant="outline" className="text-xs">
                                  {rec.category}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mb-2">
                                {rec.description}
                              </p>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 text-xs">
                                  <Zap className="w-3 h-3" />
                                  <span className="text-muted-foreground">Action:</span>
                                  <span className="font-medium">{rec.action}</span>
                                </div>
                                <Badge 
                                  variant={rec.impact === 'critical' ? 'destructive' : 
                                          rec.impact === 'high' ? 'default' : 'secondary'}
                                  className="text-xs"
                                >
                                  {rec.impact} impact
                                </Badge>
                              </div>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Brain className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="font-medium">No optimization data available</p>
                      <p className="text-sm">Click "Analyze System" to get recommendations</p>
                  </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="rag" className="space-y-6">
            <RAGContextBuilder />
          </TabsContent>
        </FilterTabs>
      </motion.div>

      {/* Configure RAG Modal */}
      <ConfigureRAGModal
        isOpen={configureModalOpen}
        onClose={() => setConfigureModalOpen(false)}
        onConfigCreated={(config) => {
          console.log('New pattern created:', config)
          setConfigureModalOpen(false)
          refetchStats() // Refresh to show new pattern
        }}
      />

      {/* Pattern Details Modal */}
      <PatternDetailsModal
        isOpen={!!selectedPattern}
        onClose={() => setSelectedPattern(null)}
        pattern={selectedPattern}
      />
    </div>
  )
}