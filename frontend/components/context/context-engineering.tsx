
'use client'

import { useState, useEffect } from 'react'
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
  RefreshCw
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, BarChart as RechartsBarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { PolicyEditor } from '@/components/context/PolicyEditor'
import { AssemblyPreview } from '@/components/context/AssemblyPreview'
import { usePolicy, useAssemble, useUpsertPolicy } from '@/hooks/use-policy'
import { contextService, type ContextStats, type ContextQuery, type ContextPattern, type ContextSource, type RAGPerformanceData } from '@/lib/context-service'
import { ConfigureRAGModal } from './configure-rag-modal'

// Real data will be loaded from backend
const initialContextStats = [
  {
    label: 'Context Queries',
    value: '0',
    change: 'Loading...',
    icon: Search,
    color: 'text-blue-400'
  },
  {
    label: 'Retrieval Success',
    value: '0%',
    change: 'Loading...',
    icon: Target,
    color: 'text-green-400'
  },
  {
    label: 'Avg Response Time',
    value: '0s',
    change: 'Loading...',
    icon: Zap,
    color: 'text-orange-400'
  },
  {
    label: 'Vector Embeddings',
    value: '0',
    change: 'Loading...',
    icon: Database,
    color: 'text-purple-400'
  }
]

// Real data will be loaded from backend - initial empty states

// Hardcoded data removed - will be loaded from backend

const confidenceColors = {
  high: 'text-green-400',
  medium: 'text-yellow-400',
  low: 'text-red-400'
}

const getConfidenceLevel = (confidence: number) => {
  if (confidence >= 0.9) return 'high'
  if (confidence >= 0.7) return 'medium'
  return 'low'
}

export function ContextEngineering() {
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h')
  const [searchTerm, setSearchTerm] = useState('')
  const [contextStats, setContextStats] = useState(initialContextStats)
  const [ragPerformanceData, setRagPerformanceData] = useState<RAGPerformanceData[]>([])
  const [contextSources, setContextSources] = useState<ContextSource[]>([])
  const [recentQueries, setRecentQueries] = useState<ContextQuery[]>([])
  const [contextPatterns, setContextPatterns] = useState<ContextPattern[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showConfigureModal, setShowConfigureModal] = useState(false)
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  })

  // Policy tab state/hooks
  const POLICY_ID = 'code_assistant'
  const { data: policyData } = usePolicy(POLICY_ID)
  const [policy, setPolicy] = useState<any>(null)
  useEffect(() => {
    if (policyData && !policy) {
      // backend may wrap in { policy }
      // @ts-ignore
      setPolicy((policyData as any)?.policy || policyData)
    }
  }, [policyData])
  const assemble = useAssemble(POLICY_ID)
  const upsert = useUpsertPolicy(POLICY_ID)
  const [testQuery, setTestQuery] = useState('Summarize main ideas from this document')

  // Load real data from backend
  useEffect(() => {
    loadContextData()
  }, [])

  const loadContextData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load all context data in parallel
      const [
        statsData,
        performanceData,
        sourcesData,
        queriesData,
        patternsData
      ] = await Promise.all([
        contextService.getContextStats(),
        contextService.getRAGPerformanceData(),
        contextService.getContextSources(),
        contextService.getRecentQueries(),
        contextService.getContextPatterns()
      ])

      // Update stats with real data
      setContextStats([
        {
          label: 'Context Queries',
          value: statsData.contextQueries.toLocaleString(),
          change: `${statsData.contextQueries > 0 ? 'Active' : 'No activity'}`,
          icon: Search,
          color: 'text-blue-400'
        },
        {
          label: 'Retrieval Success',
          value: `${statsData.retrievalSuccess.toFixed(1)}%`,
          change: statsData.retrievalSuccess > 0 ? 'System operational' : 'No RAG activity',
          icon: Target,
          color: 'text-green-400'
        },
        {
          label: 'Avg Response Time',
          value: statsData.avgResponseTime,
          change: 'Real-time measurement',
          icon: Zap,
          color: 'text-orange-400'
        },
        {
          label: 'Vector Embeddings',
          value: statsData.vectorEmbeddings.toLocaleString(),
          change: `${statsData.vectorEmbeddings} total chunks`,
          icon: Database,
          color: 'text-purple-400'
        }
      ])

      setRagPerformanceData(performanceData)
      setContextSources(sourcesData)
      setRecentQueries(queriesData)
      setContextPatterns(patternsData)

    } catch (err) {
      console.error('Error loading context data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load context data')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold mb-2">
            Context <span className="gradient-text">Engineering</span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Monitor RAG system performance and context retrieval patterns
          </p>
        </div>
        
        <div className="flex space-x-2">
          <Button variant="outline" onClick={loadContextData}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button 
            className="gradient-accent hover:opacity-90"
            onClick={() => setShowConfigureModal(true)}
          >
            <Settings className="w-4 h-4 mr-2" />
            Configure RAG
          </Button>
        </div>
      </motion.div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading context data...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <Brain className="h-8 w-8 text-red-400 mx-auto mb-4" />
            <p className="text-red-400 mb-4">Error loading context data: {error}</p>
            <Button onClick={loadContextData} variant="outline">
              Try Again
            </Button>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      {!loading && !error && (
        <motion.div
          ref={ref}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          {contextStats.map((stat, index) => (
          <motion.div
            key={stat.label}
            className="glass-card p-6 card-glow hover:border-primary/20 transition-all duration-300"
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">{stat.value}</h3>
              <p className="text-muted-foreground text-sm">{stat.label}</p>
              <p className="text-xs text-green-400">{stat.change}</p>
            </div>
          </motion.div>
        ))}
        </motion.div>
      )}

      {/* Context Engineering Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <Tabs defaultValue="performance" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid bg-secondary/50">
            <TabsTrigger value="performance" className="flex items-center space-x-2">
              <BarChart className="w-4 h-4" />
              <span className="hidden sm:inline">Performance</span>
            </TabsTrigger>
            <TabsTrigger value="queries" className="flex items-center space-x-2">
              <Search className="w-4 h-4" />
              <span className="hidden sm:inline">Query Analysis</span>
            </TabsTrigger>
            <TabsTrigger value="patterns" className="flex items-center space-x-2">
              <Network className="w-4 h-4" />
              <span className="hidden sm:inline">Patterns</span>
            </TabsTrigger>
            <TabsTrigger value="policy" className="flex items-center space-x-2">
              <FileText className="w-4 h-4" />
              <span className="hidden sm:inline">Policy</span>
            </TabsTrigger>
            <TabsTrigger value="optimization" className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Optimization</span>
            </TabsTrigger>
          </TabsList>

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
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={contextSources}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${((percent || 0) * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {contextSources.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
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
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="policy" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Policy Editor</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex gap-2 items-center">
                      <Input
                        placeholder="Test query..."
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        className="bg-secondary/50 border-secondary focus:border-primary/50"
                      />
                      <Button
                        variant="outline"
                        onClick={() => assemble.mutate({ q: testQuery })}
                        disabled={assemble.isPending}
                      >
                        Assemble
                      </Button>
                      <Button
                        onClick={() => policy && upsert.mutate({ policy })}
                        disabled={!policy || upsert.isPending}
                      >
                        Save
                      </Button>
                    </div>
                    <PolicyEditor policy={policy || {}} onChange={setPolicy} />
                  </div>
                </CardContent>
              </Card>

              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Assembly Preview</CardTitle>
                </CardHeader>
                <CardContent>
                  <AssemblyPreview result={assemble.data} />
                </CardContent>
              </Card>
            </div>
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
                <div className="space-y-4">
                  {recentQueries.map((query, index) => (
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
                          className="bg-gradient-to-r from-orange-500 to-red-500 h-1 rounded-full transition-all duration-300"
                          style={{ width: `${query.confidence * 100}%` }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="patterns" className="space-y-6">
            {/* Context Patterns */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {contextPatterns.map((pattern, index) => (
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
                    <Button variant="ghost" size="sm">
                      <Eye className="w-4 h-4 mr-1" />
                      View Details
                    </Button>
                  </div>
                </motion.div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="optimization" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>RAG System Optimization</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div className="text-center text-muted-foreground">
                    RAG system optimization tools and recommendations will be displayed here
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Configure RAG Modal */}
      <ConfigureRAGModal
        isOpen={showConfigureModal}
        onClose={() => setShowConfigureModal(false)}
        onConfigCreated={(config) => {
          console.log('RAG configuration created:', config)
          // Refresh data to show new configuration
          loadContextData()
          setShowConfigureModal(false)
        }}
      />
    </div>
  )
}
