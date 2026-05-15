'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { apiClient } from '@/lib/api-client'
import { 
  X, 
  Settings, 
  BarChart3, 
  TestTube,
  CheckCircle,
  AlertCircle,
  Edit,
  Save,
  TrendingUp,
  Clock,
  Target,
  Database
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'

interface PatternDetailsModalProps {
  isOpen: boolean
  onClose: () => void
  pattern: any
}

export function PatternDetailsModal({ isOpen, onClose, pattern }: PatternDetailsModalProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editedPattern, setEditedPattern] = useState(pattern)
  const [testQuery, setTestQuery] = useState('')
  const [testResults, setTestResults] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [recentQueries, setRecentQueries] = useState<any[]>([])
  const [queriesLoading, setQueriesLoading] = useState(false)

  // Update editedPattern when pattern changes
  useEffect(() => {
    if (pattern) {
      setEditedPattern(pattern)
    }
  }, [pattern])

  // Fetch recent queries when modal opens
  useEffect(() => {
    if (isOpen && pattern) {
      fetchRecentQueries()
    }
  }, [isOpen, pattern])

  const fetchRecentQueries = async () => {
    setQueriesLoading(true)
    try {
      const queries = await apiClient.getRecentContextQueries()
      // Filter for this pattern if we have real data
      setRecentQueries(queries.slice(0, 3))
    } catch (error) {
      console.error('Failed to fetch recent queries:', error)
      setRecentQueries([])
    } finally {
      setQueriesLoading(false)
    }
  }

  if (!isOpen || !pattern) return null

  // Mock usage history data - replace with real API call
  const usageHistory = [
    { time: '00:00', queries: 2, accuracy: 95 },
    { time: '04:00', queries: 5, accuracy: 92 },
    { time: '08:00', queries: 8, accuracy: 94 },
    { time: '12:00', queries: 12, accuracy: 91 },
    { time: '16:00', queries: 6, accuracy: 93 },
    { time: '20:00', queries: 3, accuracy: 96 },
  ]

  const handleSave = async () => {
    try {
      // Extract numeric ID
      const patternId = typeof pattern.id === 'string' && pattern.id.startsWith('pattern-')
        ? parseInt(pattern.id.replace('pattern-', ''))
        : pattern.id
      
      // Call real API
      await apiClient.updateRAGConfig(patternId, editedPattern)
      console.log('Pattern saved:', editedPattern)
      setIsEditing(false)
    } catch (error) {
      console.error('Save failed:', error)
      alert('Failed to save pattern: ' + (error instanceof Error ? error.message : 'Unknown error'))
    }
  }

  const handleTest = async () => {
    if (!testQuery.trim()) return
    
    setTestLoading(true)
    try {
      // Extract numeric ID from pattern.id (might be "pattern-1" format)
      const patternId = typeof pattern.id === 'string' && pattern.id.startsWith('pattern-')
        ? parseInt(pattern.id.replace('pattern-', ''))
        : pattern.id
      
      console.log('[Pattern Test] Testing pattern:', patternId, 'with query:', testQuery)
      
      // Call real API
      const results = await apiClient.testRAGConfig(patternId, testQuery)
      console.log('[Pattern Test] Results received:', results)
      setTestResults(results)
    } catch (error) {
      console.error('[Pattern Test] Error:', error)
      setTestResults({
        error: error instanceof Error ? error.message : 'Test failed',
        total_results: 0,
        retrieval_time: 'N/A'
      })
    } finally {
      setTestLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-card border border-border rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border sticky top-0 bg-card z-10">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Settings className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">{pattern.name}</h2>
              <p className="text-sm text-muted-foreground">
                {pattern.description}
              </p>
            </div>
            <Badge variant={pattern.status === 'active' ? 'default' : 'secondary'}>
              {pattern.status}
            </Badge>
          </div>
          <div className="flex items-center space-x-2">
            {isEditing ? (
              <>
                <Button variant="outline" size="sm" onClick={() => setIsEditing(false)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={handleSave}>
                  <Save className="w-4 h-4 mr-2" />
                  Save Changes
                </Button>
              </>
            ) : (
              <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                <Edit className="w-4 h-4 mr-2" />
                Edit
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="p-6">
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="configuration">Configuration</TabsTrigger>
              <TabsTrigger value="test">Test Pattern</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="glass-card card-glow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Usage</p>
                        <p className="text-2xl font-bold">{pattern.usage}</p>
                      </div>
                      <Target className="w-8 h-8 text-info" />
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card card-glow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Accuracy</p>
                        <p className="text-2xl font-bold">{pattern.accuracy}%</p>
                      </div>
                      <TrendingUp className="w-8 h-8 text-success" />
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card card-glow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Avg Sources</p>
                        <p className="text-2xl font-bold">{pattern.avgSources}</p>
                      </div>
                      <Database className="w-8 h-8 text-agent" />
                    </div>
                  </CardContent>
                </Card>

                <Card className="glass-card card-glow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">Category</p>
                        <p className="text-lg font-bold">{pattern.category}</p>
                      </div>
                      <Clock className="w-8 h-8 text-primary" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Usage Chart */}
              <Card className="glass-card card-glow">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="w-5 h-5" />
                    <span>Usage & Accuracy Trends</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={usageHistory}>
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
                          dataKey="queries" 
                          stroke="#60B5FF" 
                          strokeWidth={2}
                          name="Queries"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="accuracy" 
                          stroke="#72BF78" 
                          strokeWidth={2}
                          name="Accuracy (%)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Recent Queries Using This Pattern */}
              <Card className="glass-card card-glow">
                <CardHeader>
                  <CardTitle>Recent Queries</CardTitle>
                </CardHeader>
                <CardContent>
                  {queriesLoading ? (
                    <div className="text-center py-4 text-muted-foreground">
                      Loading queries...
                    </div>
                  ) : recentQueries.length === 0 ? (
                    <div className="text-center py-4 text-muted-foreground">
                      No recent queries yet
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {recentQueries.map((query, i) => (
                        <div key={query.id || i} className="p-3 rounded-lg border border-border/50 bg-secondary/20">
                          <div className="flex items-start justify-between mb-2">
                            <p className="text-sm font-medium">{query.query || query.query_text || 'Unknown query'}</p>
                            <Badge variant="outline" className="text-xs">
                              {((query.confidence || 0) * 100).toFixed(0)}% match
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Retrieved {query.sources || 0} chunks • {query.latency || 0}ms
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Configuration Tab */}
            <TabsContent value="configuration" className="space-y-6">
              <Card className="glass-card card-glow">
                <CardHeader>
                  <CardTitle>Pattern Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {isEditing && editedPattern ? (
                    <>
                      {/* Editable fields */}
                      <div>
                        <Label>Pattern Name</Label>
                        <Input
                          value={editedPattern.name || ''}
                          onChange={(e) => setEditedPattern({...editedPattern, name: e.target.value})}
                          className="mt-1"
                        />
                      </div>

                      <div>
                        <Label>Description</Label>
                        <Input
                          value={editedPattern.description || ''}
                          onChange={(e) => setEditedPattern({...editedPattern, description: e.target.value})}
                          className="mt-1"
                        />
                      </div>

                      <div>
                        <Label>Top K Results: {editedPattern.avgSources || 5}</Label>
                        <Slider
                          value={[editedPattern.avgSources || 5]}
                          onValueChange={(value) => setEditedPattern({...editedPattern, avgSources: value[0]})}
                          min={1}
                          max={20}
                          step={1}
                          className="mt-2"
                        />
                      </div>

                      <div>
                        <Label>Status</Label>
                        <Select 
                          value={editedPattern.status || 'active'}
                          onValueChange={(value) => setEditedPattern({...editedPattern, status: value})}
                        >
                          <SelectTrigger className="mt-1">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="active">Active</SelectItem>
                            <SelectItem value="inactive">Inactive</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </>
                  ) : pattern ? (
                    <>
                      {/* Read-only display */}
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Name</p>
                          <p className="font-medium">{pattern.name}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Category</p>
                          <p className="font-medium">{pattern.category}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Avg Sources</p>
                          <p className="font-medium">{pattern.avgSources}</p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Status</p>
                          <Badge variant={pattern.status === 'active' ? 'default' : 'secondary'}>
                            {pattern.status}
                          </Badge>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">Description</p>
                        <p className="text-sm">{pattern.description}</p>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-4 text-muted-foreground">
                      No pattern data available
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Test Tab */}
            <TabsContent value="test" className="space-y-6">
              <Card className="glass-card card-glow">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TestTube className="w-5 h-5" />
                    <span>Test Pattern</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="test_query">Test Query</Label>
                    <Input
                      id="test_query"
                      value={testQuery}
                      onChange={(e) => setTestQuery(e.target.value)}
                      placeholder="Enter a test query..."
                      className="mt-1"
                      onKeyDown={(e) => e.key === 'Enter' && handleTest()}
                    />
                  </div>

                  <Button
                    onClick={handleTest}
                    disabled={!testQuery.trim() || testLoading}
                    className="w-full"
                  >
                    {testLoading ? (
                      <>
                        <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                        Testing...
                      </>
                    ) : (
                      <>
                        <TestTube className="w-4 h-4 mr-2" />
                        Test Pattern
                      </>
                    )}
                  </Button>

                  {testResults && (
                    <div className="mt-4 p-4 bg-secondary/30 rounded-lg">
                      {testResults.error ? (
                        <div className="text-destructive text-sm">
                          <AlertCircle className="w-4 h-4 inline mr-2" />
                          {testResults.error}
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Results:</span>
                              <p className="font-medium">{testResults.total_results || 0}</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Time:</span>
                              <p className="font-medium">{testResults.retrieval_time || 'N/A'}</p>
                            </div>
                          </div>
                          
                          {testResults.results && testResults.results.length > 0 && (
                          <div>
                            <p className="text-xs text-muted-foreground mb-2">Sample Results:</p>
                            <div className="space-y-2">
                              {testResults.results.map((result: any, index: number) => (
                                <div key={index} className="text-xs p-2 bg-background/50 rounded border">
                                  <div className="flex justify-between items-start mb-1">
                                    <span className="font-medium text-primary">
                                      {result.source}
                                    </span>
                                    <Badge variant="outline" className="text-xs">
                                      {(result.score * 100).toFixed(1)}%
                                    </Badge>
                                  </div>
                                  <p className="text-muted-foreground line-clamp-2">
                                    {result.content}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </motion.div>
    </div>
  )
}

