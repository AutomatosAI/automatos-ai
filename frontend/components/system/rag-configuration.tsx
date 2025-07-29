'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Settings, 
  Database, 
  Search, 
  Zap, 
  Plus, 
  Edit, 
  Trash2, 
  TestTube,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { apiClient, RAGConfig } from '@/lib/api'

export function RAGConfiguration() {
  const [configs, setConfigs] = useState<RAGConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedConfig, setSelectedConfig] = useState<RAGConfig | null>(null)
  const [testQuery, setTestQuery] = useState('')
  const [testResults, setTestResults] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)

  useEffect(() => {
    fetchConfigs()
  }, [])

  const fetchConfigs = async () => {
    try {
      setLoading(true)
      const data = await apiClient.getRAGConfigs()
      setConfigs(data)
      if (data.length > 0 && !selectedConfig) {
        setSelectedConfig(data[0])
      }
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch RAG configurations')
      console.error('Error fetching RAG configs:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleTestConfig = async (configId: number) => {
    if (!testQuery.trim()) return

    try {
      setTestLoading(true)
      const results = await apiClient.testRAGConfig(configId, testQuery)
      setTestResults(results)
    } catch (err) {
      console.error('Error testing RAG config:', err)
      setTestResults({ error: err instanceof Error ? err.message : 'Test failed' })
    } finally {
      setTestLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">RAG Configuration</h2>
            <p className="text-muted-foreground">
              Configure Retrieval-Augmented Generation settings
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Array.from({ length: 2 }).map((_, index) => (
            <Card key={index} className="glass-card animate-pulse">
              <CardHeader>
                <div className="h-4 bg-secondary rounded mb-2"></div>
                <div className="h-3 bg-secondary rounded w-2/3"></div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="h-3 bg-secondary rounded"></div>
                  <div className="h-3 bg-secondary rounded w-3/4"></div>
                  <div className="h-3 bg-secondary rounded w-1/2"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">RAG Configuration</h2>
            <p className="text-muted-foreground">
              Configure Retrieval-Augmented Generation settings
            </p>
          </div>
        </div>
        <Card className="glass-card">
          <CardContent className="flex items-center justify-center p-8">
            <div className="text-center">
              <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-4" />
              <p className="text-red-500 mb-2">Error loading RAG configurations</p>
              <p className="text-sm text-muted-foreground">{error}</p>
              <Button onClick={fetchConfigs} className="mt-4">
                Try Again
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">RAG Configuration</h2>
          <p className="text-muted-foreground">
            Configure Retrieval-Augmented Generation settings
          </p>
        </div>
        <Button className="gradient-accent hover:opacity-90">
          <Plus className="w-4 h-4 mr-2" />
          New Configuration
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuration List */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database className="w-5 h-5" />
              <span>Available Configurations</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {configs.length === 0 ? (
              <div className="text-center py-8">
                <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">No configurations found</h3>
                <p className="text-muted-foreground">
                  Create your first RAG configuration to get started.
                </p>
              </div>
            ) : (
              configs.map((config, index) => (
                <motion.div
                  key={config.id}
                  className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                    selectedConfig?.id === config.id
                      ? 'border-primary/50 bg-primary/5'
                      : 'border-border/50 hover:border-primary/20'
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  onClick={() => setSelectedConfig(config)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold">{config.name}</h3>
                    <div className="flex items-center space-x-2">
                      {config.is_active && (
                        <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Active
                        </Badge>
                      )}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Model:</span>
                      <p className="font-medium truncate">
                        {config.embedding_model?.split('/').pop() || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Strategy:</span>
                      <p className="font-medium">{config.retrieval_strategy}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Chunk Size:</span>
                      <p className="font-medium">{config.chunk_size}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Top K:</span>
                      <p className="font-medium">{config.top_k}</p>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Created: {new Date(config.created_at).toLocaleDateString()}
                  </p>
                </motion.div>
              ))
            )}
          </CardContent>
        </Card>

        {/* Configuration Details */}
        {selectedConfig && (
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Settings className="w-5 h-5" />
                  <span>{selectedConfig.name}</span>
                </div>
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm">
                    <Edit className="w-4 h-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Configuration Settings */}
              <div className="space-y-4">
                <div>
                  <Label className="text-sm font-medium">Embedding Model</Label>
                  <p className="text-sm text-muted-foreground mt-1">
                    {selectedConfig.embedding_model || 'Not specified'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium">Chunk Size</Label>
                    <p className="text-lg font-bold">{selectedConfig.chunk_size}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Chunk Overlap</Label>
                    <p className="text-lg font-bold">{selectedConfig.chunk_overlap}</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium">Top K Results</Label>
                    <p className="text-lg font-bold">{selectedConfig.top_k}</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Similarity Threshold</Label>
                    <p className="text-lg font-bold">{selectedConfig.similarity_threshold}</p>
                  </div>
                </div>

                <div>
                  <Label className="text-sm font-medium">Retrieval Strategy</Label>
                  <p className="text-sm text-muted-foreground mt-1 capitalize">
                    {selectedConfig.retrieval_strategy}
                  </p>
                </div>

                {selectedConfig.configuration && (
                  <div>
                    <Label className="text-sm font-medium">Additional Configuration</Label>
                    <div className="mt-2 p-3 bg-secondary/30 rounded-lg">
                      <pre className="text-xs text-muted-foreground">
                        {JSON.stringify(selectedConfig.configuration, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>

              {/* Test Configuration */}
              <div className="border-t border-border/30 pt-6">
                <Label className="text-sm font-medium mb-3 block">Test Configuration</Label>
                <div className="space-y-3">
                  <Input
                    placeholder="Enter a test query..."
                    value={testQuery}
                    onChange={(e) => setTestQuery(e.target.value)}
                    className="bg-secondary/50"
                  />
                  <Button
                    onClick={() => handleTestConfig(selectedConfig.id)}
                    disabled={!testQuery.trim() || testLoading}
                    className="w-full"
                  >
                    {testLoading ? (
                      <>
                        <Clock className="w-4 h-4 mr-2 animate-spin" />
                        Testing...
                      </>
                    ) : (
                      <>
                        <TestTube className="w-4 h-4 mr-2" />
                        Test Configuration
                      </>
                    )}
                  </Button>

                  {testResults && (
                    <div className="mt-4 p-4 bg-secondary/30 rounded-lg">
                      {testResults.error ? (
                        <div className="text-red-500 text-sm">
                          <AlertCircle className="w-4 h-4 inline mr-2" />
                          {testResults.error}
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Results:</span>
                            <span className="font-medium">{testResults.total_results || 0}</span>
                          </div>
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Retrieval Time:</span>
                            <span className="font-medium">{testResults.retrieval_time || 'N/A'}</span>
                          </div>
                          {testResults.results && testResults.results.length > 0 && (
                            <div className="mt-3">
                              <p className="text-xs text-muted-foreground mb-2">Sample Results:</p>
                              <div className="space-y-1">
                                {testResults.results.slice(0, 3).map((result: any, index: number) => (
                                  <div key={index} className="text-xs p-2 bg-background/50 rounded">
                                    {typeof result === 'string' ? result : JSON.stringify(result)}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
