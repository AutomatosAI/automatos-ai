'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  X, 
  Settings, 
  Database, 
  Zap, 
  TestTube,
  CheckCircle,
  AlertCircle,
  Plus,
  Save
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { apiClient, type RAGConfig } from '@/lib/api'

interface ConfigureRAGModalProps {
  isOpen: boolean
  onClose: () => void
  onConfigCreated?: (config: RAGConfig) => void
}

export function ConfigureRAGModal({ isOpen, onClose, onConfigCreated }: ConfigureRAGModalProps) {
  const [formData, setFormData] = useState({
    name: '',
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: 1000,
    chunk_overlap: 200,
    retrieval_strategy: 'similarity',
    top_k: 5,
    similarity_threshold: 0.7,
    configuration: '{}'
  })
  
  const [testQuery, setTestQuery] = useState('')
  const [testResults, setTestResults] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [createdConfig, setCreatedConfig] = useState<RAGConfig | null>(null)

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleSliderChange = (field: string, values: number[]) => {
    setFormData(prev => ({
      ...prev,
      [field]: values[0]
    }))
  }

  const handleSave = async () => {
    try {
      setSaving(true)
      setError(null)

      // Validate required fields
      if (!formData.name.trim()) {
        throw new Error('Configuration name is required')
      }

      // Parse configuration JSON
      let configurationObj = {}
      if (formData.configuration.trim()) {
        try {
          configurationObj = JSON.parse(formData.configuration)
        } catch (e) {
          throw new Error('Invalid JSON in configuration field')
        }
      }

      // Create RAG configuration
      const newConfig = await apiClient.createRAGConfig({
        name: formData.name,
        embedding_model: formData.embedding_model,
        chunk_size: formData.chunk_size,
        chunk_overlap: formData.chunk_overlap,
        retrieval_strategy: formData.retrieval_strategy,
        top_k: formData.top_k,
        similarity_threshold: formData.similarity_threshold,
        configuration: configurationObj
      })

      setCreatedConfig(newConfig)
      onConfigCreated?.(newConfig)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create RAG configuration')
    } finally {
      setSaving(false)
    }
  }

  const handleTest = async () => {
    if (!createdConfig || !testQuery.trim()) return

    try {
      setTestLoading(true)
      const results = await apiClient.testRAGConfig(createdConfig.id, testQuery)
      setTestResults(results)
    } catch (err) {
      setTestResults({ 
        error: err instanceof Error ? err.message : 'Test failed' 
      })
    } finally {
      setTestLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
      chunk_size: 1000,
      chunk_overlap: 200,
      retrieval_strategy: 'similarity',
      top_k: 5,
      similarity_threshold: 0.7,
      configuration: '{}'
    })
    setTestQuery('')
    setTestResults(null)
    setError(null)
    setCreatedConfig(null)
  }

  const handleClose = () => {
    resetForm()
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="bg-card border border-border rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Settings className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Configure RAG System</h2>
              <p className="text-sm text-muted-foreground">
                Create and test a new RAG configuration
              </p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={handleClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="p-6 space-y-6">
          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-red-500" />
              <span className="text-red-500 text-sm">{error}</span>
            </div>
          )}

          {/* Success Display */}
          {createdConfig && (
            <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              <span className="text-green-500 text-sm">
                RAG configuration "{createdConfig.name}" created successfully!
              </span>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Configuration Form */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Database className="w-5 h-5" />
                  <span>Configuration Settings</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Name */}
                <div>
                  <Label htmlFor="name">Configuration Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => handleInputChange('name', e.target.value)}
                    placeholder="e.g., Production RAG Config"
                    className="mt-1"
                    disabled={!!createdConfig}
                  />
                </div>

                {/* Embedding Model */}
                <div>
                  <Label htmlFor="embedding_model">Embedding Model</Label>
                  <Select 
                    value={formData.embedding_model} 
                    onValueChange={(value) => handleInputChange('embedding_model', value)}
                    disabled={!!createdConfig}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sentence-transformers/all-MiniLM-L6-v2">
                        all-MiniLM-L6-v2 (Fast, Good Quality)
                      </SelectItem>
                      <SelectItem value="sentence-transformers/all-mpnet-base-v2">
                        all-mpnet-base-v2 (High Quality)
                      </SelectItem>
                      <SelectItem value="text-embedding-ada-002">
                        OpenAI Ada-002 (Premium)
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Chunk Size */}
                <div>
                  <Label>Chunk Size: {formData.chunk_size}</Label>
                  <Slider
                    value={[formData.chunk_size]}
                    onValueChange={(values) => handleSliderChange('chunk_size', values)}
                    min={100}
                    max={4000}
                    step={100}
                    className="mt-2"
                    disabled={!!createdConfig}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>100</span>
                    <span>4000</span>
                  </div>
                </div>

                {/* Chunk Overlap */}
                <div>
                  <Label>Chunk Overlap: {formData.chunk_overlap}</Label>
                  <Slider
                    value={[formData.chunk_overlap]}
                    onValueChange={(values) => handleSliderChange('chunk_overlap', values)}
                    min={0}
                    max={1000}
                    step={50}
                    className="mt-2"
                    disabled={!!createdConfig}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>0</span>
                    <span>1000</span>
                  </div>
                </div>

                {/* Retrieval Strategy */}
                <div>
                  <Label htmlFor="retrieval_strategy">Retrieval Strategy</Label>
                  <Select 
                    value={formData.retrieval_strategy} 
                    onValueChange={(value) => handleInputChange('retrieval_strategy', value)}
                    disabled={!!createdConfig}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="similarity">Similarity Search</SelectItem>
                      <SelectItem value="hybrid">Hybrid Search</SelectItem>
                      <SelectItem value="semantic">Semantic Search</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Top K */}
                <div>
                  <Label>Top K Results: {formData.top_k}</Label>
                  <Slider
                    value={[formData.top_k]}
                    onValueChange={(values) => handleSliderChange('top_k', values)}
                    min={1}
                    max={20}
                    step={1}
                    className="mt-2"
                    disabled={!!createdConfig}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>1</span>
                    <span>20</span>
                  </div>
                </div>

                {/* Similarity Threshold */}
                <div>
                  <Label>Similarity Threshold: {formData.similarity_threshold.toFixed(2)}</Label>
                  <Slider
                    value={[formData.similarity_threshold]}
                    onValueChange={(values) => handleSliderChange('similarity_threshold', values)}
                    min={0.0}
                    max={1.0}
                    step={0.05}
                    className="mt-2"
                    disabled={!!createdConfig}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>0.0</span>
                    <span>1.0</span>
                  </div>
                </div>

                {/* Advanced Configuration */}
                <div>
                  <Label htmlFor="configuration">Advanced Configuration (JSON)</Label>
                  <Textarea
                    id="configuration"
                    value={formData.configuration}
                    onChange={(e) => handleInputChange('configuration', e.target.value)}
                    placeholder='{"reranking": true, "diversity_threshold": 0.8}'
                    className="mt-1 font-mono text-sm"
                    rows={3}
                    disabled={!!createdConfig}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Test Configuration */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TestTube className="w-5 h-5" />
                  <span>Test Configuration</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {!createdConfig ? (
                  <div className="text-center py-8">
                    <TestTube className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">
                      Save the configuration first to enable testing
                    </p>
                  </div>
                ) : (
                  <>
                    <div>
                      <Label htmlFor="test_query">Test Query</Label>
                      <Input
                        id="test_query"
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        placeholder="Enter a test query..."
                        className="mt-1"
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
                                <div className="space-y-2 max-h-40 overflow-y-auto">
                                  {testResults.results.slice(0, 3).map((result: any, index: number) => (
                                    <div key={index} className="text-xs p-2 bg-background/50 rounded border">
                                      <div className="flex justify-between items-start mb-1">
                                        <span className="font-medium text-primary">
                                          {result.source || 'Unknown Source'}
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
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-end space-x-3 pt-4 border-t border-border">
            <Button variant="outline" onClick={handleClose}>
              {createdConfig ? 'Close' : 'Cancel'}
            </Button>
            {!createdConfig && (
              <Button onClick={handleSave} disabled={saving} className="gradient-accent">
                {saving ? (
                  <>
                    <div className="w-4 h-4 mr-2 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Create Configuration
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  )
}
