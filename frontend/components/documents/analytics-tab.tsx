
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  Search, 
  FileText, 
  Database,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle,
  Info,
  Eye,
  Download
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from '@/lib/api'

interface DocumentAnalytics {
  overview: {
    total_documents: number;
    total_chunks: number;
    total_size_mb: number;
    avg_chunks_per_doc: number;
    embedding_count: number;
    unique_sources: number;
  };
  document_types: Array<{
    type: string;
    count: number;
    percentage: number;
  }>;
  status_breakdown: Array<{
    status: string;
    count: number;
    percentage: number;
  }>;
  upload_trends: Array<{
    date: string;
    count: number;
  }>;
  size_by_type: Array<{
    type: string;
    total_size_mb: number;
    avg_size_mb: number;
    count: number;
  }>;
  performance_metrics: {
    processing_success_rate: number;
    avg_processing_time: string;
    total_processing_time: string;
    documents_per_hour: number;
  };
  insights: Array<{
    type: 'info' | 'warning' | 'success';
    title: string;
    message: string;
  }>;
  last_updated: string;
}

interface SearchPatterns {
  popular_search_terms: Array<{
    term: string;
    frequency: number;
    trend: 'up' | 'down' | 'stable';
  }>;
  most_accessed_documents: Array<{
    document_id: number;
    filename: string;
    access_count: number;
    last_accessed: string;
    avg_session_time: string;
  }>;
  search_performance: {
    avg_response_time: string;
    total_searches: number;
    successful_searches: number;
    success_rate: number;
    avg_results_per_search: number;
  };
  usage_patterns: {
    peak_hours: string[];
    most_active_day: string;
    avg_searches_per_user: number;
    common_file_types_searched: string[];
  };
  last_updated: string;
}

export function AnalyticsTab() {
  const [analyticsData, setAnalyticsData] = useState<DocumentAnalytics | null>(null)
  const [searchPatterns, setSearchPatterns] = useState<SearchPatterns | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    loadAnalyticsData()
  }, [])

  const loadAnalyticsData = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const [analytics, patterns] = await Promise.all([
        apiClient.request<DocumentAnalytics>('/api/documents/analytics/overview'),
        apiClient.request<SearchPatterns>('/api/documents/analytics/search-patterns')
      ])
      
      setAnalyticsData(analytics)
      setSearchPatterns(patterns)
    } catch (err) {
      console.error('Error loading analytics data:', err)
      setError(err instanceof Error ? err.message : 'Failed to load analytics data')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-4" />
          <p className="text-red-400 mb-4">Error: {error}</p>
          <Button onClick={loadAnalyticsData} variant="outline">
            Try Again
          </Button>
        </div>
      </div>
    )
  }

  if (!analyticsData || !searchPatterns) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No analytics data available</p>
      </div>
    )
  }

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'warning': return AlertCircle
      case 'success': return CheckCircle
      default: return Info
    }
  }

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'warning': return 'text-orange-400'
      case 'success': return 'text-green-400'
      default: return 'text-blue-400'
    }
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return '↗️'
      case 'down': return '↘️'
      default: return '→'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold mb-2">Document Analytics</h2>
          <p className="text-muted-foreground">
            Comprehensive insights into document usage and performance
          </p>
        </div>
        
        <Button onClick={loadAnalyticsData} variant="outline" size="sm">
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid bg-secondary/50">
          <TabsTrigger value="overview" className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            <span className="hidden sm:inline">Overview</span>
          </TabsTrigger>
          <TabsTrigger value="usage" className="flex items-center space-x-2">
            <Search className="w-4 h-4" />
            <span className="hidden sm:inline">Usage</span>
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            <span className="hidden sm:inline">Performance</span>
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4" />
            <span className="hidden sm:inline">Insights</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Overview Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <motion.div
              className="glass-card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-blue-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{analyticsData.overview.total_documents}</h3>
                <p className="text-muted-foreground text-sm">Total Documents</p>
                <p className="text-xs text-blue-400">
                  {analyticsData.overview.total_size_mb}MB total size
                </p>
              </div>
            </motion.div>

            <motion.div
              className="glass-card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <Database className="w-5 h-5 text-green-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{analyticsData.overview.total_chunks}</h3>
                <p className="text-muted-foreground text-sm">Vector Chunks</p>
                <p className="text-xs text-green-400">
                  {analyticsData.overview.avg_chunks_per_doc} avg per doc
                </p>
              </div>
            </motion.div>

            <motion.div
              className="glass-card p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                  <Zap className="w-5 h-5 text-purple-400" />
                </div>
              </div>
              <div className="space-y-1">
                <h3 className="text-2xl font-bold">{analyticsData.overview.embedding_count}</h3>
                <p className="text-muted-foreground text-sm">Embeddings</p>
                <p className="text-xs text-purple-400">
                  {analyticsData.overview.unique_sources} unique sources
                </p>
              </div>
            </motion.div>
          </div>

          {/* Document Types Breakdown */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <h3 className="text-lg font-semibold mb-4">Document Types</h3>
            <div className="space-y-4">
              {analyticsData.document_types.map((type, index) => (
                <div key={type.type} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 rounded-full bg-primary" style={{
                      backgroundColor: `hsl(${index * 60}, 70%, 50%)`
                    }} />
                    <span className="font-medium">{type.type.toUpperCase()}</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="w-32">
                      <Progress value={type.percentage} className="h-2" />
                    </div>
                    <div className="text-right min-w-[60px]">
                      <p className="text-sm font-medium">{type.count}</p>
                      <p className="text-xs text-muted-foreground">{type.percentage}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Status Breakdown */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <h3 className="text-lg font-semibold mb-4">Processing Status</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {analyticsData.status_breakdown.map((status, index) => (
                <div key={status.status} className="p-4 bg-secondary/30 rounded-lg text-center">
                  <h4 className="text-2xl font-bold">{status.count}</h4>
                  <p className="text-sm text-muted-foreground capitalize">{status.status}</p>
                  <p className="text-xs text-primary">{status.percentage}%</p>
                </div>
              ))}
            </div>
          </motion.div>
        </TabsContent>

        <TabsContent value="usage" className="space-y-6">
          {/* Popular Search Terms */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-4">Popular Search Terms</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {searchPatterns.popular_search_terms.map((term, index) => (
                <div key={term.term} className="flex items-center justify-between p-3 bg-secondary/20 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <span className="text-sm font-medium">#{index + 1}</span>
                    <span className="font-medium">{term.term}</span>
                    <span className="text-sm">{getTrendIcon(term.trend)}</span>
                  </div>
                  <Badge variant="outline">{term.frequency}</Badge>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Most Accessed Documents */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h3 className="text-lg font-semibold mb-4">Most Accessed Documents</h3>
            <div className="space-y-3">
              {searchPatterns.most_accessed_documents.map((doc, index) => (
                <div key={doc.document_id} className="flex items-center justify-between p-3 bg-secondary/20 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Eye className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="font-medium text-sm">{doc.filename}</p>
                      <p className="text-xs text-muted-foreground">
                        Last accessed: {new Date(doc.last_accessed).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{doc.access_count} views</p>
                    <p className="text-xs text-muted-foreground">{doc.avg_session_time} avg</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Usage Patterns */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h3 className="text-lg font-semibold mb-4">Usage Patterns</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">Peak Hours</h4>
                <div className="flex flex-wrap gap-2">
                  {searchPatterns.usage_patterns.peak_hours.map((hour) => (
                    <Badge key={hour} variant="outline">{hour}</Badge>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Most Active Day</h4>
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
                  {searchPatterns.usage_patterns.most_active_day}
                </Badge>
              </div>
              <div>
                <h4 className="font-medium mb-2">Avg Searches per User</h4>
                <p className="text-2xl font-bold">{searchPatterns.usage_patterns.avg_searches_per_user}</p>
              </div>
              <div>
                <h4 className="font-medium mb-2">Common File Types</h4>
                <div className="flex flex-wrap gap-2">
                  {searchPatterns.usage_patterns.common_file_types_searched.map((type) => (
                    <Badge key={type} variant="outline">{type}</Badge>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          {/* Search Performance */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-4">Search Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{searchPatterns.search_performance.avg_response_time}</h4>
                <p className="text-sm text-muted-foreground">Avg Response Time</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{searchPatterns.search_performance.total_searches}</h4>
                <p className="text-sm text-muted-foreground">Total Searches</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{searchPatterns.search_performance.success_rate}%</h4>
                <p className="text-sm text-muted-foreground">Success Rate</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{searchPatterns.search_performance.avg_results_per_search}</h4>
                <p className="text-sm text-muted-foreground">Avg Results</p>
              </div>
            </div>
          </motion.div>

          {/* Processing Performance */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h3 className="text-lg font-semibold mb-4">Processing Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{analyticsData.performance_metrics.processing_success_rate}%</h4>
                <p className="text-sm text-muted-foreground">Success Rate</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{analyticsData.performance_metrics.avg_processing_time}</h4>
                <p className="text-sm text-muted-foreground">Avg Processing Time</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{analyticsData.performance_metrics.documents_per_hour}</h4>
                <p className="text-sm text-muted-foreground">Docs per Hour</p>
              </div>
              <div className="text-center p-4 bg-secondary/30 rounded-lg">
                <h4 className="text-2xl font-bold">{analyticsData.performance_metrics.total_processing_time}</h4>
                <p className="text-sm text-muted-foreground">Total Time</p>
              </div>
            </div>
          </motion.div>
        </TabsContent>

        <TabsContent value="insights" className="space-y-6">
          {/* AI Insights */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-4">AI-Generated Insights</h3>
            <div className="space-y-4">
              {analyticsData.insights.map((insight, index) => {
                const Icon = getInsightIcon(insight.type)
                const color = getInsightColor(insight.type)
                
                return (
                  <div key={index} className="flex items-start space-x-3 p-4 bg-secondary/20 rounded-lg">
                    <Icon className={`w-5 h-5 mt-0.5 ${color}`} />
                    <div>
                      <h4 className="font-medium">{insight.title}</h4>
                      <p className="text-sm text-muted-foreground">{insight.message}</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>

          {/* Upload Trends */}
          <motion.div
            className="glass-card p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h3 className="text-lg font-semibold mb-4">Upload Trends (Last 30 Days)</h3>
            <div className="space-y-2">
              {analyticsData.upload_trends.slice(-7).map((trend, index) => (
                <div key={trend.date} className="flex items-center justify-between p-2 bg-secondary/10 rounded">
                  <span className="text-sm">{new Date(trend.date).toLocaleDateString()}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-secondary/30 rounded-full h-2">
                      <div 
                        className="bg-primary h-2 rounded-full" 
                        style={{ width: `${Math.min(100, (trend.count / 10) * 100)}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">{trend.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
