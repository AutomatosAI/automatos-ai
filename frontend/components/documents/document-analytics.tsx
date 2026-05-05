
'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { 
  BarChart3, 
  TrendingUp, 
  FileText, 
  Clock, 
  HardDrive, 
  Users,
  Calendar,
  Zap,
  Search,
  Activity
} from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { useUsageAnalytics } from '@/hooks/use-usage-analytics-api'
import { Skeleton } from '@/components/ui/skeleton'

interface DocumentAnalyticsProps {
  documents: any[]
  documentStats: any
}

export function DocumentAnalytics({ documents, documentStats }: DocumentAnalyticsProps) {
  const [timeRange, setTimeRange] = useState('7d')
  
  // Fetch usage analytics from backend
  const { data: usageData, isLoading: usageLoading } = useUsageAnalytics(timeRange)

  // Calculate analytics from real data
  const analytics = {
    totalDocuments: documents.length,
    totalSize: documents.reduce((acc, doc) => acc + (doc.size || 0), 0),
    documentsByType: documents.reduce((acc, doc) => {
      const type = doc.file_type || 'unknown'
      acc[type] = (acc[type] || 0) + 1
      return acc
    }, {} as Record<string, number>),
    documentsByCategory: documents.reduce((acc, doc) => {
      const category = doc.category || 'uncategorized'
      acc[category] = (acc[category] || 0) + 1
      return acc
    }, {} as Record<string, number>),
    processingStats: {
      completed: documents.filter(d => d.status === 'completed').length,
      processing: documents.filter(d => d.status === 'processing').length,
      failed: documents.filter(d => d.status === 'failed').length,
      pending: documents.filter(d => d.status === 'pending').length
    }
  }

  const formatFileSize = (bytes: number) => {
    if (!bytes) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Document Analytics</h2>
          <p className="text-muted-foreground">Insights and statistics about your document library</p>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="24h">Last 24h</SelectItem>
            <SelectItem value="7d">Last 7 Days</SelectItem>
            <SelectItem value="30d">Last 30 Days</SelectItem>
            <SelectItem value="90d">Last 90 Days</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Documents</p>
                <p className="text-2xl font-bold">{analytics.totalDocuments}</p>
                <p className="text-xs text-success mt-1">↑ 12% this week</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <FileText className="w-5 h-5 text-info" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Storage</p>
                <p className="text-2xl font-bold">{formatFileSize(analytics.totalSize)}</p>
                <p className="text-xs text-info mt-1">68% of quota used</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <HardDrive className="w-5 h-5 text-success" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Processing Rate</p>
                <p className="text-2xl font-bold">94.2%</p>
                <p className="text-xs text-success mt-1">↑ 2.1% improvement</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Zap className="w-5 h-5 text-agent" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Process Time</p>
                <p className="text-2xl font-bold">2.4s</p>
                <p className="text-xs text-success mt-1">↓ 0.3s faster</p>
              </div>
              <div className="w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center">
                <Clock className="w-5 h-5 text-orange-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Document Types */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Documents by Type</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Object.entries(analytics.documentsByType).map(([type, count]) => {
                const percentage = (count / analytics.totalDocuments) * 100
                return (
                  <div key={type} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="capitalize">{type.replace('_', ' ')}</span>
                      <span>{count} ({percentage.toFixed(1)}%)</span>
                    </div>
                    <Progress value={percentage} className="h-2" />
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Processing Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    Completed
                  </span>
                  <span>{analytics.processingStats.completed}</span>
                </div>
                <Progress value={(analytics.processingStats.completed / analytics.totalDocuments) * 100} className="h-2" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    Processing
                  </span>
                  <span>{analytics.processingStats.processing}</span>
                </div>
                <Progress value={(analytics.processingStats.processing / analytics.totalDocuments) * 100} className="h-2" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    Pending
                  </span>
                  <span>{analytics.processingStats.pending}</span>
                </div>
                <Progress value={(analytics.processingStats.pending / analytics.totalDocuments) * 100} className="h-2" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    Failed
                  </span>
                  <span>{analytics.processingStats.failed}</span>
                </div>
                <Progress value={(analytics.processingStats.failed / analytics.totalDocuments) * 100} className="h-2" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>



      {/* Recent Activity */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {documents.slice(0, 5).map((doc, index) => (
              <div key={doc.id} className="flex items-center gap-4 p-3 rounded-lg border border-border">
                <FileText className="w-8 h-8 text-primary" />
                <div className="flex-1">
                  <div className="font-medium">{doc.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {doc.status === 'completed' ? 'Processed successfully' : `Status: ${doc.status}`}
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  {doc.updated_at ? new Date(doc.updated_at).toLocaleDateString() : 'Recently'}
                </div>
                <Badge variant={doc.status === 'completed' ? 'default' : 'secondary'}>
                  {doc.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Usage Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Popular Searches */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="w-5 h-5 text-info" />
              Popular Searches (Last {timeRange === '24h' ? '24 Hours' : timeRange === '7d' ? '7 Days' : '30 Days'})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map(i => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : usageData?.popular_search_terms && usageData.popular_search_terms.length > 0 ? (
              <div className="space-y-3">
                {usageData.popular_search_terms.map((search, index) => (
                  <div 
                    key={index}
                    className="flex items-center justify-between p-3 rounded-lg bg-secondary/20 border border-border/50"
                  >
                    <div className="flex items-center gap-3 flex-1">
                      <Badge variant="outline" className="text-info bg-info/10">
                        #{index + 1}
                      </Badge>
                      <span className="font-medium">{search.query}</span>
                    </div>
                    <Badge className="bg-agent/10 text-agent">
                      {search.count} searches
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No search data yet</p>
                <p className="text-xs mt-1">Start searching documents to see popular queries</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Popular Documents */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-success" />
              Most Accessed Documents
            </CardTitle>
          </CardHeader>
          <CardContent>
            {usageLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map(i => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : usageData?.popular_documents && usageData.popular_documents.length > 0 ? (
              <div className="space-y-3">
                {usageData.popular_documents.map((doc, index) => (
                  <div 
                    key={doc.document_id}
                    className="flex items-center justify-between p-3 rounded-lg bg-secondary/20 border border-border/50"
                  >
                    <div className="flex items-center gap-3 flex-1 min-w-0">
                      <Badge variant="outline" className="text-success bg-success/10 shrink-0">
                        #{index + 1}
                      </Badge>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium truncate">{doc.filename}</div>
                        <div className="text-xs text-muted-foreground">{doc.file_type}</div>
                      </div>
                    </div>
                    <Badge className="bg-orange-500/10 text-orange-400 shrink-0">
                      {doc.view_count} views
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No document access data yet</p>
                <p className="text-xs mt-1">Documents will appear here once accessed</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Activity Timeline */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-agent" />
            Document Activity Over Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          {usageLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : usageData?.time_series && usageData.time_series.length > 0 ? (
            <div className="space-y-4">
              {/* Simple bar chart visualization */}
              <div className="grid grid-cols-1 gap-2">
                {usageData.time_series.map((day, index) => {
                  const maxCount = Math.max(...usageData.time_series.map(d => d.count))
                  const percentage = (day.count / maxCount) * 100
                  
                  return (
                    <div key={index} className="flex items-center gap-3">
                      <div className="text-xs text-muted-foreground w-24">
                        {day.date ? new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : 'N/A'}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-secondary rounded-full h-6 overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-purple-500 to-blue-500 flex items-center justify-end pr-2"
                              style={{ width: `${Math.max(percentage, 5)}%` }}
                            >
                              {day.count > 0 && (
                                <span className="text-xs font-medium text-white">
                                  {day.count}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
              
              {/* Summary */}
              {usageData.summary && (
                <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-border">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-info">
                      {usageData.summary.total_documents}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Total Documents</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-success">
                      {usageData.summary.processed_documents}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">Processed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-agent">
                      {usageData.summary.documents_this_period}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">This Period</div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No activity data yet</p>
              <p className="text-xs mt-1">Activity timeline will appear once documents are uploaded</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

