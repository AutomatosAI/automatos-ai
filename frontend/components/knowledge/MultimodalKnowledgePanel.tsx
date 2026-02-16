'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  FileText, 
  Table2, 
  Image as ImageIcon, 
  Calculator,
  Network,
  Database,
  Code2,
  Brain,
  Search,
  Filter,
  Eye,
  Download,
  Trash2,
  MoreVertical,
  ChevronRight,
  Sparkles,
  Loader2
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { apiClient } from '@/lib/api-client'
import KnowledgeGraphVisualizer from './KnowledgeGraphVisualizer'

// Knowledge type icons
const knowledgeTypeIcons = {
  document: FileText,
  codegraph: Code2,
  table: Table2,
  image: ImageIcon,
  formula: Calculator,
  diagram: Network,
  knowledge_graph: Brain,
  memory: Database
}

const knowledgeTypeColors = {
  document: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  codegraph: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  table: 'text-green-400 bg-green-500/10 border-green-500/20',
  image: 'text-orange-400 bg-orange-500/10 border-orange-500/20',
  formula: 'text-pink-400 bg-pink-500/10 border-pink-500/20',
  diagram: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20',
  knowledge_graph: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  memory: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20'
}

interface KnowledgeType {
  type_name: string
  display_name: string
  description: string
  icon: string
  count: number
  supports_search: boolean
  supports_relationships: boolean
}

interface KnowledgeItem {
  id: number
  kb_type: string  // 'formula', 'table', or 'image'
  title: string
  content_snippet?: string
  relevance_score?: number
  quality_score?: number
  created_at: string
}

interface KnowledgeStats {
  total_items: number
  by_type: Record<string, number>
  recent_uploads: number
  storage_used_mb: number
}

export function MultimodalKnowledgePanel() {
  const [activeType, setActiveType] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Data state
  const [knowledgeTypes, setKnowledgeTypes] = useState<KnowledgeType[]>([])
  const [knowledgeItems, setKnowledgeItems] = useState<KnowledgeItem[]>([])
  const [stats, setStats] = useState<KnowledgeStats | null>(null)
  const [selectedItem, setSelectedItem] = useState<KnowledgeItem | null>(null)

  // Load knowledge types on mount
  useEffect(() => {
    loadKnowledgeTypes()
    loadStats()
  }, [])

  // Load items when type filter changes
  useEffect(() => {
    loadKnowledgeItems()
  }, [activeType, searchQuery])

  const loadKnowledgeTypes = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await apiClient.request('/api/knowledge/types')
      console.log('Knowledge types response:', response)
      
      // Handle both array and object responses
      const typesArray = Array.isArray(response) ? response : (response?.types || [])
      setKnowledgeTypes(typesArray)
    } catch (e: any) {
      console.error('Error loading knowledge types:', e)
      setError(e?.message || 'Failed to load knowledge types')
      setKnowledgeTypes([]) // Set empty array on error
    } finally {
      setLoading(false)
    }
  }

  const loadKnowledgeItems = async () => {
    try {
      setLoading(true)
      setError(null)
      
      // Build search request body - query is REQUIRED by backend
      const searchRequest: any = {
        query: searchQuery && searchQuery.trim() ? searchQuery.trim() : '*', // Use wildcard for "all"
        limit: 50,
        use_semantic: true,
        use_fulltext: true,
        min_quality: 0.0
      }
      
      // kb_types is an ARRAY (plural)
      if (activeType && activeType !== 'all') {
        searchRequest.kb_types = [activeType]
      }

      console.log('Searching knowledge with request:', searchRequest)

      const response = await apiClient.request('/api/knowledge/search', {
        method: 'POST',
        body: JSON.stringify(searchRequest),
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      console.log('Knowledge search response:', response)
      
      // Response should have results array
      const items = response?.results || response || []
      setKnowledgeItems(Array.isArray(items) ? items : [])
    } catch (e: any) {
      console.error('Error loading knowledge items:', e)
      setError(e?.message || 'Failed to load knowledge items')
      setKnowledgeItems([]) // Set empty array on error
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const response = await apiClient.request('/api/knowledge/stats')
      console.log('Knowledge stats response:', response)
      
      // Transform array format to object format for easier access
      if (response && response.by_type && Array.isArray(response.by_type)) {
        const byTypeObj: Record<string, number> = {}
        response.by_type.forEach((item: any) => {
          byTypeObj[item.type] = item.count
        })
        
        setStats({
          total_items: response.totals?.total_items || 0,
          by_type: byTypeObj,
          recent_uploads: 0,
          storage_used_mb: 0
        })
      } else {
        setStats(response || null)
      }
    } catch (e: any) {
      console.error('Error loading stats:', e)
      setStats(null)
    }
  }

  const getTypeIcon = (typeName: string) => {
    const IconComponent = knowledgeTypeIcons[typeName as keyof typeof knowledgeTypeIcons] || FileText
    return IconComponent
  }

  const getTypeColor = (typeName: string) => {
    return knowledgeTypeColors[typeName as keyof typeof knowledgeTypeColors] || 'text-gray-400 bg-gray-500/10 border-gray-500/20'
  }

  const handleViewItem = async (item: KnowledgeItem) => {
    try {
      // Fetch full details from the backend
      const details = await apiClient.request(`/api/knowledge/items/${item.id}?type=${item.kb_type}`)
      console.log('Item details:', details)
      setSelectedItem(details)
    } catch (e: any) {
      console.error('Error fetching item details:', e)
      alert(`Error: ${e?.message || 'Failed to load details'}`)
    }
  }

  const handleDownloadItem = async (item: KnowledgeItem) => {
    // TODO: Implement download
    console.log('Download item:', item)
  }

  const handleDeleteItem = async (itemId: number) => {
    if (!confirm('Are you sure you want to delete this knowledge item?')) return
    
    try {
      await apiClient.request(`/api/knowledge/items/${itemId}`, { method: 'DELETE' })
      await loadKnowledgeItems()
      await loadStats()
    } catch (e: any) {
      console.error('Error deleting item:', e)
      alert(`Error: ${e?.message}`)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold mb-2">
          Multimodal <span className="gradient-text">Knowledge Base</span>
        </h2>
        <p className="text-muted-foreground">
          Browse and search tables, images, formulas, and structured knowledge extracted from your documents
        </p>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="glass-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <Database className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">{stats.total_items}</div>
                    <div className="text-sm text-muted-foreground truncate">Total Items</div>
                  </div>
                </div>
                <div className="shrink-0 text-right text-xs text-muted-foreground">
                  Across all types
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <Table2 className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">{stats.by_type?.table || 0}</div>
                    <div className="text-sm text-muted-foreground truncate">Tables</div>
                  </div>
                </div>
                <div className="shrink-0 text-right text-xs text-muted-foreground">
                  Structured data
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <ImageIcon className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">{stats.by_type?.image || 0}</div>
                    <div className="text-sm text-muted-foreground truncate">Images</div>
                  </div>
                </div>
                <div className="shrink-0 text-right text-xs text-muted-foreground">
                  Visual content
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardContent className="p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-10 h-10 rounded-2xl bg-black/20 border border-orange-500/10 flex items-center justify-center shrink-0">
                    <Calculator className="w-5 h-5 text-gray-300" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-2xl font-bold leading-none">{stats.by_type?.formula || 0}</div>
                    <div className="text-sm text-muted-foreground truncate">Formulas</div>
                  </div>
                </div>
                <div className="shrink-0 text-right text-xs text-muted-foreground">
                  Mathematical
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Filters and Search */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search knowledge items..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-secondary/50 border-secondary focus:border-primary/50"
          />
        </div>
        
        <Select value={activeType} onValueChange={setActiveType}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="All Types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            {knowledgeTypes.map(type => (
              <SelectItem key={type.type_name} value={type.type_name}>
                {type.display_name || (type.type_name ? type.type_name.charAt(0).toUpperCase() + type.type_name.slice(1) : 'Unknown')}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Knowledge Type Tabs */}
      <Tabs value={activeType} onValueChange={setActiveType} className="space-y-6">
        <TabsList className="w-full lg:w-auto justify-start gap-1 bg-secondary/50">
          <TabsTrigger value="all" className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4" />
            <span className="hidden sm:inline">All</span>
          </TabsTrigger>
          <TabsTrigger value="table" className="flex items-center space-x-2">
            <Table2 className="w-4 h-4" />
            <span className="hidden sm:inline">Tables</span>
          </TabsTrigger>
          <TabsTrigger value="image" className="flex items-center space-x-2">
            <ImageIcon className="w-4 h-4" />
            <span className="hidden sm:inline">Images</span>
          </TabsTrigger>
          <TabsTrigger value="formula" className="flex items-center space-x-2">
            <Calculator className="w-4 h-4" />
            <span className="hidden sm:inline">Formulas</span>
          </TabsTrigger>
          <TabsTrigger value="diagram" className="flex items-center space-x-2">
            <Network className="w-4 h-4" />
            <span className="hidden sm:inline">Diagrams</span>
          </TabsTrigger>
          <TabsTrigger value="graph" className="flex items-center space-x-2">
            <Brain className="w-4 h-4" />
            <span className="hidden sm:inline">Graph</span>
          </TabsTrigger>
        </TabsList>

        {/* Content Grid */}
        <div className="min-h-[400px]">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-4" />
                <p className="text-muted-foreground">Loading knowledge items...</p>
              </div>
            </div>
          ) : knowledgeItems.length === 0 && activeType !== 'graph' ? (
            <Card className="glass-card">
              <CardContent className="p-12 text-center">
                <Database className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No knowledge items found</h3>
                <p className="text-muted-foreground mb-4">
                  {searchQuery 
                    ? 'Try adjusting your search query'
                    : activeType && activeType !== 'all'
                    ? `Upload documents to extract ${activeType}s`
                    : 'Upload documents to extract multimodal knowledge'
                  }
                </p>
              </CardContent>
            </Card>
          ) : activeType === 'graph' ? (
            <div className="h-[800px]">
              <KnowledgeGraphVisualizer />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {knowledgeItems.map((item, index) => {
                const TypeIcon = getTypeIcon(item.knowledge_type)
                const typeColor = getTypeColor(item.knowledge_type)
                
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <Card className="glass-card hover:border-primary/50 transition-all cursor-pointer">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center space-x-3">
                            <div className={`w-10 h-10 rounded-lg bg-secondary/50 flex items-center justify-center ${typeColor}`}>
                              <TypeIcon className="w-5 h-5" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <CardTitle className="text-base line-clamp-1">{item.title}</CardTitle>
                              <Badge variant="outline" className={`mt-1 ${typeColor}`}>
                                {item.knowledge_type}
                              </Badge>
                            </div>
                          </div>

                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <MoreVertical className="w-4 h-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => handleViewItem(item)}>
                                <Eye className="w-4 h-4 mr-2" />
                                View Details
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={() => handleDownloadItem(item)}>
                                <Download className="w-4 h-4 mr-2" />
                                Download
                              </DropdownMenuItem>
                              <DropdownMenuItem 
                                className="text-red-400"
                                onClick={() => handleDeleteItem(item.id)}
                              >
                                <Trash2 className="w-4 h-4 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </CardHeader>

                      <CardContent onClick={() => handleViewItem(item)}>
                        {/* Description */}
                        {item.description && (
                          <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                            {item.description}
                          </p>
                        )}

                        {/* Content Preview based on type */}
                        {item.knowledge_type === 'table' && item.metadata?.preview && (
                          <div className="bg-secondary/30 rounded p-2 mb-3 overflow-x-auto">
                            {(() => {
                              const preview = item.metadata.preview
                              const lines = preview.split('\n').filter(line => line.trim())
                              if (lines.length < 2) {
                                return <div className="text-xs font-mono whitespace-pre">{preview.substring(0, 100)}...</div>
                              }
                              
                              const headers = lines[0].split('|').map(h => h.trim()).filter(h => h)
                              const rows = lines.slice(1, 3).map(line => 
                                line.split('|').map(cell => cell.trim()).filter(cell => cell)
                              )
                              
                              return (
                                <div className="overflow-x-auto">
                                  <table className="w-full text-xs border-collapse min-w-full">
                                    <thead>
                                      <tr className="border-b border-gray-600">
                                        {headers.map((header, i) => (
                                          <th key={i} className="text-left p-1 font-medium bg-gray-800/50 text-gray-200">{header}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {rows.map((row, i) => (
                                        <tr key={i} className="border-b border-gray-700">
                                          {row.map((cell, j) => (
                                            <td key={j} className="p-1 text-gray-300">{cell}</td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              )
                            })()}
                          </div>
                        )}

                        {item.knowledge_type === 'formula' && item.metadata?.latex && (
                          <div className="bg-secondary/30 rounded p-3 mb-3">
                            <code className="text-xs text-pink-400">
                              {item.metadata.latex.substring(0, 60)}...
                            </code>
                          </div>
                        )}

                        {item.knowledge_type === 'image' && item.metadata?.ai_description && (
                          <div className="text-xs text-muted-foreground italic mb-3">
                            "{item.metadata.ai_description.substring(0, 80)}..."
                          </div>
                        )}

                        {/* Metadata */}
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>
                            {new Date(item.created_at).toLocaleDateString()}
                          </span>
                          {item.source_document_id && (
                            <span className="flex items-center">
                              <FileText className="w-3 h-3 mr-1" />
                              Doc #{item.source_document_id}
                            </span>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )
              })}
            </div>
          )}
        </div>
      </Tabs>

      {/* Details Modal */}
      {selectedItem && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedItem(null)}>
          <Card className="glass-card max-w-4xl max-h-[80vh] overflow-auto m-4" onClick={(e) => e.stopPropagation()}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  {selectedItem.kb_type === 'formula' && <Calculator className="w-5 h-5 text-pink-400" />}
                  {selectedItem.kb_type === 'table' && <Table2 className="w-5 h-5 text-green-400" />}
                  {selectedItem.kb_type === 'image' && <ImageIcon className="w-5 h-5 text-orange-400" />}
                  <span>{selectedItem.kb_type === 'formula' ? 'Formula' : selectedItem.kb_type === 'table' ? 'Table' : 'Image'} Details</span>
                </CardTitle>
                <Button variant="ghost" size="sm" onClick={() => setSelectedItem(null)}>
                  ✕
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Formula Details */}
              {selectedItem.kb_type === 'formula' && (
                <>
                  {selectedItem.latex && (
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">LaTeX</h3>
                      <div className="bg-secondary/30 p-4 rounded-lg font-mono text-sm">
                        {selectedItem.latex}
                      </div>
                    </div>
                  )}
                  {selectedItem.ascii_math && (
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">ASCII Math</h3>
                      <div className="bg-secondary/30 p-4 rounded-lg text-sm">
                        {selectedItem.ascii_math}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Table Details */}
              {selectedItem.kb_type === 'table' && (
                <>
                  {selectedItem.markdown && (
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">
                        Table ({selectedItem.row_count} rows × {selectedItem.column_count} columns)
                      </h3>
                      <div className="bg-secondary/30 p-4 rounded-lg overflow-auto max-h-96">
                        {(() => {
                          const markdown = selectedItem.markdown
                          const lines = markdown.split('\n').filter(line => line.trim())
                          
                          // Parse markdown table format
                          if (lines.length >= 2 && lines[0].includes('|')) {
                            // Extract headers (first line)
                            const headerLine = lines[0]
                            const headers = headerLine.split('|')
                              .map(h => h.trim())
                              .filter(h => h && h !== '')
                            
                            // Skip separator line (second line with |---|---|)
                            // Extract data rows (lines after separator)
                            const dataLines = lines.slice(2)
                            const rows = dataLines
                              .filter(line => line.includes('|'))
                              .map(line => 
                                line.split('|')
                                  .map(cell => cell.trim())
                                  .filter(cell => cell !== '')
                              )
                            
                            return (
                              <div className="overflow-y-auto max-h-80">
                                <table className="w-full text-sm border-collapse">
                                  <thead>
                                    <tr className="border-b border-gray-600">
                                      {headers.map((header, i) => (
                                        <th key={i} className="text-left p-3 font-medium bg-gray-800/50 text-gray-200">{header}</th>
                                      ))}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {rows.map((row, i) => (
                                      <tr key={i} className="border-b border-gray-700 hover:bg-gray-800/30">
                                        {row.map((cell, j) => (
                                          <td key={j} className="p-3 text-gray-300">{cell}</td>
                                        ))}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )
                          }
                          
                          // Fallback: treat as single column data
                          return (
                            <div className="overflow-y-auto max-h-80">
                              <table className="w-full text-sm border-collapse">
                                <thead>
                                  <tr className="border-b border-gray-600">
                                    <th className="text-left p-3 font-medium bg-gray-800/50 text-gray-200">Data</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {lines.map((line, i) => (
                                    <tr key={i} className="border-b border-gray-700 hover:bg-gray-800/30">
                                      <td className="p-3 text-gray-300">{line}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )
                        })()}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* Image Details */}
              {selectedItem.kb_type === 'image' && (
                <>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    {selectedItem.width && <div><span className="text-muted-foreground">Width:</span> {selectedItem.width}px</div>}
                    {selectedItem.height && <div><span className="text-muted-foreground">Height:</span> {selectedItem.height}px</div>}
                    {selectedItem.format && <div><span className="text-muted-foreground">Format:</span> {selectedItem.format}</div>}
                  </div>
                  {selectedItem.description && (
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Description</h3>
                      <p className="text-sm">{selectedItem.description}</p>
                    </div>
                  )}
                  {selectedItem.detected_text && (
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Detected Text</h3>
                      <div className="bg-secondary/30 p-4 rounded-lg text-sm">
                        {selectedItem.detected_text}
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

