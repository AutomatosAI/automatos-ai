"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  BarChart3,
  TrendingUp,
  Users,
  DollarSign,
  Package,
  Activity,
  Clock,
  AlertTriangle,
  Search,
  Play,
  Star,
  Copy,
  Edit,
  Filter,
  FileText,
  PieChart,
  LineChart,
  Table as TableIcon
} from 'lucide-react'
import { toast } from 'sonner'

interface QueryTemplate {
  id: number
  name: string
  natural_language: string
  category: string
  dialect: string
  sql_template: string
  parameters: any[]
  visualization_type: string
  usage_count: number
  is_featured: boolean
  tags: string[]
}

interface QueryTemplatesGridProps {
  templates: any[]
  selectedSource: any
}

const categoryIcons: Record<string, any> = {
  analytics: BarChart3,
  reporting: FileText,
  monitoring: Activity,
  troubleshooting: AlertTriangle,
  exploration: Search,
  financial: DollarSign,
  customers: Users,
  inventory: Package
}

const visualizationIcons: Record<string, any> = {
  bar: BarChart3,
  line: LineChart,
  pie: PieChart,
  table: TableIcon,
  metric: TrendingUp
}

export function QueryTemplatesGrid({ templates: initialTemplates, selectedSource }: QueryTemplatesGridProps) {
  const [templates, setTemplates] = useState<QueryTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<QueryTemplate[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState<QueryTemplate | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionParams, setExecutionParams] = useState<Record<string, any>>({})

  useEffect(() => {
    loadTemplates()
  }, [selectedSource])

  useEffect(() => {
    filterTemplates()
  }, [templates, selectedCategory, searchQuery])

  const loadTemplates = async () => {
    try {
      const dialect = selectedSource?.dialect || 'postgresql'
      const response = await fetch(`/api/knowledge/sources/database/templates/list?dialect=${dialect}`)
      if (response.ok) {
        const data = await response.json()
        // Mock enriched data for demonstration
        const enrichedTemplates = data.map((t: any, idx: number) => ({
          ...t,
          sql_template: getMockSQLTemplate(t.name),
          parameters: getMockParameters(t.name),
          usage_count: Math.floor(Math.random() * 1000),
          is_featured: idx < 3,
          tags: getMockTags(t.category)
        }))
        setTemplates(enrichedTemplates)
      }
    } catch (error) {
      console.error('Failed to load templates:', error)
    }
  }

  const filterTemplates = () => {
    let filtered = [...templates]
    
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(t => t.category === selectedCategory)
    }
    
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(t => 
        t.name.toLowerCase().includes(query) ||
        t.natural_language.toLowerCase().includes(query) ||
        t.tags.some(tag => tag.toLowerCase().includes(query))
      )
    }
    
    setFilteredTemplates(filtered)
  }

  const handleExecuteTemplate = async () => {
    if (!selectedTemplate || !selectedSource) {
      toast.error('Please select a database source')
      return
    }

    setIsExecuting(true)
    try {
      const response = await fetch(`/api/knowledge/sources/database/templates/${selectedTemplate.id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: selectedSource.id,
          parameters: executionParams
        })
      })

      if (response.ok) {
        const data = await response.json()
        toast.success('Template executed successfully!')
        // Handle results display
      } else {
        toast.error('Failed to execute template')
      }
    } catch (error) {
      toast.error('Error executing template')
    } finally {
      setIsExecuting(false)
    }
  }

  const categories = ['all', 'analytics', 'reporting', 'monitoring', 'troubleshooting', 'exploration']

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle>Query Templates Library</CardTitle>
          <CardDescription>
            Pre-built, optimized queries for common business questions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Search and Filter Bar */}
          <div className="flex gap-2 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>

          {/* Category Tabs */}
          <Tabs value={selectedCategory} onValueChange={setSelectedCategory}>
            <TabsList className="w-full justify-start gap-1">
              {categories.map(cat => (
                <TabsTrigger key={cat} value={cat} className="capitalize">
                  {cat === 'all' ? 'All Templates' : cat}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {/* Templates Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredTemplates.map((template) => {
          const Icon = categoryIcons[template.category] || BarChart3
          const VizIcon = visualizationIcons[template.visualization_type] || TableIcon
          
          return (
            <Card 
              key={template.id}
              className="hover:shadow-lg transition-all cursor-pointer relative"
              onClick={() => setSelectedTemplate(template)}
            >
              {template.is_featured && (
                <div className="absolute top-2 right-2">
                  <Star className="h-4 w-4 text-[hsl(var(--warning))] fill-[hsl(var(--warning))]" />
                </div>
              )}
              
              <CardHeader>
                <div className="flex items-start justify-between">
                  <Icon className={`h-8 w-8 mb-2 ${
                    template.category === 'analytics' ? 'text-[hsl(var(--info))]' :
                    template.category === 'reporting' ? 'text-[hsl(var(--success))]' :
                    template.category === 'monitoring' ? 'text-primary' :
                    template.category === 'troubleshooting' ? 'text-[hsl(var(--destructive))]' :
                    'text-[hsl(var(--agent))]'
                  }`} />
                  <VizIcon className="h-5 w-5 text-muted-foreground" />
                </div>
                <CardTitle className="text-lg">{template.name}</CardTitle>
                <CardDescription>{template.natural_language}</CardDescription>
              </CardHeader>
              
              <CardContent>
                <div className="flex flex-wrap gap-2 mb-3">
                  {template.tags.slice(0, 3).map((tag, idx) => (
                    <Badge key={idx} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                  {template.parameters.length > 0 && (
                    <Badge variant="outline" className="text-xs">
                      {template.parameters.length} params
                    </Badge>
                  )}
                </div>
                
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Used {template.usage_count} times</span>
                  <div className="flex gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        navigator.clipboard.writeText(template.sql_template)
                        toast.success('SQL copied!')
                      }}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedTemplate(template)
                      }}
                    >
                      <Play className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Empty State */}
      {filteredTemplates.length === 0 && (
        <Card className="text-center py-12">
          <CardContent>
            <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium mb-2">No templates found</h3>
            <p className="text-muted-foreground">
              Try adjusting your search or filters
            </p>
          </CardContent>
        </Card>
      )}

      {/* Template Execution Dialog */}
      <Dialog open={!!selectedTemplate} onOpenChange={() => setSelectedTemplate(null)}>
        {selectedTemplate && (
          <DialogContent className="max-w-3xl">
            <DialogHeader>
              <DialogTitle>{selectedTemplate.name}</DialogTitle>
              <DialogDescription>
                {selectedTemplate.natural_language}
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4">
              {/* SQL Preview */}
              <div>
                <h4 className="font-medium mb-2">SQL Template</h4>
                <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm">{selectedTemplate.sql_template}</code>
                </pre>
              </div>
              
              {/* Parameters */}
              {selectedTemplate.parameters.length > 0 && (
                <div>
                  <h4 className="font-medium mb-2">Parameters</h4>
                  <div className="space-y-2">
                    {selectedTemplate.parameters.map((param: any) => (
                      <div key={param.name}>
                        <label className="text-sm font-medium">
                          {param.label || param.name}
                        </label>
                        <Input
                          type={param.type || 'text'}
                          placeholder={param.placeholder}
                          value={executionParams[param.name] || ''}
                          onChange={(e) => setExecutionParams({
                            ...executionParams,
                            [param.name]: e.target.value
                          })}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Actions */}
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setSelectedTemplate(null)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleExecuteTemplate}
                  disabled={isExecuting || !selectedSource}
                >
                  {isExecuting ? 'Executing...' : 'Execute Query'}
                </Button>
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </div>
  )
}

// Helper functions for mock data
function getMockSQLTemplate(name: string): string {
  const templates: Record<string, string> = {
    "Top N by Revenue": `
SELECT 
  customer_name,
  SUM(order_total) as total_revenue,
  COUNT(*) as order_count
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE order_date >= :start_date
GROUP BY customer_name
ORDER BY total_revenue DESC
LIMIT :limit`,
    "Revenue Trend": `
SELECT 
  DATE_TRUNC('month', order_date) as month,
  SUM(order_total) as revenue
FROM orders
WHERE order_date >= :start_date
GROUP BY month
ORDER BY month`,
    "Daily Summary": `
SELECT 
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(*) as total_orders,
  SUM(order_total) as total_revenue,
  AVG(order_total) as avg_order_value
FROM orders
WHERE DATE(order_date) = CURRENT_DATE`
  }
  return templates[name] || "SELECT * FROM table LIMIT 10"
}

function getMockParameters(name: string): any[] {
  const params: Record<string, any[]> = {
    "Top N by Revenue": [
      { name: "start_date", type: "date", label: "Start Date", placeholder: "YYYY-MM-DD" },
      { name: "limit", type: "number", label: "Number of Results", placeholder: "10" }
    ],
    "Revenue Trend": [
      { name: "start_date", type: "date", label: "Start Date", placeholder: "YYYY-MM-DD" }
    ],
    "Daily Summary": []
  }
  return params[name] || []
}

function getMockTags(category: string): string[] {
  const tags: Record<string, string[]> = {
    analytics: ["business intelligence", "KPIs", "metrics"],
    reporting: ["scheduled", "executive", "summary"],
    monitoring: ["real-time", "alerts", "performance"],
    troubleshooting: ["diagnostics", "debugging", "issues"],
    exploration: ["ad-hoc", "discovery", "analysis"]
  }
  return tags[category] || ["general"]
}
