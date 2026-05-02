'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { EnhancedPagination } from '@/components/ui/pagination'
import {
  Database, Brain, Server, Mail, Code, Key, Cloud, CreditCard,
  Briefcase, Activity, Search, ExternalLink
} from 'lucide-react'
import { ToolLogo } from '@/components/ui/tool-logo'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  listCredentialTypes,
  getCredentialCategories,
  type CredentialType
} from '@/lib/api/credentials'

const CATEGORY_ICONS: Record<string, any> = {
  database: Database,
  ai: Brain,
  infrastructure: Server,
  communication: Mail,
  code: Code,
  api: Key,
  cloud: Cloud,
  payment: CreditCard,
  crm: Briefcase,
  monitoring: Activity,
  storage: Server
}

export function CredentialTypesTab() {
  const [credentialTypes, setCredentialTypes] = useState<CredentialType[]>([])
  const [categories, setCategories] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<CredentialType | null>(null)

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(20)
  const [totalCount, setTotalCount] = useState(0)

  useEffect(() => {
    loadData()
  }, [categoryFilter])

  const loadData = async () => {
    try {
      setLoading(true)
      const [types, cats] = await Promise.all([
        listCredentialTypes({
          category: categoryFilter === 'all' ? undefined : categoryFilter,
          active_only: true
        }),
        getCredentialCategories()
      ])

      setCredentialTypes(types)
      setCategories(cats)
    } catch (error) {
      console.error('Failed to load credential types:', error)
    } finally {
      setLoading(false)
    }
  }

  const filteredTypes = credentialTypes.filter(type =>
    type.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    type.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (type.description && type.description.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // Client-side pagination
  const startIndex = (currentPage - 1) * pageSize
  const endIndex = startIndex + pageSize
  const paginatedTypes = filteredTypes.slice(startIndex, endIndex)
  const totalPages = Math.ceil(filteredTypes.length / pageSize)

  const paginationData = {
    total: filteredTypes.length,
    skip: startIndex,
    limit: pageSize,
    pages: totalPages,
    current_page: currentPage
  }

  // Reset to page 1 when search or filter changes
  useEffect(() => {
    setCurrentPage(1)
  }, [searchTerm, categoryFilter])

  const getCategoryIcon = (category: string | null) => {
    if (!category) return Key
    return CATEGORY_ICONS[category] || Key
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold">Credential Types</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Browse {filteredTypes.length} of {credentialTypes.length} available credential types for integrations
        </p>
      </div>

      {/* Filters */}
      <Card className="glass-card">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search credential types..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>

            <Select value={categoryFilter} onValueChange={setCategoryFilter}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {categories.map((cat) => (
                  <SelectItem key={cat} value={cat}>
                    {cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Credential Types Grid */}
      {loading ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground">Loading credential types...</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {paginatedTypes.map((type) => {
              const IconComponent = getCategoryIcon(type.category)

              return (
                <Card
                  key={type.id}
                  className="glass-card hover:border-primary/50 transition-all cursor-pointer"
                  onClick={() => setSelectedType(type)}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <ToolLogo
                          logo={type.logo || undefined}
                          name={type.display_name}
                          size={40}
                          fallbackIcon={type.icon || undefined}
                          showBackground={true}
                        />
                        <div>
                          <CardTitle className="text-lg">{type.display_name}</CardTitle>
                          {type.category && (
                            <Badge variant="outline" className="mt-1 text-xs">
                              {type.category}
                            </Badge>
                          )}
                        </div>
                      </div>
                      {type.is_system && (
                        <Badge className="bg-info/20 text-info text-xs">
                          System
                        </Badge>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent>
                    {type.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                        {type.description}
                      </p>
                    )}

                    <div className="text-xs text-muted-foreground">
                      {type.schema_definition.length} field{type.schema_definition.length !== 1 && 's'}
                      {type.test_endpoint && ' • Test available'}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Pagination */}
          {filteredTypes.length > pageSize && (
            <EnhancedPagination
              data={paginationData}
              onPageChange={(page) => setCurrentPage(page)}
              className="mt-6"
            />
          )}
        </>
      )}

      {filteredTypes.length === 0 && !loading && (
        <Card className="glass-card">
          <CardContent className="pt-12 pb-12 text-center">
            <Key className="w-12 h-12 mx-auto text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Credential Types Found</h3>
            <p className="text-sm text-muted-foreground">
              Try adjusting your search or filters
            </p>
          </CardContent>
        </Card>
      )}

      {/* Detail Dialog */}
      {selectedType && (
        <Dialog open={!!selectedType} onOpenChange={() => setSelectedType(null)}>
          <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3">
                <ToolLogo
                  logo={selectedType.logo || undefined}
                  name={selectedType.display_name}
                  size={24}
                  fallbackIcon={selectedType.icon || undefined}
                  showBackground={false}
                />
                {selectedType.display_name}
              </DialogTitle>
              <DialogDescription>
                {selectedType.description}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-6">
              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Internal Name</p>
                  <p className="text-sm font-mono">{selectedType.name}</p>
                </div>
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Category</p>
                  <p className="text-sm">{selectedType.category || 'None'}</p>
                </div>
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Fields</p>
                  <p className="text-sm">{selectedType.schema_definition.length}</p>
                </div>
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Test Available</p>
                  <p className="text-sm">{selectedType.test_endpoint ? 'Yes' : 'No'}</p>
                </div>
              </div>

              {/* Documentation Link */}
              {selectedType.documentation_url && (
                <a
                  href={selectedType.documentation_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
                >
                  <ExternalLink className="w-4 h-4" />
                  View Documentation
                </a>
              )}

              {/* Schema Definition */}
              <div>
                <h4 className="font-semibold mb-3">Field Schema</h4>
                <div className="space-y-3">
                  {selectedType.schema_definition.map((field, idx) => (
                    <div key={idx} className="p-3 bg-secondary/20 rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <p className="font-semibold">
                            {field.displayName}
                            {field.required && <span className="text-destructive ml-1">*</span>}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Field: <code className="text-xs">{field.name}</code>
                          </p>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {field.type}
                        </Badge>
                      </div>

                      {field.description && (
                        <p className="text-sm text-muted-foreground mt-2">{field.description}</p>
                      )}

                      {field.default !== undefined && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Default: <code>{JSON.stringify(field.default)}</code>
                        </p>
                      )}

                      {field.typeOptions?.options && (
                        <div className="mt-2">
                          <p className="text-xs font-semibold text-muted-foreground mb-1">Options:</p>
                          <div className="flex flex-wrap gap-1">
                            {field.typeOptions.options.map((opt: any, i: number) => (
                              <Badge key={i} variant="secondary" className="text-xs">
                                {opt.name}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Test Endpoint Info */}
              {selectedType.test_endpoint && (
                <div>
                  <h4 className="font-semibold mb-2">Test Configuration</h4>
                  <div className="p-3 bg-secondary/20 rounded-lg">
                    <p className="text-sm">
                      <span className="font-semibold">Method:</span> {selectedType.test_endpoint.method}
                    </p>
                    {selectedType.test_endpoint.description && (
                      <p className="text-sm mt-1 text-muted-foreground">
                        {selectedType.test_endpoint.description}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}

