import { useState, useEffect } from 'react'
import { toast } from 'sonner'
import apiClient from '@/lib/api-client'

interface DatabaseSource {
  id: number
  name: string
  dialect: string
  status: string
  tables_count: number
  last_synced: string
  credential_id: number
}

interface QueryResult {
  sql: string
  results: any[]
  validation: any
  execution_time: number
}

interface QueryTemplate {
  id: number
  name: string
  natural_language: string
  category: string
  sql_template: string
  parameters: any[]
  visualization_type: string
}

interface CacheStats {
  schema_cache_hits: number
  schema_cache_misses: number
  query_cache_hits: number
  query_cache_misses: number
  schema_hit_rate: number
  query_hit_rate: number
}

export function useDatabaseKnowledge() {
  const [sources, setSources] = useState<DatabaseSource[]>([])
  const [templates, setTemplates] = useState<QueryTemplate[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch all database sources
  const fetchSources = async () => {
    setLoading(true)
    try {
      const data = await apiClient.request<any[]>('/api/knowledge/sources/database/')
      const normalized = (Array.isArray(data) ? data : data?.items || []).map((source: any) => ({
        id: source.id,
        name: source.name,
        dialect: source.dialect,
        status: source.status,
        tables_count: source.schema_tables_count ?? source.tables_count ?? 0,
        last_synced: source.last_introspected ?? source.last_synced ?? 'Never',
        credential_id: source.credential_id ?? null,
      }))
      setSources(normalized)
    } catch (err: any) {
      setError(err.message || 'Failed to load database sources')
      toast.error(err.message || 'Failed to load database sources')
    } finally {
      setLoading(false)
    }
  }

  // Create a new database source
  const createSource = async (sourceData: {
    name: string
    credential_id: number
    description?: string
  }) => {
    setLoading(true)
    try {
      const result = await apiClient.request('/api/knowledge/sources/database/', {
        method: 'POST',
        body: sourceData,
      })

      toast.success('Database source created successfully')
      await fetchSources()
      return result
    } catch (err: any) {
      setError(err.message || 'Failed to create database source')
      toast.error(err.message || 'Failed to create database source')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Execute natural language query
  const executeQuery = async (
    sourceId: number,
    naturalLanguageQuery: string,
    options?: {
      validate?: boolean
      use_cache?: boolean
    }
  ): Promise<QueryResult> => {
    setLoading(true)
    try {
      const result = await apiClient.request(`${'/api/knowledge/sources/database'}/${sourceId}/query`, {
        method: 'POST',
        body: {
          source_id: sourceId,
          query: naturalLanguageQuery,
          ...(options || {}),
        },
      })

      toast.success(`Query executed! ${result.row_count || 0} rows returned`)
      return result
    } catch (err: any) {
      setError(err.message || 'Query execution failed')
      toast.error(err.message || 'Query execution failed')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Sync schema for a database source
  const syncSchema = async (sourceId: number) => {
    setLoading(true)
    try {
      await apiClient.request(`/api/knowledge/sources/database/${sourceId}/introspect`, {
        method: 'POST',
      })
      toast.success('Schema sync started')
      await fetchSources()
    } catch (err: any) {
      setError(err.message || 'Failed to sync schema')
      toast.error(err.message || 'Failed to sync schema')
    } finally {
      setLoading(false)
    }
  }

  // Get schema metadata
  const getSchemaMetadata = async (sourceId: number) => {
    try {
      return await apiClient.request(`/api/knowledge/sources/database/${sourceId}/schema`)
    } catch (err: any) {
      toast.error(err.message || 'Failed to load schema metadata')
      throw err
    }
  }

  // Update semantic layer
  const updateSemanticLayer = async (
    sourceId: number,
    semanticData: {
      metrics: any[]
      dimensions: any[]
    }
  ) => {
    setLoading(true)
    try {
      const result = await apiClient.request(`/api/knowledge/sources/database/${sourceId}/semantic-layer`, {
        method: 'PUT',
        body: semanticData,
      })

      toast.success('Semantic layer updated')
      return result
    } catch (err: any) {
      setError(err.message || 'Failed to update semantic layer')
      toast.error(err.message || 'Failed to update semantic layer')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Fetch query templates
  const fetchTemplates = async (dialect?: string) => {
    try {
      const url = dialect 
        ? `/api/knowledge/sources/database/templates/list?dialect=${dialect}`
        : '/api/knowledge/sources/database/templates/list'

      const data = await apiClient.request(url)
      setTemplates(data)
      return data
    } catch (err: any) {
      toast.error(err.message || 'Failed to load query templates')
      throw err
    }
  }

  // Execute a query template
  const executeTemplate = async (
    templateId: number,
    sourceId: number,
    parameters: Record<string, any>
  ) => {
    setLoading(true)
    try {
      const result = await apiClient.request(`/api/knowledge/sources/database/templates/${templateId}/execute`, {
        method: 'POST',
        body: {
          source_id: sourceId,
          parameters,
        },
      })

      toast.success('Template executed successfully')
      return result
    } catch (err: any) {
      setError(err.message || 'Failed to execute template')
      toast.error(err.message || 'Failed to execute template')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Get cache statistics
  const getCacheStats = async (): Promise<CacheStats> => {
    try {
      return await apiClient.request('/api/knowledge/sources/database/cache/stats')
    } catch (err) {
      console.error('Failed to load cache stats:', err)
      return {
        schema_cache_hits: 0,
        schema_cache_misses: 0,
        query_cache_hits: 0,
        query_cache_misses: 0,
        schema_hit_rate: 0,
        query_hit_rate: 0
      }
    }
  }

  // Initialize - fetch sources on mount
  useEffect(() => {
    fetchSources()
    fetchTemplates()
  }, [])

  return {
    sources,
    templates,
    loading,
    error,
    
    // Actions
    fetchSources,
    createSource,
    executeQuery,
    syncSchema,
    getSchemaMetadata,
    updateSemanticLayer,
    fetchTemplates,
    executeTemplate,
    getCacheStats,
    
    // Utilities
    clearError: () => setError(null)
  }
}

// Hook for managing semantic layer
export function useSemanticLayer(sourceId: number) {
  const [metrics, setMetrics] = useState<any[]>([])
  const [dimensions, setDimensions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const fetchSemanticLayer = async () => {
    if (!sourceId) return
    
    setLoading(true)
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/semantic-layer`
      )
      
      if (response.ok) {
        const data = await response.json()
        setMetrics(data.metrics || [])
        setDimensions(data.dimensions || [])
      }
    } catch (err) {
      toast.error('Failed to load semantic layer')
    } finally {
      setLoading(false)
    }
  }

  const addMetric = async (metric: any) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/metrics`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(metric)
        }
      )
      
      if (response.ok) {
        const newMetric = await response.json()
        setMetrics([...metrics, newMetric])
        toast.success('Metric added successfully')
        return newMetric
      } else {
        throw new Error('Failed to add metric')
      }
    } catch (err) {
      toast.error('Failed to add metric')
      throw err
    }
  }

  const updateMetric = async (metricId: number, updates: any) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/metrics/${metricId}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates)
        }
      )
      
      if (response.ok) {
        await fetchSemanticLayer()
        toast.success('Metric updated successfully')
      } else {
        throw new Error('Failed to update metric')
      }
    } catch (err) {
      toast.error('Failed to update metric')
      throw err
    }
  }

  const deleteMetric = async (metricId: number) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/metrics/${metricId}`,
        { method: 'DELETE' }
      )
      
      if (response.ok) {
        setMetrics(metrics.filter(m => m.id !== metricId))
        toast.success('Metric deleted')
      } else {
        throw new Error('Failed to delete metric')
      }
    } catch (err) {
      toast.error('Failed to delete metric')
      throw err
    }
  }

  const addDimension = async (dimension: any) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/dimensions`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(dimension)
        }
      )
      
      if (response.ok) {
        const newDimension = await response.json()
        setDimensions([...dimensions, newDimension])
        toast.success('Dimension added successfully')
        return newDimension
      } else {
        throw new Error('Failed to add dimension')
      }
    } catch (err) {
      toast.error('Failed to add dimension')
      throw err
    }
  }

  const updateDimension = async (dimensionId: number, updates: any) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/dimensions/${dimensionId}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates)
        }
      )
      
      if (response.ok) {
        await fetchSemanticLayer()
        toast.success('Dimension updated successfully')
      } else {
        throw new Error('Failed to update dimension')
      }
    } catch (err) {
      toast.error('Failed to update dimension')
      throw err
    }
  }

  const deleteDimension = async (dimensionId: number) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/dimensions/${dimensionId}`,
        { method: 'DELETE' }
      )
      
      if (response.ok) {
        setDimensions(dimensions.filter(d => d.id !== dimensionId))
        toast.success('Dimension deleted')
      } else {
        throw new Error('Failed to delete dimension')
      }
    } catch (err) {
      toast.error('Failed to delete dimension')
      throw err
    }
  }

  useEffect(() => {
    fetchSemanticLayer()
  }, [sourceId])

  return {
    metrics,
    dimensions,
    loading,
    
    // Actions
    fetchSemanticLayer,
    addMetric,
    updateMetric,
    deleteMetric,
    addDimension,
    updateDimension,
    deleteDimension
  }
}
