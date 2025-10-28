import { useState, useEffect } from 'react'
import { toast } from 'sonner'

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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
      const response = await fetch(`${API_BASE}/api/knowledge/sources/database`)
      if (response.ok) {
        const data = await response.json()
        setSources(data)
      } else {
        throw new Error('Failed to fetch database sources')
      }
    } catch (err) {
      setError(err.message)
      toast.error('Failed to load database sources')
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
      const response = await fetch(`${API_BASE}/api/knowledge/sources/database`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sourceData)
      })
      
      if (response.ok) {
        const newSource = await response.json()
        setSources([...sources, newSource])
        toast.success('Database source created successfully')
        return newSource
      } else {
        throw new Error('Failed to create database source')
      }
    } catch (err) {
      setError(err.message)
      toast.error('Failed to create database source')
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
      const response = await fetch(`${API_BASE}/api/knowledge/sources/database/${sourceId}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: sourceId,
          query: naturalLanguageQuery
        })
      })
      
      if (response.ok) {
        const result = await response.json()
        toast.success(`Query executed! ${result.row_count || 0} rows returned`)
        return result
      } else {
        const error = await response.json()
        throw new Error(error.detail || error.error || 'Query execution failed')
      }
    } catch (err) {
      setError(err.message)
      toast.error(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Sync schema for a database source
  const syncSchema = async (sourceId: number) => {
    setLoading(true)
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/sync`,
        { method: 'POST' }
      )
      
      if (response.ok) {
        toast.success('Schema sync started')
        await fetchSources() // Refresh sources
      } else {
        throw new Error('Failed to sync schema')
      }
    } catch (err) {
      setError(err.message)
      toast.error('Failed to sync schema')
    } finally {
      setLoading(false)
    }
  }

  // Get schema metadata
  const getSchemaMetadata = async (sourceId: number) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/schema`
      )
      
      if (response.ok) {
        return await response.json()
      } else {
        throw new Error('Failed to fetch schema metadata')
      }
    } catch (err) {
      toast.error('Failed to load schema metadata')
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
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/${sourceId}/semantic-layer`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(semanticData)
        }
      )
      
      if (response.ok) {
        toast.success('Semantic layer updated')
        return await response.json()
      } else {
        throw new Error('Failed to update semantic layer')
      }
    } catch (err) {
      setError(err.message)
      toast.error('Failed to update semantic layer')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Fetch query templates
  const fetchTemplates = async (dialect?: string) => {
    try {
      const url = dialect 
        ? `${API_BASE}/api/knowledge/sources/database/templates/list?dialect=${dialect}`
        : `${API_BASE}/api/knowledge/sources/database/templates/list`
      
      const response = await fetch(url)
      
      if (response.ok) {
        const data = await response.json()
        setTemplates(data)
        return data
      } else {
        throw new Error('Failed to fetch templates')
      }
    } catch (err) {
      toast.error('Failed to load query templates')
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
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/templates/${templateId}/execute`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_id: sourceId,
            parameters
          })
        }
      )
      
      if (response.ok) {
        const result = await response.json()
        toast.success('Template executed successfully')
        return result
      } else {
        throw new Error('Failed to execute template')
      }
    } catch (err) {
      setError(err.message)
      toast.error('Failed to execute template')
      throw err
    } finally {
      setLoading(false)
    }
  }

  // Get cache statistics
  const getCacheStats = async (): Promise<CacheStats> => {
    try {
      const response = await fetch(
        `${API_BASE}/api/knowledge/sources/database/cache/stats`
      )
      
      if (response.ok) {
        return await response.json()
      } else {
        throw new Error('Failed to fetch cache stats')
      }
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
