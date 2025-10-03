'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { apiClient } from "@/lib/api-client"
import { CodeGraphVisualization } from './CodeGraphVisualization'
import { 
  Database, FileCode, Search, GitBranch, Trash2, RotateCw,
  Activity, TrendingUp, Github, Loader2, CheckCircle2, XCircle,
  Code2, FunctionSquare, FileSearch, Network
} from 'lucide-react'

interface Project {
  id: number
  name: string
  source_type: string
  source_url: string
  branch: string
  language: string
  total_files: number
  total_symbols: number
  total_relationships: number  // NEW: Show relationship count
  last_indexed: string
  index_duration_seconds: number
  status: string
}

interface SearchResult {
  id: number
  symbol_type: string
  name: string
  qualified_name: string
  file_path: string
  line_number: number
  signature: string
  docstring: string
  code_snippet: string
  similarity?: number
}

export function CodeGraphPanel() {
  const [activeTab, setActiveTab] = useState('projects')
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Add project modal
  const [showAddModal, setShowAddModal] = useState(false)
  const [newProject, setNewProject] = useState({
    project_name: '',
    github_url: '',
    branch: 'main',
    auth_token: '',
    exclude_patterns: 'node_modules,__pycache__,.git,dist,build,.next'
  })
  
  // Search state
  const [selectedProject, setSelectedProject] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState<'symbol' | 'semantic'>('symbol')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [searchTime, setSearchTime] = useState<number>(0)

  // Load projects on mount
  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await apiClient.codegraphListProjects()
      setProjects(data || [])
      
      // Auto-select first project
      if (data && data.length > 0 && !selectedProject) {
        setSelectedProject(data[0].name)
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to load projects')
      console.error('Error loading projects:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleIndexGithub = async () => {
    if (!newProject.project_name || !newProject.github_url) {
      setError('Project name and GitHub URL are required')
      return
    }

    try {
      setLoading(true)
      setError(null)
      
      const excludeArray = newProject.exclude_patterns
        .split(',')
        .map(p => p.trim())
        .filter(p => p.length > 0)
      
      await apiClient.codegraphIndexGithub({
        project_name: newProject.project_name,
        github_url: newProject.github_url,
        branch: newProject.branch || 'main',
        auth_token: newProject.auth_token || undefined,
        exclude_patterns: excludeArray
      })
      
      setShowAddModal(false)
      setNewProject({
        project_name: '',
        github_url: '',
        branch: 'main',
        auth_token: '',
        exclude_patterns: 'node_modules,__pycache__,.git,dist,build,.next'
      })
      
      // Reload projects after indexing
      await loadProjects()
      
    } catch (e: any) {
      setError(e?.message || 'Failed to index repository')
      console.error('Error indexing:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!selectedProject || !searchQuery) {
      setError('Please select a project and enter a search query')
      return
    }

    try {
      setSearching(true)
      setError(null)
      setSearchResults([])
      
      let data
      if (searchMode === 'symbol') {
        data = await apiClient.codegraphSearchSymbols({
          project: selectedProject,
          q: searchQuery,
          limit: 20
        })
      } else {
        data = await apiClient.codegraphSearchSemantic({
          project: selectedProject,
          q: searchQuery,
          limit: 20
        })
      }
      
      setSearchResults(data.results || [])
      setSearchTime(data.execution_time_ms || 0)
      
    } catch (e: any) {
      setError(e?.message || 'Search failed')
      console.error('Error searching:', e)
    } finally {
      setSearching(false)
    }
  }

  const handleDeleteProject = async (projectId: number) => {
    if (!confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
      return
    }

    try {
      setLoading(true)
      setError(null)
      await apiClient.codegraphDeleteProject(projectId)
      await loadProjects()
    } catch (e: any) {
      setError(e?.message || 'Failed to delete project')
      console.error('Error deleting project:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleReindex = async (projectId: number) => {
    try {
      setLoading(true)
      setError(null)
      await apiClient.codegraphReindexProject(projectId)
      await loadProjects()
    } catch (e: any) {
      setError(e?.message || 'Failed to re-index project')
      console.error('Error re-indexing:', e)
    } finally {
      setLoading(false)
    }
  }

  // Calculate stats
  const stats = {
    totalProjects: projects.length,
    totalFiles: projects.reduce((sum, p) => sum + p.total_files, 0),
    totalSymbols: projects.reduce((sum, p) => sum + p.total_symbols, 0),
    totalRelationships: projects.reduce((sum, p) => sum + (p.total_relationships || 0), 0),  // NEW
    activeProjects: projects.filter(p => p.status === 'active').length
  }

  return (
    <div className="codegraph-panel space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <Database className="w-5 h-5 text-blue-400 mb-2" />
                <div className="text-2xl font-bold">{stats.totalProjects}</div>
                <div className="text-sm text-muted-foreground">Projects</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <FileCode className="w-5 h-5 text-green-400 mb-2" />
                <div className="text-2xl font-bold">{stats.totalFiles.toLocaleString()}</div>
                <div className="text-sm text-muted-foreground">Files Indexed</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <Code2 className="w-5 h-5 text-purple-400 mb-2" />
                <div className="text-2xl font-bold">{(stats.totalSymbols / 1000).toFixed(1)}K</div>
                <div className="text-sm text-muted-foreground">Code Symbols</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <Activity className="w-5 h-5 text-orange-400 mb-2" />
                <div className="text-2xl font-bold">{(stats.totalRelationships / 1000).toFixed(1)}K</div>
                <div className="text-sm text-muted-foreground">Relationships</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <XCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-red-400">{error}</span>
          </div>
        </div>
      )}

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="projects">
            <Database className="w-4 h-4 mr-2" />
            Projects
          </TabsTrigger>
          <TabsTrigger value="search">
            <Search className="w-4 h-4 mr-2" />
            Search
          </TabsTrigger>
          <TabsTrigger value="visualization">
            <Network className="w-4 h-4 mr-2" />
            Visualization
          </TabsTrigger>
        </TabsList>

        {/* Projects Tab */}
        <TabsContent value="projects" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Your Projects</h3>
            <Button onClick={() => setShowAddModal(true)}>
              <Github className="w-4 h-4 mr-2" />
              Index GitHub Repo
            </Button>
          </div>

          {loading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
              <span className="ml-2 text-sm text-muted-foreground">Loading projects...</span>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map(project => (
              <Card key={project.id} className="glass-card hover:border-primary/50 transition-all">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-lg">{project.name}</CardTitle>
                      <div className="flex items-center space-x-2 text-xs text-muted-foreground mt-1">
                        <GitBranch className="w-3 h-3" />
                        <span>{project.branch || 'main'}</span>
                      </div>
                    </div>
                    <Badge variant={project.status === 'active' ? 'default' : 'secondary'}>
                      {project.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Files:</span>
                      <span className="font-semibold">{project.total_files.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Symbols:</span>
                      <span className="font-semibold">{project.total_symbols.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Relationships:</span>
                      <span className="font-semibold">{(project.total_relationships || 0).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Language:</span>
                      <span className="font-semibold">{project.language || 'Multi'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Indexed:</span>
                      <span className="font-semibold text-xs">
                        {project.last_indexed ? new Date(project.last_indexed).toLocaleDateString() : 'Never'}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4 flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1"
                      onClick={() => {
                        setSelectedProject(project.name)
                        setActiveTab('search')
                      }}
                    >
                      <Search className="w-3 h-3 mr-1" />
                      Search
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleReindex(project.id)}
                      disabled={loading}
                      title="Re-index this project"
                    >
                      <RotateCw className="w-3 h-3" />
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleDeleteProject(project.id)}
                      disabled={loading}
                      title="Delete this project"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {projects.length === 0 && !loading && (
            <div className="text-center py-12">
              <FileCode className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No projects yet</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Index your first GitHub repository to get started
              </p>
              <Button onClick={() => setShowAddModal(true)}>
                <Github className="w-4 h-4 mr-2" />
                Index GitHub Repo
              </Button>
            </div>
          )}
        </TabsContent>

        {/* Search Tab */}
        <TabsContent value="search" className="space-y-4">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Code Search</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Select value={selectedProject} onValueChange={setSelectedProject}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select project" />
                  </SelectTrigger>
                  <SelectContent>
                    {projects.map(p => (
                      <SelectItem key={p.id} value={p.name}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select value={searchMode} onValueChange={(v) => setSearchMode(v as 'symbol' | 'semantic')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="symbol">
                      <div className="flex items-center">
                        <FunctionSquare className="w-4 h-4 mr-2" />
                        Symbol Search
                      </div>
                    </SelectItem>
                    <SelectItem value="semantic">
                      <div className="flex items-center">
                        <FileSearch className="w-4 h-4 mr-2" />
                        Semantic Search
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>

                <div className="flex space-x-2">
                  <Input
                    className="flex-1"
                    placeholder={searchMode === 'symbol' ? 'Search: function name...' : 'Search: user authentication...'}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  />
                  <Button onClick={handleSearch} disabled={searching || !selectedProject}>
                    {searching ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Search className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              </div>

              {searchResults.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  Found {searchResults.length} results in {searchTime.toFixed(0)}ms
                </div>
              )}
            </CardContent>
          </Card>

          {/* Search Results */}
          <div className="space-y-3">
            {searchResults.map((result, idx) => (
              <Card key={idx} className="glass-card hover:border-primary/50 transition-all">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="text-sm text-blue-400 font-mono">
                        {result.file_path}:{result.line_number}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {result.symbol_type}: {result.name}
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{result.symbol_type}</Badge>
                      {result.similarity && (
                        <Badge variant="secondary">
                          {(result.similarity * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="text-sm font-mono bg-secondary/30 p-2 rounded">
                      {result.signature}
                    </div>
                    {result.docstring && (
                      <div className="text-sm text-muted-foreground">
                        {result.docstring}
                      </div>
                    )}
                    <pre className="text-xs bg-black/30 p-3 rounded overflow-x-auto">
                      <code>{result.code_snippet}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {searchResults.length === 0 && searchQuery && !searching && (
            <div className="text-center py-12">
              <FileSearch className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-sm text-muted-foreground">
                No results found for "{searchQuery}"
              </p>
            </div>
          )}
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="visualization" className="space-y-4">
          {selectedProject ? (
            <CodeGraphVisualization project={selectedProject} />
          ) : (
            <Card className="glass-card">
              <CardContent className="py-12">
                <div className="text-center text-muted-foreground">
                  <Network className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Select a project from the search tab to visualize code relationships</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Add Project Modal */}
      <Dialog open={showAddModal} onOpenChange={setShowAddModal}>
        <DialogContent className="glass-card max-w-2xl">
          <DialogHeader>
            <DialogTitle>Index GitHub Repository</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div>
              <Label>Project Name</Label>
              <Input
                placeholder="my-awesome-project"
                value={newProject.project_name}
                onChange={(e) => setNewProject({...newProject, project_name: e.target.value})}
              />
            </div>

            <div>
              <Label>GitHub Repository URL</Label>
              <Input
                placeholder="https://github.com/username/repo.git"
                value={newProject.github_url}
                onChange={(e) => setNewProject({...newProject, github_url: e.target.value})}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Branch</Label>
                <Input
                  placeholder="main"
                  value={newProject.branch}
                  onChange={(e) => setNewProject({...newProject, branch: e.target.value})}
                />
              </div>

              <div>
                <Label>Auth Token (Optional)</Label>
                <Input
                  type="password"
                  placeholder="ghp_..."
                  value={newProject.auth_token}
                  onChange={(e) => setNewProject({...newProject, auth_token: e.target.value})}
                />
              </div>
            </div>

            <div>
              <Label>Exclude Patterns (comma-separated)</Label>
              <Textarea
                placeholder="node_modules,__pycache__,.git"
                value={newProject.exclude_patterns}
                onChange={(e) => setNewProject({...newProject, exclude_patterns: e.target.value})}
                rows={2}
              />
              <p className="text-xs text-muted-foreground mt-1">
                Patterns to exclude from indexing (e.g., node_modules, dist, build)
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleIndexGithub} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Indexing...
                </>
              ) : (
                <>
                  <Github className="w-4 h-4 mr-2" />
                  Index Repository
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
