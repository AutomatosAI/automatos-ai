'use client'

import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Database, AlertCircle, CheckCircle, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

interface AddDatabaseModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

export function AddDatabaseModal({ isOpen, onClose, onSuccess }: AddDatabaseModalProps) {
  const [step, setStep] = useState(1)
  const [isLoading, setIsLoading] = useState(false)
  const [credentials, setCredentials] = useState<any[]>([])
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    credential_id: '',
    dialect: 'postgresql',
    options: {
      include_samples: true,
      include_stats: true,
      timezone: 'UTC'
    }
  })
  const [testResult, setTestResult] = useState<any>(null)

  useEffect(() => {
    if (isOpen) {
      fetchCredentials()
    }
  }, [isOpen])

  const fetchCredentials = async () => {
    try {
      // Use apiClient instead of fetch to handle authentication
      const { listCredentials } = await import('@/lib/api/credentials')
      const data = await listCredentials({ active_only: true })
      
      // Filter for database credentials
      const dbCreds = (Array.isArray(data) ? data : data.items || []).filter((cred: any) => {
        const typeName = (cred.credential_type_name || '').toLowerCase()
        const displayName = (cred.credential_type_display_name || '').toLowerCase()
        const name = (cred.name || '').toLowerCase()
        
        return typeName.includes('postgres') || 
               typeName.includes('mysql') || 
               typeName.includes('redis') ||
               typeName.includes('mongo') ||
               displayName.includes('postgres') ||
               displayName.includes('mysql') ||
               displayName.includes('redis') ||
               displayName.includes('mongo') ||
               name.includes('postgres') ||
               name.includes('mysql') ||
               name.includes('database') ||
               name.includes('db')
      })
      
      console.log('Database credentials found:', dbCreds.length, dbCreds)
      setCredentials(dbCreds)
    } catch (error) {
      console.error('Failed to fetch credentials:', error)
      setCredentials([])
    }
  }

  const handleTest = async () => {
    setIsLoading(true)
    setTestResult(null)
    
    try {
      const apiClient = (await import('@/lib/api-client')).default
      
      const result = await apiClient.request(`/api/credentials/${formData.credential_id}/test`, {
        method: 'POST'
      })
      
      if (result.success) {
        setTestResult({ success: true, message: result.message || 'Connection successful!' })
        toast.success('Database connection successful!')
      } else {
        setTestResult({ 
          success: false, 
          message: result.message || 'Connection failed'
        })
        toast.error(result.message || 'Connection failed')
      }
    } catch (error: any) {
      setTestResult({ 
        success: false, 
        message: error.message || 'Failed to test connection'
      })
      toast.error(error.message || 'Failed to test connection')
    } finally {
      setIsLoading(false)
    }
  }

  const handleCreate = async () => {
    setIsLoading(true)
    
    try {
      // Use apiClient for authenticated requests
      const apiClient = (await import('@/lib/api-client')).default
      
      const source = await apiClient.request('/api/knowledge/sources/database/', {
        method: 'POST',
        body: {
          name: formData.name,
          description: formData.description,
          credential_id: parseInt(formData.credential_id),
          dialect: formData.dialect,
          ...formData.options
        }
      })
      
      toast.success('Database source created successfully!')
      
      // Introspect the schema
      setStep(3)
      await introspectSchema(source.id)
      
      onSuccess()
      handleClose()
    } catch (error) {
      toast.error('Failed to create database source')
    } finally {
      setIsLoading(false)
    }
  }

  const introspectSchema = async (sourceId: number) => {
    try {
      const apiClient = (await import('@/lib/api-client')).default
      await apiClient.request(`/api/knowledge/sources/database/${sourceId}/introspect`, {
        method: 'POST'
      })
      toast.success('Schema introspection complete!')
    } catch (error) {
      console.error('Failed to introspect schema:', error)
      toast.error('Failed to introspect schema')
    }
  }

  const handleClose = () => {
    setStep(1)
    setFormData({
      name: '',
      description: '',
      credential_id: '',
      dialect: 'postgresql',
      options: {
        include_samples: true,
        include_stats: true,
        timezone: 'UTC'
      }
    })
    setTestResult(null)
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Add Database Source
          </DialogTitle>
          <DialogDescription>
            Connect a database to query it with natural language
          </DialogDescription>
        </DialogHeader>

        <div className="py-6">
          {/* Progress Steps */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step >= 1 ? 'bg-primary text-primary-foreground' : 'bg-muted'
              }`}>
                1
              </div>
              <span className="ml-2 text-sm">Configure</span>
            </div>
            <div className="flex-1 h-[2px] bg-muted mx-4" />
            <div className="flex items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step >= 2 ? 'bg-primary text-primary-foreground' : 'bg-muted'
              }`}>
                2
              </div>
              <span className="ml-2 text-sm">Test</span>
            </div>
            <div className="flex-1 h-[2px] bg-muted mx-4" />
            <div className="flex items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                step >= 3 ? 'bg-primary text-primary-foreground' : 'bg-muted'
              }`}>
                3
              </div>
              <span className="ml-2 text-sm">Introspect</span>
            </div>
          </div>

          {/* Step 1: Configure */}
          {step === 1 && (
            <div className="space-y-4">
              <div>
                <Label htmlFor="name">Source Name</Label>
                <Input
                  id="name"
                  placeholder="Production Database"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              </div>
              
              <div>
                <Label htmlFor="description">Description (Optional)</Label>
                <Input
                  id="description"
                  placeholder="Main application database"
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                />
              </div>

              <div>
                <Label htmlFor="credential">Database Credential</Label>
                <Select 
                  value={formData.credential_id}
                  onValueChange={(value) => setFormData({ ...formData, credential_id: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a credential" />
                  </SelectTrigger>
                  <SelectContent>
                    {credentials.length === 0 ? (
                      <SelectItem value="none" disabled>
                        No database credentials found
                      </SelectItem>
                    ) : (
                      credentials.map((cred) => (
                        <SelectItem key={cred.id} value={String(cred.id)}>
                          {cred.name} ({cred.credential_type})
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
                {credentials.length === 0 && (
                  <p className="text-sm text-muted-foreground mt-1">
                    Create a database credential in Settings → Credentials first
                  </p>
                )}
              </div>

              <div>
                <Label htmlFor="dialect">Database Type</Label>
                <Select 
                  value={formData.dialect}
                  onValueChange={(value) => setFormData({ ...formData, dialect: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                    <SelectItem value="mysql">MySQL</SelectItem>
                    <SelectItem value="sqlite" disabled>SQLite (Coming Soon)</SelectItem>
                    <SelectItem value="mssql" disabled>SQL Server (Coming Soon)</SelectItem>
                    <SelectItem value="snowflake" disabled>Snowflake (Coming Soon)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Options</Label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2">
                    <input 
                      type="checkbox" 
                      checked={formData.options.include_samples}
                      onChange={(e) => setFormData({
                        ...formData,
                        options: { ...formData.options, include_samples: e.target.checked }
                      })}
                    />
                    <span className="text-sm">Include sample data in schema</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input 
                      type="checkbox" 
                      checked={formData.options.include_stats}
                      onChange={(e) => setFormData({
                        ...formData,
                        options: { ...formData.options, include_stats: e.target.checked }
                      })}
                    />
                    <span className="text-sm">Include table statistics</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Test */}
          {step === 2 && (
            <div className="space-y-4">
              <div className="bg-muted/50 rounded-lg p-4">
                <h4 className="font-medium mb-2">Connection Details</h4>
                <dl className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Name:</dt>
                    <dd>{formData.name}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Type:</dt>
                    <dd className="capitalize">{formData.dialect}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-muted-foreground">Credential:</dt>
                    <dd>{credentials.find(c => String(c.id) === formData.credential_id)?.name || 'None'}</dd>
                  </div>
                </dl>
              </div>

              {testResult && (
                <Alert variant={testResult.success ? 'default' : 'destructive'}>
                  {testResult.success ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <AlertCircle className="h-4 w-4" />
                  )}
                  <AlertDescription>{testResult.message}</AlertDescription>
                </Alert>
              )}

              <div className="flex justify-center">
                <Button 
                  onClick={handleTest}
                  disabled={isLoading || !formData.credential_id}
                  className="w-32"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Testing...
                    </>
                  ) : (
                    'Test Connection'
                  )}
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: Introspect */}
          {step === 3 && (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="w-12 h-12 animate-spin text-primary mb-4" />
                <h4 className="font-medium text-lg">Introspecting Schema...</h4>
                <p className="text-sm text-muted-foreground mt-2">
                  Discovering tables, columns, and relationships
                </p>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isLoading}>
            Cancel
          </Button>
          {step === 1 && (
            <Button 
              onClick={() => setStep(2)}
              disabled={!formData.name || !formData.credential_id}
            >
              Next: Test Connection
            </Button>
          )}
          {step === 2 && (
            <>
              <Button 
                variant="outline" 
                onClick={() => setStep(1)}
                disabled={isLoading}
              >
                Back
              </Button>
              <Button 
                onClick={handleCreate}
                disabled={isLoading || !testResult?.success}
              >
                Create & Introspect
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

