'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  listCredentialTypes,
  createCredential,
  updateCredential,
  getCredential,
  type CredentialType,
  type CredentialFieldDefinition
} from '@/lib/api/credentials'

interface DynamicCredentialFormProps {
  credentialId?: number  // If provided, edit mode
  lockCredentialTypeId?: number
  lockedCredentialType?: CredentialType
  hideMetadataSection?: boolean
  defaultName?: string
  defaultEnvironment?: string
  onSuccess?: () => void
  onCancel?: () => void
}

// Field-label overrides for credential types where the canonical schema in the
// DB ships with terse names ("Access Token", "APP Secret Key" — useless when
// the platform has multiple token types). Overlay clearer labels + inline
// hints at render time so merchants know exactly which value to paste where
// without us touching the shared credential_types DB rows.
type FieldOverride = {
  displayName?: string
  description?: string
  placeholder?: string
}
const CREDENTIAL_FIELD_OVERRIDES: Record<string, Record<string, FieldOverride>> = {
  shopifyAccessTokenApi: {
    shopSubdomain: {
      displayName: 'Shop Subdomain',
      description:
        'Just the subdomain — e.g. innobuilduk for innobuilduk.myshopify.com. No https:// and no .myshopify.com.',
      placeholder: 'innobuilduk',
    },
    accessToken: {
      displayName: 'Partner App Client ID  (or Admin API Token shpat_…)',
      description:
        'From your Shopify Partner Dashboard → Apps → your app → Configuration → Client ID. ' +
        'Alternatively, if you already have a per-store Admin API token (shpat_…) from a Custom App, paste that here instead.',
    },
    appSecretKey: {
      displayName: 'Partner App Client Secret  (leave blank if using shpat_)',
      description:
        'From the same Configuration screen, next to Client ID (shpss_…). Required when using Partner App credentials so we can complete the Shopify install bounce. Leave blank if you pasted an shpat_ token above.',
    },
  },
  shopifyOAuth2Api: {
    shopSubdomain: {
      displayName: 'Shop Subdomain',
      description:
        'Just the subdomain — e.g. innobuilduk for innobuilduk.myshopify.com.',
      placeholder: 'innobuilduk',
    },
    clientId: {
      displayName: 'Partner App Client ID',
      description:
        'From your Shopify Partner Dashboard → Apps → your app → Configuration → Client ID.',
    },
    clientSecret: {
      displayName: 'Partner App Client Secret',
      description:
        'From the same Configuration screen, next to Client ID. Treat as a password.',
    },
  },
}

function applyFieldOverride(
  typeName: string | undefined,
  field: CredentialFieldDefinition,
): CredentialFieldDefinition {
  if (!typeName) return field
  const overrides = CREDENTIAL_FIELD_OVERRIDES[typeName]?.[field.name]
  if (!overrides) return field
  return {
    ...field,
    displayName: overrides.displayName ?? field.displayName,
    description: overrides.description ?? field.description,
    placeholder: overrides.placeholder ?? field.placeholder,
  }
}

export function DynamicCredentialForm({
  credentialId,
  lockCredentialTypeId,
  lockedCredentialType,
  hideMetadataSection = false,
  defaultName,
  defaultEnvironment,
  onSuccess,
  onCancel
}: DynamicCredentialFormProps) {
  const metadataHidden = hideMetadataSection || !!lockCredentialTypeId
  const [credentialTypes, setCredentialTypes] = useState<CredentialType[]>([])
  const [selectedTypeId, setSelectedTypeId] = useState<number | null>(null)
  const [selectedType, setSelectedType] = useState<CredentialType | null>(null)
  const [formData, setFormData] = useState<Record<string, any>>({})
  const [credentialMetadata, setCredentialMetadata] = useState({
    name: '',
    environment: 'production',
    description: '',
    tags: ''
  })
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (lockCredentialTypeId && lockedCredentialType) {
      setCredentialTypes([lockedCredentialType])
      setSelectedTypeId(lockedCredentialType.id)
      setSelectedType(lockedCredentialType)
    } else if (lockCredentialTypeId) {
      loadLockedCredentialType(lockCredentialTypeId)
    } else {
      loadCredentialTypes()
    }
    if (credentialId) {
      loadExistingCredential()
    }
  }, [credentialId, lockCredentialTypeId, lockedCredentialType])

  useEffect(() => {
    if (selectedTypeId) {
      const type = credentialTypes.find(t => t.id === selectedTypeId)
      setSelectedType(type || null)
      
      // Initialize form data with defaults
      if (type && !credentialId) {
        const defaultData: Record<string, any> = {}
        type.schema_definition.forEach(field => {
          if (field.default !== undefined) {
            defaultData[field.name] = field.default
          }
        })
        setFormData(defaultData)
      }
    }
  }, [selectedTypeId, credentialTypes])

  useEffect(() => {
    if (metadataHidden && defaultName && !credentialMetadata.name) {
      setCredentialMetadata(prev => ({ ...prev, name: defaultName }))
    }
    if (metadataHidden && defaultEnvironment && !credentialMetadata.environment) {
      setCredentialMetadata(prev => ({ ...prev, environment: defaultEnvironment }))
    }
  }, [
    metadataHidden,
    defaultName,
    defaultEnvironment,
    credentialMetadata.name,
    credentialMetadata.environment
  ])

  const loadCredentialTypes = async () => {
    try {
      setLoading(true)
      const types = await listCredentialTypes({ active_only: true })
      setCredentialTypes(types)
    } catch (error) {
      console.error('Failed to load credential types:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadLockedCredentialType = async (typeId: number) => {
    try {
      setLoading(true)
      const type = await getCredentialType(typeId)
      setCredentialTypes([type])
      setSelectedTypeId(typeId)
      setSelectedType(type)
    } catch (error) {
      console.error('Failed to load credential type:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadExistingCredential = async () => {
    if (!credentialId) return

    try {
      setLoading(true)
      const credential = await getCredential(credentialId)
      
      setSelectedTypeId(lockCredentialTypeId || credential.credential_type_id)
      setCredentialMetadata({
        name: credential.name,
        environment: credential.environment,
        description: credential.description || '',
        tags: credential.tags.join(', ')
      })
      
      // Load the actual credential values for editing
      if (credential.credential_data) {
        setFormData(credential.credential_data)
      }
    } catch (error) {
      console.error('Failed to load credential:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!selectedTypeId) {
      alert('Please select a credential type')
      return
    }

    if (!credentialMetadata.name) {
      alert('Please enter a credential name')
      return
    }

    // Validate required fields
    const missingFields = selectedType?.schema_definition
      .filter(field => field.required && !formData[field.name])
      .map(field => field.displayName)

    if (missingFields && missingFields.length > 0) {
      alert(`Missing required fields:\n- ${missingFields.join('\n- ')}`)
      return
    }

    try {
      setSaving(true)

      const saved = credentialId
        ? await updateCredential(credentialId, {
            name: credentialMetadata.name,
            credential_data: formData,
            description: credentialMetadata.description || undefined,
            tags: credentialMetadata.tags
              ? credentialMetadata.tags.split(',').map(t => t.trim())
              : [],
          })
        : await createCredential({
            name: credentialMetadata.name,
            credential_type_id: selectedTypeId,
            credential_data: formData,
            environment: credentialMetadata.environment,
            description: credentialMetadata.description || undefined,
            tags: credentialMetadata.tags
              ? credentialMetadata.tags.split(',').map(t => t.trim())
              : [],
          })

      // Surface backend integration-bridge result (Shopify→Composio etc.).
      switch (saved.connection_status) {
        case 'pending_oauth':
          if (saved.oauth_redirect_url) {
            // Try popup first — but ALWAYS surface the URL on screen too. Popups
            // get blocked silently in many browsers (especially after save flows
            // not triggered by a direct click event), and merchants can't see
            // why the install screen never appears. Showing the URL in a
            // confirm() dialog lets them either keep the popup or click the
            // URL directly. The URL also lands in clipboard via prompt() fallback.
            const popup = window.open(
              saved.oauth_redirect_url,
              'shopify-oauth',
              'width=600,height=720,menubar=no,toolbar=no,location=no',
            )
            const popupBlocked = !popup || popup.closed || typeof popup.closed === 'undefined'
            if (popupBlocked) {
              // Use prompt() so the URL is selectable/copyable in a single click.
              window.prompt(
                'Shopify install popup was blocked. Copy this URL and open it in a new tab to complete Shopify install:',
                saved.oauth_redirect_url,
              )
            } else {
              // Popup opened — confirm the merchant sees it and knows what to do.
              const stayOpen = window.confirm(
                'A Shopify install popup has opened — click "Install app" on that page to finish connecting.\n\n' +
                  'If you do not see the popup, click OK to copy the install URL and open it manually.',
              )
              if (stayOpen) {
                window.prompt(
                  'Open this URL in a new tab to complete Shopify install:',
                  saved.oauth_redirect_url,
                )
              }
            }
          }
          break
        case 'unsupported':
          alert(
            saved.connection_error ||
              'This credential type isn\'t supported by the integration platform — it was saved but no live connection was created.',
          )
          break
        case 'bridge_error':
          alert(
            'Credential saved, but the platform connection failed:\n\n' +
              (saved.connection_error || 'Unknown error'),
          )
          break
        // case 'connected' or null — silent success
      }

      onSuccess?.()
    } catch (error: any) {
      console.error('Failed to save credential:', error)
      alert(`Failed to save credential: ${error.message || error}`)
    } finally {
      setSaving(false)
    }
  }

  const renderField = (field: CredentialFieldDefinition) => {
    // Check display conditions
    if (field.displayOptions?.show) {
      const conditions = field.displayOptions.show
      for (const [condField, condValues] of Object.entries(conditions)) {
        if (!condValues.includes(formData[condField])) {
          return null  // Hide this field
        }
      }
    }

    const value = formData[field.name] ?? field.default ?? ''

    switch (field.type) {
      case 'password':
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name}>
              {field.displayName}
              {field.required && <span className="text-destructive ml-1">*</span>}
            </Label>
            {field.description && (
              <p className="text-xs text-muted-foreground">{field.description}</p>
            )}
            <Input
              id={field.name}
              type="password"
              value={value}
              onChange={(e) => setFormData({ ...formData, [field.name]: e.target.value })}
              placeholder={field.placeholder || ''}
              required={field.required}
            />
          </div>
        )

      case 'number':
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name}>
              {field.displayName}
              {field.required && <span className="text-destructive ml-1">*</span>}
            </Label>
            {field.description && (
              <p className="text-xs text-muted-foreground">{field.description}</p>
            )}
            <Input
              id={field.name}
              type="number"
              value={value}
              onChange={(e) => setFormData({ ...formData, [field.name]: parseInt(e.target.value) || 0 })}
              placeholder={field.placeholder || ''}
              required={field.required}
            />
          </div>
        )

      case 'boolean':
        return (
          <div key={field.name} className="flex items-center space-x-2 py-2">
            <Checkbox
              id={field.name}
              checked={value}
              onCheckedChange={(checked) => setFormData({ ...formData, [field.name]: checked })}
            />
            <Label htmlFor={field.name} className="cursor-pointer">
              {field.displayName}
              {field.description && (
                <span className="block text-xs text-muted-foreground font-normal mt-1">
                  {field.description}
                </span>
              )}
            </Label>
          </div>
        )

      case 'options':
        const options = field.typeOptions?.options || []
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name}>
              {field.displayName}
              {field.required && <span className="text-destructive ml-1">*</span>}
            </Label>
            {field.description && (
              <p className="text-xs text-muted-foreground">{field.description}</p>
            )}
            <Select
              value={value}
              onValueChange={(val) => setFormData({ ...formData, [field.name]: val })}
            >
              <SelectTrigger>
                <SelectValue placeholder={`Select ${field.displayName}`} />
              </SelectTrigger>
              <SelectContent>
                {options.map((opt: any) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )

      case 'string':
      default:
        const rows = field.typeOptions?.rows
        
        if (rows && rows > 1) {
          return (
            <div key={field.name} className="space-y-2">
              <Label htmlFor={field.name}>
                {field.displayName}
                {field.required && <span className="text-destructive ml-1">*</span>}
              </Label>
              {field.description && (
                <p className="text-xs text-muted-foreground">{field.description}</p>
              )}
              <Textarea
                id={field.name}
                value={value}
                onChange={(e) => setFormData({ ...formData, [field.name]: e.target.value })}
                placeholder={field.placeholder || ''}
                required={field.required}
                rows={rows}
              />
            </div>
          )
        }
        
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name}>
              {field.displayName}
              {field.required && <span className="text-destructive ml-1">*</span>}
            </Label>
            {field.description && (
              <p className="text-xs text-muted-foreground">{field.description}</p>
            )}
            <Input
              id={field.name}
              type="text"
              value={value}
              onChange={(e) => setFormData({ ...formData, [field.name]: e.target.value })}
              placeholder={field.placeholder || ''}
              required={field.required}
            />
          </div>
        )
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Credential Metadata */}
      {!metadataHidden && (
        <div className="space-y-4 p-4 bg-secondary/20 rounded-lg">
          <h3 className="font-semibold">Credential Info</h3>
          
          {!credentialId && !lockCredentialTypeId && (
            <div className="space-y-2">
              <Label htmlFor="credential_type">
                Credential Type <span className="text-destructive">*</span>
              </Label>
              <Select
                value={selectedTypeId?.toString()}
                onValueChange={(val) => setSelectedTypeId(parseInt(val))}
                required
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select credential type" />
                </SelectTrigger>
                <SelectContent>
                  {credentialTypes.map((type) => (
                    <SelectItem key={type.id} value={type.id.toString()}>
                      {type.display_name} ({type.category})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="name">
              Name <span className="text-destructive">*</span>
            </Label>
            <Input
              id="name"
              value={credentialMetadata.name}
              onChange={(e) => setCredentialMetadata({ ...credentialMetadata, name: e.target.value })}
              placeholder="e.g., Production PostgreSQL"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="environment">Environment</Label>
            <Select
              value={credentialMetadata.environment}
              onValueChange={(val) => setCredentialMetadata({ ...credentialMetadata, environment: val })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="development">Development</SelectItem>
                <SelectItem value="staging">Staging</SelectItem>
                <SelectItem value="production">Production</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Input
              id="description"
              value={credentialMetadata.description}
              onChange={(e) => setCredentialMetadata({ ...credentialMetadata, description: e.target.value })}
              placeholder="Optional description"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="tags">Tags (comma-separated)</Label>
            <Input
              id="tags"
              value={credentialMetadata.tags}
              onChange={(e) => setCredentialMetadata({ ...credentialMetadata, tags: e.target.value })}
              placeholder="e.g., production, critical, database"
            />
          </div>
        </div>
      )}

      {/* Dynamic Fields Based on Credential Type */}
      {selectedType && (
        <div className="space-y-4 p-4 border border-border/30 rounded-lg">
          {!hideMetadataSection && (
            <h3 className="font-semibold">
              {selectedType.display_name} Configuration
            </h3>
          )}
          
          {credentialId && !hideMetadataSection && (
            <div className="text-sm text-warning bg-warning/10 p-3 rounded">
              ⚠️ For security reasons, existing values are not shown. Re-enter values to update.
            </div>
          )}

          {selectedType.schema_definition.map(field =>
            renderField(applyFieldOverride(selectedType.name, field))
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 justify-end pt-4 border-t border-border/30">
        {onCancel && (
          <Button type="button" variant="outline" onClick={onCancel} disabled={saving}>
            Cancel
          </Button>
        )}
        <Button type="submit" disabled={saving || !selectedTypeId}>
          {saving ? 'Saving...' : credentialId ? 'Update Credential' : 'Create Credential'}
        </Button>
      </div>
    </form>
  )
}

