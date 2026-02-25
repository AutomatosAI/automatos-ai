'use client'

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import {
  Package,
  Check,
  ChevronLeft,
  ChevronRight,
  Plus,
  X,
  Loader2,
  Image as ImageIcon,
  Send,
  Save,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared/page-header'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL || ''

const STEPS = [
  { num: 1, label: 'Basic Info' },
  { num: 2, label: 'Technical' },
  { num: 3, label: 'Media' },
  { num: 4, label: 'Pricing' },
  { num: 5, label: 'Review' },
]

const CATEGORIES = [
  'productivity',
  'analytics',
  'data',
  'development',
  'communication',
  'devops',
  'automation',
  'ai',
  'content',
]

const PERMISSIONS = [
  { value: 'chat', label: 'Chat' },
  { value: 'documents:read', label: 'Documents (Read)' },
  { value: 'documents:write', label: 'Documents (Write)' },
  { value: 'data:query', label: 'Data (Query)' },
  { value: 'data:execute', label: 'Data (Execute)' },
  { value: 'agents:read', label: 'Agents (Read)' },
  { value: 'agents:execute', label: 'Agents (Execute)' },
  { value: 'workflows:read', label: 'Workflows (Read)' },
  { value: 'workflows:execute', label: 'Workflows (Execute)' },
]

const MIN_PLANS = [
  { value: 'none', label: 'None' },
  { value: 'starter', label: 'Starter' },
  { value: 'pro', label: 'Pro' },
  { value: 'enterprise', label: 'Enterprise' },
]

const CURRENCIES = ['USD', 'EUR', 'GBP']

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/

function isValidUrl(s: string): boolean {
  try {
    new URL(s)
    return true
  } catch {
    return false
  }
}

async function apiFetch<T>(
  path: string,
  opts: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    credentials: 'include',
    ...opts,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json()
}

function formatPrice(cents: number, currency: string): string {
  const dollars = (cents / 100).toFixed(2)
  const symbols: Record<string, string> = { USD: '$', EUR: '\u20AC', GBP: '\u00A3' }
  return `${symbols[currency] || ''}${dollars} ${currency}`
}

// ---------------------------------------------------------------------------
// Progress Indicator
// ---------------------------------------------------------------------------

function ProgressIndicator({
  currentStep,
  onStepClick,
}: {
  currentStep: number
  onStepClick: (step: number) => void
}) {
  return (
    <div className="flex items-center justify-center gap-0 mb-8">
      {STEPS.map((s, i) => {
        const isCompleted = s.num < currentStep
        const isCurrent = s.num === currentStep
        return (
          <div key={s.num} className="flex items-center">
            <button
              type="button"
              onClick={() => onStepClick(s.num)}
              className={`
                flex flex-col items-center gap-1.5 group cursor-pointer
                ${s.num <= currentStep ? '' : 'opacity-50'}
              `}
            >
              <div
                className={`
                  w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold
                  transition-all duration-300 border-2
                  ${
                    isCompleted
                      ? 'bg-primary border-primary text-primary-foreground'
                      : isCurrent
                        ? 'border-primary text-primary bg-primary/10'
                        : 'border-muted-foreground/30 text-muted-foreground bg-secondary/30'
                  }
                  group-hover:scale-110
                `}
              >
                {isCompleted ? <Check className="w-5 h-5" /> : s.num}
              </div>
              <span
                className={`text-xs font-medium whitespace-nowrap ${
                  isCurrent ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                {s.label}
              </span>
            </button>
            {i < STEPS.length - 1 && (
              <div
                className={`
                  w-12 sm:w-16 md:w-20 h-0.5 mx-1 mt-[-18px] transition-colors duration-300
                  ${s.num < currentStep ? 'bg-primary' : 'bg-muted-foreground/20'}
                `}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Page Component
// ---------------------------------------------------------------------------

interface FormState {
  name: string
  display_name: string
  description: string
  categories: string[]
  bundle_url: string
  permissions: string[]
  min_plan: string
  icon_url: string
  screenshots: string[]
  readme: string
  pricing_type: string
  price_cents: number
  currency: string
}

export default function PublishWizardPage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [widgetId, setWidgetId] = useState<string | null>(null)
  const [form, setForm] = useState<FormState>({
    name: '',
    display_name: '',
    description: '',
    categories: [],
    bundle_url: '',
    permissions: [],
    min_plan: '',
    icon_url: '',
    screenshots: [],
    readme: '',
    pricing_type: 'free',
    price_cents: 0,
    currency: 'USD',
  })
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitSuccess, setSubmitSuccess] = useState(false)

  // --- field helpers -------------------------------------------------------

  const updateField = useCallback(
    <K extends keyof FormState>(key: K, value: FormState[K]) => {
      setForm((prev) => ({ ...prev, [key]: value }))
      setErrors((prev) => {
        const next = { ...prev }
        delete next[key]
        return next
      })
    },
    [],
  )

  const toggleArrayItem = useCallback(
    (key: 'categories' | 'permissions', value: string) => {
      setForm((prev) => {
        const arr = prev[key] as string[]
        return {
          ...prev,
          [key]: arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value],
        }
      })
    },
    [],
  )

  // --- screenshots helpers -------------------------------------------------

  const addScreenshot = useCallback(() => {
    setForm((prev) => {
      if (prev.screenshots.length >= 5) return prev
      return { ...prev, screenshots: [...prev.screenshots, ''] }
    })
  }, [])

  const updateScreenshot = useCallback((index: number, value: string) => {
    setForm((prev) => {
      const next = [...prev.screenshots]
      next[index] = value
      return { ...prev, screenshots: next }
    })
  }, [])

  const removeScreenshot = useCallback((index: number) => {
    setForm((prev) => ({
      ...prev,
      screenshots: prev.screenshots.filter((_, i) => i !== index),
    }))
  }, [])

  // --- validation ----------------------------------------------------------

  const validateStep = useCallback(
    (s: number): boolean => {
      const errs: Record<string, string> = {}

      if (s === 1) {
        if (!form.name.trim()) errs.name = 'Name is required'
        else if (!SLUG_RE.test(form.name)) errs.name = 'Must be lowercase slug format (e.g. my-widget)'
        if (!form.display_name.trim()) errs.display_name = 'Display name is required'
        if (!form.description.trim()) errs.description = 'Description is required'
        else if (form.description.length > 500) errs.description = 'Description must be 500 characters or fewer'
        if (form.categories.length === 0) errs.categories = 'Select at least one category'
      }

      if (s === 2) {
        if (form.bundle_url && !isValidUrl(form.bundle_url))
          errs.bundle_url = 'Must be a valid URL'
      }

      if (s === 3) {
        if (form.icon_url && !isValidUrl(form.icon_url))
          errs.icon_url = 'Must be a valid URL'
        form.screenshots.forEach((url, i) => {
          if (url && !isValidUrl(url))
            errs[`screenshot_${i}`] = 'Must be a valid URL'
        })
      }

      if (s === 4) {
        if (form.pricing_type !== 'free' && form.price_cents <= 0)
          errs.price_cents = 'Price must be greater than 0 for paid widgets'
      }

      setErrors(errs)
      return Object.keys(errs).length === 0
    },
    [form],
  )

  // --- API persistence -----------------------------------------------------

  const saveDraft = useCallback(async () => {
    setSaving(true)
    try {
      if (!widgetId) {
        // Create draft
        const data = await apiFetch<{ id: string }>(
          '/api/widget-marketplace/widgets',
          { method: 'POST', body: JSON.stringify(form) },
        )
        setWidgetId(data.id)
      } else {
        // Update draft
        await apiFetch(
          `/api/widget-marketplace/widgets/${widgetId}`,
          { method: 'PUT', body: JSON.stringify(form) },
        )
      }
    } catch (err) {
      console.error('Failed to save draft:', err)
    } finally {
      setSaving(false)
    }
  }, [form, widgetId])

  const submitForReview = useCallback(async () => {
    if (!widgetId) return
    setSubmitting(true)
    try {
      // Final save first
      await apiFetch(
        `/api/widget-marketplace/widgets/${widgetId}`,
        { method: 'PUT', body: JSON.stringify(form) },
      )
      // Submit for review
      await apiFetch(
        `/api/widget-marketplace/widgets/${widgetId}/submit`,
        { method: 'POST' },
      )
      setSubmitSuccess(true)
    } catch (err) {
      console.error('Failed to submit for review:', err)
    } finally {
      setSubmitting(false)
    }
  }, [form, widgetId])

  // --- step navigation -----------------------------------------------------

  const goNext = useCallback(async () => {
    if (!validateStep(step)) return

    // Auto-save: Create draft after step 1, update on subsequent steps
    if (step === 1 && !widgetId) {
      setSaving(true)
      try {
        const data = await apiFetch<{ id: string }>(
          '/api/widget-marketplace/widgets',
          { method: 'POST', body: JSON.stringify(form) },
        )
        setWidgetId(data.id)
      } catch (err) {
        console.error('Failed to create draft:', err)
        setSaving(false)
        return
      }
      setSaving(false)
    } else if (widgetId) {
      setSaving(true)
      try {
        await apiFetch(
          `/api/widget-marketplace/widgets/${widgetId}`,
          { method: 'PUT', body: JSON.stringify(form) },
        )
      } catch (err) {
        console.error('Failed to update draft:', err)
      }
      setSaving(false)
    }

    setStep((s) => Math.min(5, s + 1))
  }, [step, form, widgetId, validateStep])

  const goBack = useCallback(() => {
    setStep((s) => Math.max(1, s - 1))
  }, [])

  const handleStepClick = useCallback(
    (target: number) => {
      // Only allow going to completed steps or the next step
      if (target < step) setStep(target)
    },
    [step],
  )

  // --- success state -------------------------------------------------------

  if (submitSuccess) {
    return (
      <MainLayout>
        <div className="flex flex-col items-center justify-center min-h-[60vh] text-center space-y-6">
          <div className="w-20 h-20 rounded-full bg-primary/10 border-2 border-primary flex items-center justify-center">
            <Check className="w-10 h-10 text-primary" />
          </div>
          <div className="space-y-2">
            <h2 className="text-2xl font-bold">Widget Submitted for Review</h2>
            <p className="text-muted-foreground max-w-md">
              Your widget <span className="font-semibold text-foreground">{form.display_name}</span> has
              been submitted. Our team will review it and you&apos;ll be notified once it&apos;s approved.
            </p>
          </div>
          <div className="flex gap-3">
            <Button variant="outline" onClick={() => router.push('/marketplace/widgets')}>
              Back to Marketplace
            </Button>
            <Button onClick={() => router.push(`/marketplace/widgets/${widgetId}`)}>
              View Widget
            </Button>
          </div>
        </div>
      </MainLayout>
    )
  }

  // --- step renderers ------------------------------------------------------

  const renderStep1 = () => (
    <div className="space-y-6">
      {/* Name (slug) */}
      <div className="space-y-2">
        <Label htmlFor="name">
          Widget Name <span className="text-destructive">*</span>
        </Label>
        <Input
          id="name"
          placeholder="my-awesome-widget"
          value={form.name}
          onChange={(e) => updateField('name', e.target.value.toLowerCase().replace(/\s/g, '-'))}
          className={errors.name ? 'border-destructive' : ''}
        />
        <p className="text-xs text-muted-foreground">
          Lowercase slug format. Letters, numbers, and hyphens only.
        </p>
        {errors.name && <p className="text-xs text-destructive">{errors.name}</p>}
      </div>

      {/* Display Name */}
      <div className="space-y-2">
        <Label htmlFor="display_name">
          Display Name <span className="text-destructive">*</span>
        </Label>
        <Input
          id="display_name"
          placeholder="My Awesome Widget"
          value={form.display_name}
          onChange={(e) => updateField('display_name', e.target.value)}
          className={errors.display_name ? 'border-destructive' : ''}
        />
        {errors.display_name && (
          <p className="text-xs text-destructive">{errors.display_name}</p>
        )}
      </div>

      {/* Description */}
      <div className="space-y-2">
        <Label htmlFor="description">
          Description <span className="text-destructive">*</span>
        </Label>
        <Textarea
          id="description"
          placeholder="Describe what your widget does..."
          value={form.description}
          onChange={(e) => updateField('description', e.target.value)}
          className={`min-h-[120px] ${errors.description ? 'border-destructive' : ''}`}
          maxLength={500}
        />
        <div className="flex justify-between">
          <p className="text-xs text-muted-foreground">
            {form.description.length}/500 characters
          </p>
          {errors.description && (
            <p className="text-xs text-destructive">{errors.description}</p>
          )}
        </div>
      </div>

      {/* Categories */}
      <div className="space-y-3">
        <Label>
          Categories <span className="text-destructive">*</span>
        </Label>
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map((cat) => {
            const selected = form.categories.includes(cat)
            return (
              <Badge
                key={cat}
                role="button"
                tabIndex={0}
                className={`cursor-pointer capitalize transition-colors ${
                  selected
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary/50 text-muted-foreground hover:bg-secondary border border-border'
                }`}
                onClick={() => toggleArrayItem('categories', cat)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    toggleArrayItem('categories', cat)
                  }
                }}
              >
                {cat}
              </Badge>
            )
          })}
        </div>
        {errors.categories && (
          <p className="text-xs text-destructive">{errors.categories}</p>
        )}
      </div>
    </div>
  )

  const renderStep2 = () => (
    <div className="space-y-6">
      {/* Bundle URL */}
      <div className="space-y-2">
        <Label htmlFor="bundle_url">Bundle URL</Label>
        <Input
          id="bundle_url"
          type="url"
          placeholder="https://cdn.example.com/widgets/my-widget/bundle.js"
          value={form.bundle_url}
          onChange={(e) => updateField('bundle_url', e.target.value)}
          className={errors.bundle_url ? 'border-destructive' : ''}
        />
        <p className="text-xs text-muted-foreground">
          URL to the compiled widget bundle. Can be added later.
        </p>
        {errors.bundle_url && (
          <p className="text-xs text-destructive">{errors.bundle_url}</p>
        )}
      </div>

      {/* Permissions */}
      <div className="space-y-3">
        <Label>Permissions</Label>
        <p className="text-xs text-muted-foreground">
          Select the permissions your widget requires to function.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {PERMISSIONS.map((perm) => (
            <label
              key={perm.value}
              className={`
                flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors
                ${
                  form.permissions.includes(perm.value)
                    ? 'border-primary/50 bg-primary/5'
                    : 'border-border bg-secondary/20 hover:bg-secondary/40'
                }
              `}
            >
              <Checkbox
                checked={form.permissions.includes(perm.value)}
                onCheckedChange={() => toggleArrayItem('permissions', perm.value)}
              />
              <span className="text-sm">{perm.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Minimum Plan */}
      <div className="space-y-2">
        <Label htmlFor="min_plan">Minimum Plan</Label>
        <Select
          value={form.min_plan || undefined}
          onValueChange={(v) => updateField('min_plan', v)}
        >
          <SelectTrigger id="min_plan" className="w-full sm:w-[240px]">
            <SelectValue placeholder="Select minimum plan" />
          </SelectTrigger>
          <SelectContent>
            {MIN_PLANS.map((p) => (
              <SelectItem key={p.value} value={p.value}>
                {p.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          The minimum subscription plan required to install this widget.
        </p>
      </div>
    </div>
  )

  const renderStep3 = () => (
    <div className="space-y-6">
      {/* Icon URL */}
      <div className="space-y-2">
        <Label htmlFor="icon_url">Icon URL</Label>
        <div className="flex gap-3 items-start">
          <div className="flex-1 space-y-2">
            <Input
              id="icon_url"
              type="url"
              placeholder="https://example.com/icon.png"
              value={form.icon_url}
              onChange={(e) => updateField('icon_url', e.target.value)}
              className={errors.icon_url ? 'border-destructive' : ''}
            />
            {errors.icon_url && (
              <p className="text-xs text-destructive">{errors.icon_url}</p>
            )}
          </div>
          {/* Icon Preview */}
          <div className="w-16 h-16 rounded-lg border border-border bg-secondary/30 flex items-center justify-center overflow-hidden shrink-0">
            {form.icon_url && isValidUrl(form.icon_url) ? (
              <img
                src={form.icon_url}
                alt="Icon preview"
                className="w-full h-full object-cover"
                onError={(e) => {
                  ;(e.target as HTMLImageElement).style.display = 'none'
                }}
              />
            ) : (
              <ImageIcon className="w-6 h-6 text-muted-foreground" />
            )}
          </div>
        </div>
      </div>

      {/* Screenshots */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label>Screenshots</Label>
          <span className="text-xs text-muted-foreground">
            {form.screenshots.length}/5
          </span>
        </div>
        <div className="space-y-2">
          {form.screenshots.map((url, i) => (
            <div key={i} className="flex gap-2 items-center">
              <Input
                type="url"
                placeholder={`Screenshot URL ${i + 1}`}
                value={url}
                onChange={(e) => updateScreenshot(i, e.target.value)}
                className={errors[`screenshot_${i}`] ? 'border-destructive' : ''}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="shrink-0 text-muted-foreground hover:text-destructive"
                onClick={() => removeScreenshot(i)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          ))}
          {form.screenshots.length > 0 &&
            Object.entries(errors)
              .filter(([k]) => k.startsWith('screenshot_'))
              .map(([k, v]) => (
                <p key={k} className="text-xs text-destructive">{v}</p>
              ))}
        </div>
        {form.screenshots.length < 5 && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={addScreenshot}
            className="gap-1.5"
          >
            <Plus className="w-4 h-4" />
            Add Screenshot
          </Button>
        )}
      </div>

      {/* Readme */}
      <div className="space-y-2">
        <Label htmlFor="readme">Readme</Label>
        <Textarea
          id="readme"
          placeholder="# My Widget&#10;&#10;Describe your widget in detail. Supports **Markdown** formatting."
          value={form.readme}
          onChange={(e) => updateField('readme', e.target.value)}
          className="min-h-[200px] font-mono text-sm"
        />
        <p className="text-xs text-muted-foreground">
          Markdown supported. This appears on your widget&apos;s detail page.
        </p>
      </div>
    </div>
  )

  const renderStep4 = () => (
    <div className="space-y-6">
      {/* Pricing Type */}
      <div className="space-y-3">
        <Label>Pricing Type</Label>
        <RadioGroup
          value={form.pricing_type}
          onValueChange={(v) => {
            updateField('pricing_type', v)
            if (v === 'free') updateField('price_cents', 0)
          }}
          className="grid gap-3"
        >
          {[
            { value: 'free', label: 'Free', desc: 'Available at no cost to all users' },
            { value: 'one_time', label: 'One-time Purchase', desc: 'Single payment for permanent access' },
            { value: 'subscription', label: 'Subscription', desc: 'Recurring monthly payment' },
          ].map((opt) => (
            <label
              key={opt.value}
              className={`
                flex items-start gap-3 p-4 rounded-lg border cursor-pointer transition-colors
                ${
                  form.pricing_type === opt.value
                    ? 'border-primary/50 bg-primary/5'
                    : 'border-border bg-secondary/20 hover:bg-secondary/40'
                }
              `}
            >
              <RadioGroupItem value={opt.value} className="mt-0.5" />
              <div>
                <div className="text-sm font-medium">{opt.label}</div>
                <div className="text-xs text-muted-foreground">{opt.desc}</div>
              </div>
            </label>
          ))}
        </RadioGroup>
      </div>

      {/* Price (shown only for paid types) */}
      {form.pricing_type !== 'free' && (
        <div className="space-y-4 p-4 rounded-lg border border-border bg-secondary/10">
          <div className="space-y-2">
            <Label htmlFor="price">
              Price <span className="text-destructive">*</span>
            </Label>
            <div className="flex gap-3">
              <div className="relative flex-1 max-w-[200px]">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">
                  {form.currency === 'USD' ? '$' : form.currency === 'EUR' ? '\u20AC' : '\u00A3'}
                </span>
                <Input
                  id="price"
                  type="number"
                  min="0"
                  step="0.01"
                  placeholder="9.99"
                  value={form.price_cents > 0 ? (form.price_cents / 100).toFixed(2) : ''}
                  onChange={(e) => {
                    const cents = Math.round(parseFloat(e.target.value || '0') * 100)
                    updateField('price_cents', isNaN(cents) ? 0 : cents)
                  }}
                  className={`pl-7 ${errors.price_cents ? 'border-destructive' : ''}`}
                />
              </div>
              <Select
                value={form.currency}
                onValueChange={(v) => updateField('currency', v)}
              >
                <SelectTrigger className="w-[100px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {CURRENCIES.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {errors.price_cents && (
              <p className="text-xs text-destructive">{errors.price_cents}</p>
            )}
            {form.pricing_type === 'subscription' && (
              <p className="text-xs text-muted-foreground">
                Price is per month, billed monthly.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )

  const renderStep5 = () => {
    const sections = [
      {
        title: 'Basic Info',
        items: [
          { label: 'Name', value: form.name },
          { label: 'Display Name', value: form.display_name },
          { label: 'Description', value: form.description },
          {
            label: 'Categories',
            value: form.categories.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {form.categories.map((c) => (
                  <Badge key={c} variant="secondary" className="capitalize text-xs">
                    {c}
                  </Badge>
                ))}
              </div>
            ) : (
              <span className="text-muted-foreground italic">None</span>
            ),
          },
        ],
      },
      {
        title: 'Technical Details',
        items: [
          { label: 'Bundle URL', value: form.bundle_url || <span className="text-muted-foreground italic">Not set</span> },
          {
            label: 'Permissions',
            value: form.permissions.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {form.permissions.map((p) => (
                  <Badge key={p} variant="outline" className="text-xs">
                    {PERMISSIONS.find((x) => x.value === p)?.label || p}
                  </Badge>
                ))}
              </div>
            ) : (
              <span className="text-muted-foreground italic">None</span>
            ),
          },
          {
            label: 'Minimum Plan',
            value: form.min_plan
              ? MIN_PLANS.find((p) => p.value === form.min_plan)?.label || form.min_plan
              : <span className="text-muted-foreground italic">Not set</span>,
          },
        ],
      },
      {
        title: 'Media',
        items: [
          {
            label: 'Icon',
            value: form.icon_url ? (
              <div className="flex items-center gap-2">
                <img
                  src={form.icon_url}
                  alt=""
                  className="w-8 h-8 rounded object-cover"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
                />
                <span className="text-xs text-muted-foreground truncate max-w-[300px]">
                  {form.icon_url}
                </span>
              </div>
            ) : (
              <span className="text-muted-foreground italic">Not set</span>
            ),
          },
          {
            label: 'Screenshots',
            value: form.screenshots.filter(Boolean).length > 0
              ? `${form.screenshots.filter(Boolean).length} screenshot(s)`
              : <span className="text-muted-foreground italic">None</span>,
          },
          {
            label: 'Readme',
            value: form.readme
              ? `${form.readme.length} characters`
              : <span className="text-muted-foreground italic">Not written</span>,
          },
        ],
      },
      {
        title: 'Pricing',
        items: [
          {
            label: 'Type',
            value:
              form.pricing_type === 'free'
                ? 'Free'
                : form.pricing_type === 'one_time'
                  ? 'One-time Purchase'
                  : 'Subscription',
          },
          ...(form.pricing_type !== 'free'
            ? [{ label: 'Price', value: formatPrice(form.price_cents, form.currency) }]
            : []),
        ],
      },
    ]

    return (
      <div className="space-y-6">
        <p className="text-sm text-muted-foreground">
          Review your widget details before submitting for review.
        </p>

        {sections.map((section) => (
          <Card key={section.title} className="glass-card">
            <CardHeader className="pb-3">
              <h3 className="text-sm font-semibold">{section.title}</h3>
            </CardHeader>
            <CardContent className="space-y-3">
              {section.items.map((item) => (
                <div key={item.label} className="flex flex-col sm:flex-row sm:items-start gap-1 sm:gap-4">
                  <span className="text-xs font-medium text-muted-foreground w-32 shrink-0 pt-0.5">
                    {item.label}
                  </span>
                  <div className="text-sm flex-1 break-words">
                    {typeof item.value === 'string' ? (
                      <span className={item.value ? '' : 'text-muted-foreground italic'}>
                        {item.value || 'Not set'}
                      </span>
                    ) : (
                      item.value
                    )}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  // --- main render ---------------------------------------------------------

  return (
    <MainLayout>
      <div className="max-w-3xl mx-auto space-y-6">
        <PageHeader
          title="Publish"
          titleAccent="Widget"
          subtitle="Submit your widget to the marketplace for review"
        />

        <ProgressIndicator currentStep={step} onStepClick={handleStepClick} />

        <Card className="glass-card">
          <CardHeader className="pb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Package className="w-5 h-5 text-primary" />
              {STEPS[step - 1].label}
            </h2>
          </CardHeader>
          <CardContent>
            {step === 1 && renderStep1()}
            {step === 2 && renderStep2()}
            {step === 3 && renderStep3()}
            {step === 4 && renderStep4()}
            {step === 5 && renderStep5()}
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            onClick={goBack}
            disabled={step === 1}
            className="gap-1.5"
          >
            <ChevronLeft className="w-4 h-4" />
            Back
          </Button>

          <div className="flex items-center gap-3">
            {/* Save Draft */}
            {widgetId && step < 5 && (
              <Button
                variant="ghost"
                onClick={saveDraft}
                disabled={saving}
                className="gap-1.5 text-muted-foreground"
              >
                {saving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                Save Draft
              </Button>
            )}

            {step < 5 ? (
              <Button onClick={goNext} disabled={saving} className="gap-1.5">
                {saving ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    Next
                    <ChevronRight className="w-4 h-4" />
                  </>
                )}
              </Button>
            ) : (
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={saveDraft}
                  disabled={saving || submitting}
                  className="gap-1.5"
                >
                  {saving ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4" />
                  )}
                  Save Draft
                </Button>
                <Button
                  onClick={submitForReview}
                  disabled={saving || submitting}
                  className="gap-1.5"
                >
                  {submitting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                  Submit for Review
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
