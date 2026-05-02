'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Upload,
  FileArchive,
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  Link2,
  ArrowLeft,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { usePageAPI } from '@/hooks/use-page-api'
import { apiClient } from '@/lib/api-client'

// ===================================================================
// Types
// ===================================================================

interface ScanFinding {
  type?: string
  category?: string
  severity: string
  file?: string
  line?: number
  pattern?: string
  matched_text?: string
  description?: string
  evidence?: string
}

interface ScanResult {
  id: string
  plugin_slug: string
  plugin_version: string
  static_scan_status: string | null
  static_findings: ScanFinding[] | null
  blocked_patterns_found: string[] | null
  llm_scan_status: string | null
  llm_risk_score: number | null
  llm_findings: ScanFinding[] | null
  llm_summary: string | null
  llm_model_used: string | null
  llm_tokens_used: number | null
  overall_verdict: string | null
  scanned_at: string | null
  scanned_by: string | null
}

interface UploadResult {
  plugin_id: string
  scan_id: string | null
  slug: string
  name: string
  version: string
  security_status: string
  approval_status: string
  status: string
}

type UploadMethod = 'zip' | 'github'

type ScanPhase =
  | 'idle'
  | 'uploading'
  | 'extracting'
  | 'validating'
  | 'static_scan'
  | 'llm_scan'
  | 'uploading_s3'
  | 'complete'
  | 'error'

const PHASE_LABELS: Record<ScanPhase, string> = {
  idle: 'Ready',
  uploading: 'Uploading file...',
  extracting: 'Extracting archive...',
  validating: 'Validating manifest...',
  static_scan: 'Running static analysis...',
  llm_scan: 'Running LLM security scan...',
  uploading_s3: 'Uploading to S3...',
  complete: 'Scan complete',
  error: 'Error occurred',
}

const PHASE_PROGRESS: Record<ScanPhase, number> = {
  idle: 0,
  uploading: 15,
  extracting: 30,
  validating: 40,
  static_scan: 55,
  llm_scan: 75,
  uploading_s3: 90,
  complete: 100,
  error: 0,
}

// ===================================================================
// Helpers
// ===================================================================

function verdictBadge(verdict: string | null) {
  switch (verdict) {
    case 'safe':
      return (
        <Badge className="bg-success/10 text-success border-success/20">
          <ShieldCheck className="h-3 w-3 mr-1" />
          Safe
        </Badge>
      )
    case 'review_required':
      return (
        <Badge className="bg-warning/10 text-warning border-warning/20">
          <ShieldAlert className="h-3 w-3 mr-1" />
          Review Required
        </Badge>
      )
    case 'blocked':
      return (
        <Badge className="bg-destructive/20 text-destructive border-destructive/30">
          <ShieldX className="h-3 w-3 mr-1" />
          Blocked
        </Badge>
      )
    default:
      return (
        <Badge className="bg-secondary/50 text-muted-foreground border-border/30">
          <Shield className="h-3 w-3 mr-1" />
          Pending
        </Badge>
      )
  }
}

function severityColor(severity: string) {
  switch (severity) {
    case 'critical':
      return 'text-destructive'
    case 'high':
      return 'text-orange-400'
    case 'medium':
      return 'text-warning'
    case 'low':
      return 'text-info'
    default:
      return 'text-muted-foreground'
  }
}

function formatFileSize(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${bytes} B`
}

// ===================================================================
// Component
// ===================================================================

export default function AdminPluginUploadPage() {
  usePageAPI('admin')

  // Upload method
  const [uploadMethod, setUploadMethod] = useState<UploadMethod>('zip')
  const [githubUrl, setGithubUrl] = useState('')

  // File state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  // Scan progress
  const [scanPhase, setScanPhase] = useState<ScanPhase>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Results
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null)
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)

  // Approve/Reject
  const [actionLoading, setActionLoading] = useState(false)
  const [actionDone, setActionDone] = useState<'approved' | 'rejected' | null>(null)

  // Expand findings
  const [showStaticFindings, setShowStaticFindings] = useState(false)
  const [showLlmFindings, setShowLlmFindings] = useState(false)

  // Dropzone
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      if (file.size > 10 * 1024 * 1024) {
        setErrorMessage('File too large. Maximum size is 10 MB.')
        return
      }
      setSelectedFile(file)
      setErrorMessage(null)
      setScanPhase('idle')
      setUploadResult(null)
      setScanResult(null)
      setActionDone(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/zip': ['.zip'] },
    maxFiles: 1,
    disabled: scanPhase !== 'idle' && scanPhase !== 'complete' && scanPhase !== 'error',
  })

  // Upload & Scan
  async function handleUploadAndScan() {
    if (uploadMethod === 'zip' && !selectedFile) return
    if (uploadMethod === 'github' && !githubUrl.trim()) return

    setErrorMessage(null)
    setScanResult(null)
    setUploadResult(null)
    setActionDone(null)

    try {
      // Phase 1: Upload
      setScanPhase('uploading')

      const formData = new FormData()
      if (uploadMethod === 'zip' && selectedFile) {
        formData.append('file', selectedFile)
        formData.append('source_type', 'upload')
      } else {
        // For GitHub URL, we still need a file — create a placeholder
        // The backend expects a zip file; for GitHub, admin should download & upload
        formData.append('source_type', 'github')
        formData.append('source_url', githubUrl)
        // Cannot upload without a file — show error
        setErrorMessage('GitHub URL import requires a downloaded zip file. Please download the repo as a zip and upload it.')
        setScanPhase('error')
        return
      }

      // Simulate progress phases
      setScanPhase('extracting')

      const headers = await apiClient.getAuthHeaders()
      // Don't set Content-Type — browser sets multipart boundary
      delete (headers as Record<string, string>)['Content-Type']

      const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || ''
      const response = await fetch(`${BACKEND_URL}/api/admin/plugins/upload`, {
        method: 'POST',
        headers,
        body: formData,
      })

      setScanPhase('validating')

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(errData.detail || `Upload failed: ${response.statusText}`)
      }

      const result: UploadResult = await response.json()
      setUploadResult(result)

      setScanPhase('static_scan')

      // Fetch scan results
      if (result.scan_id) {
        setScanPhase('llm_scan')

        const scanData: ScanResult = await apiClient.request(
          `/api/admin/plugins/${result.plugin_id}/scan`
        )
        setScanResult(scanData)

        setScanPhase('uploading_s3')
      }

      setScanPhase('complete')
    } catch (err: any) {
      console.error('Upload error:', err)
      setErrorMessage(err?.message || 'An unexpected error occurred')
      setScanPhase('error')
    }
  }

  // Approve
  async function handleApprove() {
    if (!uploadResult) return
    setActionLoading(true)
    try {
      await apiClient.request(`/api/admin/plugins/${uploadResult.plugin_id}/approve`, {
        method: 'POST',
      })
      setActionDone('approved')
    } catch (err: any) {
      setErrorMessage(err?.message || 'Failed to approve plugin')
    } finally {
      setActionLoading(false)
    }
  }

  // Reject
  async function handleReject() {
    if (!uploadResult) return
    const reason = window.prompt('Enter rejection reason:')
    if (!reason) return

    setActionLoading(true)
    try {
      await apiClient.request(`/api/admin/plugins/${uploadResult.plugin_id}/reject`, {
        method: 'POST',
        body: { reason },
      })
      setActionDone('rejected')
    } catch (err: any) {
      setErrorMessage(err?.message || 'Failed to reject plugin')
    } finally {
      setActionLoading(false)
    }
  }

  // Reset
  function handleReset() {
    setSelectedFile(null)
    setScanPhase('idle')
    setErrorMessage(null)
    setUploadResult(null)
    setScanResult(null)
    setActionDone(null)
    setGithubUrl('')
    setShowStaticFindings(false)
    setShowLlmFindings(false)
  }

  const isScanning = scanPhase !== 'idle' && scanPhase !== 'complete' && scanPhase !== 'error'

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => window.history.back()}
            className="shrink-0"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold">
              Plugin <span className="gradient-text">Upload</span>
            </h1>
            <p className="text-muted-foreground mt-1">
              Upload, scan, and approve plugins for the marketplace
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Upload */}
          <div className="space-y-6">
            {/* Upload Method Selection */}
            <Card className="glass-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg">Upload Method</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Button
                    variant={uploadMethod === 'zip' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setUploadMethod('zip')}
                    disabled={isScanning}
                  >
                    <FileArchive className="h-4 w-4 mr-2" />
                    Zip File
                  </Button>
                  <Button
                    variant={uploadMethod === 'github' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setUploadMethod('github')}
                    disabled={isScanning}
                  >
                    <Link2 className="h-4 w-4 mr-2" />
                    GitHub URL
                  </Button>
                </div>

                {uploadMethod === 'zip' ? (
                  <>
                    {/* Dropzone */}
                    <div
                      {...getRootProps()}
                      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                        isDragActive
                          ? 'border-orange-500 bg-orange-500/5'
                          : selectedFile
                          ? 'border-success/50 bg-success/5'
                          : 'border-border hover:border-orange-500/50'
                      } ${isScanning ? 'opacity-50 pointer-events-none' : ''}`}
                    >
                      <input {...getInputProps()} />
                      {selectedFile ? (
                        <div className="space-y-2">
                          <FileArchive className="h-10 w-10 mx-auto text-success" />
                          <p className="font-medium text-white">{selectedFile.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {formatFileSize(selectedFile.size)}
                          </p>
                          {!isScanning && (
                            <p className="text-xs text-muted-foreground">
                              Drop a new file to replace
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Upload className="h-10 w-10 mx-auto text-muted-foreground" />
                          <p className="font-medium text-white">
                            {isDragActive
                              ? 'Drop your plugin zip here'
                              : 'Drag & drop your plugin zip'}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            or click to browse (.zip only, max 10 MB)
                          </p>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="space-y-2">
                    <Input
                      type="url"
                      placeholder="https://github.com/user/repo"
                      value={githubUrl}
                      onChange={(e) => setGithubUrl(e.target.value)}
                      disabled={isScanning}
                      className="bg-secondary/50 border-secondary"
                    />
                    <p className="text-xs text-muted-foreground">
                      Enter a GitHub repository URL. The repo must contain a manifest.json at root.
                    </p>
                  </div>
                )}

                {/* Upload Button */}
                <Button
                  className="w-full"
                  onClick={handleUploadAndScan}
                  disabled={
                    isScanning ||
                    (uploadMethod === 'zip' && !selectedFile) ||
                    (uploadMethod === 'github' && !githubUrl.trim())
                  }
                >
                  {isScanning ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Shield className="h-4 w-4 mr-2" />
                      Upload & Scan
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Scan Progress */}
            {scanPhase !== 'idle' && (
              <Card className="glass-card">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Scan Progress</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Progress value={PHASE_PROGRESS[scanPhase]} />
                  <div className="space-y-2">
                    {(
                      [
                        'uploading',
                        'extracting',
                        'validating',
                        'static_scan',
                        'llm_scan',
                        'uploading_s3',
                        'complete',
                      ] as ScanPhase[]
                    ).map((phase) => {
                      const currentIdx = Object.keys(PHASE_PROGRESS).indexOf(scanPhase)
                      const phaseIdx = Object.keys(PHASE_PROGRESS).indexOf(phase)
                      const isActive = scanPhase === phase
                      const isDone = currentIdx > phaseIdx
                      const isError = scanPhase === 'error'

                      return (
                        <div
                          key={phase}
                          className={`flex items-center gap-2 text-sm ${
                            isActive
                              ? 'text-orange-400'
                              : isDone
                              ? 'text-success'
                              : isError && isActive
                              ? 'text-destructive'
                              : 'text-muted-foreground'
                          }`}
                        >
                          {isDone ? (
                            <CheckCircle2 className="h-4 w-4 shrink-0" />
                          ) : isActive && !isError ? (
                            <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                          ) : (
                            <div className="h-4 w-4 rounded-full border border-current shrink-0" />
                          )}
                          {PHASE_LABELS[phase]}
                        </div>
                      )
                    })}
                  </div>

                  {errorMessage && (
                    <div className="flex items-start gap-2 p-3 rounded-md bg-destructive/10 border border-destructive/30 text-destructive text-sm">
                      <XCircle className="h-4 w-4 mt-0.5 shrink-0" />
                      <span>{errorMessage}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="space-y-6">
            {/* Upload Result */}
            {uploadResult && (
              <Card className="glass-card">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">Plugin Info</CardTitle>
                    {verdictBadge(scanResult?.overall_verdict || null)}
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-muted-foreground">Name</span>
                      <p className="text-white font-medium">{uploadResult.name}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Version</span>
                      <p className="text-white font-medium">{uploadResult.version}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Slug</span>
                      <p className="text-white font-medium font-mono text-xs">
                        {uploadResult.slug}
                      </p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Security Status</span>
                      <p className="text-white font-medium capitalize">
                        {uploadResult.security_status}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Scan Results */}
            {scanResult && (
              <>
                {/* Risk Score */}
                <Card className="glass-card">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg">Security Scan Results</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Risk Score */}
                    {scanResult.llm_risk_score !== null && (
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-muted-foreground">Risk Score</span>
                          <span
                            className={`text-2xl font-bold ${
                              scanResult.llm_risk_score < 20
                                ? 'text-success'
                                : scanResult.llm_risk_score < 70
                                ? 'text-warning'
                                : 'text-destructive'
                            }`}
                          >
                            {scanResult.llm_risk_score}/100
                          </span>
                        </div>
                        <Progress
                          value={scanResult.llm_risk_score}
                          className={
                            scanResult.llm_risk_score < 20
                              ? '[&>div]:bg-green-500'
                              : scanResult.llm_risk_score < 70
                              ? '[&>div]:bg-yellow-500'
                              : '[&>div]:bg-red-500'
                          }
                        />
                      </div>
                    )}

                    {/* LLM Summary */}
                    {scanResult.llm_summary && (
                      <div>
                        <span className="text-sm text-muted-foreground">LLM Analysis Summary</span>
                        <p className="text-sm text-white mt-1">{scanResult.llm_summary}</p>
                        {scanResult.llm_model_used && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Model: {scanResult.llm_model_used}
                            {scanResult.llm_tokens_used
                              ? ` | Tokens: ${scanResult.llm_tokens_used}`
                              : ''}
                          </p>
                        )}
                      </div>
                    )}

                    {/* Static Scan Summary */}
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm text-muted-foreground">Static Analysis</span>
                        <p className="text-sm text-white capitalize">
                          {scanResult.static_scan_status || 'N/A'}
                        </p>
                      </div>
                      <Badge
                        variant="outline"
                        className={`${
                          scanResult.static_scan_status === 'passed'
                            ? 'border-success/30 text-success'
                            : 'border-warning/30 text-warning'
                        }`}
                      >
                        {scanResult.static_findings?.length || 0} findings
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                {/* Static Findings */}
                {scanResult.static_findings && scanResult.static_findings.length > 0 && (
                  <Card className="glass-card">
                    <CardHeader className="pb-3">
                      <button
                        className="flex items-center justify-between w-full text-left"
                        onClick={() => setShowStaticFindings(!showStaticFindings)}
                      >
                        <CardTitle className="text-lg">
                          Static Findings ({scanResult.static_findings.length})
                        </CardTitle>
                        {showStaticFindings ? (
                          <ChevronUp className="h-5 w-5 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-muted-foreground" />
                        )}
                      </button>
                    </CardHeader>
                    {showStaticFindings && (
                      <CardContent className="space-y-2 max-h-64 overflow-y-auto">
                        {scanResult.static_findings.map((finding, idx) => (
                          <div
                            key={idx}
                            className="p-2 rounded bg-secondary/50 border border-secondary text-sm"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <AlertTriangle
                                className={`h-3 w-3 ${severityColor(finding.severity)}`}
                              />
                              <span className={`font-medium ${severityColor(finding.severity)}`}>
                                {finding.severity}
                              </span>
                              <span className="text-muted-foreground">|</span>
                              <span className="text-foreground/90">{finding.type || finding.category}</span>
                            </div>
                            <p className="text-foreground/90">{finding.description}</p>
                            {finding.file && (
                              <p className="text-xs text-muted-foreground mt-1 font-mono">
                                {finding.file}
                                {finding.line ? `:${finding.line}` : ''}
                              </p>
                            )}
                            {finding.matched_text && (
                              <p className="text-xs text-muted-foreground mt-1 font-mono truncate">
                                Match: {finding.matched_text}
                              </p>
                            )}
                          </div>
                        ))}
                      </CardContent>
                    )}
                  </Card>
                )}

                {/* LLM Findings */}
                {scanResult.llm_findings && scanResult.llm_findings.length > 0 && (
                  <Card className="glass-card">
                    <CardHeader className="pb-3">
                      <button
                        className="flex items-center justify-between w-full text-left"
                        onClick={() => setShowLlmFindings(!showLlmFindings)}
                      >
                        <CardTitle className="text-lg">
                          LLM Findings ({scanResult.llm_findings.length})
                        </CardTitle>
                        {showLlmFindings ? (
                          <ChevronUp className="h-5 w-5 text-muted-foreground" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-muted-foreground" />
                        )}
                      </button>
                    </CardHeader>
                    {showLlmFindings && (
                      <CardContent className="space-y-2 max-h-64 overflow-y-auto">
                        {scanResult.llm_findings.map((finding, idx) => (
                          <div
                            key={idx}
                            className="p-2 rounded bg-secondary/50 border border-secondary text-sm"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <AlertTriangle
                                className={`h-3 w-3 ${severityColor(finding.severity)}`}
                              />
                              <span className={`font-medium ${severityColor(finding.severity)}`}>
                                {finding.severity}
                              </span>
                              <span className="text-muted-foreground">|</span>
                              <span className="text-foreground/90">
                                {finding.category || finding.type}
                              </span>
                            </div>
                            <p className="text-foreground/90">{finding.description}</p>
                            {finding.file && (
                              <p className="text-xs text-muted-foreground mt-1 font-mono">
                                {finding.file}
                              </p>
                            )}
                            {finding.evidence && (
                              <p className="text-xs text-muted-foreground mt-1 font-mono truncate">
                                Evidence: {finding.evidence}
                              </p>
                            )}
                          </div>
                        ))}
                      </CardContent>
                    )}
                  </Card>
                )}
              </>
            )}

            {/* Approve / Reject Actions */}
            {scanPhase === 'complete' && uploadResult && !actionDone && (
              <Card className="glass-card">
                <CardContent className="p-4">
                  <div className="flex gap-3">
                    <Button
                      className="flex-1 bg-success hover:bg-success/80 text-white"
                      onClick={handleApprove}
                      disabled={actionLoading}
                    >
                      {actionLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4 mr-2" />
                      )}
                      Approve
                    </Button>
                    <Button
                      variant="destructive"
                      className="flex-1"
                      onClick={handleReject}
                      disabled={actionLoading}
                    >
                      {actionLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <XCircle className="h-4 w-4 mr-2" />
                      )}
                      Reject
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Action Result */}
            {actionDone && (
              <Card className="glass-card">
                <CardContent className="p-4">
                  <div
                    className={`flex items-center gap-2 ${
                      actionDone === 'approved' ? 'text-success' : 'text-destructive'
                    }`}
                  >
                    {actionDone === 'approved' ? (
                      <CheckCircle2 className="h-5 w-5" />
                    ) : (
                      <XCircle className="h-5 w-5" />
                    )}
                    <span className="font-medium">
                      Plugin {actionDone === 'approved' ? 'approved' : 'rejected'} successfully
                    </span>
                  </div>
                  <Button
                    variant="outline"
                    className="w-full mt-3"
                    onClick={handleReset}
                  >
                    Upload Another Plugin
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Error state reset */}
            {scanPhase === 'error' && (
              <Card className="glass-card">
                <CardContent className="p-4">
                  <Button variant="outline" className="w-full" onClick={handleReset}>
                    Try Again
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
