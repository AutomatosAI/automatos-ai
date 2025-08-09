
'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  Activity, 
  Zap, 
  Waves, 
  Grid3x3, 
  Play, 
  Pause, 
  RefreshCw,
  Loader2,
  Eye,
  EyeOff,
  Settings
} from 'lucide-react'
import { useFieldOperations, useFieldInteractions, useFieldStatistics } from '@/hooks/use-field-theory'
import { toast } from 'sonner'

// Simple field visualization component
function FieldCanvas({ 
  fieldData, 
  dimensions = { width: 400, height: 300 }, 
  showVectors = true,
  fieldType = 'scalar'
}: {
  fieldData?: any
  dimensions?: { width: number; height: number }
  showVectors?: boolean
  fieldType?: 'scalar' | 'vector' | 'tensor'
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !fieldData) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height)

    // Draw field visualization based on field data
    if (fieldType === 'scalar') {
      drawScalarField(ctx, fieldData, dimensions)
    } else if (fieldType === 'vector' && showVectors) {
      drawVectorField(ctx, fieldData, dimensions)
    } else if (fieldType === 'tensor') {
      drawTensorField(ctx, fieldData, dimensions)
    }
  }, [fieldData, dimensions, showVectors, fieldType])

  const drawScalarField = (ctx: CanvasRenderingContext2D, data: any, dims: { width: number; height: number }) => {
    const gridSize = 20
    const field_value = data.field_value || 0.5

    for (let x = 0; x < dims.width; x += gridSize) {
      for (let y = 0; y < dims.height; y += gridSize) {
        // Create a field intensity based on distance from center and field value
        const centerX = dims.width / 2
        const centerY = dims.height / 2
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
        const maxDistance = Math.sqrt(centerX ** 2 + centerY ** 2)
        const intensity = field_value * (1 - distance / maxDistance)
        
        // Color based on field intensity
        const red = Math.floor(255 * intensity)
        const blue = Math.floor(255 * (1 - intensity))
        const green = Math.floor(128 * intensity)
        
        ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${0.3 + intensity * 0.7})`
        ctx.fillRect(x, y, gridSize - 1, gridSize - 1)
      }
    }
  }

  const drawVectorField = (ctx: CanvasRenderingContext2D, data: any, dims: { width: number; height: number }) => {
    const gridSize = 30
    const gradient = data.gradient || [1, 0, 0]

    for (let x = gridSize; x < dims.width - gridSize; x += gridSize) {
      for (let y = gridSize; y < dims.height - gridSize; y += gridSize) {
        // Draw vector arrows
        const vectorLength = 15
        const angle = Math.atan2(gradient[1], gradient[0])
        
        ctx.strokeStyle = '#ff6b35'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.lineTo(
          x + vectorLength * Math.cos(angle),
          y + vectorLength * Math.sin(angle)
        )
        ctx.stroke()

        // Arrow head
        ctx.beginPath()
        ctx.moveTo(x + vectorLength * Math.cos(angle), y + vectorLength * Math.sin(angle))
        ctx.lineTo(
          x + vectorLength * Math.cos(angle) - 5 * Math.cos(angle - Math.PI / 6),
          y + vectorLength * Math.sin(angle) - 5 * Math.sin(angle - Math.PI / 6)
        )
        ctx.moveTo(x + vectorLength * Math.cos(angle), y + vectorLength * Math.sin(angle))
        ctx.lineTo(
          x + vectorLength * Math.cos(angle) - 5 * Math.cos(angle + Math.PI / 6),
          y + vectorLength * Math.sin(angle) - 5 * Math.sin(angle + Math.PI / 6)
        )
        ctx.stroke()
      }
    }
  }

  const drawTensorField = (ctx: CanvasRenderingContext2D, data: any, dims: { width: number; height: number }) => {
    // Draw tensor field as ellipses representing tensor components
    const gridSize = 40

    for (let x = gridSize; x < dims.width - gridSize; x += gridSize) {
      for (let y = gridSize; y < dims.height - gridSize; y += gridSize) {
        ctx.strokeStyle = '#60B5FF'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.ellipse(x, y, 10, 5, Math.PI / 4, 0, 2 * Math.PI)
        ctx.stroke()
      }
    }
  }

  return (
    <canvas
      ref={canvasRef}
      width={dimensions.width}
      height={dimensions.height}
      className="border border-border/30 rounded-lg bg-secondary/10"
    />
  )
}

export function FieldVisualization() {
  const [sessionId] = useState(() => `session_${Date.now()}`)
  const [fieldType, setFieldType] = useState<'scalar' | 'vector' | 'tensor'>('scalar')
  const [showVectors, setShowVectors] = useState(true)
  const [propagationSteps, setPropagationSteps] = useState([3])
  const [isPlaying, setIsPlaying] = useState(false)
  const [context, setContext] = useState('Advanced multi-agent coordination and field theory integration')

  // Field theory hooks
  const { fieldData, updateField, propagateField, optimizeField, isLoading: fieldLoading } = useFieldOperations()
  const { interactionData, modelInteractions, dynamicManagement, isLoading: interactionLoading } = useFieldInteractions()
  const { statistics, isLoading: statsLoading, refetch } = useFieldStatistics()

  // Initialize field on component mount
  useEffect(() => {
    handleUpdateField()
  }, [])

  const handleUpdateField = async () => {
    try {
      await updateField(
        sessionId,
        context,
        fieldType,
        fieldType === 'vector' ? [1, 0.5, -0.3] : undefined
      )
      toast.success('Field updated successfully!')
    } catch (error) {
      toast.error('Failed to update field')
    }
  }

  const handlePropagateField = async () => {
    try {
      await propagateField(sessionId, propagationSteps[0])
      toast.success('Field propagated successfully!')
    } catch (error) {
      toast.error('Failed to propagate field')
    }
  }

  const handleOptimizeField = async () => {
    try {
      await optimizeField(sessionId, ['performance', 'stability', 'efficiency'])
      toast.success('Field optimized successfully!')
    } catch (error) {
      toast.error('Failed to optimize field')
    }
  }

  const handleModelInteractions = async () => {
    try {
      const session2 = `session_${Date.now()}_2`
      await updateField(session2, 'Secondary context for interaction modeling', fieldType)
      await modelInteractions(sessionId, session2, 'similarity')
      toast.success('Field interactions modeled successfully!')
    } catch (error) {
      toast.error('Failed to model interactions')
    }
  }

  const togglePlayback = () => {
    setIsPlaying(!isPlaying)
    if (!isPlaying) {
      // Start automatic field updates
      const interval = setInterval(async () => {
        await handlePropagateField()
      }, 2000)
      
      setTimeout(() => {
        clearInterval(interval)
        setIsPlaying(false)
      }, 10000) // Stop after 10 seconds
    }
  }

  const isLoading = fieldLoading || interactionLoading || statsLoading

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Field Theory Visualization</h2>
          <p className="text-muted-foreground">
            Visualize and interact with field-based context representations
          </p>
          {statistics?.fieldStats && (
            <div className="flex items-center space-x-4 mt-2 text-sm text-muted-foreground">
              <span>Active Fields: {statistics.fieldStats.active_fields || 1}</span>
              <span>Field Value: {fieldData?.field_value?.toFixed(3) || 'N/A'}</span>
              <span>Stability: {fieldData?.stability?.toFixed(3) || 'N/A'}</span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            onClick={refetch}
            disabled={isLoading}
            className="hover:border-primary/50"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            variant={isPlaying ? 'secondary' : 'default'}
            onClick={togglePlayback}
            className={isPlaying ? '' : 'gradient-accent hover:opacity-90'}
          >
            {isPlaying ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
            {isPlaying ? 'Stop' : 'Animate'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Field Visualization */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center">
                <Grid3x3 className="w-5 h-5 mr-2" />
                Field Visualization
              </span>
              <Badge className={`${fieldType === 'scalar' ? 'bg-orange-500/10 text-orange-400' : 
                                 fieldType === 'vector' ? 'bg-blue-500/10 text-blue-400' : 
                                 'bg-purple-500/10 text-purple-400'} border-current/20`}>
                {fieldType}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-center">
              <FieldCanvas
                fieldData={fieldData}
                fieldType={fieldType}
                showVectors={showVectors}
                dimensions={{ width: 400, height: 300 }}
              />
            </div>
            
            {isLoading && (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                <span className="ml-2 text-muted-foreground">Processing field operations...</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Field Controls */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="w-5 h-5 mr-2" />
              Field Controls
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Field Type Selection */}
            <div className="space-y-2">
              <Label>Field Type</Label>
              <Select value={fieldType} onValueChange={(value: any) => setFieldType(value)}>
                <SelectTrigger className="bg-secondary/50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="scalar">Scalar Field</SelectItem>
                  <SelectItem value="vector">Vector Field</SelectItem>
                  <SelectItem value="tensor">Tensor Field</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Propagation Steps */}
            <div className="space-y-2">
              <Label>Propagation Steps: {propagationSteps[0]}</Label>
              <Slider
                value={propagationSteps}
                onValueChange={setPropagationSteps}
                max={10}
                min={1}
                step={1}
                className="w-full"
              />
            </div>

            {/* Vector Field Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <Label htmlFor="show-vectors">Show Vectors</Label>
                <p className="text-sm text-muted-foreground">Display vector field components</p>
              </div>
              <Switch
                id="show-vectors"
                checked={showVectors}
                onCheckedChange={setShowVectors}
              />
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-3">
              <Button
                onClick={handleUpdateField}
                disabled={isLoading}
                className="gradient-accent hover:opacity-90"
              >
                {isLoading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Waves className="w-4 h-4 mr-2" />}
                Update
              </Button>
              <Button
                onClick={handlePropagateField}
                disabled={isLoading}
                variant="outline"
                className="hover:border-primary/50"
              >
                {isLoading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Activity className="w-4 h-4 mr-2" />}
                Propagate
              </Button>
              <Button
                onClick={handleOptimizeField}
                disabled={isLoading}
                variant="outline"
                className="hover:border-primary/50"
              >
                {isLoading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Zap className="w-4 h-4 mr-2" />}
                Optimize
              </Button>
              <Button
                onClick={handleModelInteractions}
                disabled={isLoading}
                variant="outline"
                className="hover:border-primary/50"
              >
                {isLoading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Grid3x3 className="w-4 h-4 mr-2" />}
                Interact
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Field Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Waves className="w-8 h-8 text-orange-400" />
              <Badge variant="outline">Field Value</Badge>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">
                {fieldData?.field_value?.toFixed(3) || '0.000'}
              </h3>
              <p className="text-muted-foreground text-sm">Current field strength</p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-green-400" />
              <Badge variant="outline">Stability</Badge>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">
                {fieldData?.stability?.toFixed(3) || 'N/A'}
              </h3>
              <p className="text-muted-foreground text-sm">Field stability measure</p>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <Zap className="w-8 h-8 text-blue-400" />
              <Badge variant="outline">Interactions</Badge>
            </div>
            <div className="space-y-1">
              <h3 className="text-2xl font-bold">
                {statistics?.interactions?.length || 0}
              </h3>
              <p className="text-muted-foreground text-sm">Active field interactions</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Real-time Field Data */}
      {fieldData && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Field Data</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-semibold mb-2">Field Properties</h4>
                <div className="space-y-1 text-muted-foreground">
                  <p>Type: {fieldType}</p>
                  <p>Value: {fieldData.field_value?.toFixed(6) || 'N/A'}</p>
                  <p>Stability: {fieldData.stability?.toFixed(6) || 'N/A'}</p>
                  {fieldData.gradient && (
                    <p>Gradient: [{fieldData.gradient.map((g: number) => g.toFixed(3)).join(', ')}]</p>
                  )}
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-2">System Status</h4>
                <div className="space-y-1 text-muted-foreground">
                  <p>Session ID: {sessionId}</p>
                  <p>Last Updated: {new Date().toLocaleTimeString()}</p>
                  <p>Propagation Steps: {propagationSteps[0]}</p>
                  <p>Status: {isLoading ? 'Processing...' : 'Ready'}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
