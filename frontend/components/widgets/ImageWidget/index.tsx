'use client'

/**
 * ImageWidget Component for PRD-38.1 Widget Architecture
 *
 * Displays images with zoom, rotation, pan controls,
 * and download functionality.
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import {
  Image as ImageIcon,
  Download,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize2,
  RefreshCw,
} from 'lucide-react'
import { WidgetBase } from '../WidgetBase'
import { registerWidget } from '../registry'
import type { WidgetBaseProps, ImageWidgetData, WidgetDefinition } from '../types'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'

export function ImageWidget({
  id,
  title,
  data,
  metadata,
  isActive,
  isLoading,
  error,
  onClose,
  onMaximize,
}: WidgetBaseProps<ImageWidgetData>) {
  const [zoom, setZoom] = useState(1)
  const [rotation, setRotation] = useState(0)
  const [imageError, setImageError] = useState(false)
  const [isImageLoading, setIsImageLoading] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)

  // Reset state when data changes
  useEffect(() => {
    setZoom(1)
    setRotation(0)
    setImageError(false)
    setIsImageLoading(true)
  }, [data.src])

  // Handle download
  const handleDownload = useCallback(async () => {
    try {
      // For base64 images, create download link directly
      if (data.src.startsWith('data:')) {
        const link = document.createElement('a')
        link.href = data.src
        link.download = `${title.replace(/\s+/g, '_')}.png`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        toast.success('Image downloaded')
        return
      }

      // For URLs, fetch and download
      const response = await fetch(data.src)
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${title.replace(/\s+/g, '_')}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      toast.success('Image downloaded')
    } catch (err) {
      toast.error('Failed to download image')
    }
  }, [data.src, title])

  // Zoom controls
  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 4))
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.25))
  const handleRotate = () => setRotation((r) => (r + 90) % 360)
  const handleReset = () => {
    setZoom(1)
    setRotation(0)
  }

  // Handle image load
  const handleImageLoad = () => {
    setIsImageLoading(false)
    setImageError(false)
  }

  // Handle image error
  const handleImageError = () => {
    setIsImageLoading(false)
    setImageError(true)
  }

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      const delta = e.deltaY > 0 ? -0.1 : 0.1
      setZoom((z) => Math.min(Math.max(z + delta, 0.25), 4))
    }
  }, [])

  return (
    <WidgetBase
      title={title}
      icon={<ImageIcon className="h-4 w-4" />}
      metadata={metadata}
      isActive={isActive}
      isLoading={isLoading}
      error={error}
      onClose={onClose}
      onMaximize={onMaximize}
      onDownload={handleDownload}
      canDownload
      customActions={[
        {
          label: 'Zoom In',
          icon: <ZoomIn className="h-4 w-4 mr-2" />,
          onClick: handleZoomIn,
        },
        {
          label: 'Zoom Out',
          icon: <ZoomOut className="h-4 w-4 mr-2" />,
          onClick: handleZoomOut,
        },
        {
          label: 'Rotate',
          icon: <RotateCw className="h-4 w-4 mr-2" />,
          onClick: handleRotate,
        },
        {
          label: 'Reset View',
          icon: <RefreshCw className="h-4 w-4 mr-2" />,
          onClick: handleReset,
        },
      ]}
    >
      <div className="flex flex-col h-full">
        {/* Prompt info for generated images */}
        {data.prompt && (
          <div className="px-3 py-2 bg-muted/30 border-b border-border/30">
            <p className="text-xs text-muted-foreground line-clamp-2">
              <span className="font-medium text-foreground">Prompt:</span>{' '}
              {data.prompt}
            </p>
            {data.model && (
              <p className="text-xs text-muted-foreground mt-1">
                <span className="font-medium text-foreground">Model:</span>{' '}
                {data.model}
              </p>
            )}
          </div>
        )}

        {/* Image viewport */}
        <div
          ref={containerRef}
          className="flex-1 overflow-hidden flex items-center justify-center bg-muted/10 relative"
          onDoubleClick={handleReset}
          onWheel={handleWheel}
        >
          {isImageLoading && !imageError && (
            <div className="absolute inset-0 flex items-center justify-center bg-muted/20">
              <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          )}

          {imageError ? (
            <div className="text-center text-muted-foreground p-4">
              <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm font-medium">Failed to load image</p>
              <p className="text-xs mt-1">The image could not be displayed</p>
            </div>
          ) : (
            <img
              src={data.src}
              alt={data.alt || title}
              style={{
                transform: `scale(${zoom}) rotate(${rotation}deg)`,
                transition: 'transform 0.2s ease',
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
              }}
              className={cn(
                'rounded shadow-lg',
                isImageLoading && 'opacity-0'
              )}
              onLoad={handleImageLoad}
              onError={handleImageError}
              draggable={false}
            />
          )}
        </div>

        {/* Controls footer */}
        <div className="flex items-center justify-between px-3 py-2 border-t border-border/30 bg-muted/20">
          {/* Zoom slider */}
          <div className="flex items-center gap-2 flex-1 max-w-[200px]">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={handleZoomOut}
              disabled={zoom <= 0.25}
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </Button>
            <Slider
              value={[zoom]}
              onValueChange={([value]) => setZoom(value)}
              min={0.25}
              max={4}
              step={0.1}
              className="flex-1"
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={handleZoomIn}
              disabled={zoom >= 4}
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </Button>
          </div>

          {/* Zoom percentage and rotate */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-12 text-center font-mono">
              {Math.round(zoom * 100)}%
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={handleRotate}
            >
              <RotateCw className="h-3.5 w-3.5" />
            </Button>
          </div>

          {/* Dimensions */}
          {(data.width || data.height) && (
            <div className="text-xs text-muted-foreground ml-4">
              {data.width} × {data.height}
            </div>
          )}
        </div>
      </div>
    </WidgetBase>
  )
}

/**
 * Widget definition for registry
 */
export const ImageWidgetDef: WidgetDefinition<ImageWidgetData> = {
  type: 'image',
  displayName: 'Image',
  description: 'Display images with zoom and rotation controls',
  icon: ImageIcon,
  component: ImageWidget,
  defaultSize: { width: 4, height: 4 },
  minSize: { width: 2, height: 2 },
  capabilities: ['downloadable', 'fullscreen'],
}

// Register the widget
registerWidget(ImageWidgetDef)
