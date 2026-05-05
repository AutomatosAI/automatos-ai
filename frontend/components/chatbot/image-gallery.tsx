'use client'

import { useState } from 'react'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { ChevronLeft, ChevronRight, Download, X } from 'lucide-react'

export interface ChatImage {
  src: string
  alt: string
}

interface ImageGalleryProps {
  images: ChatImage[]
}

export function ImageGallery({ images }: ImageGalleryProps) {
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null)

  if (!images.length) return null

  const openLightbox = (idx: number) => setLightboxIndex(idx)
  const closeLightbox = () => setLightboxIndex(null)
  const prev = () =>
    setLightboxIndex((i) => (i !== null ? (i - 1 + images.length) % images.length : null))
  const next = () =>
    setLightboxIndex((i) => (i !== null ? (i + 1) % images.length : null))

  const handleDownload = (src: string, alt: string) => {
    const a = document.createElement('a')
    a.href = src
    a.download = alt || 'generated-image'
    a.click()
  }

  // Layout: 1 image = full width, 2+ = 2-column grid
  const gridClass =
    images.length === 1
      ? 'grid grid-cols-1'
      : 'grid grid-cols-2 gap-2'

  // For 3+ images, show "+N" overlay on the last visible cell
  const maxVisible = images.length <= 4 ? images.length : 4
  const visibleImages = images.slice(0, maxVisible)
  const overflow = images.length - maxVisible

  return (
    <>
      <div className={`${gridClass} my-3`}>
        {visibleImages.map((img, idx) => (
          <button
            key={idx}
            onClick={() => openLightbox(idx)}
            className="group relative overflow-hidden rounded-xl border border-orange-500/10 bg-black/20 hover:shadow-[0_0_20px_rgba(249,115,22,0.15)] transition-all duration-220"
          >
            <img
              src={img.src}
              alt={img.alt}
              className={`w-full object-contain ${
                images.length === 1 ? 'max-h-[480px]' : 'max-h-[280px]'
              }`}
              loading="lazy"
            />

            {/* Download overlay */}
            <div className="absolute inset-0 flex items-end justify-end p-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <span
                role="button"
                onClick={(e) => {
                  e.stopPropagation()
                  handleDownload(img.src, img.alt)
                }}
                className="rounded-lg bg-black/60 p-1.5 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
              >
                <Download className="w-4 h-4" />
              </span>
            </div>

            {/* Overflow badge on last cell */}
            {idx === maxVisible - 1 && overflow > 0 && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <span className="text-2xl font-bold text-white">+{overflow}</span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Lightbox */}
      <Dialog open={lightboxIndex !== null} onOpenChange={() => closeLightbox()}>
        <DialogContent className="max-w-[95vw] max-h-[95vh] p-0 border-orange-500/20 bg-black/95 backdrop-blur-xl overflow-hidden [&>button]:hidden">
          {lightboxIndex !== null && (
            <div className="relative flex items-center justify-center w-full h-[90vh]">
              <img
                src={images[lightboxIndex].src}
                alt={images[lightboxIndex].alt}
                className="max-w-full max-h-full object-contain"
              />

              {/* Close */}
              <button
                onClick={closeLightbox}
                className="absolute top-4 right-4 rounded-full bg-black/60 p-2 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>

              {/* Download */}
              <button
                onClick={() =>
                  handleDownload(
                    images[lightboxIndex].src,
                    images[lightboxIndex].alt
                  )
                }
                className="absolute top-4 right-16 rounded-full bg-black/60 p-2 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
              >
                <Download className="w-5 h-5" />
              </button>

              {/* Prev/Next */}
              {images.length > 1 && (
                <>
                  <button
                    onClick={prev}
                    className="absolute left-4 top-1/2 -translate-y-1/2 rounded-full bg-black/60 p-2 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
                  >
                    <ChevronLeft className="w-6 h-6" />
                  </button>
                  <button
                    onClick={next}
                    className="absolute right-4 top-1/2 -translate-y-1/2 rounded-full bg-black/60 p-2 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
                  >
                    <ChevronRight className="w-6 h-6" />
                  </button>
                </>
              )}

              {/* Counter */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-3 py-1 text-sm text-white/70">
                {lightboxIndex + 1} / {images.length}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
