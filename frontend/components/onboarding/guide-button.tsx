'use client'

import { useState, useRef, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { BookOpen, RotateCcw, Compass, Map } from 'lucide-react'
import { cn } from '@/lib/utils'
import { getTourForRoute, getAllTours, createWelcomeTour } from '@/lib/shepherd/tour-registry'
import { hasSeenTour, resetAllTours, resetTour } from '@/lib/shepherd/tour-storage'

export function GuideButton() {
  const [open, setOpen] = useState(false)
  const [showPulse, setShowPulse] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const pathname = usePathname()
  const { user } = useUser()

  // Show pulse animation once after welcome tour completes
  useEffect(() => {
    if (!user?.id) return
    const welcomeDone = hasSeenTour('welcome', user.id)
    const pulseSeen = typeof window !== 'undefined'
      ? localStorage.getItem(`automatos-tour:guide-pulse:${user.id}`)
      : null
    if (welcomeDone && !pulseSeen) {
      setShowPulse(true)
      // Auto-dismiss pulse after 8 seconds
      const timer = setTimeout(() => {
        setShowPulse(false)
        localStorage.setItem(`automatos-tour:guide-pulse:${user.id}`, 'true')
      }, 8000)
      return () => clearTimeout(timer)
    }
  }, [user?.id])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const handleTourThisPage = async () => {
    if (!user?.id || !pathname) return
    setOpen(false)
    dismissPulse()

    const entry = getTourForRoute(pathname)
    if (!entry) return

    // Reset so it can replay
    resetTour(entry.id, user.id)
    const tour = await entry.factory(user.id)
    tour.start()
  }

  const handleWelcomeTour = async () => {
    if (!user?.id) return
    setOpen(false)
    dismissPulse()

    resetTour('welcome', user.id)
    const tour = await createWelcomeTour(user.id)
    tour.start()
  }

  const handleResetAll = () => {
    if (!user?.id) return
    setOpen(false)
    dismissPulse()
    resetAllTours(user.id)
  }

  const dismissPulse = () => {
    if (showPulse && user?.id) {
      setShowPulse(false)
      localStorage.setItem(`automatos-tour:guide-pulse:${user.id}`, 'true')
    }
  }

  const currentPageTour = pathname ? getTourForRoute(pathname) : undefined

  return (
    <div ref={menuRef} className="fixed bottom-6 left-6 z-50">
      {/* Dropdown Menu */}
      {open && (
        <div className="absolute bottom-14 left-0 w-56 rounded-xl border border-primary/15 bg-background/80 backdrop-blur-xl shadow-[0_0_40px_hsla(var(--primary)/0.1)] p-1.5 animate-in fade-in zoom-in-95 slide-in-from-bottom-2 duration-200">
          {/* Tour this page */}
          {currentPageTour && (
            <button
              onClick={handleTourThisPage}
              className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-primary/10 hover:text-white transition-colors"
            >
              <Map className="w-4 h-4 text-primary" />
              Tour this page
              <span className="ml-auto text-xs text-gray-500">{currentPageTour.label}</span>
            </button>
          )}

          {currentPageTour && <div className="my-1 border-t border-border/30" />}

          {/* Welcome Orientation */}
          <button
            onClick={handleWelcomeTour}
            className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-primary/10 hover:text-white transition-colors"
          >
            <Compass className="w-4 h-4 text-orange-400" />
            Welcome Orientation
          </button>

          <div className="my-1 border-t border-border/30" />

          {/* Reset All Tours */}
          <button
            onClick={handleResetAll}
            className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg text-sm text-gray-400 hover:bg-secondary/50 hover:text-gray-200 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset All Tours
          </button>
        </div>
      )}

      {/* FAB Button */}
      <button
        onClick={() => {
          setOpen(!open)
          dismissPulse()
        }}
        className={cn(
          'relative flex items-center justify-center w-10 h-10 rounded-full',
          'border border-primary/20 bg-background/60 backdrop-blur-xl',
          'shadow-[0_0_20px_hsla(var(--primary)/0.08)]',
          'hover:bg-primary/10 hover:border-primary/30 hover:shadow-[0_0_30px_hsla(var(--primary)/0.15)]',
          'transition-all duration-300',
          open && 'bg-primary/10 border-primary/30'
        )}
        aria-label="Guide - Tour help"
      >
        <BookOpen className="w-4 h-4 text-primary" />

        {/* Pulse ring */}
        {showPulse && (
          <span className="absolute inset-0 rounded-full animate-ping border-2 border-primary/40 pointer-events-none" />
        )}
      </button>
    </div>
  )
}
