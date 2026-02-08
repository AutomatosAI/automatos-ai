import { useEffect } from 'react'
import { TOUR_EVENTS } from '@/lib/shepherd/tour-bridge'

/**
 * Hook that lets the Shepherd tour drive which tab is active in a modal.
 * Listens for `tour:set-modal-tab` events, calls setStep(), then confirms
 * with `tour:modal-tab-ready` after React commits the render.
 */
export function useTourTabBridge(setStep: (step: number) => void) {
  useEffect(() => {
    const handler = (e: Event) => {
      const { step } = (e as CustomEvent).detail
      setStep(step)
      // Confirm after React renders
      requestAnimationFrame(() => {
        window.dispatchEvent(new CustomEvent(TOUR_EVENTS.MODAL_TAB_READY))
      })
    }
    window.addEventListener(TOUR_EVENTS.SET_MODAL_TAB, handler)
    return () => window.removeEventListener(TOUR_EVENTS.SET_MODAL_TAB, handler)
  }, [setStep])
}
