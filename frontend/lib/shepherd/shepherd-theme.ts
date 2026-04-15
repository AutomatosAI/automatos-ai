/** True when viewport is ≤ 640px (mobile). */
export function isMobile(): boolean {
  return typeof window !== 'undefined' && window.innerWidth <= 640
}

/**
 * On mobile, override `attachTo.on` to 'bottom' to prevent tooltips
 * from being pushed off-screen by 'right'/'left' placements.
 */
export function mobilePosition(desktop: string): string {
  if (!isMobile()) return desktop
  if (desktop === 'left' || desktop === 'right') return 'bottom'
  return desktop
}

export const shepherdTheme = {
  defaultStepOptions: {
    classes: 'shepherd-theme-automatos',
    scrollTo: { behavior: 'smooth' as const, block: 'center' as const },
    cancelIcon: {
      enabled: true,
    },
    modalOverlayOpeningPadding: isMobile() ? 4 : 8,
    modalOverlayOpeningRadius: 8,
    when: {
      show(this: any) {
        const el = this.el as HTMLElement | undefined
        if (!el) return
        el.classList.add('animate-in', 'fade-in', 'zoom-in-95')
        // Mark unattached steps so CSS can center them on mobile
        if (!this.options?.attachTo?.element) {
          el.classList.add('shepherd-centered')
        } else {
          el.classList.remove('shepherd-centered')
        }
      },
    },
  },
  useModalOverlay: true,
}
