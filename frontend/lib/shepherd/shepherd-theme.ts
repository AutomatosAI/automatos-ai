export const shepherdTheme = {
  defaultStepOptions: {
    classes: 'shepherd-theme-automatos',
    scrollTo: { behavior: 'smooth' as const, block: 'center' as const },
    cancelIcon: {
      enabled: true,
    },
    modalOverlayOpeningPadding: 8,
    modalOverlayOpeningRadius: 8,
    when: {
      show() {
        const currentStepElement = document.querySelector('.shepherd-element')
        if (currentStepElement) {
          currentStepElement.classList.add('animate-in', 'fade-in', 'zoom-in-95')
        }
      },
    },
  },
  useModalOverlay: true,
}
