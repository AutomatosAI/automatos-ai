import '@testing-library/jest-dom/vitest'

// jsdom implements no layout and no media queries, so `window.matchMedia` does
// not exist — and `hooks/use-mobile` calls it on mount, which is now on the
// mount path of any component that reads a width (PRD-246). Report "does not
// match", i.e. a desktop viewport; a test that needs a narrow one mocks the
// hook itself: vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: … })).
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia
}
