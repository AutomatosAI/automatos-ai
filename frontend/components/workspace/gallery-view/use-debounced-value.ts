/**
 * useDebouncedValue — the value once it has stopped changing for `delay` ms.
 * Shared by the Deliverables filter bar's search and tag inputs (moved out of
 * filter-bar.tsx when the tag filter needed it too, 7 Oct).
 */

import { useEffect, useState } from 'react'

export function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState<T>(value)
  useEffect(() => {
    const handle = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(handle)
  }, [value, delay])
  return debounced
}
