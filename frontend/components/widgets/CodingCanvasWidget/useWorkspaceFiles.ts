'use client'

/**
 * useWorkspaceFiles — API hook for workspace filesystem browsing
 * PRD-66 Phase 1: Code Viewer Widget
 *
 * Lazily fetches directory contents when expanded.
 * Caches fetched directories to avoid re-fetching on collapse/expand.
 */

import { useCallback, useRef, useState } from 'react'
import { apiClient } from '@/lib/api-client'
import type { WorkspaceFileEntry, OpenFileTab } from '../types'

interface FileContentResponse {
  path: string
  name: string
  content: string
  size: number
  language: string
  mime_type: string
}

interface DirListResponse {
  path: string
  entries: Array<{
    name: string
    path: string
    type: 'file' | 'directory'
    size: number
    modified_at?: number
  }>
  truncated: boolean
}

/**
 * @param rootPath PRD-235 W2: the folder the tree is rooted at ('.' = the workspace
 *   root; e.g. 'sessions/71' or 'projects/my-repo'). Paths stay workspace-relative,
 *   so the editor, saves and the terminal work unchanged under any root.
 */
export function useWorkspaceFiles(workspaceId: string | undefined, rootPath: string = '.') {
  const root = rootPath || '.'
  const [tree, setTree] = useState<WorkspaceFileEntry[]>([])
  const [isLoadingTree, setIsLoadingTree] = useState(false)
  const [treeError, setTreeError] = useState<string | null>(null)
  const dirCacheRef = useRef<Map<string, WorkspaceFileEntry[]>>(new Map())

  /**
   * Fetch directory listing and update the tree.
   * For root ('.'), replaces the top-level tree.
   * For subdirectories, merges children into the existing tree.
   */
  const fetchDirectory = useCallback(
    async (dirPath: string = root) => {
      if (!workspaceId) return

      // Check cache
      const cached = dirCacheRef.current.get(dirPath)
      if (cached) {
        if (dirPath === root) {
          setTree(cached)
        } else {
          setTree((prev) => mergeChildren(prev, dirPath, cached))
        }
        return
      }

      if (dirPath === root) setIsLoadingTree(true)

      // Mark directory as loading in the tree
      if (dirPath !== root) {
        setTree((prev) => setNodeLoading(prev, dirPath, true))
      }

      try {
        const res: DirListResponse = await apiClient.request(
          `/api/workspaces/${workspaceId}/files?path=${encodeURIComponent(dirPath)}`
        )

        const entries: WorkspaceFileEntry[] = res.entries.map((e) => ({
          name: e.name,
          path: e.path,
          type: e.type,
          size: e.size,
          modified_at: e.modified_at,
          children: e.type === 'directory' ? undefined : undefined,
        }))

        // Cache it
        dirCacheRef.current.set(dirPath, entries)

        if (dirPath === root) {
          setTree(entries)
        } else {
          setTree((prev) => mergeChildren(prev, dirPath, entries))
        }

        setTreeError(null)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Failed to load directory'
        if (dirPath === root) setTreeError(msg)
        if (dirPath !== root) {
          setTree((prev) => setNodeLoading(prev, dirPath, false))
        }
      } finally {
        if (dirPath === root) setIsLoadingTree(false)
      }
    },
    [workspaceId, root]
  )

  /**
   * Fetch file content for the editor.
   */
  const fetchFileContent = useCallback(
    async (filePath: string): Promise<OpenFileTab | null> => {
      if (!workspaceId) return null

      try {
        const res: FileContentResponse = await apiClient.request(
          `/api/workspaces/${workspaceId}/files/content?path=${encodeURIComponent(filePath)}`
        )
        return {
          path: res.path,
          name: res.name,
          language: res.language,
          content: res.content,
          isLoading: false,
        }
      } catch {
        return null
      }
    },
    [workspaceId]
  )

  /**
   * Invalidate cache for a specific directory (or all).
   */
  const invalidateCache = useCallback((dirPath?: string) => {
    if (dirPath) {
      dirCacheRef.current.delete(dirPath)
    } else {
      dirCacheRef.current.clear()
    }
  }, [])

  return {
    tree,
    isLoadingTree,
    treeError,
    fetchDirectory,
    fetchFileContent,
    invalidateCache,
  }
}

// ---------------------------------------------------------------------------
// Tree helpers
// ---------------------------------------------------------------------------

function mergeChildren(
  nodes: WorkspaceFileEntry[],
  parentPath: string,
  children: WorkspaceFileEntry[]
): WorkspaceFileEntry[] {
  return nodes.map((node) => {
    if (node.path === parentPath) {
      return { ...node, children, isLoading: false }
    }
    if (node.children) {
      return { ...node, children: mergeChildren(node.children, parentPath, children) }
    }
    return node
  })
}

function setNodeLoading(
  nodes: WorkspaceFileEntry[],
  targetPath: string,
  loading: boolean
): WorkspaceFileEntry[] {
  return nodes.map((node) => {
    if (node.path === targetPath) {
      return { ...node, isLoading: loading }
    }
    if (node.children) {
      return { ...node, children: setNodeLoading(node.children, targetPath, loading) }
    }
    return node
  })
}
