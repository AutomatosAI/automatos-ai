# PRD-129: Workspace Outputs Hub

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Claude
**Created:** 2026-04-10
**Updated:** 2026-04-10
**Dependencies:** PRD-66 (Code Canvas — COMPLETE), PRD-76 (Agent Reports — COMPLETE), PRD-128 (Notifications — DRAFT)

---

## Executive Summary

The current Workspace tab is a VS Code-style file explorer: tree view, code editor, terminal. Perfect for developers, intimidating for non-technical users. When an agent produces a report, image, or document, users can't easily find it without knowing the exact path.

This PRD transforms the Workspace into an **Outputs Hub** — a consumer-friendly gallery where users can discover, preview, and manage everything their agents produce. The VS Code experience remains available as "Explorer Mode" for developers, but the default becomes a filterable grid of output cards.

### What We're Building

1. **Deliverables concept** — extends `agent_reports` pattern to ALL output types (images, documents, code, slides)
2. **Gallery View** — grid of preview cards, default for non-technical users
3. **Filter bar** — date range, type, agent, source (chat/task/mission/heartbeat)
4. **Preview slide-over** — inline render of markdown, images, code, PDFs
5. **View toggle** — switch between Gallery / Explorer / Activity views
6. **S3 document integration** — surface RAG-uploaded documents alongside workspace files

### What We're NOT Building

- Document editor (view only, edit externally or in Code Canvas)
- Collaboration/sharing features (v2 scope)
- File versioning/history (git handles this for code)
- New file upload UI (existing attachment flow is sufficient)

### Design System

| Element | Value |
|---------|-------|
| **Style** | Dark Mode (OLED) — high contrast, eye-friendly |
| **Colors** | Primary: `#18181B`, Secondary: `#27272A`, CTA: `#F8FAFC`, Background: `#FAFAFA` (light mode fallback) |
| **Typography** | Plus Jakarta Sans — friendly, modern, SaaS-appropriate |
| **Effects** | Minimal glow, dark-to-light transitions, high readability |
| **Icons** | Lucide React (no emojis) |

### Reuse Strategy

| Component | Reuse | Notes |
|-----------|-------|-------|
| `agent_reports` table | **Extend** | Add artifact types beyond reports, or create `deliverables` table |
| `WorkspaceExplorer` | **Keep** | Becomes Explorer View, toggle to access |
| `useWorkspaceFiles` hook | **Reuse** | Fetch file listings for Gallery |
| `WorkspaceClient` | **100%** | All file operations go through existing proxy |
| `report_service.py` | **Extend** | Pattern for DB + file hybrid storage |

---

## 1. Data Architecture

### 1.1 Option A: Extend `agent_reports` Table

Add new `artifact_type` values and make `agent_id` optional (for user uploads):

```sql
-- Expand artifact_type enum for agent_reports
ALTER TABLE agent_reports 
    ALTER COLUMN agent_id DROP NOT NULL;

-- Add source tracking
ALTER TABLE agent_reports 
    ADD COLUMN source_type VARCHAR(30) DEFAULT 'heartbeat',
    ADD COLUMN source_id VARCHAR(255);
    
-- source_type values: chat, task, mission, heartbeat, playbook, trigger, upload
-- source_id: the ID of the source (task_id, mission_id, etc.)

-- Expand report_type to artifact_type (rename for clarity)
-- Or add new column:
ALTER TABLE agent_reports
    ADD COLUMN artifact_type VARCHAR(30);
    
-- artifact_type values: report, image, document, code, slide, spreadsheet, archive, audio, video
```

### 1.2 Option B: Create `deliverables` Table (Recommended)

Cleaner separation, avoids overloading `agent_reports`:

```sql
CREATE TABLE deliverables (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id        UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    
    -- Source tracking
    source_type         VARCHAR(30) NOT NULL,
        -- chat, task, mission, heartbeat, playbook, trigger, upload
    source_id           VARCHAR(255),           -- FK to source (nullable for uploads)
    
    -- Agent attribution (nullable for user uploads)
    agent_id            INTEGER REFERENCES agents(id) ON DELETE SET NULL,
    agent_name          VARCHAR(100),
    
    -- Artifact info
    artifact_type       VARCHAR(30) NOT NULL,
        -- report, image, document, code, slide, spreadsheet, archive, audio, video
    title               VARCHAR(255) NOT NULL,
    summary             VARCHAR(500),
    
    -- File reference
    storage_type        VARCHAR(20) NOT NULL DEFAULT 'workspace',
        -- workspace: file in workspace volume
        -- s3: file in S3 (RAG uploads)
        -- external: external URL
    file_path           VARCHAR(1024) NOT NULL,
    file_name           VARCHAR(255),           -- Original filename
    file_type           VARCHAR(50),            -- MIME type
    file_size_bytes     INTEGER,
    
    -- Preview
    preview_url         VARCHAR(1024),          -- Thumbnail or preview image URL
    preview_type        VARCHAR(30),            -- image, text_snippet, code_snippet
    
    -- Metadata
    metadata            JSONB DEFAULT '{}',
        -- Flexible: { language: "python", lines: 150 } for code
        -- { dimensions: "1920x1080" } for images
        -- { pages: 12 } for PDFs
    
    -- Status
    status              VARCHAR(20) DEFAULT 'ready',
        -- ready, processing, error
    
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indices
CREATE INDEX ix_deliverables_workspace ON deliverables(workspace_id);
CREATE INDEX ix_deliverables_agent ON deliverables(agent_id);
CREATE INDEX ix_deliverables_type ON deliverables(workspace_id, artifact_type);
CREATE INDEX ix_deliverables_source ON deliverables(workspace_id, source_type);
CREATE INDEX ix_deliverables_created ON deliverables(workspace_id, created_at DESC);
```

### 1.3 Auto-Population Strategy

Deliverables get created automatically when:

| Source | Trigger | What Gets Registered |
|--------|---------|---------------------|
| `platform_submit_report` | Agent calls tool | Report deliverable |
| `workspace_write_file` | Agent writes file with image/doc extension | Image/document deliverable |
| Chat with attachment | User uploads file in chat | Upload deliverable |
| RAG document sync | Document synced from cloud | Document deliverable |
| Mission completion | Mission produces artifacts | All artifacts from mission |

---

## 2. Deliverables Service

File: `orchestrator/services/deliverable_service.py`

```python
"""
Deliverable Service (PRD-129)

Manages deliverables — outputs produced by agents or uploaded by users.
Provides unified discovery layer over workspace files, S3 documents,
and agent reports.
"""

import logging
import mimetypes
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from core.workspace_client import WorkspaceClient

logger = logging.getLogger(__name__)

# Map file extensions to artifact types
EXTENSION_TO_ARTIFACT = {
    # Images
    '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image',
    '.webp': 'image', '.svg': 'image', '.ico': 'image',
    # Documents
    '.md': 'report', '.txt': 'document', '.pdf': 'document',
    '.docx': 'document', '.doc': 'document',
    # Slides
    '.pptx': 'slide', '.ppt': 'slide', '.key': 'slide',
    # Spreadsheets
    '.xlsx': 'spreadsheet', '.xls': 'spreadsheet', '.csv': 'spreadsheet',
    # Code
    '.py': 'code', '.ts': 'code', '.tsx': 'code', '.js': 'code',
    '.jsx': 'code', '.go': 'code', '.rs': 'code', '.java': 'code',
    '.rb': 'code', '.php': 'code', '.swift': 'code', '.kt': 'code',
    # Archives
    '.zip': 'archive', '.tar': 'archive', '.gz': 'archive',
    # Audio/Video
    '.mp3': 'audio', '.wav': 'audio', '.m4a': 'audio',
    '.mp4': 'video', '.mov': 'video', '.webm': 'video',
}


def _slugify(value: str) -> str:
    """Convert string to kebab-case slug."""
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")[:80]


def _infer_artifact_type(file_path: str) -> str:
    """Infer artifact type from file extension."""
    ext = '.' + file_path.split('.')[-1].lower() if '.' in file_path else ''
    return EXTENSION_TO_ARTIFACT.get(ext, 'document')


class DeliverableService:
    """Service for managing deliverables."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id

    async def register(
        self,
        file_path: str,
        title: str,
        source_type: str,
        source_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
        summary: Optional[str] = None,
        storage_type: str = "workspace",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new deliverable.

        Called automatically when agents write files or submit reports,
        or manually when users upload files.
        """
        # Infer artifact type if not provided
        if not artifact_type:
            artifact_type = _infer_artifact_type(file_path)

        # Get file info
        file_name = file_path.split('/')[-1]
        file_type, _ = mimetypes.guess_type(file_name)
        
        # Get file size from workspace
        file_size = None
        if storage_type == "workspace":
            ws_client = WorkspaceClient(str(self.workspace_id))
            try:
                # List parent directory to get file size
                parent = '/'.join(file_path.split('/')[:-1]) or '.'
                listing = await ws_client.list_dir(parent)
                for entry in listing.get('entries', []):
                    if entry.get('name') == file_name:
                        file_size = entry.get('size')
                        break
            except Exception as e:
                logger.debug("Could not get file size: %s", e)

        # Generate preview URL for images
        preview_url = None
        preview_type = None
        if artifact_type == 'image':
            preview_url = f"/api/workspaces/{self.workspace_id}/files/content?path={file_path}"
            preview_type = 'image'
        elif artifact_type in ('report', 'code'):
            preview_type = 'text_snippet'

        # Insert deliverable
        try:
            result = self.db.execute(
                text("""
                    INSERT INTO deliverables
                        (workspace_id, source_type, source_id, agent_id, agent_name,
                         artifact_type, title, summary, storage_type, file_path,
                         file_name, file_type, file_size_bytes, preview_url, preview_type,
                         metadata, created_at, updated_at)
                    VALUES
                        (:workspace_id, :source_type, :source_id, :agent_id, :agent_name,
                         :artifact_type, :title, :summary, :storage_type, :file_path,
                         :file_name, :file_type, :file_size_bytes, :preview_url, :preview_type,
                         :metadata, NOW(), NOW())
                    RETURNING id
                """),
                {
                    "workspace_id": str(self.workspace_id),
                    "source_type": source_type,
                    "source_id": source_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "artifact_type": artifact_type,
                    "title": title,
                    "summary": summary,
                    "storage_type": storage_type,
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_size_bytes": file_size,
                    "preview_url": preview_url,
                    "preview_type": preview_type,
                    "metadata": metadata or {},
                },
            )
            row = result.fetchone()
            self.db.commit()

            deliverable_id = str(row[0]) if row else None
            logger.info(
                "[DeliverableService] Registered deliverable %s: %s (%s)",
                deliverable_id, title, artifact_type,
            )

            return {
                "success": True,
                "deliverable_id": deliverable_id,
                "artifact_type": artifact_type,
            }

        except Exception as e:
            self.db.rollback()
            logger.error("[DeliverableService] Failed to register: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def list_deliverables(
        self,
        artifact_type: Optional[str] = None,
        source_type: Optional[str] = None,
        agent_id: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 24,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List deliverables with filters."""
        
        conditions = ["d.workspace_id = :workspace_id"]
        params: Dict[str, Any] = {
            "workspace_id": str(self.workspace_id),
            "limit": limit,
            "offset": offset,
        }

        if artifact_type:
            conditions.append("d.artifact_type = :artifact_type")
            params["artifact_type"] = artifact_type

        if source_type:
            conditions.append("d.source_type = :source_type")
            params["source_type"] = source_type

        if agent_id:
            conditions.append("d.agent_id = :agent_id")
            params["agent_id"] = agent_id

        if date_from:
            conditions.append("d.created_at >= :date_from")
            params["date_from"] = date_from

        if date_to:
            conditions.append("d.created_at <= :date_to")
            params["date_to"] = date_to

        if search:
            conditions.append("(d.title ILIKE :search OR d.summary ILIKE :search)")
            params["search"] = f"%{search}%"

        where = " AND ".join(conditions)

        # Count total
        count = self.db.execute(
            text(f"SELECT COUNT(*) FROM deliverables d WHERE {where}"),
            params,
        ).scalar() or 0

        # Fetch deliverables
        rows = self.db.execute(
            text(f"""
                SELECT
                    d.id, d.source_type, d.source_id, d.agent_id,
                    COALESCE(d.agent_name, a.name) AS agent_name,
                    d.artifact_type, d.title, d.summary,
                    d.storage_type, d.file_path, d.file_name, d.file_type,
                    d.file_size_bytes, d.preview_url, d.preview_type,
                    d.metadata, d.status, d.created_at
                FROM deliverables d
                LEFT JOIN agents a ON a.id = d.agent_id
                WHERE {where}
                ORDER BY d.created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()

        deliverables = []
        for row in rows:
            deliverables.append({
                "id": str(row.id),
                "source_type": row.source_type,
                "source_id": row.source_id,
                "agent_id": row.agent_id,
                "agent_name": row.agent_name,
                "artifact_type": row.artifact_type,
                "title": row.title,
                "summary": row.summary,
                "storage_type": row.storage_type,
                "file_path": row.file_path,
                "file_name": row.file_name,
                "file_type": row.file_type,
                "file_size_bytes": row.file_size_bytes,
                "preview_url": row.preview_url,
                "preview_type": row.preview_type,
                "metadata": row.metadata or {},
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })

        return {
            "success": True,
            "deliverables": deliverables,
            "total": count,
            "limit": limit,
            "offset": offset,
        }

    async def get_deliverable(self, deliverable_id: str, include_content: bool = False) -> Dict[str, Any]:
        """Get a single deliverable with optional content."""
        
        row = self.db.execute(
            text("""
                SELECT
                    d.id, d.workspace_id, d.source_type, d.source_id, d.agent_id,
                    COALESCE(d.agent_name, a.name) AS agent_name,
                    d.artifact_type, d.title, d.summary,
                    d.storage_type, d.file_path, d.file_name, d.file_type,
                    d.file_size_bytes, d.preview_url, d.preview_type,
                    d.metadata, d.status, d.created_at
                FROM deliverables d
                LEFT JOIN agents a ON a.id = d.agent_id
                WHERE d.id = :id AND d.workspace_id = :workspace_id
            """),
            {"id": deliverable_id, "workspace_id": str(self.workspace_id)},
        ).fetchone()

        if not row:
            return {"success": False, "error": "Deliverable not found"}

        deliverable = {
            "id": str(row.id),
            "workspace_id": str(row.workspace_id),
            "source_type": row.source_type,
            "source_id": row.source_id,
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "artifact_type": row.artifact_type,
            "title": row.title,
            "summary": row.summary,
            "storage_type": row.storage_type,
            "file_path": row.file_path,
            "file_name": row.file_name,
            "file_type": row.file_type,
            "file_size_bytes": row.file_size_bytes,
            "preview_url": row.preview_url,
            "preview_type": row.preview_type,
            "metadata": row.metadata or {},
            "status": row.status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

        # Fetch content if requested
        if include_content and row.storage_type == "workspace":
            ws_client = WorkspaceClient(str(self.workspace_id))
            
            if row.artifact_type == 'image':
                # For images, return download URL
                deliverable["content_url"] = f"/api/workspaces/{self.workspace_id}/files/download?path={row.file_path}"
            else:
                # For text-based, fetch content
                result = await ws_client.read_file(row.file_path)
                if result.get("success"):
                    deliverable["content"] = result.get("content", "")
                else:
                    deliverable["content"] = None
                    deliverable["content_error"] = result.get("error", "Could not read file")

        return {"success": True, "deliverable": deliverable}

    async def get_stats(self) -> Dict[str, Any]:
        """Get deliverable statistics for workspace."""
        
        by_type = self.db.execute(
            text("""
                SELECT artifact_type, COUNT(*) AS count
                FROM deliverables
                WHERE workspace_id = :workspace_id
                GROUP BY artifact_type
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchall()

        by_agent = self.db.execute(
            text("""
                SELECT COALESCE(d.agent_name, a.name, 'Unknown') AS agent_name, COUNT(*) AS count
                FROM deliverables d
                LEFT JOIN agents a ON a.id = d.agent_id
                WHERE d.workspace_id = :workspace_id
                GROUP BY COALESCE(d.agent_name, a.name, 'Unknown')
                ORDER BY count DESC
                LIMIT 10
            """),
            {"workspace_id": str(self.workspace_id)},
        ).fetchall()

        total = self.db.execute(
            text("SELECT COUNT(*) FROM deliverables WHERE workspace_id = :workspace_id"),
            {"workspace_id": str(self.workspace_id)},
        ).scalar() or 0

        return {
            "success": True,
            "total": total,
            "by_type": {r.artifact_type: r.count for r in by_type},
            "by_agent": {r.agent_name: r.count for r in by_agent},
        }
```

---

## 3. API Endpoints

File: `orchestrator/api/deliverables.py`

```python
"""
Deliverables API (PRD-129)
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import SessionLocal
from services.deliverable_service import DeliverableService

router = APIRouter(prefix="/api/deliverables", tags=["deliverables"])


@router.get("")
async def list_deliverables(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    artifact_type: Optional[str] = Query(None),
    source_type: Optional[str] = Query(None),
    agent_id: Optional[int] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(24, le=100),
    offset: int = Query(0),
):
    """List deliverables with filters."""
    db = SessionLocal()
    try:
        service = DeliverableService(db, ctx.workspace_id)
        return await service.list_deliverables(
            artifact_type=artifact_type,
            source_type=source_type,
            agent_id=agent_id,
            date_from=date_from,
            date_to=date_to,
            search=search,
            limit=limit,
            offset=offset,
        )
    finally:
        db.close()


@router.get("/stats")
async def get_stats(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Get deliverable statistics."""
    db = SessionLocal()
    try:
        service = DeliverableService(db, ctx.workspace_id)
        return await service.get_stats()
    finally:
        db.close()


@router.get("/{deliverable_id}")
async def get_deliverable(
    deliverable_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    include_content: bool = Query(False),
):
    """Get a single deliverable."""
    db = SessionLocal()
    try:
        service = DeliverableService(db, ctx.workspace_id)
        result = await service.get_deliverable(deliverable_id, include_content)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error"))
        return result
    finally:
        db.close()
```

---

## 4. Frontend Components

### 4.1 Component Architecture

```
/workspace
├── page.tsx                    # Main workspace page
├── workspace-view-toggle.tsx   # Gallery / Explorer / Activity toggle
├── gallery-view/
│   ├── index.tsx              # Gallery grid container
│   ├── filter-bar.tsx         # Date, type, agent, search filters
│   ├── deliverable-card.tsx   # Individual output card
│   └── deliverable-preview.tsx # Slide-over preview panel
├── explorer-view/
│   └── index.tsx              # Existing WorkspaceExplorer
└── activity-view/
    └── index.tsx              # Timeline of recent outputs
```

### 4.2 Gallery View Component

File: `frontend/components/workspace/gallery-view/index.tsx`

```tsx
'use client'

import { useState, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2 } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { FilterBar, FilterState } from './filter-bar'
import { DeliverableCard } from './deliverable-card'
import { DeliverablePreview } from './deliverable-preview'

interface Deliverable {
  id: string
  source_type: string
  source_id: string | null
  agent_id: number | null
  agent_name: string | null
  artifact_type: string
  title: string
  summary: string | null
  file_path: string
  file_name: string | null
  file_type: string | null
  file_size_bytes: number | null
  preview_url: string | null
  preview_type: string | null
  created_at: string
}

interface GalleryViewProps {
  workspaceId: string
}

export function GalleryView({ workspaceId }: GalleryViewProps) {
  const [filters, setFilters] = useState<FilterState>({
    artifact_type: null,
    source_type: null,
    agent_id: null,
    date_range: 'all',
    search: '',
  })

  const [selectedId, setSelectedId] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['deliverables', workspaceId, filters],
    queryFn: () => {
      const params = new URLSearchParams()
      if (filters.artifact_type) params.set('artifact_type', filters.artifact_type)
      if (filters.source_type) params.set('source_type', filters.source_type)
      if (filters.agent_id) params.set('agent_id', String(filters.agent_id))
      if (filters.search) params.set('search', filters.search)
      // Date range handling
      if (filters.date_range === 'today') {
        const today = new Date()
        today.setHours(0, 0, 0, 0)
        params.set('date_from', today.toISOString())
      } else if (filters.date_range === 'week') {
        const week = new Date()
        week.setDate(week.getDate() - 7)
        params.set('date_from', week.toISOString())
      } else if (filters.date_range === 'month') {
        const month = new Date()
        month.setMonth(month.getMonth() - 1)
        params.set('date_from', month.toISOString())
      }
      params.set('limit', '48')
      return apiClient.get(`/api/deliverables?${params.toString()}`)
    },
  })

  const deliverables: Deliverable[] = data?.deliverables ?? []
  const total = data?.total ?? 0

  const handleCardClick = useCallback((id: string) => {
    setSelectedId(id)
  }, [])

  const handlePreviewClose = useCallback(() => {
    setSelectedId(null)
  }, [])

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-destructive">
        Failed to load outputs
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Filter Bar */}
      <FilterBar
        filters={filters}
        onFiltersChange={setFilters}
        total={total}
      />

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : deliverables.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="rounded-full bg-muted p-4 mb-4">
              <FileIcon className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-medium">No outputs yet</h3>
            <p className="text-sm text-muted-foreground mt-1">
              When your agents produce reports, images, or documents, they'll appear here.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {deliverables.map((item) => (
              <DeliverableCard
                key={item.id}
                deliverable={item}
                onClick={() => handleCardClick(item.id)}
                isSelected={selectedId === item.id}
              />
            ))}
          </div>
        )}
      </div>

      {/* Preview Slide-over */}
      {selectedId && (
        <DeliverablePreview
          workspaceId={workspaceId}
          deliverableId={selectedId}
          onClose={handlePreviewClose}
        />
      )}
    </div>
  )
}

function FileIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  )
}
```

### 4.3 Filter Bar Component

File: `frontend/components/workspace/gallery-view/filter-bar.tsx`

```tsx
'use client'

import { Search, Calendar, FileType, User, Zap } from 'lucide-react'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { useDebounce } from '@/hooks/use-debounce'
import { useState, useEffect } from 'react'

export interface FilterState {
  artifact_type: string | null
  source_type: string | null
  agent_id: number | null
  date_range: 'all' | 'today' | 'week' | 'month'
  search: string
}

interface FilterBarProps {
  filters: FilterState
  onFiltersChange: (filters: FilterState) => void
  total: number
}

const ARTIFACT_TYPES = [
  { value: 'all', label: 'All Types' },
  { value: 'report', label: 'Reports' },
  { value: 'image', label: 'Images' },
  { value: 'document', label: 'Documents' },
  { value: 'code', label: 'Code' },
  { value: 'slide', label: 'Slides' },
  { value: 'spreadsheet', label: 'Spreadsheets' },
]

const SOURCE_TYPES = [
  { value: 'all', label: 'All Sources' },
  { value: 'chat', label: 'Chat' },
  { value: 'task', label: 'Tasks' },
  { value: 'mission', label: 'Missions' },
  { value: 'heartbeat', label: 'Heartbeats' },
  { value: 'playbook', label: 'Playbooks' },
  { value: 'upload', label: 'Uploads' },
]

const DATE_RANGES = [
  { value: 'all', label: 'All Time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'This Week' },
  { value: 'month', label: 'This Month' },
]

export function FilterBar({ filters, onFiltersChange, total }: FilterBarProps) {
  const [searchInput, setSearchInput] = useState(filters.search)
  const debouncedSearch = useDebounce(searchInput, 300)

  useEffect(() => {
    if (debouncedSearch !== filters.search) {
      onFiltersChange({ ...filters, search: debouncedSearch })
    }
  }, [debouncedSearch])

  const updateFilter = <K extends keyof FilterState>(key: K, value: FilterState[K]) => {
    onFiltersChange({ ...filters, [key]: value === 'all' ? null : value })
  }

  const hasActiveFilters = filters.artifact_type || filters.source_type || filters.date_range !== 'all' || filters.search

  const clearFilters = () => {
    setSearchInput('')
    onFiltersChange({
      artifact_type: null,
      source_type: null,
      agent_id: null,
      date_range: 'all',
      search: '',
    })
  }

  return (
    <div className="border-b px-4 py-3 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search outputs..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        {/* Type Filter */}
        <Select
          value={filters.artifact_type ?? 'all'}
          onValueChange={(v) => updateFilter('artifact_type', v)}
        >
          <SelectTrigger className="w-[140px] h-9">
            <FileType className="mr-2 h-4 w-4 text-muted-foreground" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {ARTIFACT_TYPES.map((type) => (
              <SelectItem key={type.value} value={type.value}>
                {type.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Source Filter */}
        <Select
          value={filters.source_type ?? 'all'}
          onValueChange={(v) => updateFilter('source_type', v)}
        >
          <SelectTrigger className="w-[140px] h-9">
            <Zap className="mr-2 h-4 w-4 text-muted-foreground" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SOURCE_TYPES.map((type) => (
              <SelectItem key={type.value} value={type.value}>
                {type.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Date Filter */}
        <Select
          value={filters.date_range}
          onValueChange={(v) => updateFilter('date_range', v as FilterState['date_range'])}
        >
          <SelectTrigger className="w-[130px] h-9">
            <Calendar className="mr-2 h-4 w-4 text-muted-foreground" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DATE_RANGES.map((range) => (
              <SelectItem key={range.value} value={range.value}>
                {range.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Clear / Count */}
        <div className="flex items-center gap-2 ml-auto">
          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={clearFilters} className="h-9">
              Clear
            </Button>
          )}
          <span className="text-sm text-muted-foreground">
            {total} {total === 1 ? 'output' : 'outputs'}
          </span>
        </div>
      </div>
    </div>
  )
}
```

### 4.4 Deliverable Card Component

File: `frontend/components/workspace/gallery-view/deliverable-card.tsx`

```tsx
'use client'

import { memo } from 'react'
import {
  FileText, Image, FileCode, Presentation, Sheet, Archive,
  Music, Video, File, Zap, MessageSquare, ClipboardList, Calendar
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { formatDistanceToNow } from 'date-fns'

interface Deliverable {
  id: string
  artifact_type: string
  title: string
  summary: string | null
  agent_name: string | null
  source_type: string
  preview_url: string | null
  preview_type: string | null
  file_size_bytes: number | null
  created_at: string
}

interface DeliverableCardProps {
  deliverable: Deliverable
  onClick: () => void
  isSelected: boolean
}

const ARTIFACT_ICONS: Record<string, React.ElementType> = {
  report: FileText,
  image: Image,
  document: FileText,
  code: FileCode,
  slide: Presentation,
  spreadsheet: Sheet,
  archive: Archive,
  audio: Music,
  video: Video,
}

const ARTIFACT_COLORS: Record<string, string> = {
  report: 'bg-blue-500/10 text-blue-500',
  image: 'bg-purple-500/10 text-purple-500',
  document: 'bg-amber-500/10 text-amber-500',
  code: 'bg-emerald-500/10 text-emerald-500',
  slide: 'bg-orange-500/10 text-orange-500',
  spreadsheet: 'bg-green-500/10 text-green-500',
  archive: 'bg-gray-500/10 text-gray-500',
  audio: 'bg-pink-500/10 text-pink-500',
  video: 'bg-red-500/10 text-red-500',
}

const SOURCE_ICONS: Record<string, React.ElementType> = {
  chat: MessageSquare,
  task: ClipboardList,
  mission: Calendar,
  heartbeat: Zap,
  playbook: File,
  upload: File,
  trigger: Zap,
}

function formatFileSize(bytes: number | null): string {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export const DeliverableCard = memo(function DeliverableCard({
  deliverable,
  onClick,
  isSelected,
}: DeliverableCardProps) {
  const Icon = ARTIFACT_ICONS[deliverable.artifact_type] || File
  const colorClass = ARTIFACT_COLORS[deliverable.artifact_type] || 'bg-muted text-muted-foreground'
  const SourceIcon = SOURCE_ICONS[deliverable.source_type] || File

  const isImage = deliverable.artifact_type === 'image' && deliverable.preview_url

  return (
    <button
      onClick={onClick}
      className={cn(
        'group relative flex flex-col overflow-hidden rounded-lg border bg-card text-left transition-all cursor-pointer',
        'hover:border-primary/50 hover:shadow-md',
        isSelected && 'ring-2 ring-primary border-primary'
      )}
    >
      {/* Preview area */}
      <div className="relative aspect-[4/3] bg-muted/50 overflow-hidden">
        {isImage ? (
          <img
            src={deliverable.preview_url}
            alt={deliverable.title}
            className="h-full w-full object-cover transition-transform group-hover:scale-105"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <div className={cn('rounded-xl p-4', colorClass)}>
              <Icon className="h-8 w-8" />
            </div>
          </div>
        )}

        {/* Source badge */}
        <div className="absolute top-2 right-2 flex h-6 w-6 items-center justify-center rounded-full bg-background/80 backdrop-blur">
          <SourceIcon className="h-3 w-3 text-muted-foreground" />
        </div>
      </div>

      {/* Info */}
      <div className="flex flex-col gap-1 p-3">
        <h4 className="text-sm font-medium leading-tight line-clamp-2">
          {deliverable.title}
        </h4>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {deliverable.agent_name && (
            <>
              <span className="truncate max-w-[80px]">{deliverable.agent_name}</span>
              <span>·</span>
            </>
          )}
          <span>{formatDistanceToNow(new Date(deliverable.created_at), { addSuffix: true })}</span>
        </div>

        {deliverable.file_size_bytes && (
          <span className="text-[10px] text-muted-foreground/70">
            {formatFileSize(deliverable.file_size_bytes)}
          </span>
        )}
      </div>
    </button>
  )
})
```

### 4.5 Preview Slide-over Component

File: `frontend/components/workspace/gallery-view/deliverable-preview.tsx`

```tsx
'use client'

import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { X, Download, ExternalLink, FileText, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { apiClient } from '@/lib/api-client'
import { formatDistanceToNow } from 'date-fns'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface DeliverablePreviewProps {
  workspaceId: string
  deliverableId: string
  onClose: () => void
}

export function DeliverablePreview({
  workspaceId,
  deliverableId,
  onClose,
}: DeliverablePreviewProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['deliverable', deliverableId],
    queryFn: () => apiClient.get(`/api/deliverables/${deliverableId}?include_content=true`),
    enabled: !!deliverableId,
  })

  const deliverable = data?.deliverable

  // Close on escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose])

  const handleDownload = () => {
    if (deliverable?.file_path) {
      window.open(
        `/api/workspaces/${workspaceId}/files/download?path=${encodeURIComponent(deliverable.file_path)}`,
        '_blank'
      )
    }
  }

  const handleOpenInCanvas = () => {
    // Navigate to Code Canvas with file path
    window.location.href = `/workspace?view=explorer&path=${encodeURIComponent(deliverable?.file_path || '')}`
  }

  return (
    <Sheet open={true} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-2xl overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : deliverable ? (
          <>
            <SheetHeader className="space-y-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <SheetTitle className="text-xl">{deliverable.title}</SheetTitle>
                  <div className="flex items-center gap-2 mt-1 text-sm text-muted-foreground">
                    {deliverable.agent_name && (
                      <>
                        <span>{deliverable.agent_name}</span>
                        <span>·</span>
                      </>
                    )}
                    <span>{formatDistanceToNow(new Date(deliverable.created_at), { addSuffix: true })}</span>
                    <span>·</span>
                    <span className="capitalize">{deliverable.artifact_type}</span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={handleDownload}>
                  <Download className="mr-2 h-4 w-4" />
                  Download
                </Button>
                <Button variant="outline" size="sm" onClick={handleOpenInCanvas}>
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open in Canvas
                </Button>
              </div>
            </SheetHeader>

            {/* Content */}
            <div className="mt-6">
              {deliverable.artifact_type === 'image' ? (
                <div className="rounded-lg overflow-hidden border bg-muted/30">
                  <img
                    src={deliverable.content_url || deliverable.preview_url}
                    alt={deliverable.title}
                    className="w-full h-auto"
                  />
                </div>
              ) : deliverable.artifact_type === 'code' ? (
                <div className="rounded-lg overflow-hidden border">
                  <SyntaxHighlighter
                    language={getLanguageFromPath(deliverable.file_path)}
                    style={oneDark}
                    customStyle={{ margin: 0, borderRadius: 0 }}
                  >
                    {deliverable.content || ''}
                  </SyntaxHighlighter>
                </div>
              ) : deliverable.artifact_type === 'report' || deliverable.file_type?.includes('markdown') ? (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown>{deliverable.content || ''}</ReactMarkdown>
                </div>
              ) : deliverable.content ? (
                <pre className="p-4 rounded-lg bg-muted/30 border overflow-x-auto text-sm">
                  {deliverable.content}
                </pre>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-center border rounded-lg bg-muted/30">
                  <FileText className="h-12 w-12 text-muted-foreground/50 mb-4" />
                  <p className="text-muted-foreground">
                    Preview not available for this file type
                  </p>
                  <Button variant="link" size="sm" onClick={handleDownload} className="mt-2">
                    Download to view
                  </Button>
                </div>
              )}
            </div>

            {/* Summary */}
            {deliverable.summary && (
              <div className="mt-6 p-4 rounded-lg bg-muted/30 border">
                <h4 className="text-sm font-medium mb-2">Summary</h4>
                <p className="text-sm text-muted-foreground">{deliverable.summary}</p>
              </div>
            )}
          </>
        ) : (
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            Not found
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}

function getLanguageFromPath(path: string): string {
  const ext = path.split('.').pop()?.toLowerCase()
  const langMap: Record<string, string> = {
    py: 'python',
    ts: 'typescript',
    tsx: 'tsx',
    js: 'javascript',
    jsx: 'jsx',
    go: 'go',
    rs: 'rust',
    java: 'java',
    rb: 'ruby',
    php: 'php',
    swift: 'swift',
    kt: 'kotlin',
    css: 'css',
    html: 'html',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    md: 'markdown',
    sql: 'sql',
    sh: 'bash',
    bash: 'bash',
  }
  return langMap[ext || ''] || 'text'
}
```

### 4.6 View Toggle Component

File: `frontend/components/workspace/workspace-view-toggle.tsx`

```tsx
'use client'

import { LayoutGrid, FolderTree, Activity } from 'lucide-react'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { cn } from '@/lib/utils'

export type WorkspaceView = 'gallery' | 'explorer' | 'activity'

interface WorkspaceViewToggleProps {
  view: WorkspaceView
  onViewChange: (view: WorkspaceView) => void
}

export function WorkspaceViewToggle({ view, onViewChange }: WorkspaceViewToggleProps) {
  return (
    <ToggleGroup
      type="single"
      value={view}
      onValueChange={(v) => v && onViewChange(v as WorkspaceView)}
      className="bg-muted/50 p-1 rounded-lg"
    >
      <ToggleGroupItem
        value="gallery"
        aria-label="Gallery view"
        className={cn(
          'px-3 py-1.5 text-xs font-medium',
          view === 'gallery' && 'bg-background shadow-sm'
        )}
      >
        <LayoutGrid className="h-4 w-4 mr-1.5" />
        Outputs
      </ToggleGroupItem>
      <ToggleGroupItem
        value="explorer"
        aria-label="Explorer view"
        className={cn(
          'px-3 py-1.5 text-xs font-medium',
          view === 'explorer' && 'bg-background shadow-sm'
        )}
      >
        <FolderTree className="h-4 w-4 mr-1.5" />
        Explorer
      </ToggleGroupItem>
      <ToggleGroupItem
        value="activity"
        aria-label="Activity view"
        className={cn(
          'px-3 py-1.5 text-xs font-medium',
          view === 'activity' && 'bg-background shadow-sm'
        )}
      >
        <Activity className="h-4 w-4 mr-1.5" />
        Activity
      </ToggleGroupItem>
    </ToggleGroup>
  )
}
```

### 4.7 Main Workspace Page

File: `frontend/app/workspace/page.tsx` (updated)

```tsx
'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { useWorkspaceId } from '@/hooks/use-workspace-id'
import { WorkspaceViewToggle, WorkspaceView } from '@/components/workspace/workspace-view-toggle'
import { GalleryView } from '@/components/workspace/gallery-view'
import { WorkspaceExplorer } from '@/components/workspace/WorkspaceExplorer'
import { ActivityView } from '@/components/workspace/activity-view'

export default function WorkspacePage() {
  const workspaceId = useWorkspaceId()
  const searchParams = useSearchParams()
  
  // Default to gallery for non-tech users, but respect URL param
  const initialView = (searchParams.get('view') as WorkspaceView) || 'gallery'
  const [view, setView] = useState<WorkspaceView>(initialView)

  // Handle path param for explorer mode
  const pathParam = searchParams.get('path')
  
  useEffect(() => {
    if (pathParam) {
      setView('explorer')
    }
  }, [pathParam])

  if (!workspaceId) {
    return (
      <div className="flex items-center justify-center h-screen text-muted-foreground">
        Loading workspace...
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b shrink-0">
        <h1 className="text-lg font-semibold">Workspace</h1>
        <WorkspaceViewToggle view={view} onViewChange={setView} />
      </header>

      {/* Content */}
      <main className="flex-1 min-h-0">
        {view === 'gallery' && (
          <GalleryView workspaceId={workspaceId} />
        )}
        {view === 'explorer' && (
          <WorkspaceExplorer
            workspaceId={workspaceId}
            className="h-full"
          />
        )}
        {view === 'activity' && (
          <ActivityView workspaceId={workspaceId} />
        )}
      </main>
    </div>
  )
}
```

---

## 5. Integration Points

### 5.1 Auto-Register Deliverables

When agents write files, auto-register as deliverables:

**In `exec_workspace.py`** (workspace_write_file handler):

```python
async def execute_workspace_action(...):
    # ... existing write logic ...
    
    # After successful write, register deliverable for non-code files
    if action_name == "workspace_write_file" and result.get("success"):
        file_path = params.get("path", "")
        artifact_type = _infer_artifact_type(file_path)
        
        # Only register meaningful outputs (not .cache, .log, etc.)
        if artifact_type in ("report", "image", "document", "slide", "spreadsheet"):
            from services.deliverable_service import DeliverableService
            
            service = DeliverableService(db, workspace_id)
            await service.register(
                file_path=file_path,
                title=_generate_title_from_path(file_path),
                source_type=execution_context.get("source_type", "chat"),
                source_id=execution_context.get("source_id"),
                agent_id=agent_id,
                agent_name=agent_name,
            )
```

### 5.2 Wire Report Service

**In `report_service.py`** (after `create_report`):

```python
async def create_report(self, ...):
    # ... existing logic ...
    
    # After DB insert, also register as deliverable
    from services.deliverable_service import DeliverableService
    
    deliv_service = DeliverableService(self.db, self.workspace_id)
    await deliv_service.register(
        file_path=file_path,
        title=title,
        source_type="heartbeat" if heartbeat_result_id else "task",
        source_id=str(heartbeat_result_id) if heartbeat_result_id else None,
        agent_id=agent_id,
        agent_name=agent_name,
        artifact_type="report",
        summary=summary,
    )
```

---

## 6. Migration

File: `orchestrator/alembic/versions/prd129_deliverables.py`

```python
"""PRD-129: Deliverables Table

Revision ID: prd129_deliverables
Revises: prd128_notifications
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = 'prd129_deliverables'
down_revision = 'prd128_notifications'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'deliverables',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_type', sa.String(30), nullable=False),
        sa.Column('source_id', sa.String(255), nullable=True),
        sa.Column('agent_id', sa.Integer, sa.ForeignKey('agents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('agent_name', sa.String(100), nullable=True),
        sa.Column('artifact_type', sa.String(30), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('summary', sa.String(500), nullable=True),
        sa.Column('storage_type', sa.String(20), nullable=False, server_default='workspace'),
        sa.Column('file_path', sa.String(1024), nullable=False),
        sa.Column('file_name', sa.String(255), nullable=True),
        sa.Column('file_type', sa.String(50), nullable=True),
        sa.Column('file_size_bytes', sa.Integer, nullable=True),
        sa.Column('preview_url', sa.String(1024), nullable=True),
        sa.Column('preview_type', sa.String(30), nullable=True),
        sa.Column('metadata', JSONB, server_default='{}'),
        sa.Column('status', sa.String(20), server_default='ready'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    op.create_index('ix_deliverables_workspace', 'deliverables', ['workspace_id'])
    op.create_index('ix_deliverables_agent', 'deliverables', ['agent_id'])
    op.create_index('ix_deliverables_type', 'deliverables', ['workspace_id', 'artifact_type'])
    op.create_index('ix_deliverables_source', 'deliverables', ['workspace_id', 'source_type'])
    op.create_index('ix_deliverables_created', 'deliverables', ['workspace_id', sa.desc('created_at')])


def downgrade():
    op.drop_table('deliverables')
```

---

## 7. Implementation Phases

### Phase 1: Database & Service (2 days)
- [ ] Create `deliverables` table migration
- [ ] Implement `DeliverableService`
- [ ] Add `/api/deliverables` endpoints
- [ ] Unit tests for service

### Phase 2: Auto-Registration (1 day)
- [ ] Wire `workspace_write_file` to auto-register deliverables
- [ ] Wire `report_service.create_report()` to register
- [ ] Backfill existing reports as deliverables (one-time migration)

### Phase 3: Frontend Gallery (3 days)
- [ ] Implement `GalleryView` component
- [ ] Implement `FilterBar` component
- [ ] Implement `DeliverableCard` component
- [ ] Implement `DeliverablePreview` slide-over
- [ ] Implement `WorkspaceViewToggle`
- [ ] Update `/workspace` page

### Phase 4: Activity View (1 day)
- [ ] Implement `ActivityView` timeline component
- [ ] Wire to deliverables + notifications feed

### Phase 5: Polish & Testing (1 day)
- [ ] E2E test: agent writes file → appears in gallery → preview works
- [ ] Performance test: 500 deliverables grid scroll
- [ ] Responsive testing: mobile grid layout

---

## 8. UI/UX Checklist

Based on design system guidelines:

- [ ] No emojis as icons — using Lucide React throughout
- [ ] `cursor-pointer` on all clickable cards
- [ ] Hover states with smooth transitions (150-300ms)
- [ ] Text contrast 4.5:1 minimum in both light/dark modes
- [ ] Focus states visible for keyboard navigation
- [ ] `prefers-reduced-motion` respected on animations
- [ ] Responsive breakpoints: 375px, 768px, 1024px, 1440px
- [ ] Skeleton loading states for async content
- [ ] No horizontal scroll on mobile
- [ ] Grid columns: 2 (mobile) → 3 (sm) → 4 (md) → 5 (lg) → 6 (xl)

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Gallery load time | < 500ms | P95 for 100 deliverables |
| Card render performance | 60fps scroll | Chrome DevTools |
| Discovery success rate | > 80% | Users find recent output within 30s (user testing) |
| View mode preference | Track split | Gallery vs Explorer usage ratio |
| Preview engagement | > 50% | Deliverables previewed / deliverables viewed |

---

## 10. Future Enhancements (Not in Scope)

- **Bulk actions** — Select multiple deliverables for download/delete
- **Favorites/pinning** — Pin important outputs to top
- **Collections** — Group deliverables into user-defined collections
- **Sharing** — Generate public links to deliverables
- **Versioning** — Track versions of same-named files
- **AI summary** — Auto-generate summaries for documents/code
- **Thumbnail generation** — Generate previews for PDFs, slides
