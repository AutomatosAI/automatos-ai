# PRD-63: Document Generation Module

**Status:** Draft
**Priority:** P0 — Critical
**Created:** 2026-02-18
**Dependencies:** PRD-60 (RAG v3), PRD-08 (Document System)
**Estimated Effort:** MVP 16h | Core 28h | Full 42h

---

## Executive Summary

Automatos agents can **ingest** documents (PDF, DOCX, Markdown) and **query** them via RAG, but cannot **produce** polished business documents. This PRD adds a Document Generation module that enables agents and workflows to output professional PDF, DOCX, and XLSX files from data + templates — the "last mile" that turns AI analysis into deliverables businesses actually send to clients.

---

## Part 1: Competitive Landscape — Top 10 Document Generation Tools

### Comparison Matrix

| # | Tool | Stars | Language | License | Output Formats | Template Approach | FastAPI Fit |
|---|------|-------|----------|---------|----------------|-------------------|-------------|
| 1 | **Typst** | 46K | Rust | Apache-2.0 | PDF | Programmable markup | Moderate (CLI) |
| 2 | **Pandoc** | 41.5K | Haskell | GPL-2.0 | 40+ formats | Markdown conversion | Moderate (CLI) |
| 3 | **Gotenberg** | 11.3K | Go | MIT | PDF (from anything) | HTTP API + Chromium/LibreOffice | Excellent |
| 4 | **WeasyPrint** | 8.5K | Python | BSD | PDF | HTML + CSS | Excellent |
| 5 | **python-docx** | 5.4K | Python | MIT | DOCX | Programmatic API | Excellent |
| 6 | **pdfme** | 4K | TypeScript | MIT | PDF | JSON schema + WYSIWYG designer | Frontend only |
| 7 | **Docxtemplater** | 3.5K | JavaScript | MIT (core) | DOCX, PPTX | {tag} placeholders | Poor (Node) |
| 8 | **XlsxWriter** | 3.5K | Python | BSD | XLSX | Programmatic API | Excellent |
| 9 | **python-docx-template** | 2.3K | Python | LGPL-2.1 | DOCX | Jinja2 in Word files | Excellent |
| 10 | **Carbone** | 1.6K | JavaScript | CCL (restrictive) | 15+ formats | Tags in Office files | Blocked (license) |

### How the Best Tools Work

#### Typst (46K stars) — The Modern LaTeX Killer
- Programmable markup language compiled to PDF in milliseconds
- Loops, conditionals, functions — code meets document
- Incremental compilation, WASM build available
- **Takeaway:** Watch for future; ecosystem too young for enterprise templates today

#### Gotenberg (11.3K stars) — The Conversion Swiss Army Knife
- Docker container with Chromium + LibreOffice inside
- Send HTML/DOCX/XLSX/Markdown via HTTP → get PDF back
- Stateless, scales horizontally, 50M+ Docker pulls
- Python client: `gotenberg-client` on PyPI
- **Takeaway:** Perfect conversion sidecar — don't build what Gotenberg already does

#### WeasyPrint (8.5K stars) — HTML/CSS → PDF, Pure Python
- pip-installable, BSD license, actively maintained (Feb 2026 release)
- Full CSS print support: `@page` rules, headers/footers, page breaks, margins
- Pairs with Jinja2 for template rendering
- **Takeaway:** The go-to Python PDF engine. Jinja2 + WeasyPrint is the standard pattern.

#### python-docx-template (2.3K stars) — Jinja2 in Word Files
- Business users design templates in Microsoft Word
- Developers add `{{ variable }}` and `{% for item in items %}` tags
- Renders to DOCX with full formatting preserved
- Built on python-docx (already in our requirements.txt)
- **Takeaway:** Best DOCX solution. Non-technical users can design templates.

#### pdfme (4K stars) — WYSIWYG PDF Template Designer
- React component for visual PDF template design
- Drag-and-drop fields, JSON schema output
- MIT license, browser-based
- **Takeaway:** Could power a frontend template designer. Generation stays in Python.

### Tools We Skip (and Why)

| Tool | Reason |
|------|--------|
| **Carbone** | CCL license prohibits hosted SaaS without commercial license |
| **Docxtemplater** | Node.js only; paid modules for images/charts/XLSX — adds language complexity |
| **ReportLab** | Too low-level; Jinja2+WeasyPrint achieves same output with 1/5 the code |
| **Pandoc** | Document converter, not report generator; GPL friction; requires LaTeX for PDF |
| **Typst** | Promising but ecosystem too young; revisit in 6-12 months |

### 2026 Industry Pattern: LLM + Templates

The emerging standard for AI document generation:

```
Agent produces structured JSON  →  Template engine merges with design  →  Polished document
```

No single open-source project packages this end-to-end. The opportunity is to build the pipeline from composable, best-in-class tools.

---

## Part 2: Current Automatos State

### What Exists Today

#### Backend

| Component | File | Status |
|-----------|------|--------|
| **ReportGenerator** | `orchestrator/modules/tools/services/report_generator.py` | ✅ Working — Markdown reports with YAML frontmatter |
| **python-docx** | `requirements.txt` | ✅ Installed (v1.1.0) — but **unused** for generation |
| **PandasAI charts** | `orchestrator/modules/tools/services/pandas_ai_service.py` | ✅ Working — matplotlib/seaborn PNG charts |
| **Document upload/storage** | `orchestrator/api/documents.py` | ✅ Full CRUD, S3 optional |
| **PDF text extraction** | `pdfplumber`, `PyPDF2` | ✅ Working — ingestion only, not generation |
| **Artifact model** | `orchestrator/core/models/core.py` | ✅ DB model with kinds: code, text, image, sheet |

#### Frontend

| Component | File | Status |
|-----------|------|--------|
| **Artifact viewer** | `frontend/components/chatbot/artifact-viewer.tsx` | ✅ Fullscreen viewer with download (text blob only) |
| **Document management** | `frontend/components/documents/document-management.tsx` | ✅ Upload, list, search, tabs |
| **Chart rendering** | `chart.js`, `plotly.js`, `d3`, `recharts` | ✅ All installed |
| **Markdown rendering** | `react-markdown`, `remark-gfm` | ✅ Working |

#### What's Missing

| Capability | Gap |
|------------|-----|
| **PDF generation** | No WeasyPrint, no HTML→PDF pipeline |
| **DOCX generation** | python-docx installed but no template system |
| **XLSX export** | No XlsxWriter or openpyxl |
| **Template management** | No template CRUD, no template storage |
| **Template designer UI** | No visual template editor |
| **Multi-format download** | Artifact viewer only exports text blobs |
| **Agent document tool** | No `generate_document` tool for agents |
| **Workflow document step** | No document generation step in recipes |
| **Document artifact kind** | Artifact model lacks "document" kind |

### Architecture Assessment

**Strengths we build on:**
- ReportGenerator already creates markdown reports agents can reference
- Artifact model + viewer provides the delivery mechanism
- PandasAI already generates charts that could embed in documents
- python-docx already installed — just needs template layer
- Document storage (local + S3) ready for generated files

**Critical gap:** There is no path from "agent has data" → "user gets a polished PDF/DOCX." Everything stops at Markdown or raw text.

---

## Part 3: Recommended Tech Stack

### 3-Layer Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Template Engines (per format)          │
│  ├── PDF:   Jinja2 + WeasyPrint (HTML/CSS)      │
│  ├── DOCX:  python-docx-template (Jinja2 tags)  │
│  └── XLSX:  XlsxWriter (programmatic)           │
├─────────────────────────────────────────────────┤
│  Layer 2: Conversion Sidecar                     │
│  └── Gotenberg (Docker) — DOCX/XLSX → PDF       │
├─────────────────────────────────────────────────┤
│  Layer 3: AI Integration                         │
│  └── Agent → JSON schema → Template → Document   │
└─────────────────────────────────────────────────┘
```

### New Dependencies

**Backend (pip):**
```
weasyprint>=62.0          # HTML/CSS → PDF
docxtpl>=0.18.0           # Jinja2 in Word templates
xlsxwriter>=3.2.0         # Excel generation
gotenberg-client>=1.0.0   # Gotenberg API client (optional)
```

**System (for WeasyPrint):**
```bash
# macOS
brew install pango cairo libffi

# Ubuntu/Docker
apt-get install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0
```

**Frontend (npm) — Phase 7 only:**
```
@pdfme/ui          # WYSIWYG PDF template designer (optional, future)
```

---

## Part 4: Implementation Phases

### Phase 1: Template Management System (4h)

**Goal:** CRUD for document templates with versioning

#### 1.1 Database Schema

```sql
CREATE TABLE document_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    format VARCHAR(20) NOT NULL CHECK (format IN ('pdf', 'docx', 'xlsx')),

    -- Template content
    template_content TEXT,                    -- HTML/CSS for PDF, null for DOCX
    template_file_path VARCHAR(500),          -- Path to .docx template file

    -- Schema definition: what variables the template expects
    data_schema JSONB NOT NULL DEFAULT '{}',  -- JSON Schema for input data

    -- Sample data for preview
    sample_data JSONB DEFAULT '{}',

    -- Metadata
    category VARCHAR(100) DEFAULT 'general',  -- report, invoice, contract, letter, proposal
    tags TEXT[] DEFAULT '{}',
    thumbnail_url VARCHAR(500),

    -- Versioning
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,

    -- Audit
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(workspace_id, name, version)
);

CREATE INDEX idx_templates_workspace ON document_templates(workspace_id);
CREATE INDEX idx_templates_format ON document_templates(format);
CREATE INDEX idx_templates_category ON document_templates(category);
```

#### 1.2 API Endpoints

```
POST   /api/documents/templates              — Create template
GET    /api/documents/templates              — List templates (filter by format, category)
GET    /api/documents/templates/{id}         — Get template with schema
PUT    /api/documents/templates/{id}         — Update template
DELETE /api/documents/templates/{id}         — Delete template
POST   /api/documents/templates/{id}/preview — Preview with sample data
POST   /api/documents/templates/upload       — Upload .docx template file
```

#### 1.3 Built-in Starter Templates

Ship with 5 pre-built templates:

| Template | Format | Category | Variables |
|----------|--------|----------|-----------|
| Basic Report | PDF | report | title, date, sections[], author |
| Invoice | PDF | invoice | company, client, line_items[], total, due_date |
| Meeting Notes | DOCX | report | title, date, attendees[], agenda[], action_items[] |
| Data Export | XLSX | data | title, columns[], rows[] |
| Executive Summary | PDF | report | title, date, highlights[], metrics{}, recommendations[] |

---

### Phase 2: PDF Generation Engine (4h)

**Goal:** Jinja2 + WeasyPrint pipeline for HTML/CSS → PDF

#### 2.1 Service: `DocumentGenerationService`

**File:** `orchestrator/modules/documents/generation_service.py`

```python
class DocumentGenerationService:
    """Generates documents from templates + data"""

    async def generate_pdf(
        self,
        template_id: UUID,
        data: dict,
        workspace_id: UUID
    ) -> GeneratedDocument:
        """
        1. Load template HTML/CSS from document_templates
        2. Validate data against template's data_schema
        3. Render Jinja2 template with data
        4. Convert HTML → PDF via WeasyPrint
        5. Store PDF in document storage
        6. Return GeneratedDocument with download URL
        """

    async def generate_docx(
        self,
        template_id: UUID,
        data: dict,
        workspace_id: UUID
    ) -> GeneratedDocument:
        """Uses python-docx-template (docxtpl)"""

    async def generate_xlsx(
        self,
        data: dict,
        workspace_id: UUID,
        template_id: UUID = None
    ) -> GeneratedDocument:
        """Uses XlsxWriter"""

    async def convert_to_pdf(
        self,
        source_path: str,
        source_format: str
    ) -> str:
        """DOCX/XLSX → PDF via Gotenberg (if available) or LibreOffice CLI fallback"""
```

#### 2.2 PDF Template System

Templates are HTML + CSS with Jinja2, stored in `template_content`:

```html
<!DOCTYPE html>
<html>
<head>
<style>
  @page {
    size: A4;
    margin: 2cm;
    @top-center { content: "{{ company_name }}"; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); }
  }
  body { font-family: 'Inter', sans-serif; color: #1a1a2e; }
  .header { border-bottom: 3px solid #ff6b35; padding-bottom: 1rem; }
  .metric-card {
    display: inline-block; width: 30%;
    background: #f8f9fa; border-radius: 8px; padding: 1rem;
  }
  table { width: 100%; border-collapse: collapse; }
  th { background: #1a1a2e; color: white; padding: 0.75rem; }
  td { padding: 0.5rem; border-bottom: 1px solid #eee; }
</style>
</head>
<body>
  <div class="header">
    <h1>{{ title }}</h1>
    <p>Generated: {{ date }} | Author: {{ author }}</p>
  </div>

  {% for section in sections %}
  <h2>{{ section.title }}</h2>
  <p>{{ section.content }}</p>
  {% endfor %}

  {% if metrics %}
  <div class="metrics">
    {% for key, value in metrics.items() %}
    <div class="metric-card">
      <div class="label">{{ key }}</div>
      <div class="value">{{ value }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
</body>
</html>
```

#### 2.3 Chart Embedding

Leverage existing PandasAI chart generation:

```python
async def _embed_charts(self, html: str, data: dict) -> str:
    """Replace {{ chart:field_name }} tags with base64 PNG images"""
    for chart_ref in re.findall(r'\{\{ chart:(\w+) \}\}', html):
        if chart_ref in data.get('_charts', {}):
            b64 = data['_charts'][chart_ref]
            html = html.replace(
                f'{{{{ chart:{chart_ref} }}}}',
                f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
            )
    return html
```

---

### Phase 3: DOCX Generation Engine (3h)

**Goal:** python-docx-template for Word document output

#### 3.1 Template Upload Flow

1. User uploads `.docx` file with Jinja2 tags via `/api/documents/templates/upload`
2. Backend extracts variable names from `{{ }}` tags
3. Auto-generates `data_schema` from discovered variables
4. Stores `.docx` in template storage path
5. User can preview with sample data

#### 3.2 DOCX Rendering

```python
from docxtpl import DocxTemplate

async def generate_docx(self, template_id, data, workspace_id):
    template = await self._load_template(template_id)

    doc = DocxTemplate(template.template_file_path)

    # Handle images if present
    for key, value in data.items():
        if isinstance(value, dict) and value.get('_type') == 'image':
            data[key] = InlineImage(doc, value['path'], width=Mm(value.get('width', 150)))

    doc.render(data)

    output_path = self._generate_output_path(workspace_id, template.name, 'docx')
    doc.save(output_path)

    return GeneratedDocument(
        path=output_path,
        format='docx',
        filename=f"{template.name}_{datetime.now().strftime('%Y%m%d')}.docx",
        size=os.path.getsize(output_path)
    )
```

---

### Phase 4: XLSX Export Engine (2h)

**Goal:** XlsxWriter for structured data export

#### 4.1 Two Modes

**Mode A: Data Export** — NL2SQL results, database queries, any tabular data
```python
async def generate_xlsx_from_data(self, title, columns, rows, workspace_id):
    """Direct data → Excel without template"""
    output_path = self._generate_output_path(workspace_id, title, 'xlsx')

    workbook = xlsxwriter.Workbook(output_path)
    worksheet = workbook.add_worksheet(title[:31])  # Excel 31-char limit

    # Header formatting
    header_fmt = workbook.add_format({
        'bold': True, 'bg_color': '#1a1a2e', 'font_color': 'white',
        'border': 1, 'text_wrap': True
    })

    # Write headers
    for col, name in enumerate(columns):
        worksheet.write(0, col, name, header_fmt)

    # Write data with auto-formatting
    for row_idx, row in enumerate(rows, 1):
        for col_idx, value in enumerate(row):
            if isinstance(value, (int, float)):
                worksheet.write_number(row_idx, col_idx, value)
            elif isinstance(value, datetime):
                worksheet.write_datetime(row_idx, col_idx, value)
            else:
                worksheet.write_string(row_idx, col_idx, str(value))

    # Auto-fit columns
    for col, name in enumerate(columns):
        max_width = max(len(str(name)), max(len(str(r[col])) for r in rows) if rows else 0)
        worksheet.set_column(col, col, min(max_width + 2, 50))

    workbook.close()
    return GeneratedDocument(path=output_path, format='xlsx', ...)
```

**Mode B: Template-based** — Formatted reports with charts, multiple sheets, styling

---

### Phase 5: Agent Tool Integration (3h)

**Goal:** Agents can generate documents via function calling

#### 5.1 Agent Tool Schema

```python
GENERATE_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_document",
        "description": "Generate a polished PDF, DOCX, or XLSX document from data. Use this when the user asks for a report, invoice, export, or any formatted document.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Document title"
                },
                "format": {
                    "type": "string",
                    "enum": ["pdf", "docx", "xlsx"],
                    "description": "Output format"
                },
                "template_name": {
                    "type": "string",
                    "description": "Template to use (e.g. 'Basic Report', 'Invoice'). Omit for auto-selection."
                },
                "data": {
                    "type": "object",
                    "description": "Data to populate the template — must match the template's expected schema"
                }
            },
            "required": ["title", "format", "data"]
        }
    }
}
```

#### 5.2 Integration in Agent Factory

**File to modify:** `orchestrator/modules/agents/services/agent_platform_tools.py`

Add `generate_document` alongside existing tools (`search_codebase`, `query_database`, etc.):

```python
async def _handle_generate_document(self, args: dict) -> str:
    service = DocumentGenerationService(self.db, self.workspace_id)

    result = await service.generate(
        title=args['title'],
        format=args['format'],
        data=args['data'],
        template_name=args.get('template_name'),
    )

    return json.dumps({
        "status": "success",
        "filename": result.filename,
        "format": result.format,
        "download_url": result.download_url,
        "size_kb": result.size // 1024
    })
```

#### 5.3 New Artifact Kind

Extend the Artifact model:

```sql
ALTER TABLE artifacts DROP CONSTRAINT IF EXISTS artifacts_kind_check;
ALTER TABLE artifacts ADD CONSTRAINT artifacts_kind_check
    CHECK (kind IN ('code', 'text', 'image', 'sheet', 'document'));
```

Frontend type update:
```typescript
type ArtifactKind = 'code' | 'text' | 'image' | 'sheet' | 'document'
```

---

### Phase 6: Workflow Document Step (2h)

**Goal:** Document generation as a recipe step type

#### 6.1 New Step Type: `generate_document`

```python
# In recipe executor
async def _execute_step(self, step, context):
    if step['type'] == 'generate_document':
        service = DocumentGenerationService(self.db, self.workspace_id)

        # Resolve template variables from prior step outputs
        data = self._resolve_variables(step['config']['data'], context)

        result = await service.generate(
            title=step['config']['title'],
            format=step['config']['format'],
            data=data,
            template_name=step['config'].get('template_name'),
        )

        return {"document_url": result.download_url, "filename": result.filename}
```

#### 6.2 Example Recipe: Weekly Report

```json
{
    "name": "Weekly Team Report",
    "steps": [
        {
            "type": "agent",
            "config": {
                "prompt": "Query the database for this week's metrics: tickets closed, PRs merged, incidents resolved. Return as JSON with keys: tickets_closed, prs_merged, incidents, highlights (array of strings)."
            }
        },
        {
            "type": "generate_document",
            "config": {
                "title": "Weekly Team Report",
                "format": "pdf",
                "template_name": "Basic Report",
                "data": {
                    "title": "Weekly Team Report — {{ current_date }}",
                    "author": "Automatos AI",
                    "sections": [
                        {"title": "Key Metrics", "content": "{{ step_1.output }}"},
                        {"title": "Highlights", "content": "{{ step_1.highlights }}"}
                    ]
                }
            }
        },
        {
            "type": "agent",
            "config": {
                "prompt": "Email the report at {{ step_2.document_url }} to the team distribution list."
            }
        }
    ]
}
```

---

### Phase 7: Frontend — Template Manager & Document Viewer (5h)

**Goal:** UI for managing templates, previewing documents, and downloading generated files

#### 7.1 Template Manager Component

**File:** `frontend/components/documents/template-manager.tsx`

Features:
- Grid view of templates with thumbnails, format badges, category filters
- Create/edit template modal:
  - PDF: HTML/CSS code editor (Monaco) with live preview
  - DOCX: Upload .docx file, auto-detect variables, show schema
  - XLSX: Column/sheet configurator
- Sample data editor (JSON) with "Preview" button
- Template versioning (view history, rollback)
- Duplicate/export template

#### 7.2 Document Artifact Renderer

**File:** Update `frontend/components/chatbot/artifact-viewer.tsx`

Add `'document'` kind handler:

```typescript
case 'document':
  return (
    <div className="document-artifact">
      <div className="flex items-center gap-3 mb-4">
        <FileText className="h-8 w-8 text-orange-500" />
        <div>
          <h3 className="font-semibold">{artifact.title}</h3>
          <p className="text-sm text-muted-foreground">
            {artifact.metadata?.format?.toUpperCase()} • {artifact.metadata?.size_kb}KB
          </p>
        </div>
      </div>

      {/* PDF preview via iframe or embedded viewer */}
      {artifact.metadata?.format === 'pdf' && (
        <iframe
          src={artifact.metadata?.preview_url}
          className="w-full h-[600px] rounded-lg border"
        />
      )}

      <div className="flex gap-2 mt-4">
        <Button onClick={() => downloadDocument(artifact.metadata?.download_url)}>
          <Download className="h-4 w-4 mr-2" /> Download {artifact.metadata?.format?.toUpperCase()}
        </Button>
        {artifact.metadata?.format !== 'pdf' && (
          <Button variant="outline" onClick={() => convertToPdf(artifact.id)}>
            <FileText className="h-4 w-4 mr-2" /> Convert to PDF
          </Button>
        )}
      </div>
    </div>
  )
```

#### 7.3 Integration Points

- **Document Management tab:** Add "Templates" sub-tab alongside existing Documents/CodeGraph tabs
- **Chat:** Agent returns document artifacts inline with preview + download
- **Workflow builder:** "Generate Document" step type in recipe editor with template picker and data mapper

---

### Phase 8: Gotenberg Sidecar (2h) — Optional

**Goal:** DOCX/XLSX → PDF conversion via Docker sidecar

#### 8.1 Docker Compose Addition

```yaml
services:
  gotenberg:
    image: gotenberg/gotenberg:8
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GOTENBERG_API_TIMEOUT=120s
      - GOTENBERG_LOG_LEVEL=info
```

#### 8.2 Conversion Service

```python
from gotenberg_client import GotenbergClient

class ConversionService:
    def __init__(self):
        self.gotenberg_url = os.getenv('GOTENBERG_URL', 'http://gotenberg:3000')

    async def docx_to_pdf(self, docx_path: str) -> str:
        async with GotenbergClient(self.gotenberg_url) as client:
            with open(docx_path, 'rb') as f:
                response = await client.libre_office.convert(f)
                pdf_path = docx_path.replace('.docx', '.pdf')
                with open(pdf_path, 'wb') as out:
                    out.write(response.content)
                return pdf_path

    async def xlsx_to_pdf(self, xlsx_path: str) -> str:
        # Same pattern with LibreOffice conversion
        ...
```

#### 8.3 Graceful Degradation

If Gotenberg is unavailable, fall back to:
1. LibreOffice CLI (`libreoffice --headless --convert-to pdf`)
2. Or simply return the native format with a "PDF conversion unavailable" message

---

## File Change Summary

### New Files

**Backend:**
| File | Purpose |
|------|---------|
| `orchestrator/modules/documents/__init__.py` | Module init |
| `orchestrator/modules/documents/generation_service.py` | Core generation engine |
| `orchestrator/modules/documents/conversion_service.py` | Gotenberg/LibreOffice conversion |
| `orchestrator/modules/documents/template_service.py` | Template CRUD operations |
| `orchestrator/api/document_generation.py` | REST API endpoints |
| `orchestrator/alembic/versions/YYYYMMDD_add_document_templates.py` | DB migration |
| `orchestrator/modules/documents/templates/` | Built-in starter templates (HTML/CSS) |

**Frontend:**
| File | Purpose |
|------|---------|
| `frontend/components/documents/template-manager.tsx` | Template CRUD UI |
| `frontend/components/documents/template-editor.tsx` | HTML/CSS + preview editor |
| `frontend/components/documents/template-picker.tsx` | Template selection modal (for workflows) |

### Modified Files

| File | Change |
|------|--------|
| `orchestrator/requirements.txt` | Add weasyprint, docxtpl, xlsxwriter, gotenberg-client |
| `orchestrator/core/models/core.py` | Add DocumentTemplate model, update Artifact kind constraint |
| `orchestrator/modules/agents/services/agent_platform_tools.py` | Add generate_document tool |
| `orchestrator/api/workflow_recipes.py` | Add generate_document step type |
| `frontend/types/chat.ts` | Add 'document' to ArtifactKind |
| `frontend/components/chatbot/artifact-viewer.tsx` | Add document renderer |
| `frontend/components/documents/document-management.tsx` | Add Templates tab |
| `docker-compose.yml` | Add Gotenberg service (Phase 8) |

---

## Priority Matrix

| Phase | What | Effort | Value | Priority |
|-------|------|--------|-------|----------|
| **Phase 1** | Template Management | 4h | Foundation for everything | P0 |
| **Phase 2** | PDF Generation (WeasyPrint) | 4h | Highest-demand format | P0 |
| **Phase 5** | Agent Tool Integration | 3h | "Generate a report" in chat | P0 |
| **Phase 3** | DOCX Generation | 3h | Business document output | P1 |
| **Phase 4** | XLSX Export | 2h | Data export capability | P1 |
| **Phase 6** | Workflow Document Step | 2h | Automated report pipelines | P1 |
| **Phase 7** | Frontend Template Manager | 5h | Full self-service UI | P2 |
| **Phase 8** | Gotenberg Sidecar | 2h | Cross-format conversion | P2 |

**MVP (Phases 1+2+5): 11h** — Templates + PDF generation + agent integration
**Core (+ Phases 3+4+6): 18h** — Add DOCX, XLSX, workflow step
**Full (+ Phases 7+8): 25h** — Template designer UI + conversion sidecar

---

## Success Criteria

- [ ] Agent can generate a PDF report from chat: "Generate a monthly report from our sales data"
- [ ] Generated documents are professional quality (headers, footers, page numbers, branding)
- [ ] Templates are workspace-scoped and manageable via API
- [ ] DOCX templates support business user design (edit in Word, upload)
- [ ] XLSX export works for NL2SQL query results
- [ ] Workflow recipes can include document generation steps
- [ ] Documents appear as artifacts in chat with inline preview + download
- [ ] At least 5 starter templates ship out of the box

---

## User Stories

**As a sales manager**, I want to say "Generate this quarter's sales report as a PDF" and get a polished document with charts and metrics, so I can send it to leadership without manual formatting.

**As an operations lead**, I want a weekly recipe that automatically pulls Jira data, generates a status report, and emails it to stakeholders every Monday at 9am.

**As a finance analyst**, I want to export my NL2SQL query results as a formatted Excel spreadsheet with proper headers and number formatting.

**As an admin**, I want to upload our company's report template (branded DOCX) and have agents use it automatically when generating documents.

---

## Integration with Existing PRDs

| PRD | Integration Point |
|-----|-------------------|
| **PRD-60 (RAG v3)** | RAG query results → document sections. "Summarize our docs into a report" |
| **PRD-61 (NL2SQL v2)** | Query results → XLSX export or PDF table. "Export this query as Excel" |
| **PRD-62 (CodeGraph v2)** | Code analysis → technical documentation. "Generate API docs from codebase" |
| **PRD-09 (Context Engineering)** | Context data feeds document content |
| **PRD-58 (Prompt Management)** | System prompts for document generation style/tone |

---

## Out of Scope (Future)

- Real-time collaborative document editing (Google Docs-style)
- Presentation/slide generation (PPTX) — could add via python-pptx later
- pdfme WYSIWYG frontend designer — revisit after core engine ships
- Document signing / approval workflows
- Batch document generation (mail merge for 1000+ documents)
- Custom fonts/branding per workspace (use CSS for now)

---

**Estimated Total Effort:** MVP 11h | Core 18h | Full 25h
**Priority:** P0 — Critical (enables the "AI → deliverable" pipeline)
**Dependencies:** PRD-08 Document System (completed ✅), WeasyPrint system deps
