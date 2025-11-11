-- Migration to add tools_schema field to skills table
-- This enables skills to define executable tool functions
-- PRD-22: Skills as Executable Tools

-- Add tools_schema column to skills table
ALTER TABLE skills
ADD COLUMN IF NOT EXISTS tools_schema JSONB DEFAULT NULL;

-- Add index for querying skills with tools
CREATE INDEX IF NOT EXISTS idx_skills_tools_schema ON skills USING GIN (tools_schema);

-- Add comment
COMMENT ON COLUMN skills.tools_schema IS 'OpenAI-compatible tool/function schemas that this skill provides';

-- Example tools_schema format:
-- {
--   "tools": [
--     {
--       "name": "create_pdf",
--       "description": "Create a PDF file from markdown content",
--       "parameters": {
--         "type": "object",
--         "properties": {
--           "source_file": {"type": "string", "description": "Path to markdown source"},
--           "output_file": {"type": "string", "description": "Path to output PDF"}
--         },
--         "required": ["source_file", "output_file"]
--       }
--     }
--   ]
-- }

-- Seed tools_schema for existing document creation skills
-- PDF Skill
UPDATE skills
SET tools_schema = '{
  "tools": [
    {
      "name": "create_pdf",
      "description": "Create a PDF file from markdown or text content using pandoc. Automatically handles formatting and styling.",
      "parameters": {
        "type": "object",
        "properties": {
          "source_file": {
            "type": "string",
            "description": "Path to the source markdown (.md) or text file to convert"
          },
          "output_file": {
            "type": "string",
            "description": "Path where the PDF file should be created (must end with .pdf)"
          },
          "title": {
            "type": "string",
            "description": "Document title (optional, will be extracted from file if not provided)"
          }
        },
        "required": ["source_file", "output_file"]
      }
    }
  ]
}'::jsonb
WHERE name = 'pdf' AND tools_schema IS NULL;

-- DOCX Skill
UPDATE skills
SET tools_schema = '{
  "tools": [
    {
      "name": "create_docx",
      "description": "Create a Microsoft Word (.docx) document from markdown or text content.",
      "parameters": {
        "type": "object",
        "properties": {
          "source_file": {
            "type": "string",
            "description": "Path to the source markdown (.md) or text file"
          },
          "output_file": {
            "type": "string",
            "description": "Path for the output Word document (must end with .docx)"
          },
          "title": {
            "type": "string",
            "description": "Document title"
          }
        },
        "required": ["source_file", "output_file"]
      }
    }
  ]
}'::jsonb
WHERE name = 'docx' AND tools_schema IS NULL;

-- XLSX Skill  
UPDATE skills
SET tools_schema = '{
  "tools": [
    {
      "name": "create_xlsx",
      "description": "Create an Excel (.xlsx) spreadsheet from CSV or structured data.",
      "parameters": {
        "type": "object",
        "properties": {
          "source_file": {
            "type": "string",
            "description": "Path to source CSV file or JSON data file"
          },
          "output_file": {
            "type": "string",
            "description": "Path for the output Excel file (must end with .xlsx)"
          },
          "sheet_name": {
            "type": "string",
            "description": "Name for the worksheet (optional, defaults to Sheet1)"
          }
        },
        "required": ["source_file", "output_file"]
      }
    }
  ]
}'::jsonb
WHERE name = 'xlsx' AND tools_schema IS NULL;

-- PPTX Skill
UPDATE skills
SET tools_schema = '{
  "tools": [
    {
      "name": "create_pptx",
      "description": "Create a PowerPoint (.pptx) presentation from markdown content.",
      "parameters": {
        "type": "object",
        "properties": {
          "source_file": {
            "type": "string",
            "description": "Path to source markdown file (# headers become slides)"
          },
          "output_file": {
            "type": "string",
            "description": "Path for the output PowerPoint file (must end with .pptx)"
          },
          "title": {
            "type": "string",
            "description": "Presentation title"
          }
        },
        "required": ["source_file", "output_file"]
      }
    }
  ]
}'::jsonb
WHERE name = 'pptx' AND tools_schema IS NULL;

-- Log the migration
INSERT INTO skill_audit_log (skill_id, action, action_details, status)
SELECT 
    id,
    'schema_update',
    '{"migration": "002_add_skill_tools_schema", "added": "tools_schema field and tool definitions"}'::jsonb,
    'success'
FROM skills
WHERE name IN ('pdf', 'docx', 'xlsx', 'pptx');

