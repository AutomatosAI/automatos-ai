"""
Document tool executors -- PDF, DOCX, XLSX, PPTX creation and generation.
Extracted from unified_executor.py.
"""

import logging
from pathlib import Path as _Path
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


async def execute_generate_document(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    PRD-63: Generate a polished document via DocumentGenerationService.

    Routes to AgentPlatformTools.execute_tool which handles workspace
    resolution, template selection, and file generation.
    """
    return await executor.platform_tools.execute_tool(
        tool_name="generate_document",
        parameters=parameters,
        agent_id=agent_id,
    )


async def execute_document_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    PRD-22: Execute document creation tools (PDF, DOCX, XLSX, PPTX).

    Uses pandoc for conversions and python libraries for document generation.
    """
    import subprocess

    source_file = parameters.get('source_file')
    output_file = parameters.get('output_file')
    title = parameters.get('title', 'Document')

    if not source_file or not output_file:
        return {
            "success": False,
            "error": "Missing required parameters: source_file and output_file",
            "tool": tool_name
        }

    # Resolve paths within workspace (prevent path traversal)
    workspace = _Path(executor.workspace_dir).resolve()
    resolved_source = (workspace / source_file).resolve()
    resolved_output = (workspace / output_file).resolve()
    source_file = str(resolved_source)
    output_file = str(resolved_output)

    if not resolved_source.is_relative_to(workspace) or not resolved_output.is_relative_to(workspace):
        return {
            "success": False,
            "error": "File paths must be within the workspace directory",
            "tool": tool_name
        }

    try:
        if tool_name == 'create_pdf':
            # Use pandoc to convert markdown to PDF
            cmd = [
                'pandoc',
                source_file,
                '-o', output_file,
                '--pdf-engine=pdflatex',
                '--metadata', f'title={title}',
                '--standalone'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {
                    "success": True,
                    "action": "create_pdf",
                    "params": parameters,
                    "result": f"PDF created successfully: {output_file}",
                    "output_file": output_file
                }
            else:
                # Fallback: Create simple text PDF if pandoc fails
                logger.warning(f"Pandoc failed, trying fallback method: {result.stderr}")
                return _create_simple_pdf_fallback(source_file, output_file, title, parameters)

        elif tool_name == 'create_docx':
            # Use pandoc for DOCX
            cmd = [
                'pandoc',
                source_file,
                '-o', output_file,
                '--metadata', f'title={title}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {
                    "success": True,
                    "action": "create_docx",
                    "params": parameters,
                    "result": f"DOCX created successfully: {output_file}",
                    "output_file": output_file
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create DOCX: {result.stderr}",
                    "tool": tool_name
                }

        elif tool_name == 'create_xlsx':
            # Create Excel file from CSV or JSON
            try:
                import pandas as pd

                # Detect source file type
                if source_file.endswith('.csv'):
                    df = pd.read_csv(source_file)
                elif source_file.endswith('.json'):
                    df = pd.read_json(source_file)
                else:
                    return {
                        "success": False,
                        "error": "Source file must be CSV or JSON for XLSX creation",
                        "tool": tool_name
                    }

                # Write to Excel
                sheet_name = parameters.get('sheet_name', 'Sheet1')
                df.to_excel(output_file, sheet_name=sheet_name, index=False)

                return {
                    "success": True,
                    "action": "create_xlsx",
                    "params": parameters,
                    "result": f"Excel file created successfully: {output_file}",
                    "output_file": output_file
                }
            except ImportError:
                return {
                    "success": False,
                    "error": "pandas library not available. Install with: pip install pandas openpyxl",
                    "tool": tool_name
                }

        elif tool_name == 'create_pptx':
            # Use pandoc for PPTX
            cmd = [
                'pandoc',
                source_file,
                '-o', output_file,
                '--metadata', f'title={title}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {
                    "success": True,
                    "action": "create_pptx",
                    "params": parameters,
                    "result": f"PowerPoint created successfully: {output_file}",
                    "output_file": output_file
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to create PPTX: {result.stderr}",
                    "tool": tool_name
                }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Document creation timed out after 60 seconds",
            "tool": tool_name
        }
    except Exception as e:
        logger.error(f"Document creation error: {e}")
        return {
            "success": False,
            "error": f"Document creation failed: {str(e)}",
            "tool": tool_name
        }


def _create_simple_pdf_fallback(
    source_file: str,
    output_file: str,
    title: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fallback method to create PDF using reportlab if pandoc fails.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        # Read source content
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create PDF
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 0.2*inch))

        # Add content (basic formatting)
        for line in content.split('\n'):
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

        doc.build(story)

        return {
            "success": True,
            "action": "create_pdf",
            "params": parameters,
            "result": f"PDF created successfully (using fallback method): {output_file}",
            "output_file": output_file,
            "method": "reportlab_fallback"
        }
    except ImportError:
        return {
            "success": False,
            "error": "Neither pandoc nor reportlab is available. Install pandoc or reportlab.",
            "tool": "create_pdf"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Fallback PDF creation failed: {str(e)}",
            "tool": "create_pdf"
        }
