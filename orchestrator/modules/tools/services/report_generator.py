"""
Report Generation Tool
======================

Allows LLM to create markdown reports that appear in the artifact viewer.
Reports are saved to /var/automatos/documents/ for download and reuse.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = "/var/automatos/documents/reports"


class ReportGenerator:
    """Generates and saves markdown reports to the artifact viewer"""
    
    def __init__(self):
        # Ensure reports directory exists
        os.makedirs(REPORTS_DIR, exist_ok=True)
        logger.info(f"ReportGenerator initialized, reports dir: {REPORTS_DIR}")
    
    async def create_report(
        self,
        title: str,
        content: str,
        category: str = "general"
    ) -> Dict[str, Any]:
        """
        Create a markdown report and save it to the documents directory.
        
        Args:
            title: Report title (used for filename)
            content: Markdown content of the report
            category: Report category (general, technical, analytics, etc.)
            
        Returns:
            Dict with success status, file path, and download URL
        """
        try:
            # Generate filename from title
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
            safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
            filename = f"{timestamp}_{safe_title}.md"
            
            # Full path
            file_path = os.path.join(REPORTS_DIR, filename)
            
            # Create markdown with frontmatter
            markdown_content = f"""---
title: {title}
category: {category}
generated: {datetime.now().isoformat()}
type: report
---

# {title}

{content}
"""
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Report created: {file_path}")
            
            # Return success with download info
            return {
                "success": True,
                "file_path": file_path,
                "filename": filename,
                "download_url": f"/api/documents/download?path={file_path}",
                "message": f"Report '{title}' created successfully and is available in the artifacts panel"
            }
            
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create report: {str(e)}"
            }
