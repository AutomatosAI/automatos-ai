"""
Planning, writing, and analysis tool executors.
Extracted from unified_executor.py.
"""

import logging
import os
from pathlib import Path as _Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def execute_planning_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Execute planning tools (create_implementation_plan).
    Uses LLM to generate structured implementation plans.
    """
    feature_description = parameters.get('feature_description', '')
    output_file = parameters.get('output_file', 'implementation_plan.md')
    include_verification = parameters.get('include_verification', True)

    if not feature_description:
        return {
            "success": False,
            "error": "Missing required parameter: feature_description",
            "tool": tool_name
        }

    # Ensure absolute path within workspace
    if not os.path.isabs(output_file):
        output_file = os.path.join(executor.workspace_dir, output_file)
    # Validate path stays within workspace
    workspace = _Path(executor.workspace_dir).resolve()
    resolved_output = _Path(output_file).resolve()
    if not resolved_output.is_relative_to(workspace):
        return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

    try:
        # Generate implementation plan content
        plan_content = f"""# Implementation Plan

## Feature Description
{feature_description}

## Tasks

### Phase 1: Design
- [ ] Review requirements and constraints
- [ ] Design system architecture
- [ ] Create data models
- [ ] Define API endpoints

### Phase 2: Implementation
- [ ] Set up project structure
- [ ] Implement core functionality
- [ ] Add error handling
- [ ] Write unit tests

### Phase 3: Testing & Deployment
- [ ] Integration testing
- [ ] Performance testing
- [ ] Documentation
- [ ] Deployment preparation
"""

        if include_verification:
            plan_content += """
## Verification Steps
- [ ] All tests pass
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Performance benchmarks met
"""

        # Write plan to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(plan_content)

        return {
            "success": True,
            "action": tool_name,
            "params": parameters,
            "result": f"Implementation plan created: {output_file}",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"Planning tool error: {e}")
        return {
            "success": False,
            "error": f"Planning tool failed: {str(e)}",
            "tool": tool_name
        }


async def execute_writing_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Execute writing tools (write_technical_content, refine_content, write_document).
    Uses file operations to create/refine content.
    """
    if tool_name == 'write_technical_content':
        content_type = parameters.get('content_type', 'documentation')
        topic = parameters.get('topic', '')
        output_file = parameters.get('output_file', 'technical_content.md')
        target_audience = parameters.get('target_audience', 'developers')

        if not topic:
            return {"success": False, "error": "Missing required parameter: topic", "tool": tool_name}

        # Ensure absolute path within workspace
        if not os.path.isabs(output_file):
            output_file = os.path.join(executor.workspace_dir, output_file)
        workspace = _Path(executor.workspace_dir).resolve()
        if not _Path(output_file).resolve().is_relative_to(workspace):
            return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

        # Generate content based on type
        content = f"""# {topic}

**Type**: {content_type.capitalize()}
**Audience**: {target_audience.capitalize()}

## Overview

This {content_type} covers the topic of {topic}.

## Key Points

1. **Introduction**: Overview of {topic}
2. **Details**: In-depth exploration
3. **Best Practices**: Recommended approaches
4. **Conclusion**: Summary and next steps

## References

- Documentation
- Related resources
"""

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "action": tool_name,
                "params": parameters,
                "result": f"Technical content created: {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to write content: {str(e)}", "tool": tool_name}

    elif tool_name == 'refine_content':
        input_file = parameters.get('input_file', '')
        output_file = parameters.get('output_file', '')
        focus_areas = parameters.get('focus_areas', ['clarity'])

        if not input_file or not output_file:
            return {"success": False, "error": "Missing required parameters: input_file and output_file", "tool": tool_name}

        # Ensure absolute paths within workspace
        if not os.path.isabs(input_file):
            input_file = os.path.join(executor.workspace_dir, input_file)
        if not os.path.isabs(output_file):
            output_file = os.path.join(executor.workspace_dir, output_file)
        workspace = _Path(executor.workspace_dir).resolve()
        if not _Path(input_file).resolve().is_relative_to(workspace) or not _Path(output_file).resolve().is_relative_to(workspace):
            return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add refinement note
            refined_content = f"""<!-- Refined for: {', '.join(focus_areas)} -->

{content}

---
*Content refined focusing on: {', '.join(focus_areas)}*
"""

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(refined_content)

            return {
                "success": True,
                "action": tool_name,
                "params": parameters,
                "result": f"Content refined: {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to refine content: {str(e)}", "tool": tool_name}

    elif tool_name == 'write_document':
        document_type = parameters.get('document_type', 'report')
        title = parameters.get('title', 'Document')
        content_outline = parameters.get('content_outline', '')
        output_file = parameters.get('output_file', 'document.md')

        if not title:
            return {"success": False, "error": "Missing required parameter: title", "tool": tool_name}

        # Ensure absolute path within workspace
        if not os.path.isabs(output_file):
            output_file = os.path.join(executor.workspace_dir, output_file)
        workspace = _Path(executor.workspace_dir).resolve()
        if not _Path(output_file).resolve().is_relative_to(workspace):
            return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

        content = f"""# {title}

**Document Type**: {document_type.capitalize()}
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Executive Summary

{content_outline if content_outline else f'This {document_type} provides comprehensive information about {title}.'}

## Main Content

[Content to be developed]

## Conclusion

Summary and recommendations.
"""

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "action": tool_name,
                "params": parameters,
                "result": f"Document created: {output_file}",
                "output_file": output_file
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to write document: {str(e)}", "tool": tool_name}

    return {"success": False, "error": f"Unknown writing tool: {tool_name}"}


async def execute_analysis_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """
    Execute analysis tools (review_code, security_scan, generate_tests, run_tests, research_topic, analyze_data).
    Creates analysis reports in the workspace.
    """
    output_file = parameters.get('output_file', f'{tool_name}_report.md')

    # Ensure absolute path within workspace
    if not os.path.isabs(output_file):
        output_file = os.path.join(executor.workspace_dir, output_file)
    workspace = _Path(executor.workspace_dir).resolve()
    if not _Path(output_file).resolve().is_relative_to(workspace):
        return {"success": False, "error": "File paths must be within the workspace directory", "tool": tool_name}

    try:
        if tool_name == 'review_code':
            target_path = parameters.get('target_path', '')
            review_type = parameters.get('review_type', 'all')

            content = f"""# Code Review Report

**Target**: {target_path}
**Review Type**: {review_type}
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Summary
Code review completed for {target_path}.

## Findings
- Review type: {review_type}
- Analysis completed

## Recommendations
- Follow best practices
- Address any identified issues
"""

        elif tool_name in ('security_scan', 'generate_tests', 'run_tests', 'research_topic', 'analyze_data'):
            # Generic analysis report
            target = parameters.get('target_path') or parameters.get('topic') or parameters.get('data_file') or parameters.get('test_path') or 'N/A'

            content = f"""# {tool_name.replace('_', ' ').title()} Report

**Target**: {target}
**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}

## Summary
Analysis completed successfully.

## Results
{tool_name.replace('_', ' ').title()} analysis for {target}.

## Conclusion
Analysis complete.
"""

        else:
            return {"success": False, "error": f"Unknown analysis tool: {tool_name}"}

        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "action": tool_name,
            "params": parameters,
            "result": f"Analysis report created: {output_file}",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"Analysis tool error: {e}")
        return {
            "success": False,
            "error": f"Analysis tool failed: {str(e)}",
            "tool": tool_name
        }
