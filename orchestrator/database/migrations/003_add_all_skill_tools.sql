-- Migration to add tools_schema to ALL key skills
-- This enables ALL skills to define executable tools for agents (PRD-22 expansion)

-- Update 'writing-plans' skill to include a 'create_plan' tool
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "create_implementation_plan",
                "description": "Create a detailed implementation plan for a feature or project with tasks, dependencies, and verification steps",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "feature_description": {
                            "type": "string",
                            "description": "Description of the feature or project to plan"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the output plan file (e.g., implementation_plan.md)"
                        },
                        "include_verification": {
                            "type": "boolean",
                            "description": "Include verification steps for each task",
                            "default": true
                        }
                    },
                    "required": ["feature_description", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'writing-plans' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'writing-skills' skill to include content creation tools
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "write_technical_content",
                "description": "Write technical documentation, reports, or articles with proper structure and formatting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "enum": ["documentation", "report", "article", "tutorial"],
                            "description": "Type of content to write"
                        },
                        "topic": {
                            "type": "string",
                            "description": "Main topic or subject"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the output file (e.g., technical_report.md)"
                        },
                        "target_audience": {
                            "type": "string",
                            "description": "Target audience (e.g., developers, managers, general)",
                            "default": "developers"
                        }
                    },
                    "required": ["content_type", "topic", "output_file"]
                }
            },
            {
                "name": "refine_content",
                "description": "Refine and improve existing written content for clarity, structure, and quality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_file": {
                            "type": "string",
                            "description": "Path to the input file to refine"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the refined output file"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["clarity", "structure", "grammar", "technical_accuracy", "tone"]
                            },
                            "description": "Areas to focus on during refinement"
                        }
                    },
                    "required": ["input_file", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'writing-skills' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Code Reviewer' skill
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "review_code",
                "description": "Perform code review on files or directories, checking for issues, best practices, and security",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Path to file or directory to review"
                        },
                        "review_type": {
                            "type": "string",
                            "enum": ["security", "performance", "style", "all"],
                            "description": "Type of review to perform",
                            "default": "all"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the review report (e.g., code_review.md)"
                        }
                    },
                    "required": ["target_path", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Code Reviewer' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Security Expert' skill
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "security_scan",
                "description": "Scan code for security vulnerabilities, including SQL injection, XSS, and other common issues",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "string",
                            "description": "Path to file or directory to scan"
                        },
                        "scan_depth": {
                            "type": "string",
                            "enum": ["quick", "standard", "deep"],
                            "description": "Depth of security scan",
                            "default": "standard"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the security report (e.g., security_scan.md)"
                        }
                    },
                    "required": ["target_path", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Security Expert' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Tester' skill
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "generate_tests",
                "description": "Generate unit tests for code files or modules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_file": {
                            "type": "string",
                            "description": "Path to source file to generate tests for"
                        },
                        "test_file": {
                            "type": "string",
                            "description": "Path for the generated test file"
                        },
                        "test_framework": {
                            "type": "string",
                            "enum": ["pytest", "unittest", "jest", "mocha"],
                            "description": "Test framework to use",
                            "default": "pytest"
                        }
                    },
                    "required": ["source_file", "test_file"]
                }
            },
            {
                "name": "run_tests",
                "description": "Run tests in a directory or file and generate a report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_path": {
                            "type": "string",
                            "description": "Path to test file or directory"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the test report (e.g., test_results.txt)"
                        }
                    },
                    "required": ["test_path", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Tester' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Researcher' skill
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "research_topic",
                "description": "Research a topic using available knowledge sources and create a comprehensive report",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to research"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the research report (e.g., research_findings.md)"
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["overview", "standard", "comprehensive"],
                            "description": "Depth of research",
                            "default": "standard"
                        }
                    },
                    "required": ["topic", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Researcher' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Data Analysit' skill (note: name has typo in DB)
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "analyze_data",
                "description": "Analyze data from CSV, JSON, or other files and generate insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file": {
                            "type": "string",
                            "description": "Path to data file to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["summary", "trends", "anomalies", "correlations"],
                            "description": "Type of analysis to perform",
                            "default": "summary"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the analysis report (e.g., data_analysis.md)"
                        }
                    },
                    "required": ["data_file", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Data Analysit' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Update 'Document Writer' skill
UPDATE skills
SET
    tools_schema = '{
        "tools": [
            {
                "name": "write_document",
                "description": "Write a structured document (report, specification, proposal)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "enum": ["report", "specification", "proposal", "memo"],
                            "description": "Type of document to write"
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title"
                        },
                        "content_outline": {
                            "type": "string",
                            "description": "Brief outline or key points to cover"
                        },
                        "output_file": {
                            "type": "string",
                            "description": "Path for the output document (e.g., report.md)"
                        }
                    },
                    "required": ["document_type", "title", "output_file"]
                }
            }
        ]
    }'::jsonb,
    updated_at = NOW()
WHERE name = 'Document Writer' AND (tools_schema IS NULL OR tools_schema = '{}'::jsonb);

-- Log the update
DO $$
BEGIN
    RAISE NOTICE 'Added tools_schema to 9 skills: writing-plans, writing-skills, Code Reviewer, Security Expert, Tester, Researcher, Data Analysit, Document Writer, and existing document skills (pdf, docx, xlsx, pptx from previous migration)';
END $$;

