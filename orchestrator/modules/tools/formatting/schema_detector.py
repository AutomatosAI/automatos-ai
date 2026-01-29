"""
Schema-Driven Response Type Detection (PRD-38 Enhancement)
==========================================================

Detects widget type from Composio action response_schema, not action names.
Works for ALL 863+ Composio apps without hardcoding.

Instead of:
    if "GMAIL" in action and "LIST" in action:  # BAD - hardcoded

We do:
    widget_type = ResponseTypeDetector.detect(response_schema)  # GOOD - generic

This module provides:
1. ResponseTypeDetector - Detect widget type from schema or result shape
2. ParameterHintExtractor - Generate LLM hints from parameter schemas
3. Generic data extractors for emails, tables, documents, etc.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Tuple

logger = logging.getLogger(__name__)


class ResponseTypeDetector:
    """
    Detect widget type from Composio action response_schema.

    Instead of checking "GMAIL" in action_name, we inspect
    the response_schema to understand what data structure
    the action returns. This scales to 15,000+ actions.
    """

    # Key patterns that indicate specific data types
    # These are generic across ALL providers (Gmail, Outlook, Yahoo, etc.)
    EMAIL_KEYS = {
        "emails", "messages", "threads", "inbox", "mail", "mailbox",
        "email_list", "message_list", "email_messages"
    }
    CODE_KEYS = {
        "code", "source", "snippet", "language", "syntax", "file_content",
        "source_code", "script", "program", "function_code"
    }
    DOCUMENT_KEYS = {
        "content", "text", "body", "document", "chunks", "pages", "markdown",
        "article", "post", "note", "description", "readme"
    }
    TABLE_KEYS = {
        "rows", "columns", "records", "items", "results", "data", "entries",
        "list", "array", "collection", "objects"
    }
    IMAGE_KEYS = {
        "image", "images", "url", "base64", "thumbnail", "photo", "picture",
        "attachment", "media", "file_url", "image_url"
    }
    FILE_KEYS = {
        "files", "file", "path", "filename", "directory", "folders", "tree",
        "file_list", "contents", "file_path"
    }
    CALENDAR_KEYS = {
        "events", "calendar", "meetings", "appointments", "schedule",
        "event_list", "calendar_events"
    }
    CONTACT_KEYS = {
        "contacts", "people", "users", "members", "participants",
        "contact_list", "person", "profile"
    }

    # Widget type mapping
    WIDGET_TYPES = {
        "email": EMAIL_KEYS,
        "code": CODE_KEYS,
        "document": DOCUMENT_KEYS,
        "data": TABLE_KEYS,
        "image": IMAGE_KEYS,
        "file": FILE_KEYS,
        "calendar": CALENDAR_KEYS,
        "contact": CONTACT_KEYS,
    }

    @classmethod
    def detect(
        cls,
        response_schema: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        action_name: Optional[str] = None
    ) -> str:
        """
        Detect widget type using multiple strategies.

        Priority:
        1. response_schema (most reliable - from Composio API)
        2. result shape (fallback - inspect actual data)
        3. action_name hints (last resort - but NOT hardcoded providers)

        Args:
            response_schema: The response_schema from composio_actions_cache
            result: The actual result data (for fallback detection)
            action_name: Action name (for hint extraction, not provider matching)

        Returns:
            Widget type: 'email', 'code', 'document', 'data', 'image', 'file', 'generic'
        """
        # Strategy 1: Schema-based detection (best)
        if response_schema:
            detected = cls._detect_from_schema(response_schema)
            if detected != "generic":
                logger.debug(f"[SchemaDetector] Detected '{detected}' from response_schema")
                return detected

        # Strategy 2: Result shape detection (fallback)
        if result:
            detected = cls._detect_from_result(result)
            if detected != "generic":
                logger.debug(f"[SchemaDetector] Detected '{detected}' from result shape")
                return detected

        # Strategy 3: Action name hints (last resort - generic patterns only)
        if action_name:
            detected = cls._detect_from_action_hints(action_name)
            if detected != "generic":
                logger.debug(f"[SchemaDetector] Detected '{detected}' from action name hints")
                return detected

        return "generic"

    @classmethod
    def _detect_from_schema(cls, response_schema: Dict[str, Any]) -> str:
        """Detect widget type from response_schema JSONB."""
        if not response_schema:
            return "generic"

        properties = response_schema.get("properties", {})
        if not properties:
            # Try nested structures (array items, etc.)
            if "items" in response_schema:
                properties = response_schema["items"].get("properties", {})
            elif response_schema.get("type") == "array":
                items = response_schema.get("items", {})
                properties = items.get("properties", {})

        if not properties:
            return "generic"

        keys = set(k.lower() for k in properties.keys())

        # Check each widget type's key patterns
        for widget_type, key_patterns in cls.WIDGET_TYPES.items():
            matching_keys = keys & key_patterns
            if matching_keys:
                # Email and code need stronger signals
                if widget_type == "email" and matching_keys:
                    return "email"
                if widget_type == "code" and ("code" in keys or "language" in keys):
                    return "code"
                if widget_type in ("data", "file", "calendar", "contact"):
                    return widget_type

        # Check for array-of-objects pattern (table-like data)
        if cls._has_array_of_objects(response_schema):
            return "data"

        # Document detection needs multiple signals
        if len(keys & cls.DOCUMENT_KEYS) >= 2:
            return "document"

        return "generic"

    @classmethod
    def _detect_from_result(cls, result: Dict[str, Any]) -> str:
        """Fallback: detect type from actual result data shape."""
        if not result or not isinstance(result, dict):
            return "generic"

        # Unwrap common nested structures (Composio often wraps in "data")
        data = result
        for unwrap_key in ["data", "result", "response", "output"]:
            if isinstance(data, dict) and unwrap_key in data:
                nested = data[unwrap_key]
                if isinstance(nested, (dict, list)):
                    data = nested
                    break

        # If still a dict with nested "data", unwrap again
        if isinstance(data, dict) and "data" in data:
            nested = data["data"]
            if isinstance(nested, (dict, list)):
                data = nested

        # Get keys for matching
        if isinstance(data, dict):
            keys = set(k.lower() for k in data.keys())
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Array of objects - get keys from first item
            keys = set(k.lower() for k in data[0].keys())
        else:
            return "generic"

        # Check email patterns
        if keys & cls.EMAIL_KEYS:
            return "email"
        # Check if items have email-like structure
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0] if isinstance(data[0], dict) else {}
            item_keys = set(k.lower() for k in first_item.keys())
            # Email items typically have: subject, from/sender, to, date/timestamp
            email_item_keys = {"subject", "from", "to", "sender", "date", "timestamp", "snippet", "body"}
            if len(item_keys & email_item_keys) >= 3:
                return "email"

        # Check for array of objects (table-like)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return "data"

        # Check nested arrays
        for key in ["results", "items", "data", "records", "list", "entries"]:
            nested = data.get(key) if isinstance(data, dict) else None
            if isinstance(nested, list) and len(nested) > 0 and isinstance(nested[0], dict):
                # Check if nested items are email-like
                item_keys = set(k.lower() for k in nested[0].keys())
                email_item_keys = {"subject", "from", "to", "sender", "date", "snippet"}
                if len(item_keys & email_item_keys) >= 3:
                    return "email"
                return "data"

        # Check code patterns
        if keys & cls.CODE_KEYS:
            return "code"

        # Check document patterns
        if len(keys & cls.DOCUMENT_KEYS) >= 2:
            return "document"

        # Check file patterns
        if keys & cls.FILE_KEYS:
            return "file"

        return "generic"

    @classmethod
    def _detect_from_action_hints(cls, action_name: str) -> str:
        """
        Last resort: detect from action name patterns.

        NOTE: This does NOT hardcode provider names (Gmail, Outlook, etc.)
        Instead, it looks for generic action verbs/nouns.
        """
        action_lower = action_name.lower()

        # Email actions (generic patterns)
        email_patterns = ["_email", "email_", "message_", "_message", "mail_", "_mail", "inbox", "send_email", "list_emails"]
        if any(p in action_lower for p in email_patterns):
            return "email"

        # Code actions
        code_patterns = ["_code", "code_", "snippet", "source_", "_source", "script", "function"]
        if any(p in action_lower for p in code_patterns):
            return "code"

        # File actions
        file_patterns = ["_file", "file_", "download", "upload", "list_files", "get_file"]
        if any(p in action_lower for p in file_patterns):
            return "file"

        # Calendar actions
        calendar_patterns = ["calendar", "event", "meeting", "schedule", "appointment"]
        if any(p in action_lower for p in calendar_patterns):
            return "calendar"

        # List/search actions often return tables
        list_patterns = ["list_", "_list", "search_", "_search", "get_all", "fetch_"]
        if any(p in action_lower for p in list_patterns):
            return "data"

        return "generic"

    @classmethod
    def _has_array_of_objects(cls, schema: Dict[str, Any]) -> bool:
        """Check if schema describes an array of objects (table-like)."""
        if schema.get("type") == "array":
            items = schema.get("items", {})
            return items.get("type") == "object"

        # Check properties for array types
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name.lower() in cls.TABLE_KEYS:
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "array":
                    items = prop_schema.get("items", {})
                    if isinstance(items, dict) and items.get("type") == "object":
                        return True

        return False


class ParameterHintExtractor:
    """
    Extract LLM-friendly parameter hints from action parameter schemas.

    Instead of hardcoding "Gmail needs maxResults", we read the
    parameters JSONB and generate helpful hints dynamically.
    """

    @classmethod
    def extract_hints(cls, parameters: Optional[Dict[str, Any]], max_params: int = 10) -> str:
        """
        Convert JSON Schema parameters to human-readable hints.

        Args:
            parameters: The parameters JSONB from composio_actions_cache
            max_params: Maximum number of parameters to include

        Returns:
            Formatted string of parameter hints for LLM context
        """
        if not parameters:
            return ""

        if parameters.get("type") != "object":
            return ""

        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))

        if not properties:
            return ""

        hints = []
        # Sort: required first, then alphabetically
        sorted_params = sorted(
            properties.items(),
            key=lambda x: (x[0] not in required, x[0])
        )

        for param_name, param_def in sorted_params[:max_params]:
            hint = cls._format_param_hint(param_name, param_def, param_name in required)
            hints.append(hint)

        if len(properties) > max_params:
            hints.append(f"  ... and {len(properties) - max_params} more parameters")

        return "\n".join(hints)

    @classmethod
    def _format_param_hint(cls, name: str, definition: Dict[str, Any], is_required: bool) -> str:
        """Format a single parameter as a hint line."""
        req_str = "REQUIRED" if is_required else "optional"
        param_type = definition.get("type", "any")
        description = definition.get("description", "")

        # Truncate long descriptions
        if len(description) > 100:
            description = description[:97] + "..."

        # Handle enums
        enum_vals = definition.get("enum", [])
        if enum_vals:
            enum_preview = ", ".join(str(v) for v in enum_vals[:4])
            if len(enum_vals) > 4:
                enum_preview += f", ... ({len(enum_vals)} options)"
            return f"  - {name} ({req_str}): {description} [options: {enum_preview}]"

        # Handle defaults
        default = definition.get("default")
        if default is not None:
            return f"  - {name} ({req_str}, {param_type}): {description} [default: {default}]"

        return f"  - {name} ({req_str}, {param_type}): {description}"

    @classmethod
    def get_required_params(cls, parameters: Optional[Dict[str, Any]]) -> List[str]:
        """Get list of required parameter names."""
        if not parameters:
            return []
        return list(parameters.get("required", []))

    @classmethod
    def get_param_names(cls, parameters: Optional[Dict[str, Any]]) -> List[str]:
        """Get all parameter names."""
        if not parameters:
            return []
        properties = parameters.get("properties", {})
        return list(properties.keys())


class GenericDataExtractor:
    """
    Extract data for widgets in a generic way based on detected type.

    Instead of hardcoded Gmail/Outlook extraction, we use the detected
    widget type and apply generic extraction patterns.
    """

    @classmethod
    def extract_emails(cls, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract email data from any email-like result.

        Works for Gmail, Outlook, Yahoo, or any provider that returns
        email-like structures.
        """
        emails = []

        # Check for Composio error responses first
        data = cls._unwrap_result(result)
        if isinstance(data, dict):
            error_msg = data.get("composio_execution_message") or data.get("error") or data.get("message")
            if error_msg and isinstance(error_msg, str):
                if "ERROR" in error_msg.upper() or "INVALID" in error_msg.upper() or "NO RESULTS" in error_msg.upper():
                    logger.warning(f"[DataExtractor] Composio returned error: {error_msg[:200]}")
                    return emails  # Return empty - this is an error response, not email data

        # Common keys where emails might be stored
        email_list_keys = ["emails", "messages", "mail", "items", "data", "results", "value"]

        email_list = None
        if isinstance(data, list):
            email_list = data
        elif isinstance(data, dict):
            for key in email_list_keys:
                if key in data and isinstance(data[key], list):
                    email_list = data[key]
                    break

        if not email_list:
            return emails

        for item in email_list:
            if not isinstance(item, dict):
                continue

            # Skip ID-only items (no content)
            has_content = any(
                item.get(k) for k in ["snippet", "body", "subject", "payload", "content", "bodyPreview"]
            )
            if not has_content:
                logger.debug(f"[DataExtractor] Skipping email with no content: {item.get('id', 'unknown')}")
                continue

            email = cls._extract_single_email(item)
            if email:
                emails.append(email)

        logger.info(f"[DataExtractor] Extracted {len(emails)} emails from result")
        return emails

    @classmethod
    def _extract_single_email(cls, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a single email from various provider formats."""
        # Common field mappings across providers
        # Note: Gmail uses messageText for body, Outlook uses body/content
        email = {
            "id": item.get("id") or item.get("messageId") or item.get("uid"),
            "threadId": item.get("threadId") or item.get("conversationId"),
            "subject": cls._extract_field(item, ["subject", "Subject"]),
            "from": cls._extract_field(item, ["from", "sender", "From", "senderEmail"]),
            "to": cls._extract_field(item, ["to", "recipients", "To", "toRecipients"]),
            "date": cls._extract_field(item, ["date", "receivedDateTime", "internalDate", "timestamp", "sentDateTime", "receivedAt"]),
            "snippet": cls._extract_field(item, ["snippet", "bodyPreview", "preview", "summary"]),
            # Gmail uses messageText for body content (HTML)
            "body": cls._extract_field(item, ["messageText", "body", "content", "text", "htmlBody", "textBody", "bodyText"]),
            "isRead": cls._extract_read_status(item),
            "hasAttachments": cls._extract_has_attachments(item),
            "labels": item.get("labelIds") or item.get("categories") or item.get("labels") or [],
            "attachments": cls._extract_attachments(item),
        }

        # Extract from nested payload (Gmail format)
        if "payload" in item and isinstance(item["payload"], dict):
            payload = item["payload"]
            headers = payload.get("headers", [])
            if isinstance(headers, list):
                for header in headers:
                    name = header.get("name", "").lower()
                    value = header.get("value")
                    if name == "subject" and not email["subject"]:
                        email["subject"] = value
                    elif name == "from" and not email["from"]:
                        email["from"] = value
                    elif name == "to" and not email["to"]:
                        email["to"] = value
                    elif name == "date" and not email["date"]:
                        email["date"] = value

            # Body from payload - Gmail uses base64 encoding
            if not email["body"]:
                body_data = payload.get("body", {})
                if isinstance(body_data, dict):
                    raw_body = body_data.get("data") or body_data.get("content")
                    if raw_body:
                        email["body"] = cls._decode_base64_body(raw_body)

        # Decode body if it looks like base64 (Gmail format)
        if email.get("body") and cls._looks_like_base64(email["body"]):
            email["body"] = cls._decode_base64_body(email["body"])

        # Detect HTML content and set bodyHtml for proper rendering
        body_content = email.get("body", "")
        logger.debug(f"[DataExtractor] Email {email.get('id')}: body length={len(body_content) if body_content else 0}, is_html={cls._is_html_content(body_content) if body_content else False}")

        if body_content and cls._is_html_content(body_content):
            # Store raw HTML for rendering
            email["bodyHtml"] = body_content
            # Create plain text version for fallback/preview
            plain_text = cls._html_to_text(body_content)
            email["body"] = plain_text
            logger.debug(f"[DataExtractor] Email {email.get('id')}: converted HTML to text, plain_text length={len(plain_text)}")

        # Generate snippet from body if not present
        if not email.get("snippet") and email.get("body"):
            email["snippet"] = email["body"][:200].strip()
            logger.debug(f"[DataExtractor] Email {email.get('id')}: generated snippet from body")

        # Clean up None values
        return {k: v for k, v in email.items() if v is not None}

    @classmethod
    def _looks_like_base64(cls, text: str) -> bool:
        """Check if text looks like base64 encoded content."""
        if not text or len(text) < 50:
            return False
        # Base64 typically has no spaces, lots of uppercase, and specific chars
        import re
        # Check for base64-like pattern (alphanumeric, +, /, =, -)
        if re.match(r'^[A-Za-z0-9+/=_-]+$', text[:100].replace('\n', '').replace('\r', '')):
            # If it's mostly uppercase/numbers and no spaces, likely base64
            return True
        return False

    @classmethod
    def _decode_base64_body(cls, encoded: str) -> str:
        """Decode base64 email body (Gmail uses URL-safe base64)."""
        import base64
        try:
            # Gmail uses URL-safe base64 (- instead of +, _ instead of /)
            # Also need to handle padding
            encoded_clean = encoded.replace('-', '+').replace('_', '/')
            # Add padding if needed
            padding = 4 - len(encoded_clean) % 4
            if padding != 4:
                encoded_clean += '=' * padding

            decoded_bytes = base64.b64decode(encoded_clean)
            decoded_str = decoded_bytes.decode('utf-8', errors='replace')

            # Return the decoded content (HTML or plain text)
            # The caller will determine if it's HTML and handle accordingly
            return decoded_str[:50000]  # Reasonable limit for email body
        except Exception as e:
            logger.warning(f"[DataExtractor] Failed to decode base64 body: {e}")
            return encoded[:200] + "..." if len(encoded) > 200 else encoded

    @classmethod
    def _is_html_content(cls, content: str) -> bool:
        """Check if content appears to be HTML."""
        if not content:
            return False
        content_start = content.strip()[:500].lower()
        return (
            content_start.startswith('<!doctype html') or
            content_start.startswith('<html') or
            '<body' in content_start or
            ('<div' in content_start and '</' in content_start) or
            ('<p>' in content_start or '<br' in content_start)
        )

    @classmethod
    def _html_to_text(cls, html: str) -> str:
        """Convert HTML to plain text for preview/fallback."""
        import re
        if not html:
            return ""
        try:
            # Remove style and script tags with their content
            text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove head section
            text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Replace common block elements with newlines
            text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
            text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
            # Remove all remaining tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Decode HTML entities
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
            text = text.replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'")
            # Normalize whitespace
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n\s*\n+', '\n\n', text)
            text = text.strip()
            return text[:5000]  # Reasonable limit
        except Exception as e:
            logger.warning(f"[DataExtractor] Failed to convert HTML to text: {e}")
            return html[:500]

    @classmethod
    def extract_table_data(cls, result: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract table data (columns, rows) from any list-like result.

        Returns:
            Tuple of (column_names, rows)
        """
        data = cls._unwrap_result(result)

        # Find the data array
        data_list = None
        if isinstance(data, list):
            data_list = data
        elif isinstance(data, dict):
            for key in ["data", "results", "items", "records", "rows", "entries", "list", "value"]:
                if key in data and isinstance(data[key], list):
                    data_list = data[key]
                    break

        if not data_list or not isinstance(data_list, list):
            return [], []

        if len(data_list) == 0:
            return [], []

        # Extract columns from first row
        first_row = data_list[0]
        if not isinstance(first_row, dict):
            return [], []

        columns = list(first_row.keys())
        rows = [row for row in data_list if isinstance(row, dict)]

        return columns, rows

    @classmethod
    def _unwrap_result(cls, result: Dict[str, Any]) -> Any:
        """Unwrap nested result structures."""
        data = result

        # Unwrap common nesting patterns
        for _ in range(3):  # Max 3 levels of unwrapping
            if isinstance(data, dict):
                # Check for single-key wrapper
                if len(data) == 1:
                    data = list(data.values())[0]
                    continue

                # Check for "data" key wrapper
                if "data" in data:
                    nested = data["data"]
                    if isinstance(nested, (dict, list)):
                        data = nested
                        continue

                # Check for "result" key wrapper
                if "result" in data and isinstance(data["result"], (dict, list)):
                    data = data["result"]
                    continue

            break

        return data

    @classmethod
    def _extract_field(cls, item: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        """Extract a field trying multiple possible key names."""
        for key in keys:
            if key in item and item[key] is not None:
                value = item[key]
                # Handle nested structures
                if isinstance(value, dict):
                    # Outlook format: {"emailAddress": {"address": "...", "name": "..."}}
                    if "emailAddress" in value:
                        addr = value["emailAddress"]
                        return addr.get("address") or addr.get("name")
                    # Generic nested
                    return value.get("content") or value.get("value") or value.get("address")
                return value
        return None

    @classmethod
    def _extract_read_status(cls, item: Dict[str, Any]) -> bool:
        """Extract read/unread status."""
        # Gmail: "UNREAD" in labelIds means unread
        labels = item.get("labelIds", [])
        if isinstance(labels, list) and "UNREAD" in labels:
            return False

        # Outlook: isRead field
        if "isRead" in item:
            return bool(item["isRead"])

        # Default to read
        return True

    @classmethod
    def _extract_has_attachments(cls, item: Dict[str, Any]) -> bool:
        """Extract whether email has attachments."""
        # Direct field
        if "hasAttachments" in item:
            return bool(item["hasAttachments"])

        # Gmail: attachmentList
        if item.get("attachmentList"):
            return True

        # Gmail: check payload.parts
        payload = item.get("payload", {})
        if isinstance(payload, dict) and payload.get("parts"):
            return True

        # Check for attachments array
        if item.get("attachments"):
            return True

        return False

    @classmethod
    def _extract_attachments(cls, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract attachment details from email.

        Returns list of attachments with:
        - id: attachment ID
        - filename: file name
        - mimeType: MIME type
        - size: file size in bytes
        - downloadUrl: URL to download (if available)
        """
        attachments = []

        # Gmail format: attachmentList
        attachment_list = item.get("attachmentList", [])
        if attachment_list and isinstance(attachment_list, list):
            for att in attachment_list:
                if isinstance(att, dict):
                    attachments.append({
                        "id": att.get("attachmentId") or att.get("id") or att.get("fileId"),
                        "filename": att.get("filename") or att.get("name") or "attachment",
                        "mimeType": att.get("mimeType") or att.get("contentType") or "application/octet-stream",
                        "size": att.get("size") or att.get("fileSize") or 0,
                        "downloadUrl": att.get("downloadUrl") or att.get("url"),
                    })

        # Outlook format: attachments array
        outlook_attachments = item.get("attachments", [])
        if outlook_attachments and isinstance(outlook_attachments, list) and not attachments:
            for att in outlook_attachments:
                if isinstance(att, dict):
                    attachments.append({
                        "id": att.get("id"),
                        "filename": att.get("name") or att.get("filename") or "attachment",
                        "mimeType": att.get("contentType") or att.get("mimeType") or "application/octet-stream",
                        "size": att.get("size") or 0,
                        "downloadUrl": att.get("contentLocation") or att.get("downloadUrl"),
                    })

        # Gmail payload.parts (for inline attachments)
        payload = item.get("payload", {})
        if isinstance(payload, dict) and not attachments:
            parts = payload.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    # Skip text/html and text/plain parts (body, not attachments)
                    mime_type = part.get("mimeType", "")
                    if mime_type in ("text/plain", "text/html"):
                        continue
                    filename = part.get("filename")
                    if filename:  # Only include parts with filenames
                        body = part.get("body", {})
                        attachments.append({
                            "id": body.get("attachmentId") or part.get("partId"),
                            "filename": filename,
                            "mimeType": mime_type,
                            "size": body.get("size") or 0,
                            "downloadUrl": None,  # Gmail requires separate API call
                        })

        return attachments if attachments else None
