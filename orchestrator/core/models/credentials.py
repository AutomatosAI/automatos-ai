"""
Credential Management Database Models
=====================================

SQLAlchemy models for credential types, stored credentials, and audit logs.
Inspired by n8n's credential architecture.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from .core import Base


# ============================================================================
# SQLAlchemy Database Models
# ============================================================================

class CredentialType(Base):
    """
    Credential type definitions (like n8n's credential types).
    Defines the schema for a type of credential (e.g., PostgreSQL, OpenAI, SSH).
    """
    __tablename__ = 'credential_types'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)  # 'postgres_credentials', 'openai_api'
    display_name = Column(String(255), nullable=False)  # 'PostgreSQL', 'OpenAI API'
    category = Column(String(100), index=True)  # 'database', 'ai', 'infrastructure', 'api'
    icon = Column(String(50))  # Icon name for UI
    logo = Column(String(255))  # Logo file path, e.g. "/logos/OpenAI.png"
    description = Column(Text)
    
    # Schema definition (JSON array of field definitions)
    # Each field: {displayName, name, type, required, default, description, typeOptions}
    schema_definition = Column(JSON, nullable=False)
    
    # Test endpoint configuration (JSON object)
    # {method: 'api_call', url: '...', headers: {...}}
    test_endpoint = Column(JSON)
    
    documentation_url = Column(String(500))
    is_system = Column(Boolean, default=False)  # System-defined vs user-created
    is_active = Column(Boolean, default=True, index=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    credentials = relationship("Credential", back_populates="credential_type", cascade="all, delete-orphan")


class Credential(Base):
    """
    Stored credentials (encrypted).
    Each credential is an instance of a credential type with encrypted values.
    """
    __tablename__ = 'credentials'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)  # User-friendly name: "Production PostgreSQL"
    credential_type_id = Column(Integer, ForeignKey('credential_types.id', ondelete='CASCADE'), nullable=False)
    
    # Encrypted JSON blob of credential field values
    # e.g., {"host": "localhost", "user": "postgres", "password": "secret"}
    encrypted_data = Column(Text, nullable=False)
    
    environment = Column(String(50), default='production', index=True)  # dev, staging, production
    description = Column(Text)
    tags = Column(JSON, default=list)  # For organization and filtering
    
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(DateTime)  # Optional expiration
    
    # Testing status
    last_tested = Column(DateTime)
    test_status = Column(String(50))  # 'passed', 'failed', 'pending', 'not_tested'
    test_message = Column(Text)  # Test result message
    
    created_by = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    credential_type = relationship("CredentialType", back_populates="credentials")
    # Note: cascade removed - audit logs deleted via database CASCADE to avoid SQLAlchemy type mismatch
    # Database has ondelete='CASCADE' so audit logs are deleted automatically
    audit_logs = relationship("CredentialAuditLog", back_populates="credential", lazy='noload', passive_deletes=True)
    # Note: tool_assignments relationship removed - AgentToolAssignment.credential_id is optional and may not always reference credentials
    
    # Unique constraint: one credential name per environment
    __table_args__ = (
        {'extend_existing': True}
    )


class CredentialAuditLog(Base):
    """
    Audit log for credential operations.
    Tracks all access, modifications, and testing of credentials.
    """
    __tablename__ = 'credential_audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    credential_id = Column(Integer, ForeignKey('credentials.id', ondelete='CASCADE'), index=True)
    
    action = Column(String(50), nullable=False, index=True)  # created, updated, deleted, accessed, tested
    user_id = Column(String(255))
    ip_address = Column(String(45))  # IPv4 or IPv6
    
    success = Column(Boolean)
    error_message = Column(Text)
    audit_metadata = Column('metadata', JSON)  # Additional context (renamed column to avoid SQLAlchemy reserved word)
    
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    credential = relationship("Credential", back_populates="audit_logs")
    
    __table_args__ = (
        {'extend_existing': True}
    )


# ============================================================================
# Pydantic Models for API
# ============================================================================

class CredentialFieldType(str, Enum):
    """Field types for credential schemas (matching n8n)"""
    STRING = "string"
    PASSWORD = "password"  # Masked in UI
    NUMBER = "number"
    BOOLEAN = "boolean"
    OPTIONS = "options"  # Dropdown
    HIDDEN = "hidden"  # Not shown in UI (auto-populated)


class CredentialFieldDefinition(BaseModel):
    """Definition of a single field in a credential schema"""
    displayName: str = Field(..., description="Label shown in UI")
    name: str = Field(..., description="Internal field name")
    type: CredentialFieldType = Field(..., description="Field type")
    required: bool = Field(default=False, description="Is field required")
    default: Optional[Any] = Field(None, description="Default value")
    description: Optional[str] = Field(None, description="Help text for field")
    placeholder: Optional[str] = Field(None, description="Placeholder text")
    
    # Type-specific options
    typeOptions: Optional[Dict[str, Any]] = Field(None, description="Type-specific options")
    # For 'options' type: typeOptions = {options: [{name: 'Option 1', value: 'opt1'}, ...]}
    # For 'password' type: typeOptions = {password: true}
    
    # Conditional display
    displayOptions: Optional[Dict[str, Any]] = Field(None, description="When to show this field")
    
    class Config:
        use_enum_values = True


class CredentialTestEndpoint(BaseModel):
    """Configuration for testing a credential"""
    method: str = Field(..., description="Test method: 'api_call', 'connect', 'custom'")
    url: Optional[str] = Field(None, description="API endpoint for testing")
    headers: Optional[Dict[str, str]] = Field(None, description="Headers to send")
    description: Optional[str] = Field(None, description="What the test validates")


class CredentialTypeCreate(BaseModel):
    """Create a new credential type"""
    name: str = Field(..., min_length=1, max_length=255, description="Unique identifier")
    display_name: str = Field(..., description="Display name in UI")
    category: Optional[str] = Field(None, description="Category for organization")
    icon: Optional[str] = Field(None, description="Icon name")
    description: Optional[str] = Field(None, description="Description of credential type")
    schema_definition: List[CredentialFieldDefinition] = Field(..., description="Field definitions")
    test_endpoint: Optional[CredentialTestEndpoint] = Field(None, description="How to test credential")
    documentation_url: Optional[str] = Field(None, description="Link to documentation")
    is_system: bool = Field(default=False, description="System-defined type")


class CredentialTypeResponse(BaseModel):
    """Credential type response"""
    id: int
    name: str
    display_name: str
    category: Optional[str]
    icon: Optional[str]
    logo: Optional[str]  # Logo file path
    description: Optional[str]
    schema_definition: Any = Field(..., description="Field definitions")
    test_endpoint: Optional[Dict[str, Any]]
    documentation_url: Optional[str]
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    @field_validator('schema_definition', mode='before')
    @classmethod
    def convert_schema_definition(cls, v):
        """Convert JSON to list if needed"""
        if isinstance(v, str):
            import json
            v = json.loads(v)
        
        if isinstance(v, dict):
            # If it's a JSON Schema object with 'properties', extract the properties as a list
            if 'properties' in v:
                properties = v['properties']
                required_fields = v.get('required', [])
                result = []
                for k, prop in properties.items():
                    field_def = {
                        'name': k,
                        'displayName': prop.get('title', k),
                        'type': prop.get('type', 'string'),
                        'required': k in required_fields,
                        'description': prop.get('description', '')
                    }
                    # Add any additional properties
                    for key, value in prop.items():
                        if key not in ['title', 'type', 'description']:
                            field_def[key] = value
                    result.append(field_def)
                return result
            else:
                return [v]  # Convert single dict to list
        elif isinstance(v, list):
            return v  # Already a list
        
        return v
    
    class Config:
        from_attributes = True


class CredentialCreate(BaseModel):
    """Create a new credential"""
    name: str = Field(..., min_length=1, max_length=255, description="User-friendly name")
    credential_type_id: int = Field(..., description="ID of credential type")
    credential_data: Dict[str, Any] = Field(..., description="Credential field values (will be encrypted)")
    environment: str = Field(default="production", description="Target environment")
    description: Optional[str] = Field(None, description="Description")
    tags: Optional[List[str]] = Field(default=[], description="Tags for organization")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")


class CredentialUpdate(BaseModel):
    """Update an existing credential"""
    name: Optional[str] = Field(None, description="New name")
    credential_data: Optional[Dict[str, Any]] = Field(None, description="Updated credential values")
    description: Optional[str] = Field(None, description="Updated description")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    is_active: Optional[bool] = Field(None, description="Active status")
    expires_at: Optional[datetime] = Field(None, description="Expiration date")


class CredentialResponse(BaseModel):
    """Credential response with decrypted values for editing"""
    id: int
    name: str
    credential_type_id: int
    credential_type_name: str
    credential_type_display_name: str
    environment: str
    description: Optional[str]
    tags: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_tested: Optional[datetime]
    test_status: Optional[str]
    test_message: Optional[str]
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    # Metadata about credential
    has_credentials: bool = Field(default=True, description="Indicates encrypted data exists")
    field_names: List[str] = Field(default=[], description="List of field names")
    credential_data: Optional[Dict[str, Any]] = Field(default=None, description="Decrypted credential values for editing")
    
    class Config:
        from_attributes = True


class CredentialTestRequest(BaseModel):
    """Request to test a credential"""
    credential_id: int = Field(..., description="Credential to test")


class CredentialTestResponse(BaseModel):
    """Result of credential test"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    tested_at: datetime


class CredentialAuditLogResponse(BaseModel):
    """Audit log entry"""
    id: int
    credential_id: str
    tool_id: Optional[str]
    action: str
    user_id: Optional[str]
    ip_address: Optional[str]
    success: Optional[bool]
    error_message: Optional[str]
    details: Optional[Dict[str, Any]] = Field(None, alias='metadata')
    created_at: datetime = Field(alias='timestamp')
    
    class Config:
        from_attributes = True
        populate_by_name = True


class CredentialResolveRequest(BaseModel):
    """Internal request to resolve credential (for service use)"""
    credential_id: Optional[int] = Field(None, description="Credential ID")
    credential_name: Optional[str] = Field(None, description="Credential name")
    environment: str = Field(default="production", description="Environment")
    service_name: str = Field(..., description="Requesting service name")


class CredentialResolveResponse(BaseModel):
    """Decrypted credential data (internal use only)"""
    credential_id: int
    credential_name: str
    credential_type: str
    data: Dict[str, Any]  # Decrypted credential values
    environment: str
    resolved_at: datetime

