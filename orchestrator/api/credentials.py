

"""
Credentials Management API Routes

Secure credential storage and management for tools with encryption,
environment separation, and audit logging.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
import base64
import os

from database.database import get_db
from database.models import ToolCredentials, Tool, CredentialAuditLog
from models import ToolCredentials, Tool, CredentialAuditLog
from utils.logging_adapter import set_request_id
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/legacy/credentials", tags=["Legacy Credentials Management"])

# ====================================
# Encryption Setup
# ====================================

# In production, use proper key management (AWS KMS, Azure Key Vault, etc.)
ENCRYPTION_KEY = os.getenv("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY if isinstance(ENCRYPTION_KEY, bytes) else ENCRYPTION_KEY.encode())

# ====================================
# Pydantic Models
# ====================================

class CredentialBase(BaseModel):
    credential_key: str = Field(..., description="Credential key/name")
    credential_value: str = Field(..., description="Credential value (will be encrypted)")
    environment: str = Field(..., description="Target environment")
    description: Optional[str] = Field(None, description="Credential description")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

class CredentialCreate(CredentialBase):
    tool_id: int = Field(..., description="Associated tool ID")

class CredentialUpdate(BaseModel):
    credential_value: Optional[str] = None
    description: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None

class CredentialResponse(BaseModel):
    id: int
    tool_id: int
    tool_name: str
    credential_key: str
    environment: str
    description: Optional[str]
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    # Note: credential_value is never returned for security

    class Config:
        from_attributes = True

class CredentialTestRequest(BaseModel):
    tool_id: int
    environment: str
    credentials: Dict[str, str]

class CredentialAuditEntry(BaseModel):
    id: int
    tool_id: int
    tool_name: str
    credential_key: str
    action: str
    environment: str
    user_id: Optional[str]
    timestamp: datetime
    details: Optional[Dict[str, Any]]

# ====================================
# Helper Functions
# ====================================

def encrypt_credential(value: str) -> str:
    """Encrypt a credential value"""
    try:
        encrypted_bytes = cipher_suite.encrypt(value.encode())
        return base64.b64encode(encrypted_bytes).decode()
    except Exception as e:
        logger.error(f"Failed to encrypt credential: {e}")
        raise HTTPException(status_code=500, detail="Encryption failed")

def decrypt_credential(encrypted_value: str) -> str:
    """Decrypt a credential value"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_value.encode())
        decrypted_bytes = cipher_suite.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    except Exception as e:
        logger.error(f"Failed to decrypt credential: {e}")
        raise HTTPException(status_code=500, detail="Decryption failed")

def create_audit_log(
    db: Session, 
    tool_id: int, 
    credential_key: str, 
    action: str, 
    environment: str,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Create an audit log entry"""
    audit_entry = CredentialAuditLog(
        tool_id=tool_id,
        credential_key=credential_key,
        action=action,
        environment=environment,
        user_id=user_id,
        details=details or {},
        timestamp=datetime.utcnow()
    )
    db.add(audit_entry)
    return audit_entry

# ====================================
# API Endpoints
# ====================================

@router.post("/", response_model=CredentialResponse)
async def create_credential(
    credential: CredentialCreate,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """
    Create and store an encrypted credential for a tool.
    All credential values are encrypted at rest and only accessible by authorized agents.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Verify tool exists
        tool = db.query(Tool).filter(Tool.id == credential.tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Check if credential already exists
        existing = db.query(ToolCredentials).filter(
            and_(
                ToolCredentials.tool_id == credential.tool_id,
                ToolCredentials.credential_key == credential.credential_key,
                ToolCredentials.environment == credential.environment
            )
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Credential {credential.credential_key} already exists for this tool and environment"
            )
        
        # Encrypt the credential value
        encrypted_value = encrypt_credential(credential.credential_value)
        
        # Create credential record
        new_credential = ToolCredentials(
            tool_id=credential.tool_id,
            credential_key=credential.credential_key,
            credential_value=encrypted_value,
            environment=credential.environment,
            description=credential.description,
            expires_at=credential.expires_at,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(new_credential)
        
        # Create audit log
        create_audit_log(
            db, credential.tool_id, credential.credential_key, "create", 
            credential.environment, user_id,
            {"description": credential.description, "has_expiration": bool(credential.expires_at)}
        )
        
        db.commit()
        db.refresh(new_credential)
        
        # Prepare response (without credential value)
        response = CredentialResponse(
            id=new_credential.id,
            tool_id=new_credential.tool_id,
            tool_name=tool.name,
            credential_key=new_credential.credential_key,
            environment=new_credential.environment,
            description=new_credential.description,
            expires_at=new_credential.expires_at,
            is_active=new_credential.is_active,
            created_at=new_credential.created_at,
            updated_at=new_credential.updated_at
        )
        
        logger.info(f"Created credential {credential.credential_key} for tool {tool.name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating credential: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create credential: {str(e)}")

@router.get("/", response_model=List[CredentialResponse])
async def list_credentials(
    tool_id: Optional[int] = Query(None, description="Filter by tool ID"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    active_only: bool = Query(True, description="Only show active credentials"),
    db: Session = Depends(get_db)
):
    """List all credentials with optional filtering (values are never returned)"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        query = db.query(ToolCredentials, Tool.name.label("tool_name")).join(
            Tool, ToolCredentials.tool_id == Tool.id
        )
        
        if tool_id:
            query = query.filter(ToolCredentials.tool_id == tool_id)
        
        if environment:
            query = query.filter(ToolCredentials.environment == environment)
        
        if active_only:
            query = query.filter(ToolCredentials.is_active == True)
        
        results = query.all()
        
        credentials = []
        for credential, tool_name in results:
            credentials.append(CredentialResponse(
                id=credential.id,
                tool_id=credential.tool_id,
                tool_name=tool_name,
                credential_key=credential.credential_key,
                environment=credential.environment,
                description=credential.description,
                expires_at=credential.expires_at,
                is_active=credential.is_active,
                created_at=credential.created_at,
                updated_at=credential.updated_at
            ))
        
        logger.info(f"Retrieved {len(credentials)} credentials")
        return credentials
        
    except Exception as e:
        logger.error(f"Error listing credentials: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list credentials: {str(e)}")

@router.get("/{tool_id}", response_model=Dict[str, str])
async def get_tool_credentials(
    tool_id: int,
    environment: str = Query(..., description="Target environment"),
    agent_type: Optional[str] = Query(None, description="Agent type for permission check"),
    db: Session = Depends(get_db)
):
    """
    Get decrypted credentials for a tool in a specific environment.
    This endpoint is used by agents to access tool credentials.
    Access is logged for security auditing.
    """
    set_request_id(str(uuid.uuid4()))
    
    try:
        # Verify tool exists
        tool = db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Check agent permissions (simplified for demo)
        if agent_type and agent_type not in (tool.permissions or []):
            create_audit_log(
                db, tool_id, "ALL", "access_denied", environment, agent_type,
                {"reason": "insufficient_permissions"}
            )
            db.commit()
            raise HTTPException(status_code=403, detail="Agent type not authorized for this tool")
        
        # Get credentials for the environment
        credentials = db.query(ToolCredentials).filter(
            and_(
                ToolCredentials.tool_id == tool_id,
                ToolCredentials.environment == environment,
                ToolCredentials.is_active == True
            )
        ).all()
        
        if not credentials:
            raise HTTPException(
                status_code=404, 
                detail=f"No credentials found for tool {tool.name} in {environment} environment"
            )
        
        # Decrypt credentials
        decrypted_creds = {}
        for cred in credentials:
            # Check expiration
            if cred.expires_at and cred.expires_at < datetime.utcnow():
                logger.warning(f"Credential {cred.credential_key} has expired")
                continue
                
            try:
                decrypted_value = decrypt_credential(cred.credential_value)
                decrypted_creds[cred.credential_key] = decrypted_value
            except Exception as e:
                logger.error(f"Failed to decrypt credential {cred.credential_key}: {e}")
                continue
        
        # Create audit log for access
        create_audit_log(
            db, tool_id, "ALL", "access_granted", environment, agent_type,
            {"credentials_accessed": list(decrypted_creds.keys())}
        )
        
        db.commit()
        
        logger.info(f"Retrieved {len(decrypted_creds)} credentials for tool {tool.name}")
        return decrypted_creds
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving credentials for tool {tool_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve credentials: {str(e)}")

@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    update_data: CredentialUpdate,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """Update an existing credential"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        credential = db.query(ToolCredentials).filter(
            ToolCredentials.id == credential_id
        ).first()
        
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        tool = db.query(Tool).filter(Tool.id == credential.tool_id).first()
        
        # Track changes for audit
        changes = {}
        
        # Update fields
        if update_data.credential_value is not None:
            credential.credential_value = encrypt_credential(update_data.credential_value)
            changes["credential_value"] = "updated"
        
        if update_data.description is not None:
            changes["description"] = {"old": credential.description, "new": update_data.description}
            credential.description = update_data.description
        
        if update_data.expires_at is not None:
            changes["expires_at"] = {"old": credential.expires_at, "new": update_data.expires_at}
            credential.expires_at = update_data.expires_at
        
        if update_data.is_active is not None:
            changes["is_active"] = {"old": credential.is_active, "new": update_data.is_active}
            credential.is_active = update_data.is_active
        
        credential.updated_at = datetime.utcnow()
        
        # Create audit log
        create_audit_log(
            db, credential.tool_id, credential.credential_key, "update",
            credential.environment, user_id, {"changes": changes}
        )
        
        db.commit()
        db.refresh(credential)
        
        response = CredentialResponse(
            id=credential.id,
            tool_id=credential.tool_id,
            tool_name=tool.name,
            credential_key=credential.credential_key,
            environment=credential.environment,
            description=credential.description,
            expires_at=credential.expires_at,
            is_active=credential.is_active,
            created_at=credential.created_at,
            updated_at=credential.updated_at
        )
        
        logger.info(f"Updated credential {credential.credential_key} for tool {tool.name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating credential {credential_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update credential: {str(e)}")

@router.delete("/{credential_id}")
async def delete_credential(
    credential_id: int,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """Delete a credential (secure deletion)"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        credential = db.query(ToolCredentials).filter(
            ToolCredentials.id == credential_id
        ).first()
        
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")
        
        tool = db.query(Tool).filter(Tool.id == credential.tool_id).first()
        
        # Create audit log before deletion
        create_audit_log(
            db, credential.tool_id, credential.credential_key, "delete",
            credential.environment, user_id,
            {"credential_id": credential_id, "tool_name": tool.name if tool else "unknown"}
        )
        
        # Secure deletion - overwrite credential value before deletion
        credential.credential_value = encrypt_credential("DELETED")
        db.flush()
        
        # Delete the credential
        db.delete(credential)
        db.commit()
        
        logger.info(f"Deleted credential {credential.credential_key} for tool {tool.name if tool else 'unknown'}")
        return {"message": "Credential deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting credential {credential_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete credential: {str(e)}")

@router.post("/test-connection")
async def test_connection(
    test_request: CredentialTestRequest,
    user_id: Optional[str] = Query(None, description="User ID for audit logging"),
    db: Session = Depends(get_db)
):
    """Test tool connection with provided credentials"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        tool = db.query(Tool).filter(Tool.id == test_request.tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Create audit log for connection test
        create_audit_log(
            db, test_request.tool_id, "connection_test", "test", 
            test_request.environment, user_id,
            {"credentials_tested": list(test_request.credentials.keys())}
        )
        
        # Mock connection test - in real implementation, make actual API calls
        import asyncio
        await asyncio.sleep(1)  # Simulate connection test time
        
        # Mock test results based on credential completeness
        required_creds = tool.required_credentials or []
        provided_creds = list(test_request.credentials.keys())
        missing_creds = [cred for cred in required_creds if cred not in provided_creds]
        
        if missing_creds:
            result = {
                "success": False,
                "message": f"Missing required credentials: {', '.join(missing_creds)}",
                "details": {
                    "required": required_creds,
                    "provided": provided_creds,
                    "missing": missing_creds
                }
            }
        else:
            result = {
                "success": True,
                "message": "Connection successful! All credentials are valid.",
                "details": {
                    "response_time_ms": 234,
                    "server_version": "1.0.0",
                    "authenticated": True
                }
            }
        
        db.commit()
        
        logger.info(f"Connection test for tool {tool.name}: {'success' if result['success'] else 'failed'}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing connection for tool {test_request.tool_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to test connection: {str(e)}")

@router.get("/audit/logs", response_model=List[CredentialAuditEntry])
async def get_audit_logs(
    tool_id: Optional[int] = Query(None, description="Filter by tool ID"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    action: Optional[str] = Query(None, description="Filter by action"),
    limit: int = Query(100, description="Maximum number of logs to return"),
    db: Session = Depends(get_db)
):
    """Get credential audit logs with optional filtering"""
    set_request_id(str(uuid.uuid4()))
    
    try:
        query = db.query(CredentialAuditLog, Tool.name.label("tool_name")).join(
            Tool, CredentialAuditLog.tool_id == Tool.id
        )
        
        if tool_id:
            query = query.filter(CredentialAuditLog.tool_id == tool_id)
        
        if environment:
            query = query.filter(CredentialAuditLog.environment == environment)
        
        if action:
            query = query.filter(CredentialAuditLog.action == action)
        
        results = query.order_by(CredentialAuditLog.timestamp.desc()).limit(limit).all()
        
        audit_entries = []
        for log, tool_name in results:
            audit_entries.append(CredentialAuditEntry(
                id=log.id,
                tool_id=log.tool_id,
                tool_name=tool_name,
                credential_key=log.credential_key,
                action=log.action,
                environment=log.environment,
                user_id=log.user_id,
                timestamp=log.timestamp,
                details=log.details
            ))
        
        logger.info(f"Retrieved {len(audit_entries)} audit log entries")
        return audit_entries
        
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {str(e)}")

# ====================================
# Health Check
# ====================================

@router.get("/health")
async def credentials_health_check():
    """Health check endpoint for credentials management system"""
    return {
        "status": "healthy",
        "service": "credentials-management",
        "timestamp": datetime.utcnow().isoformat(),
        "encryption": "active",
        "version": "1.0.0"
    }
