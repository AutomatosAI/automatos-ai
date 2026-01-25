"""
Credential Management Service
=============================

Comprehensive credential management inspired by n8n's architecture.
Handles CRUD operations, encryption, testing, and audit logging.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from core.models.credentials import (
    CredentialType,
    Credential,
    CredentialAuditLog,
    CredentialCreate,
    CredentialUpdate,
    CredentialResponse,
    CredentialTestRequest,
    CredentialTestResponse
)
from core.credentials.encryption import get_encryption_service, EncryptionKeyError
from core.credentials.tester import CredentialTester

logger = logging.getLogger(__name__)


class CredentialNotFoundError(Exception):
    """Raised when credential is not found"""
    pass


class CredentialValidationError(Exception):
    """Raised when credential validation fails"""
    pass


class CredentialStore:
    """
    Manages credential storage, retrieval, and lifecycle.
    Handles encryption, validation, testing, and audit logging.
    """
    
    def __init__(self, db: Session):
        """
        Initialize credential store.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.encryption_service = get_encryption_service()
    
    # ========================================================================
    # Credential Type Management
    # ========================================================================
    
    def get_credential_type(self, type_id: int) -> Optional[CredentialType]:
        """Get credential type by ID"""
        return self.db.query(CredentialType).filter(CredentialType.id == type_id).first()
    
    def get_credential_type_by_name(self, name: str) -> Optional[CredentialType]:
        """Get credential type by name"""
        return self.db.query(CredentialType).filter(CredentialType.name == name).first()
    
    def list_credential_types(
        self,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> List[CredentialType]:
        """
        List all credential types with optional filtering.
        
        Args:
            category: Filter by category
            active_only: Only return active types
            
        Returns:
            List of credential types
        """
        query = self.db.query(CredentialType)
        
        if category:
            query = query.filter(CredentialType.category == category)
        
        if active_only:
            query = query.filter(CredentialType.is_active == True)
        
        return query.order_by(CredentialType.display_name).all()
    
    # ========================================================================
    # Credential CRUD Operations
    # ========================================================================
    
    def create_credential(
        self,
        credential_data: CredentialCreate,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Credential:
        """
        Create a new credential with encryption.
        
        Args:
            credential_data: Credential creation data
            user_id: User creating the credential
            ip_address: IP address of requester
            
        Returns:
            Created credential
            
        Raises:
            CredentialValidationError: If validation fails
            EncryptionKeyError: If encryption fails
        """
        # Validate credential type exists
        cred_type = self.get_credential_type(credential_data.credential_type_id)
        if not cred_type:
            raise CredentialValidationError(f"Credential type {credential_data.credential_type_id} not found")
        
        # Validate required fields
        self._validate_credential_data(credential_data.credential_data, cred_type.schema_definition)
        
        # Check for duplicate name in same environment
        existing = self.db.query(Credential).filter(
            and_(
                Credential.name == credential_data.name,
                Credential.environment == credential_data.environment,
                Credential.is_active == True  # Only block duplicates for active creds
            )
        ).first()
        
        if existing:
            raise CredentialValidationError(
                f"Credential '{credential_data.name}' already exists in {credential_data.environment} environment"
            )
        
        # Encrypt credential data
        try:
            encrypted_data = self.encryption_service.encrypt_dict(credential_data.credential_data)
        except Exception as e:
            logger.error(f"Failed to encrypt credential data: {e}")
            raise EncryptionKeyError(f"Encryption failed: {e}")
        
        # Create credential record
        credential = Credential(
            name=credential_data.name,
            credential_type_id=credential_data.credential_type_id,
            encrypted_data=encrypted_data,
            environment=credential_data.environment,
            description=credential_data.description,
            tags=credential_data.tags or [],
            expires_at=credential_data.expires_at,
            is_active=True,
            test_status='not_tested',
            created_by=user_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(credential)
        
        # Create audit log
        self._create_audit_log(
            credential=credential,
            action='created',
            user_id=user_id,
            ip_address=ip_address,
            success=True,
            metadata={"credential_type": cred_type.name, "environment": credential_data.environment}
        )
        
        self.db.commit()
        self.db.refresh(credential)
        
        logger.info(f"Created credential '{credential.name}' of type '{cred_type.name}'")
        return credential
    
    def get_credential(self, credential_id: int) -> Optional[Credential]:
        """Get credential by ID (without decrypting)"""
        return self.db.query(Credential).filter(Credential.id == credential_id).first()
    
    def get_credential_by_name(
        self,
        name: str,
        environment: str = "production"
    ) -> Optional[Credential]:
        """Get credential by name (environment-agnostic for MVP single environment)"""
        # For MVP: Just find the credential by name, ignore environment
        return self.db.query(Credential).filter(
            and_(
                Credential.name == name,
                Credential.is_active == True
            )
        ).first()
    
    def list_credentials(
        self,
        credential_type_id: Optional[int] = None,
        environment: Optional[str] = None,
        active_only: bool = True,
        tags: Optional[List[str]] = None
    ) -> List[Credential]:
        """
        List credentials with optional filtering.
        
        Args:
            credential_type_id: Filter by credential type
            environment: Filter by environment
            active_only: Only return active credentials
            tags: Filter by tags (match any)
            
        Returns:
            List of credentials (encrypted data NOT decrypted)
        """
        query = self.db.query(Credential)
        
        if credential_type_id:
            query = query.filter(Credential.credential_type_id == credential_type_id)
        
        if environment:
            query = query.filter(Credential.environment == environment)
        
        if active_only:
            query = query.filter(Credential.is_active == True)
        
        if tags:
            # Match any tag
            tag_filters = [Credential.tags.contains([tag]) for tag in tags]
            query = query.filter(or_(*tag_filters))
        
        return query.order_by(Credential.name).all()
    
    def update_credential(
        self,
        credential_id: int,
        update_data: CredentialUpdate,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Credential:
        """
        Update an existing credential.
        
        Args:
            credential_id: ID of credential to update
            update_data: Update data
            user_id: User performing update
            ip_address: IP address of requester
            
        Returns:
            Updated credential
            
        Raises:
            CredentialNotFoundError: If credential not found
            CredentialValidationError: If validation fails
        """
        credential = self.get_credential(credential_id)
        if not credential:
            raise CredentialNotFoundError(f"Credential {credential_id} not found")
        
        cred_type = self.get_credential_type(credential.credential_type_id)
        changes = {}
        
        # Update name
        if update_data.name is not None:
            changes['name'] = {'old': credential.name, 'new': update_data.name}
            credential.name = update_data.name
        
        # Update credential data (requires re-encryption)
        if update_data.credential_data is not None:
            self._validate_credential_data(update_data.credential_data, cred_type.schema_definition)
            encrypted_data = self.encryption_service.encrypt_dict(update_data.credential_data)
            credential.encrypted_data = encrypted_data
            credential.test_status = 'not_tested'  # Reset test status
            changes['credential_data'] = 'updated'
        
        # Update other fields
        if update_data.description is not None:
            changes['description'] = {'old': credential.description, 'new': update_data.description}
            credential.description = update_data.description
        
        if update_data.tags is not None:
            changes['tags'] = {'old': credential.tags, 'new': update_data.tags}
            credential.tags = update_data.tags
        
        if update_data.is_active is not None:
            changes['is_active'] = {'old': credential.is_active, 'new': update_data.is_active}
            credential.is_active = update_data.is_active
        
        if update_data.expires_at is not None:
            changes['expires_at'] = {'old': credential.expires_at, 'new': update_data.expires_at}
            credential.expires_at = update_data.expires_at
        
        credential.updated_at = datetime.utcnow()
        
        # Create audit log
        self._create_audit_log(
            credential=credential,
            action='updated',
            user_id=user_id,
            ip_address=ip_address,
            success=True,
            metadata={"changes": changes}
        )
        
        self.db.commit()
        self.db.refresh(credential)
        
        logger.info(f"Updated credential '{credential.name}' (ID: {credential_id})")
        return credential
    
    def delete_credential(
        self,
        credential_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Securely delete a credential.
        
        Args:
            credential_id: ID of credential to delete
            user_id: User performing deletion
            ip_address: IP address of requester
            
        Returns:
            True if deleted successfully
            
        Raises:
            CredentialNotFoundError: If credential not found
        """
        credential = self.get_credential(credential_id)
        if not credential:
            raise CredentialNotFoundError(f"Credential {credential_id} not found")
        
        cred_name = credential.name
        
        # Create audit log before deletion
        self._create_audit_log(
            credential=credential,
            action='deleted',
            user_id=user_id,
            ip_address=ip_address,
            success=True,
            metadata={"credential_name": cred_name}
        )
        self.db.flush()  # Ensure audit log is saved
        
        # Note: Audit logs will be deleted automatically by database CASCADE constraint
        # Relationship is set to lazy='noload' to prevent SQLAlchemy from querying it
        # (which would cause VARCHAR/INTEGER type mismatch)
        
        # Secure deletion: overwrite encrypted data first
        credential.encrypted_data = self.encryption_service.encrypt('DELETED')
        self.db.flush()
        
        # Delete credential using raw SQL to avoid SQLAlchemy relationship loading
        # which causes VARCHAR/INTEGER type mismatch
        # Database CASCADE will automatically delete audit logs
        from sqlalchemy import text
        self.db.execute(
            text("DELETE FROM credentials WHERE id = :cred_id"),
            {"cred_id": credential_id}
        )
        self.db.commit()
        
        logger.info(f"Deleted credential '{cred_name}' (ID: {credential_id})")
        return True
    
    # ========================================================================
    # Credential Data Access (Decryption)
    # ========================================================================
    
    def get_decrypted_credential(
        self,
        credential_id: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get decrypted credential data.
        
        SECURITY: This method returns plaintext credentials!
        Only use for legitimate service operations. All access is audited.
        
        Args:
            credential_id: ID of credential
            user_id: User/service requesting access
            ip_address: IP address of requester
            service_name: Name of service requesting access
            
        Returns:
            Dict of decrypted credential values
            
        Raises:
            CredentialNotFoundError: If credential not found
            EncryptionKeyError: If decryption fails
        """
        credential = self.get_credential(credential_id)
        if not credential:
            raise CredentialNotFoundError(f"Credential {credential_id} not found")
        
        # Check if active
        if not credential.is_active:
            self._create_audit_log(
                credential=credential,
                action='access_denied',
                user_id=user_id,
                ip_address=ip_address,
                success=False,
                metadata={"reason": "credential_inactive", "service": service_name}
            )
            self.db.commit()
            raise CredentialValidationError(f"Credential '{credential.name}' is inactive")
        
        # Check expiration
        if credential.expires_at and credential.expires_at < datetime.utcnow():
            self._create_audit_log(
                credential=credential,
                action='access_denied',
                user_id=user_id,
                ip_address=ip_address,
                success=False,
                metadata={"reason": "credential_expired", "service": service_name}
            )
            self.db.commit()
            raise CredentialValidationError(f"Credential '{credential.name}' has expired")
        
        # Decrypt credential data
        try:
            decrypted_data = self.encryption_service.decrypt_dict(credential.encrypted_data)
        except Exception as e:
            # Don't log error - expected during bootstrap or with wrong key
            self._create_audit_log(
                credential=credential,
                action='access_failed',
                user_id=user_id,
                ip_address=ip_address,
                success=False,
                error_message=str(e),
                metadata={"service": service_name}
            )
            self.db.commit()
            raise EncryptionKeyError(f"Failed to decrypt credential: {e}")
        
        # Create audit log for successful access
        self._create_audit_log(
            credential=credential,
            action='accessed',
            user_id=user_id,
            ip_address=ip_address,
            success=True,
            metadata={"service": service_name, "fields_accessed": list(decrypted_data.keys())}
        )
        
        self.db.commit()
        
        logger.info(f"Credential '{credential.name}' accessed by {service_name or user_id or 'unknown'}")
        return decrypted_data
    
    def get_decrypted_credential_by_name(
        self,
        name: str,
        environment: str = "production",
        user_id: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get decrypted credential by name and environment.
        
        Args:
            name: Credential name
            environment: Environment
            user_id: User requesting access
            service_name: Service requesting access
            
        Returns:
            Decrypted credential data
            
        Raises:
            CredentialNotFoundError: If not found
        """
        credential = self.get_credential_by_name(name, environment)
        if not credential:
            raise CredentialNotFoundError(
                f"Credential '{name}' not found in {environment} environment"
            )
        
        return self.get_decrypted_credential(
            credential.id,
            user_id=user_id,
            service_name=service_name
        )
    
    # ========================================================================
    # Credential Testing
    # ========================================================================
    
    async def test_credential(
        self,
        credential_id: int,
        user_id: Optional[str] = None
    ) -> CredentialTestResponse:
        """
        Test a credential to verify it works.
        
        Args:
            credential_id: ID of credential to test
            user_id: User requesting test
            
        Returns:
            Test result
        """
        credential = self.get_credential(credential_id)
        if not credential:
            raise CredentialNotFoundError(f"Credential {credential_id} not found")
        
        cred_type = self.get_credential_type(credential.credential_type_id)
        
        # Decrypt credential data for testing
        try:
            decrypted_data = self.encryption_service.decrypt_dict(credential.encrypted_data)
        except Exception as e:
            return CredentialTestResponse(
                success=False,
                message=f"Failed to decrypt credential: {e}",
                tested_at=datetime.utcnow()
            )
        
        # Perform test based on credential type
        test_result = await self._perform_credential_test(cred_type, decrypted_data)
        
        # Update credential with test results
        credential.last_tested = datetime.utcnow()
        credential.test_status = 'passed' if test_result.success else 'failed'
        credential.test_message = test_result.message
        
        # Create audit log
        self._create_audit_log(
            credential=credential,
            action='tested',
            user_id=user_id,
            success=test_result.success,
            error_message=None if test_result.success else test_result.message,
            metadata={"test_result": test_result.details}
        )
        
        self.db.commit()
        
        logger.info(
            f"Tested credential '{credential.name}': "
            f"{'✅ PASSED' if test_result.success else '❌ FAILED'}"
        )
        if not test_result.success:
            logger.error(f"Test failure details: {test_result.message}")
        
        return test_result
    
    async def _perform_credential_test(
        self,
        cred_type: CredentialType,
        decrypted_data: Dict[str, Any]
    ) -> CredentialTestResponse:
        """
        Perform actual credential test using n8n-style validation.
        
        Args:
            cred_type: Credential type definition
            decrypted_data: Decrypted credential values
            
        Returns:
            Test result
        """
        try:
            async with CredentialTester() as tester:
                result = await tester.test_credential(cred_type.name, decrypted_data)
                
                return CredentialTestResponse(
                    success=result['success'],
                    message=result['message'],
                    details=result.get('details', {}),
                    tested_at=datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Credential test failed: {e}")
            return CredentialTestResponse(
                success=False,
                message=f"Test failed with error: {str(e)}",
                details={'error': str(e)},
                tested_at=datetime.utcnow()
            )
    
    async def _test_database_connection(
        self,
        cred_type_name: str,
        data: Dict[str, Any]
    ) -> CredentialTestResponse:
        """Test database connection"""
        import asyncio
        
        try:
            if cred_type_name == 'postgres_credentials':
                import psycopg2
                conn = psycopg2.connect(
                    host=data.get('host'),
                    port=data.get('port', 5432),
                    database=data.get('database'),
                    user=data.get('user'),
                    password=data.get('password'),
                    connect_timeout=5
                )
                conn.close()
                return CredentialTestResponse(
                    success=True,
                    message="PostgreSQL connection successful",
                    details={"database": data.get('database'), "host": data.get('host')},
                    tested_at=datetime.utcnow()
                )
            
            elif cred_type_name == 'redis_credentials':
                import redis
                client = redis.Redis(
                    host=data.get('host'),
                    port=data.get('port', 6379),
                    password=data.get('password'),
                    db=data.get('database', 0),
                    socket_connect_timeout=5
                )
                client.ping()
                client.close()
                return CredentialTestResponse(
                    success=True,
                    message="Redis connection successful",
                    details={"host": data.get('host'), "db": data.get('database', 0)},
                    tested_at=datetime.utcnow()
                )
            
            else:
                return CredentialTestResponse(
                    success=False,
                    message=f"No test implementation for {cred_type_name}",
                    tested_at=datetime.utcnow()
                )
        
        except Exception as e:
            return CredentialTestResponse(
                success=False,
                message=f"Connection failed: {str(e)}",
                details={"error": str(e)},
                tested_at=datetime.utcnow()
            )
    
    async def _test_api_call(
        self,
        test_config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> CredentialTestResponse:
        """Test API credential with HTTP call"""
        import httpx
        
        # Replace placeholders in URL and headers
        url = self._replace_placeholders(test_config.get('url', ''), data)
        headers = {}
        
        if test_config.get('headers'):
            for key, value in test_config['headers'].items():
                headers[key] = self._replace_placeholders(value, data)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 200:
                    return CredentialTestResponse(
                        success=True,
                        message="API connection successful",
                        details={"status_code": response.status_code, "url": url},
                        tested_at=datetime.utcnow()
                    )
                else:
                    return CredentialTestResponse(
                        success=False,
                        message=f"API returned status {response.status_code}",
                        details={"status_code": response.status_code, "response": response.text[:200]},
                        tested_at=datetime.utcnow()
                    )
        
        except Exception as e:
            return CredentialTestResponse(
                success=False,
                message=f"API test failed: {str(e)}",
                details={"error": str(e)},
                tested_at=datetime.utcnow()
            )
    
    async def _test_oauth2(
        self,
        test_config: Dict[str, Any],
        data: Dict[str, Any]
    ) -> CredentialTestResponse:
        """Test OAuth2 token"""
        # Basic OAuth2 token validation (check if token exists)
        if data.get('access_token'):
            return CredentialTestResponse(
                success=True,
                message="OAuth2 token present (full validation not implemented in MVP)",
                tested_at=datetime.utcnow()
            )
        else:
            return CredentialTestResponse(
                success=False,
                message="OAuth2 access token missing",
                tested_at=datetime.utcnow()
            )
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _validate_credential_data(
        self,
        data: Dict[str, Any],
        schema: List[Dict[str, Any]]
    ) -> None:
        """
        Validate credential data against schema.
        
        Args:
            data: Credential data to validate
            schema: Schema definition (can be JSON string or list)
            
        Raises:
            CredentialValidationError: If validation fails
        """
        # Handle schema as string (from database JSON column)
        import json
        
        if isinstance(schema, str):
            schema = json.loads(schema)
        
        # Convert JSON Schema format to list format if needed
        if isinstance(schema, dict) and 'properties' in schema:
            # This is a JSON Schema object, convert to list format
            properties = schema['properties']
            required_fields = schema.get('required', [])
            schema_list = []
            for key, prop in properties.items():
                field_def = {
                    'name': key,
                    'displayName': prop.get('title', key),
                    'type': prop.get('type', 'string'),
                    'required': key in required_fields,
                    'description': prop.get('description', ''),
                    'default': prop.get('default')
                }
                schema_list.append(field_def)
            schema = schema_list
        
        # Check required fields
        for field in schema:
            field_name = field.get('name')
            is_required = field.get('required', False)
            
            if is_required and field_name not in data:
                raise CredentialValidationError(f"Required field '{field_name}' is missing")
            
            # Type validation (basic)
            if field_name in data:
                value = data[field_name]
                field_type = field.get('type')
                
                if field_type == 'number' and not isinstance(value, (int, float)):
                    raise CredentialValidationError(f"Field '{field_name}' must be a number")
                
                if field_type == 'boolean' and not isinstance(value, bool):
                    raise CredentialValidationError(f"Field '{field_name}' must be a boolean")
    
    def _replace_placeholders(self, template: str, data: Dict[str, Any]) -> str:
        """Replace {field_name} placeholders in template string"""
        result = template
        for key, value in data.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def _create_audit_log(
        self,
        credential: Credential,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: Optional[bool] = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CredentialAuditLog:
        """Create audit log entry"""
        audit_log = CredentialAuditLog(
            credential_id=credential.id,
            action=action,
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            audit_metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        
        self.db.add(audit_log)
        return audit_log
    
    def get_audit_logs(
        self,
        credential_id: Optional[int] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[CredentialAuditLog]:
        """
        Get credential audit logs with filtering.
        
        Args:
            credential_id: Filter by credential
            action: Filter by action type
            user_id: Filter by user
            limit: Maximum results
            
        Returns:
            List of audit log entries
        """
        query = self.db.query(CredentialAuditLog)
        
        if credential_id:
            # Handle potential VARCHAR/INTEGER mismatch by casting
            from sqlalchemy import cast, Integer, text
            # Use cast to ensure type compatibility
            query = query.filter(cast(CredentialAuditLog.credential_id, Integer) == credential_id)
        
        if action:
            query = query.filter(CredentialAuditLog.action == action)
        
        if user_id:
            query = query.filter(CredentialAuditLog.user_id == user_id)
        
        return query.order_by(CredentialAuditLog.created_at.desc()).limit(limit).all()
    
    # ========================================================================
    # Bulk Operations
    # ========================================================================
    
    def export_credentials(
        self,
        environment: Optional[str] = None,
        include_encrypted: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Export credentials for backup/migration.
        
        Args:
            environment: Filter by environment
            include_encrypted: Include encrypted data (for backup)
            
        Returns:
            List of credential data
        """
        credentials = self.list_credentials(environment=environment, active_only=False)
        
        export_data = []
        for cred in credentials:
            cred_type = self.get_credential_type(cred.credential_type_id)
            
            export_item = {
                "name": cred.name,
                "credential_type": cred_type.name,
                "environment": cred.environment,
                "description": cred.description,
                "tags": cred.tags,
                "is_active": cred.is_active,
                "created_at": cred.created_at.isoformat(),
            }
            
            if include_encrypted:
                export_item["encrypted_data"] = cred.encrypted_data
            
            export_data.append(export_item)
        
        return export_data

