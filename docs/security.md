---
title: Security & Compliance Guide
description: Comprehensive security configuration, credential encryption, audit logging, and compliance standards for Automatos AI
---

# 🔐 Security & Compliance Guide

*Enterprise-grade security with defense-in-depth architecture*

---

## 📖 Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Credential Encryption](#credential-encryption)
4. [Audit Logging](#audit-logging)
5. [Network Security](#network-security)
6. [Data Protection](#data-protection)
7. [Compliance Standards](#compliance-standards)
8. [Security Best Practices](#security-best-practices)

---

## Security Overview

### Defense-in-Depth Architecture

Automatos AI implements **7 layers of security**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LAYER 1: NETWORK SECURITY                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Firewall (UFW)                                   │         │
│  │ • Rate limiting (100 req/min default)              │         │
│  │ • DDoS protection (fail2ban)                       │         │
│  │ • TLS 1.3 encryption                               │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 2: APPLICATION AUTHENTICATION                             │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • API key authentication (X-API-Key header)        │         │
│  │ • JWT tokens for sessions                          │         │
│  │ • OAuth2 integration (future)                      │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 3: AUTHORIZATION & ACCESS CONTROL                         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Role-based access control (RBAC)                 │         │
│  │ • Agent-tool permission matrix                     │         │
│  │ • Least privilege principle                        │         │
│  │ • Multi-tenant isolation                           │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 4: INPUT VALIDATION                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Pydantic schema validation                       │         │
│  │ • SQL injection prevention                         │         │
│  │ • XSS sanitization                                 │         │
│  │ • Command injection protection                     │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 5: DATA ENCRYPTION                                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Credentials: Fernet AES-128-CBC + HMAC-SHA256    │         │
│  │ • Database: AES-256 at rest                        │         │
│  │ • Transit: TLS 1.3                                 │         │
│  │ • API Keys: Hashed with bcrypt                     │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 6: AUDIT & MONITORING                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • All API calls logged                             │         │
│  │ • Credential access tracked                        │         │
│  │ • Tool execution audited                           │         │
│  │ • Security events monitored                        │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  LAYER 7: INCIDENT RESPONSE                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • Automated alerting                               │         │
│  │ • Anomaly detection                                │         │
│  │ • Automatic account lockout                        │         │
│  │ • Security event correlation                       │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Authentication & Authorization

### API Key Authentication

**All API requests require authentication**:

```bash
curl -H "X-API-Key: your_secure_api_key" \
     ${API_URL}/api/v1/agents
```

**API Key Generation**:
```bash
# Generate secure API key (32 bytes = 256 bits)
openssl rand -hex 32

# Add to .env
API_KEY=your_generated_key_here
```

### JWT Session Tokens

For web UI sessions:

```python
# Token structure
{
  "sub": "user_id",
  "exp": timestamp + 3600,  # 1 hour expiration
  "iat": timestamp,
  "permissions": ["read", "write", "admin"]
}
```

### Role-Based Access Control

```python
class UserRole(Enum):
    ADMIN = "admin"           # Full access
    DEVELOPER = "developer"   # Code & workflow access
    OPERATOR = "operator"     # Execute workflows only
    VIEWER = "viewer"         # Read-only access

# Permission matrix
ROLE_PERMISSIONS = {
    "admin": ["*"],
    "developer": ["agents.*", "workflows.*", "tools.*"],
    "operator": ["workflows.execute", "agents.execute"],
    "viewer": ["*.read"]
}
```

---

## Credential Encryption

### Fernet Encryption (Symmetric)

**Algorithm**: AES-128-CBC with HMAC-SHA256 authentication

**Key Generation** (one-time setup):
```python
from cryptography.fernet import Fernet

# Generate key
key = Fernet.generate_key()

# Save securely (NOT in code!)
with open('.credential_key', 'wb') as f:
    f.write(key)

# Restrict permissions
os.chmod('.credential_key', 0o600)  # Only owner can read
```

### Encryption Process

```python
from cryptography.fernet import Fernet
import json

# Load key
with open('.credential_key', 'rb') as f:
    key = f.read()

fernet = Fernet(key)

# Encrypt credential data
credential_data = {
    "access_key_id": "AKIA...",
    "secret_access_key": "...",
    "region": "us-east-1"
}

encrypted_data = fernet.encrypt(
    json.dumps(credential_data).encode()
)

# Store in database (encrypted)
credential.encrypted_data = encrypted_data.decode()
```

### Decryption (Runtime Only)

```python
# Decrypt only when needed
def get_credential_data(credential_id: int):
    credential = db.query(Credential).filter(id=credential_id).first()
    
    # Decrypt
    decrypted = fernet.decrypt(credential.encrypted_data.encode())
    
    # Return as dict
    return json.loads(decrypted)

# Decrypted data NEVER stored, only used immediately
```

### Key Rotation

```bash
# Generate new key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Rotate credentials:
# 1. Decrypt all with old key
# 2. Re-encrypt with new key
# 3. Update .credential_key
# 4. Restart services
```

---

## Audit Logging

### What Gets Logged

**ALL security-sensitive operations**:

```sql
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,  -- 'credential_access', 'api_call', 'tool_execution'
    actor VARCHAR(255),                 -- User or agent_id
    resource_type VARCHAR(100),         -- 'credential', 'agent', 'workflow'
    resource_id VARCHAR(255),
    action VARCHAR(100),                -- 'create', 'read', 'update', 'delete', 'execute'
    success BOOLEAN,
    ip_address VARCHAR(50),
    user_agent TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_actor ON audit_logs(actor);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_success ON audit_logs(success);
```

### Example Audit Trail

```sql
-- View recent security events
SELECT 
    created_at,
    event_type,
    actor,
    action,
    success
FROM audit_logs
WHERE event_type IN ('credential_access', 'tool_execution', 'shell_command')
ORDER BY created_at DESC
LIMIT 20;

-- Results:
2025-01-15 10:35:28 | credential_access | agent_5  | read    | true
2025-01-15 10:35:29 | tool_execution    | agent_5  | execute | true
2025-01-15 10:36:45 | credential_access | admin    | create  | true
2025-01-15 10:37:12 | shell_command     | agent_8  | execute | false  ← BLOCKED!
```

### Audit Log Retention

```python
# Retention policy
AUDIT_LOG_RETENTION = {
    'security_events': 365,      # 1 year
    'api_calls': 90,            # 3 months  
    'tool_executions': 180,     # 6 months
    'credential_access': 365    # 1 year
}

# Automated cleanup job (nightly)
async def cleanup_old_audit_logs():
    for event_type, days in AUDIT_LOG_RETENTION.items():
        cutoff = datetime.now() - timedelta(days=days)
        
        deleted = db.query(AuditLog).filter(
            AuditLog.event_type == event_type,
            AuditLog.created_at < cutoff
        ).delete()
        
        logger.info(f"Cleaned up {deleted} old {event_type} logs")
```

---

## Network Security

### Firewall Configuration

```bash
# UFW (Ubuntu)
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw enable

# Verify
ufw status numbered
```

### Rate Limiting

**Nginx Configuration**:
```nginx
# Rate limit zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=10r/s;

# Apply to API endpoints
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    limit_req_status 429;
}

# Strict limit for authentication
location /api/auth/ {
    limit_req zone=auth_limit burst=5 nodelay;
}
```

### DDoS Protection

```bash
# Install fail2ban
apt install fail2ban

# Configure for Nginx
cat > /etc/fail2ban/jail.local <<EOF
[nginx-limit-req]
enabled = true
filter = nginx-limit-req
port = http,https
logpath = /var/log/nginx/*error.log
maxretry = 10
findtime = 60
bantime = 3600
EOF

systemctl restart fail2ban
```

### SSL/TLS Configuration

```nginx
# TLS 1.3 only (most secure)
ssl_protocols TLSv1.3;

# Strong cipher suites
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

# Prefer server ciphers
ssl_prefer_server_ciphers off;

# HSTS (force HTTPS)
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
```

---

## Data Protection

### Database Encryption

**At Rest**:
```sql
-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt sensitive columns
CREATE TABLE sensitive_data (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    encrypted_field BYTEA,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Encrypt on insert
INSERT INTO sensitive_data (user_id, encrypted_field)
VALUES (1, pgp_sym_encrypt('sensitive data', 'encryption-key'));

-- Decrypt on select
SELECT pgp_sym_decrypt(encrypted_field, 'encryption-key') 
FROM sensitive_data WHERE id = 1;
```

**In Transit**:
```bash
# Require SSL for PostgreSQL connections
# In postgresql.conf:
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
```

### Redis Security

```bash
# Require authentication
requirepass your_secure_redis_password

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
rename-command SHUTDOWN ""

# Bind to localhost only (if on same server as API)
bind 127.0.0.1

# Enable persistence
save 900 1      # Save after 900 sec if 1 key changed
save 300 10     # Save after 300 sec if 10 keys changed
save 60 10000   # Save after 60 sec if 10000 keys changed
```

---

## Compliance Standards

### SOC 2 Type II

**Controls Implemented**:

✅ **Access Controls**
- Multi-factor authentication
- Role-based access control
- Least privilege principle
- Regular access reviews

✅ **Change Management**
- Version control (Git)
- Code reviews required
- Deployment approvals
- Rollback procedures

✅ **Data Protection**
- Encryption at rest and in transit
- Secure key management
- Data classification
- Retention policies

✅ **Monitoring & Logging**
- Comprehensive audit logs
- Real-time alerting
- Log retention (365 days for security events)
- SIEM integration ready

✅ **Incident Response**
- Automated detection
- Escalation procedures
- Recovery procedures
- Post-incident reviews

### GDPR Compliance

✅ **Data Subject Rights**:
```python
# Right to access
GET /api/users/{id}/data-export

# Right to deletion
DELETE /api/users/{id}/data

# Right to portability
GET /api/users/{id}/data-export?format=json
```

✅ **Data Processing**:
- Purpose limitation
- Storage minimization
- Consent management
- Processing records

✅ **Security Measures**:
- Encryption
- Pseudonymization
- Access controls
- Breach notification (<72 hours)

### PCI-DSS (If Processing Payments)

**NOT storing payment card data** - using tokens from payment processors

**Controls**:
- TLS 1.3 for transmission
- No card data storage
- Audit logging
- Access controls

---

## Security Best Practices

### 1. Credential Management

**Do's** ✅:
- Use unique passwords (>16 characters)
- Rotate API keys quarterly
- Store in encrypted credential system
- Use separate credentials per environment

**Don'ts** ❌:
- Never commit credentials to Git
- Never log decrypted credentials
- Never share credentials between environments
- Never reuse passwords

### 2. Tool Permissions

**Principle of Least Privilege**:

```python
# Agent should only have tools it needs
code_review_agent_tools = [
    'search_knowledge',   # ✅ Needs for research
    'read_file',          # ✅ Needs to view code
    'github_pr_mcp'       # ✅ Needs to comment on PRs
    # ❌ NOT write_file (shouldn't modify during review)
    # ❌ NOT execute_command (no shell access needed)
]
```

### 3. Shell Command Safety

**Command Whitelist** (only allow safe commands):
```python
ALLOWED_COMMANDS = [
    'npm test',
    'npm run build',
    'git status',
    'git log',
    'pytest',
    'docker ps',
    'docker logs'
]

FORBIDDEN_COMMANDS = [
    'rm -rf /',
    'dd if=',
    'mkfs',
    'format',
    ':(){ :|:& };:',  # Fork bomb
    'chmod 777',
    'chown root',
    'sudo'
]
```

### 4. Environment Separation

```bash
# Development
DATABASE_URL=postgresql://localhost:5432/automatos_dev
OPENAI_API_KEY=sk-test-key-for-development

# Staging
DATABASE_URL=postgresql://staging-db:5432/automatos_staging
OPENAI_API_KEY=sk-staging-key-limited-spend

# Production
DATABASE_URL=postgresql://prod-db:5432/automatos_prod
OPENAI_API_KEY=sk-prod-key-with-spend-limits
```

### 5. Secrets Management

**Never commit secrets**:
```bash
# .gitignore
.env
.env.*
.credential_key
*.pem
*.key
credentials.json
secrets/
```

**Use environment variables**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

# ✅ Good
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# ❌ Bad
OPENAI_API_KEY = "sk-actual-key-here"
```

---

## Security Monitoring

### Key Metrics to Monitor

```python
# Failed authentication attempts
SELECT COUNT(*) as failed_attempts
FROM audit_logs
WHERE event_type = 'authentication'
AND success = false
AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY ip_address
HAVING COUNT(*) > 5;

# Suspicious tool executions
SELECT *
FROM audit_logs
WHERE event_type = 'tool_execution'
AND action = 'execute_command'
AND created_at > NOW() - INTERVAL '24 hours';

# Unusual credential access patterns
SELECT credential_id, COUNT(*) as access_count
FROM audit_logs
WHERE event_type = 'credential_access'
AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY credential_id
HAVING COUNT(*) > 50;  # >50 accesses per hour = suspicious
```

### Automated Alerts

```python
# Alert on suspicious activity
async def check_security_events():
    # Check failed logins
    failed_logins = await count_failed_logins_last_hour()
    if failed_logins > 10:
        await send_alert("High failed login rate", severity="high")
    
    # Check unusual credential access
    credential_access = await check_credential_access_patterns()
    if credential_access.is_anomalous:
        await send_alert("Unusual credential access", severity="critical")
    
    # Check shell command attempts
    shell_commands = await count_shell_command_attempts()
    if shell_commands > 0:
        await send_alert("Shell command execution detected", severity="medium")
```

---

## Incident Response

### Security Incident Levels

| Level | Description | Response Time | Actions |
|-------|-------------|---------------|---------|
| **Critical** | Data breach, system compromise | <15 minutes | Immediate lockdown, investigation |
| **High** | Unauthorized access attempt | <1 hour | Block IP, investigate, alert admin |
| **Medium** | Suspicious activity | <4 hours | Monitor, log, review |
| **Low** | Policy violation | <24 hours | Log, notify user |

### Automated Response

```python
async def handle_security_incident(incident: SecurityIncident):
    if incident.level == "critical":
        # Immediate lockdown
        await disable_all_api_keys()
        await lock_all_credentials()
        await notify_admin_immediately()
        await create_incident_report()
    
    elif incident.level == "high":
        # Block and investigate
        await block_ip_address(incident.ip)
        await disable_affected_credentials()
        await notify_admin()
        await log_incident()
    
    elif incident.level == "medium":
        # Monitor and log
        await increase_monitoring(incident.resource)
        await log_incident()
    
    # Always log
    await audit_log(incident)
```

---

## Security Checklist

### Pre-Production

- [ ] All credentials encrypted
- [ ] API keys rotated
- [ ] SSL certificates installed
- [ ] Firewall configured
- [ ] Rate limiting enabled
- [ ] Audit logging active
- [ ] Backups tested
- [ ] Security scan passed
- [ ] Penetration test completed

### Ongoing (Monthly)

- [ ] Review audit logs
- [ ] Rotate API keys
- [ ] Update SSL certificates
- [ ] Patch dependencies
- [ ] Review user permissions
- [ ] Test backup restore
- [ ] Security scan
- [ ] Compliance review

---

## Vulnerability Management

### Dependency Scanning

```bash
# Python dependencies
pip install safety
safety check

# Node dependencies
npm audit
npm audit fix

# Container scanning
docker scan automatos_backend_api
```

### Security Updates

```bash
# Update system packages
apt update && apt upgrade -y

# Update Python packages
pip install --upgrade -r requirements.txt

# Update Node packages
npm update
```

---

## Compliance Reporting

### Generate Compliance Report

```bash
curl -X GET ${API_URL}/api/compliance/report?standard=soc2 \
  -H "X-API-Key: admin_key"
```

**Response**:
```json
{
  "standard": "SOC2",
  "controls": {
    "total": 45,
    "implemented": 43,
    "partial": 2,
    "not_implemented": 0
  },
  "compliance_score": 0.956,
  "last_audit": "2025-01-01",
  "next_review": "2025-04-01"
}
```

---

## Next Steps

1. **🚀 [Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production security hardening
2. **🔒 [Credential System](CREDENTIAL_SYSTEM_GUIDE.md)** - Credential management details
3. **🔧 [Tools Guide](TOOLS_INTEGRATION_GUIDE.md)** - Tool permission management
4. **📊 [Monitoring](MONITORING.md)** - Security monitoring setup

---

**Built with ❤️ for enterprise-grade security**

*Last updated: January 2025*

