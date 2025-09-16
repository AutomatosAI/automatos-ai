
"""
Enhanced Security and Audit Logging System
==========================================

Provides comprehensive security monitoring, audit logging, and threat detection
for the orchestration system.
"""

import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import os
import sqlite3
from contextlib import contextmanager

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    COMMAND_EXECUTION = "command_execution"
    FILE_ACCESS = "file_access"
    NETWORK_ACCESS = "network_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"

@dataclass
class SecurityEvent:
    event_id: str
    event_type: EventType
    security_level: SecurityLevel
    timestamp: datetime
    user_id: str
    source_ip: str
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_score: int = 0

@dataclass
class ThreatIndicator:
    indicator_type: str
    value: str
    severity: SecurityLevel
    description: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1

class SecurityAuditLogger:
    """Comprehensive security audit logging system"""
    
    def __init__(self, db_path: str = "security_audit.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("security_audit")
        self._setup_database()
        self._setup_logging()
        
        # Threat detection
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.suspicious_patterns = {
            'failed_login_threshold': 5,
            'command_injection_patterns': [
                ';', '&&', '||', '`', '$(', '${', '../', '/etc/passwd'
            ],
            'suspicious_commands': [
                'rm -rf', 'dd if=', 'mkfs', 'fdisk', 'format'
            ],
            'rate_limit_threshold': 100  # requests per minute
        }
        
        # Rate limiting tracking
        self.request_counts = defaultdict(lambda: deque())
        self.failed_attempts = defaultdict(int)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def _setup_database(self):
        """Initialize SQLite database for audit logs"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    security_level TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    source_ip TEXT,
                    resource TEXT,
                    action TEXT,
                    result TEXT,
                    details TEXT,
                    risk_score INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS threat_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    UNIQUE(indicator_type, value)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON security_events(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_user 
                ON security_events(user_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_events_ip 
                ON security_events(source_ip)
            ''')
    
    def _setup_logging(self):
        """Setup structured logging"""
        handler = logging.FileHandler('security_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def log_event(self, event: SecurityEvent):
        """Log security event with threat detection"""
        with self.lock:
            # Store in database
            with self.get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO security_events 
                    (event_id, event_type, security_level, timestamp, user_id, 
                     source_ip, resource, action, result, details, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type.value,
                    event.security_level.value,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.source_ip,
                    event.resource,
                    event.action,
                    event.result,
                    json.dumps(event.details),
                    event.risk_score
                ))
                conn.commit()
            
            # Log to file
            log_data = asdict(event)
            log_data['timestamp'] = event.timestamp.isoformat()
            log_data['event_type'] = event.event_type.value
            log_data['security_level'] = event.security_level.value
            
            if event.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.logger.error(f"SECURITY ALERT: {json.dumps(log_data)}")
            else:
                self.logger.info(f"Security Event: {json.dumps(log_data)}")
            
            # Perform threat detection
            self._detect_threats(event)
    
    def _detect_threats(self, event: SecurityEvent):
        """Detect potential security threats"""
        
        # Failed authentication attempts
        if event.event_type == EventType.AUTHENTICATION and event.result == "failed":
            self.failed_attempts[event.source_ip] += 1
            
            if self.failed_attempts[event.source_ip] >= self.suspicious_patterns['failed_login_threshold']:
                self._add_threat_indicator(
                    "suspicious_ip",
                    event.source_ip,
                    SecurityLevel.HIGH,
                    f"Multiple failed login attempts: {self.failed_attempts[event.source_ip]}"
                )
        
        # Command injection detection
        if event.event_type == EventType.COMMAND_EXECUTION:
            command = event.details.get('command', '')
            for pattern in self.suspicious_patterns['command_injection_patterns']:
                if pattern in command:
                    self._add_threat_indicator(
                        "command_injection",
                        command,
                        SecurityLevel.CRITICAL,
                        f"Potential command injection detected: {pattern}"
                    )
        
        # Suspicious command detection
        if event.event_type == EventType.COMMAND_EXECUTION:
            command = event.details.get('command', '').lower()
            for suspicious_cmd in self.suspicious_patterns['suspicious_commands']:
                if suspicious_cmd in command:
                    self._add_threat_indicator(
                        "dangerous_command",
                        command,
                        SecurityLevel.HIGH,
                        f"Dangerous command executed: {suspicious_cmd}"
                    )
        
        # Rate limiting detection
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        self.request_counts[event.source_ip] = deque([
            req_time for req_time in self.request_counts[event.source_ip]
            if req_time > minute_ago
        ])
        
        # Add current request
        self.request_counts[event.source_ip].append(current_time)
        
        # Check rate limit
        if len(self.request_counts[event.source_ip]) > self.suspicious_patterns['rate_limit_threshold']:
            self._add_threat_indicator(
                "rate_limit_exceeded",
                event.source_ip,
                SecurityLevel.MEDIUM,
                f"Rate limit exceeded: {len(self.request_counts[event.source_ip])} requests/minute"
            )
    
    def _add_threat_indicator(self, indicator_type: str, value: str, severity: SecurityLevel, description: str):
        """Add or update threat indicator"""
        key = f"{indicator_type}:{value}"
        current_time = datetime.now()
        
        if key in self.threat_indicators:
            # Update existing indicator
            indicator = self.threat_indicators[key]
            indicator.last_seen = current_time
            indicator.count += 1
            indicator.severity = max(indicator.severity, severity, key=lambda x: x.value)
        else:
            # Create new indicator
            indicator = ThreatIndicator(
                indicator_type=indicator_type,
                value=value,
                severity=severity,
                description=description,
                first_seen=current_time,
                last_seen=current_time
            )
            self.threat_indicators[key] = indicator
        
        # Store in database
        with self.get_db_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO threat_indicators 
                (indicator_type, value, severity, description, first_seen, last_seen, count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                indicator.indicator_type,
                indicator.value,
                indicator.severity.value,
                indicator.description,
                indicator.first_seen.isoformat(),
                indicator.last_seen.isoformat(),
                indicator.count
            ))
            conn.commit()
        
        # Log threat indicator
        self.logger.warning(f"THREAT INDICATOR: {json.dumps(asdict(indicator), default=str)}")
    
    def get_security_events(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        security_level: Optional[SecurityLevel] = None,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Query security events with filters"""
        
        query = "SELECT * FROM security_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if security_level:
            query += " AND security_level = ?"
            params.append(security_level.value)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if source_ip:
            query += " AND source_ip = ?"
            params.append(source_ip)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        with self.get_db_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                event = SecurityEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    security_level=SecurityLevel(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    user_id=row[4],
                    source_ip=row[5],
                    resource=row[6],
                    action=row[7],
                    result=row[8],
                    details=json.loads(row[9]) if row[9] else {},
                    risk_score=row[10]
                )
                events.append(event)
        
        return events
    
    def get_threat_indicators(self, severity: Optional[SecurityLevel] = None) -> List[ThreatIndicator]:
        """Get threat indicators"""
        query = "SELECT * FROM threat_indicators WHERE 1=1"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        query += " ORDER BY last_seen DESC"
        
        indicators = []
        with self.get_db_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                indicator = ThreatIndicator(
                    indicator_type=row[1],
                    value=row[2],
                    severity=SecurityLevel(row[3]),
                    description=row[4],
                    first_seen=datetime.fromisoformat(row[5]),
                    last_seen=datetime.fromisoformat(row[6]),
                    count=row[7]
                )
                indicators.append(indicator)
        
        return indicators
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        events = self.get_security_events(start_time=start_time, end_time=end_time, limit=1000)
        threat_indicators = self.get_threat_indicators()
        
        # Analyze events
        event_counts = defaultdict(int)
        security_level_counts = defaultdict(int)
        user_activity = defaultdict(int)
        ip_activity = defaultdict(int)
        
        for event in events:
            event_counts[event.event_type.value] += 1
            security_level_counts[event.security_level.value] += 1
            if event.user_id:
                user_activity[event.user_id] += 1
            if event.source_ip:
                ip_activity[event.source_ip] += 1
        
        # Calculate risk score
        total_risk_score = sum(event.risk_score for event in events)
        avg_risk_score = total_risk_score / len(events) if events else 0
        
        # Top threats
        critical_threats = [t for t in threat_indicators if t.severity == SecurityLevel.CRITICAL]
        high_threats = [t for t in threat_indicators if t.severity == SecurityLevel.HIGH]
        
        report = {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": hours
            },
            "summary": {
                "total_events": len(events),
                "total_risk_score": total_risk_score,
                "average_risk_score": round(avg_risk_score, 2),
                "critical_threats": len(critical_threats),
                "high_threats": len(high_threats)
            },
            "event_breakdown": dict(event_counts),
            "security_level_breakdown": dict(security_level_counts),
            "top_users": dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_source_ips": dict(sorted(ip_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "critical_threats": [asdict(t) for t in critical_threats[:10]],
            "high_threats": [asdict(t) for t in high_threats[:10]],
            "recommendations": self._generate_recommendations(events, threat_indicators)
        }
        
        return report
    
    def _generate_recommendations(self, events: List[SecurityEvent], threats: List[ThreatIndicator]) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        
        # Check for high-risk IPs
        high_risk_ips = [t for t in threats if t.indicator_type == "suspicious_ip"]
        if high_risk_ips:
            recommendations.append(
                f"Consider blocking {len(high_risk_ips)} suspicious IP addresses with multiple failed login attempts"
            )
        
        # Check for command injection attempts
        injection_attempts = [t for t in threats if t.indicator_type == "command_injection"]
        if injection_attempts:
            recommendations.append(
                "Implement stricter input validation to prevent command injection attacks"
            )
        
        # Check for dangerous commands
        dangerous_commands = [t for t in threats if t.indicator_type == "dangerous_command"]
        if dangerous_commands:
            recommendations.append(
                "Review and restrict dangerous command execution permissions"
            )
        
        # Check for rate limiting issues
        rate_limit_violations = [t for t in threats if t.indicator_type == "rate_limit_exceeded"]
        if rate_limit_violations:
            recommendations.append(
                "Implement more aggressive rate limiting for high-traffic sources"
            )
        
        # Check authentication failures
        auth_failures = [e for e in events if e.event_type == EventType.AUTHENTICATION and e.result == "failed"]
        if len(auth_failures) > 50:
            recommendations.append(
                "High number of authentication failures detected - consider implementing CAPTCHA or account lockouts"
            )
        
        return recommendations

# Utility functions for easy integration
def create_security_event(
    event_type: EventType,
    user_id: str,
    source_ip: str,
    resource: str,
    action: str,
    result: str,
    details: Dict[str, Any] = None,
    security_level: SecurityLevel = SecurityLevel.MEDIUM
) -> SecurityEvent:
    """Create a security event with auto-generated ID"""
    
    event_id = hashlib.sha256(
        f"{time.time()}{user_id}{source_ip}{action}".encode()
    ).hexdigest()[:16]
    
    return SecurityEvent(
        event_id=event_id,
        event_type=event_type,
        security_level=security_level,
        timestamp=datetime.now(),
        user_id=user_id,
        source_ip=source_ip,
        resource=resource,
        action=action,
        result=result,
        details=details or {}
    )

# Global audit logger instance
_audit_logger = None

def get_audit_logger() -> SecurityAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger

# Example usage
if __name__ == "__main__":
    audit_logger = SecurityAuditLogger()
    
    # Test event logging
    event = create_security_event(
        event_type=EventType.AUTHENTICATION,
        user_id="test_user",
        source_ip="192.168.1.100",
        resource="/api/login",
        action="login_attempt",
        result="failed",
        details={"reason": "invalid_password"},
        security_level=SecurityLevel.MEDIUM
    )
    
    audit_logger.log_event(event)
    
    # Generate report
    report = audit_logger.generate_security_report(hours=1)
    print(json.dumps(report, indent=2, default=str))
