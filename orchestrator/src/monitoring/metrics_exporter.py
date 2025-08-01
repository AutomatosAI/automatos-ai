"""
Orchestrator AI Metrics Exporter for Prometheus
Exposes security audit logs and workflow metrics
"""

import json
import sqlite3
import time
from flask import Flask, Response
from collections import defaultdict, Counter
import os

app = Flask(__name__)

def get_security_metrics():
    """Extract metrics from security audit database"""
    try:
        db_path = '/app/security_audit.db'
        if not os.path.exists(db_path):
            return {}
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get metrics from last 24 hours
        cursor.execute("""
            SELECT event_type, COUNT(*) as count, AVG(risk_score) as avg_risk_score
            FROM security_events 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY event_type
        """)
        
        metrics = {}
        for row in cursor.fetchall():
            event_type, count, avg_risk_score = row
            metrics[f'orchestrator_security_events_total{{event_type="{event_type}"}}'] = count
            metrics[f'orchestrator_security_risk_score_avg{{event_type="{event_type}"}}'] = avg_risk_score or 0
            
        conn.close()
        return metrics
    except Exception as e:
        print(f"Error getting security metrics: {e}")
        return {}

def get_workflow_metrics():
    """Extract workflow metrics from container logs"""
    try:
        # Count recent workflow requests from logs  
        import subprocess
        result = subprocess.run([
            'docker', 'logs', '--since', '1h', 'enhanced_mcp_bridge'
        ], capture_output=True, text=True)
        
        logs = result.stdout
        workflow_requests = logs.count('POST /workflow')
        workflow_errors = logs.count('error')
        
        return {
            'orchestrator_workflow_requests_total': workflow_requests,
            'orchestrator_workflow_errors_total': workflow_errors,
            'orchestrator_uptime_seconds': time.time()
        }
    except Exception as e:
        print(f"Error getting workflow metrics: {e}")
        return {
            'orchestrator_uptime_seconds': time.time()
        }

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    all_metrics = {}
    all_metrics.update(get_security_metrics())
    all_metrics.update(get_workflow_metrics())
    
    # Format as Prometheus metrics
    output = []
    for metric_name, value in all_metrics.items():
        output.append(f"{metric_name} {value}")
    
    return Response('\n'.join(output) + '\n', mimetype='text/plain')

@app.route('/health')  
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': time.time()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=False)
