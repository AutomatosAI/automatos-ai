---
title: Playbooks - Automated Pattern Discovery Guide
description: Learn how Automatos AI automatically discovers workflow patterns using FP-Growth algorithm and creates reusable playbook templates
---

# 📚 Playbooks - Automated Pattern Discovery Guide

*"GitHub Copilot for Workflows" - Automatically learn and reuse successful patterns*

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [FP-Growth Algorithm](#fp-growth-algorithm)
4. [Pattern Discovery](#pattern-discovery)
5. [Playbook Generation](#playbook-generation)
6. [Using Playbooks](#using-playbooks)
7. [Real-World Examples](#real-world-examples)
8. [API Reference](#api-reference)
9. [UI Guide](#ui-guide)

---

## Overview

### What are Playbooks?

Playbooks are **automatically discovered workflow patterns** that the system learns from historical executions. Instead of manually creating workflows from scratch, users can leverage proven patterns that have worked successfully in the past.

**The Vision**:
```
Traditional Workflow Creation ❌:
  User creates workflow manually
  → Trial and error
  → Repeats mistakes
  → Organizational knowledge lost

Playbook-Driven Creation ✅:
  System observes what works
  → Extracts successful patterns
  → Creates reusable templates
  → Self-improving system
```

### Key Features

| Feature | Description | Impact |
|---------|-------------|--------|
| **Automated Discovery** | No manual pattern creation needed | Zero effort |
| **FP-Growth Mining** | Efficient pattern extraction | 2-3x faster than alternatives |
| **Confidence Scoring** | Validates patterns statistically | 80%+ success rate |
| **1-Click Creation** | Convert playbook to workflow instantly | 70% time savings |
| **Continuous Learning** | Improves from every execution | Gets better over time |
| **Multi-Tenant Isolation** | Org-specific patterns only | Secure & relevant |

---

## How It Works

### The Pattern Discovery Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│           AUTOMATED PATTERN DISCOVERY PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: DATA COLLECTION                                         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Query: workflow_executions (last 30-90 days)       │         │
│  │ Filter: status = 'completed'                       │         │
│  │ Extract: [agents, sequence, time, tokens, cost]    │         │
│  │ Result: 50-500 completed workflows                 │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 2: SEQUENCE EXTRACTION                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Convert workflows to agent sequences:              │         │
│  │ Example: ["CodeArchitect",                         │         │
│  │          "SecurityExpert",                         │         │
│  │          "DocumentGenerator"]                      │         │
│  │ Group by: category, tags, goal                     │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 3: FREQUENT PATTERN MINING (FP-Growth)                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Algorithm: FP-Growth                               │         │
│  │ Min Support: 5 occurrences (configurable)          │         │
│  │ Find: Patterns that appear ≥5 times                │         │
│  │ Output: [(pattern, frequency)]                     │         │
│  │ Time: ~10 seconds for 500 workflows                │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 4: PATTERN VALIDATION                                      │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Calculate metrics:                                 │         │
│  │ - Support (how often pattern occurs)               │         │
│  │ - Confidence (success rate)                        │         │
│  │ - Lift (vs random selection)                       │         │
│  │ - Avg execution time, tokens, cost                 │         │
│  │ Filter: confidence ≥ 80%                           │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 5: PLAYBOOK GENERATION                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Create workflow_template record                    │         │
│  │ Auto-name: "auto_security_audit_003"               │         │
│  │ Store metadata: confidence, perf, use cases        │         │
│  │ Status: active                                     │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  RESULT: Ready-to-use Playbooks                                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Users can:                                         │         │
│  │ - Browse discovered patterns                       │         │
│  │ - Preview pattern details                          │         │
│  │ - Create workflow from playbook (1-click)          │         │
│  │ - Customize before execution                       │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Context Engineering Alignment

Playbooks follow the progressive complexity model:

- **Atoms**: Individual agent actions
- **Molecules**: Agent sequences (2-3 agents)
- **Cells**: Validated workflow patterns (playbooks)
- **Organs**: Pattern-based workflow optimization
- **Organisms**: Self-improving orchestration system

---

## FP-Growth Algorithm

### Why FP-Growth?

**FP-Growth (Frequent Pattern Growth)** is chosen over alternatives because:

| Algorithm | Speed | Memory | Best For |
|-----------|-------|--------|----------|
| **Apriori** | Slow | High | Small datasets |
| **Eclat** | Medium | Medium | Dense datasets |
| **FP-Growth** | **Fast** | **Low** | **Sequences** ✅ |

**Performance**:
- 2-3x faster than Apriori
- No candidate generation (memory efficient)
- Compressed FP-tree structure
- Perfect for agent sequences

### Algorithm Implementation

```python
class WorkflowPatternMiner:
    """
    FP-Growth algorithm for workflow pattern mining
    """
    
    def mine_patterns(
        self,
        min_support: int = 5,
        min_confidence: float = 0.80,
        execution_window_days: int = 30
    ) -> List[PlaybookPattern]:
        """
        Discover workflow patterns using FP-Growth
        
        Parameters:
        - min_support: Minimum occurrences required
        - min_confidence: Minimum success rate
        - execution_window_days: Historical window
        
        Returns:
        - List of validated playbook patterns
        """
        
        # Step 1: Collect workflow executions
        cutoff_date = datetime.now() - timedelta(days=execution_window_days)
        executions = db.query(WorkflowExecution).filter(
            WorkflowExecution.status == 'completed',
            WorkflowExecution.created_at >= cutoff_date
        ).all()
        
        # Step 2: Extract agent sequences
        transactions = []
        for execution in executions:
            agents = self._extract_agent_sequence(execution)
            transactions.append({
                'sequence': agents,
                'success': execution.quality_score >= 0.7,
                'execution_time': execution.duration_seconds,
                'tokens': execution.total_tokens,
                'cost': execution.total_cost
            })
        
        # Step 3: Build FP-tree
        fp_tree = self._build_fp_tree(transactions, min_support)
        
        # Step 4: Mine frequent patterns
        patterns = fp_tree.mine_patterns(min_support)
        
        # Step 5: Calculate metrics and validate
        validated_patterns = []
        for pattern, support in patterns:
            metrics = self._calculate_pattern_metrics(
                pattern, transactions
            )
            
            if metrics['confidence'] >= min_confidence:
                validated_patterns.append({
                    'pattern': pattern,
                    'support': support,
                    'confidence': metrics['confidence'],
                    'lift': metrics['lift'],
                    'avg_execution_time': metrics['avg_time'],
                    'avg_tokens': metrics['avg_tokens'],
                    'avg_cost': metrics['avg_cost']
                })
        
        return validated_patterns
```

### FP-Tree Construction

```python
class FPTree:
    """Compressed tree structure for pattern mining"""
    
    def __init__(self):
        self.root = FPNode(None, None, None)
        self.header_table = {}
    
    def insert(self, transaction: List[str]):
        """
        Insert transaction into FP-tree
        
        Example transaction: ['CodeArchitect', 'SecurityExpert', 'DocumentGen']
        """
        current_node = self.root
        
        for item in transaction:
            # Check if item node exists as child
            child = current_node.get_child(item)
            
            if child:
                # Increment count if exists
                child.count += 1
            else:
                # Create new node
                child = FPNode(item, current_node, None)
                current_node.add_child(child)
                
                # Update header table
                if item in self.header_table:
                    self.header_table[item].append(child)
                else:
                    self.header_table[item] = [child]
            
            current_node = child
    
    def mine_patterns(self, min_support: int) -> List[Tuple[List[str], int]]:
        """
        Extract frequent patterns from FP-tree
        
        Returns:
        - List of (pattern, support_count) tuples
        """
        patterns = []
        
        # Mine patterns for each item in header table
        for item in sorted(self.header_table.keys()):
            # Get conditional pattern base
            conditional_base = self._get_conditional_base(item)
            
            # Build conditional FP-tree
            conditional_tree = FPTree()
            for pattern, count in conditional_base:
                if count >= min_support:
                    conditional_tree.insert(pattern)
            
            # Recursively mine patterns
            sub_patterns = conditional_tree.mine_patterns(min_support)
            
            # Add item to each sub-pattern
            for sub_pattern, support in sub_patterns:
                patterns.append(([item] + sub_pattern, support))
            
            # Add item itself as pattern
            total_support = sum(node.count for node in self.header_table[item])
            if total_support >= min_support:
                patterns.append(([item], total_support))
        
        return patterns
```

---

## Pattern Discovery

### Pattern Metrics

#### 1. Support

**Definition**: How often a pattern occurs

**Formula**:
```
support(A→B→C) = count(A→B→C) / total_workflows

Example:
15 workflows with pattern [CodeArchitect, SecurityExpert, DocumentGen]
100 total workflows
support = 15 / 100 = 0.15 (15%)
```

**Interpretation**:
- High support (>10%): Very common pattern
- Medium support (5-10%): Moderately common
- Low support (<5%): Rare but possibly valuable

#### 2. Confidence

**Definition**: Success rate when pattern is used

**Formula**:
```
confidence(A→B→C) = successful(A→B→C) / count(A→B→C)

Example:
15 workflows with pattern
14 succeeded (quality ≥ 0.7)
confidence = 14 / 15 = 0.933 (93.3%)
```

**Interpretation**:
- High confidence (≥90%): Very reliable pattern
- Medium confidence (80-90%): Reliable
- Low confidence (<80%): Not validated (filtered out)

#### 3. Lift

**Definition**: How much better than random selection

**Formula**:
```
lift(A→B) = P(B|A) / P(B)
         = confidence(A→B) / support(B)

Example:
P(SecurityExpert | CodeArchitect) = 0.85
P(SecurityExpert alone) = 0.30
lift = 0.85 / 0.30 = 2.83

Interpretation: Using CodeArchitect increases likelihood 
of using SecurityExpert by 2.83x
```

**Interpretation**:
- lift > 1.0: Positive correlation (agents work well together)
- lift = 1.0: No correlation (independent)
- lift < 1.0: Negative correlation (avoid combination)

### Pattern Example

**Discovered Pattern**: Security Audit Workflow

```json
{
  "id": 42,
  "name": "auto_security_audit_003",
  "pattern": [
    "CodeArchitect",
    "SecurityExpert", 
    "VulnerabilityAnalyzer",
    "ComplianceChecker"
  ],
  "metrics": {
    "support": 23,           // Occurred 23 times
    "confidence": 0.957,     // 95.7% success rate
    "lift": 2.8,             // 2.8x better than random
    "usage_count": 12        // Used 12 times after discovery
  },
  "performance": {
    "avg_execution_time": 245,  // 4 min 5 sec
    "avg_token_usage": 8234,
    "avg_cost": 0.12,
    "historical_success_rate": 0.957
  },
  "metadata": {
    "category": "security",
    "tenant_id": "org-acme",
    "created_at": "2025-10-15T10:30:00Z",
    "last_mined_at": "2025-11-01T02:00:00Z"
  }
}
```

---

## Playbook Generation

### Automatic Naming

Playbooks are auto-named using the pattern:
```
auto_{category}_{sequence_number}

Examples:
- auto_security_audit_003
- auto_code_review_015
- auto_deployment_007
- auto_data_analysis_002
```

### Template Structure

```json
{
  "name": "auto_security_audit_003",
  "description": "Automatically discovered pattern for security audits",
  "workflow_definition": {
    "goal": "Perform comprehensive security audit",
    "category": "security",
    "steps": [
      {
        "agent_type": "code_architect",
        "description": "Analyze code structure and architecture",
        "skills": ["code_analysis", "architecture"],
        "priority": "high"
      },
      {
        "agent_type": "security_expert",
        "description": "Scan for security vulnerabilities",
        "skills": ["security_audit", "owasp"],
        "priority": "critical"
      },
      {
        "agent_type": "vulnerability_analyzer",
        "description": "Deep dive into identified vulnerabilities",
        "skills": ["vulnerability_analysis", "risk_assessment"],
        "priority": "high"
      },
      {
        "agent_type": "compliance_checker",
        "description": "Validate against compliance standards",
        "skills": ["compliance", "standards"],
        "priority": "medium"
      }
    ]
  },
  "pattern_metadata": {
    "support": 23,
    "confidence": 0.957,
    "lift": 2.8,
    "avg_execution_time": 245,
    "avg_cost": 0.12
  }
}
```

---

## Using Playbooks

### Discovery

**Step 1**: Browse available playbooks

```http
GET /api/playbooks?category=security&min_confidence=0.80

Response:
{
  "items": [
    {
      "id": 42,
      "name": "auto_security_audit_003",
      "pattern": ["CodeArchitect", "SecurityExpert", ...],
      "confidence": 0.957,
      "support": 23,
      "avg_cost": 0.12
    },
    ...
  ],
  "total": 15
}
```

**Step 2**: Preview playbook details

```http
GET /api/playbooks/42

Response:
{
  "id": 42,
  "name": "auto_security_audit_003",
  "pattern": [...],
  "metrics": {...},
  "performance": {...},
  "historical_executions": [
    {
      "execution_id": 789,
      "quality_score": 0.94,
      "duration": 238,
      "cost": 0.11
    },
    ...
  ],
  "recommended_use_cases": [
    "Pre-production security review",
    "Compliance audit preparation",
    "Quarterly security assessment"
  ]
}
```

### 1-Click Workflow Creation

**Step 3**: Create workflow from playbook

```http
POST /api/playbooks/42/create-workflow
Content-Type: application/json

{
  "name": "Q4 Security Audit",
  "context": {
    "codegraph_project": "backend-service",
    "compliance_standards": ["SOC2", "GDPR"]
  },
  "customize": {
    "add_agent": {
      "agent_type": "documentation_expert",
      "description": "Generate security documentation"
    }
  }
}

Response: 201 Created
{
  "workflow_id": 567,
  "name": "Q4 Security Audit",
  "status": "active",
  "based_on_playbook": 42,
  "estimated_execution_time": 245,
  "estimated_cost": 0.12,
  "agents_assigned": 5
}
```

**Step 4**: Execute workflow

```http
POST /api/workflows/567/execute

Response:
{
  "execution_id": 890,
  "status": "running",
  "estimated_duration": 245,
  "websocket_url": "wss://api.automatos.app/ws/executions/890"
}
```

---

## Real-World Examples

### Example 1: Code Review Pattern

**Discovered Pattern**:
```
[CodeArchitect, SecurityExpert, PerformanceOptimizer, DocumentationExpert]
```

**Statistics**:
- Support: 34 occurrences
- Confidence: 91.2%
- Avg execution time: 5m 30s
- Avg cost: $0.08

**Use Cases**:
- Pull request reviews
- Pre-merge quality checks
- Code audit workflows

### Example 2: Deployment Pipeline

**Discovered Pattern**:
```
[InfrastructureValidator, SecurityChecker, DeploymentExecutor, MonitoringSetup]
```

**Statistics**:
- Support: 18 occurrences
- Confidence: 94.4%
- Avg execution time: 8m 15s
- Avg cost: $0.15

**Use Cases**:
- Production deployments
- Staging environment setup
- Infrastructure updates

### Example 3: Data Analysis Workflow

**Discovered Pattern**:
```
[DataValidator, StatisticsExpert, DataAnalyst, InsightsGenerator]
```

**Statistics**:
- Support: 27 occurrences
- Confidence: 88.9%
- Avg execution time: 4m 20s
- Avg cost: $0.06

**Use Cases**:
- Business intelligence reports
- Customer behavior analysis
- Performance metrics review

---

## API Reference

### List Playbooks

```http
GET /api/playbooks

Query Parameters:
- tenant_id (optional): Filter by organization
- category (optional): Filter by workflow category
- min_confidence (optional): Minimum confidence threshold
- min_support (optional): Minimum occurrence count
- limit (default: 50): Results per page
- offset (default: 0): Pagination offset

Response: 200 OK
{
  "items": [
    {
      "id": 42,
      "name": "auto_security_audit_003",
      "pattern": ["CodeArchitect", "SecurityExpert", ...],
      "support": 23,
      "confidence": 0.957,
      "lift": 2.8,
      "category": "security",
      "avg_execution_time": 245,
      "avg_cost": 0.12,
      "usage_count": 12,
      "created_at": "2025-10-15T10:30:00Z"
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

### Get Playbook Details

```http
GET /api/playbooks/{playbook_id}

Response: 200 OK
{
  "id": 42,
  "name": "auto_security_audit_003",
  "pattern": [...],
  "metrics": {
    "support": 23,
    "confidence": 0.957,
    "lift": 2.8
  },
  "performance": {
    "avg_execution_time": 245,
    "avg_token_usage": 8234,
    "avg_cost": 0.12,
    "historical_success_rate": 0.957
  },
  "historical_executions": [
    {
      "execution_id": 789,
      "workflow_id": 123,
      "quality_score": 0.94,
      "duration": 238,
      "tokens": 8123,
      "cost": 0.11,
      "executed_at": "2025-10-20T14:30:00Z"
    }
  ],
  "recommended_use_cases": [...],
  "workflow_template": {...}
}
```

### Create Workflow from Playbook

```http
POST /api/playbooks/{playbook_id}/create-workflow
Content-Type: application/json

{
  "name": "My Custom Workflow",
  "description": "Based on discovered pattern",
  "context": {
    "key": "value"
  },
  "customize": {
    "add_agent": {...},
    "remove_agent_index": 2,
    "modify_agent": {...}
  }
}

Response: 201 Created
{
  "workflow_id": 567,
  "name": "My Custom Workflow",
  "status": "active",
  "based_on_playbook": 42,
  "agents_assigned": 5,
  "estimated_cost": 0.12
}
```

### Mine New Patterns

```http
POST /api/playbooks/mine
Content-Type: application/json

{
  "tenant_id": "org-acme",
  "min_support": 5,
  "min_confidence": 0.80,
  "execution_window_days": 30,
  "top_k": 50,
  "name_prefix": "auto"
}

Response: 202 Accepted
{
  "job_id": 789,
  "status": "pending",
  "estimated_duration": 15,
  "message": "Pattern mining job queued"
}
```

### Get Mining Job Status

```http
GET /api/playbooks/mining-jobs/{job_id}

Response: 200 OK
{
  "id": 789,
  "status": "completed",
  "patterns_discovered": 12,
  "execution_window_days": 30,
  "min_support": 5,
  "min_confidence": 0.80,
  "started_at": "2025-11-01T02:00:00Z",
  "completed_at": "2025-11-01T02:00:15Z",
  "created_playbooks": [42, 43, 44, ...]
}
```

---

## UI Guide

### Playbooks Page

**Location**: Dashboard > Playbooks

```
┌─────────────────────────────────────────────────────────────────┐
│ DISCOVERED PLAYBOOKS                        [Mine New Patterns]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Filter: [All Categories ▼] [Min Confidence: 80% ▼]              │
│ Sort: [Confidence ▼]                                             │
│                                                                  │
│ Showing 12 playbooks                                             │
│                                                                  │
│ ┌────────────────────────────────────────────────────┐         │
│ │ 🔒 auto_security_audit_003                          │         │
│ │ Security • Confidence: 95.7% • Support: 23          │         │
│ │                                                    │         │
│ │ Pattern:                                           │         │
│ │ CodeArchitect → SecurityExpert →                   │         │
│ │ VulnerabilityAnalyzer → ComplianceChecker          │         │
│ │                                                    │         │
│ │ Performance:                                       │         │
│ │ ⏱️ 4m 5s avg  |  💰 $0.12 avg  |  🎯 12 uses        │         │
│ │                                                    │         │
│ │ [View Details] [Create Workflow]                   │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ 📝 auto_code_review_015                             │         │
│ │ Code Review • Confidence: 91.2% • Support: 34       │         │
│ │                                                    │         │
│ │ Pattern:                                           │         │
│ │ CodeArchitect → SecurityExpert →                   │         │
│ │ PerformanceOptimizer → DocumentationExpert         │         │
│ │                                                    │         │
│ │ Performance:                                       │         │
│ │ ⏱️ 5m 30s avg  |  💰 $0.08 avg  |  🎯 45 uses        │         │
│ │                                                    │         │
│ │ [View Details] [Create Workflow]                   │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Playbook Details Modal

```
┌─────────────────────────────────────────────────────────────────┐
│ PLAYBOOK DETAILS: auto_security_audit_003              [✕ Close] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Pattern Flow                                                     │
│ ┌────────────────────────────────────────────────────┐         │
│ │ 1. CodeArchitect                                   │         │
│ │    Analyze code structure and architecture         │         │
│ │    ↓                                               │         │
│ │ 2. SecurityExpert                                  │         │
│ │    Scan for security vulnerabilities               │         │
│ │    ↓                                               │         │
│ │ 3. VulnerabilityAnalyzer                           │         │
│ │    Deep dive into identified issues                │         │
│ │    ↓                                               │         │
│ │ 4. ComplianceChecker                               │         │
│ │    Validate against standards                      │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
│ Metrics                                                          │
│ Support: 23 occurrences | Confidence: 95.7% | Lift: 2.8x        │
│                                                                  │
│ Performance (Average)                                            │
│ Execution Time: 4m 5s | Tokens: 8,234 | Cost: $0.12             │
│                                                                  │
│ Historical Success (Last 10 Uses)                                │
│ [Bar chart showing quality scores: 0.94, 0.96, 0.93, ...]       │
│                                                                  │
│ Recommended For                                                  │
│ • Pre-production security reviews                                │
│ • Compliance audit preparation                                   │
│ • Quarterly security assessments                                 │
│                                                                  │
│ [Create Workflow from This Pattern]                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced Features

### Scheduled Pattern Mining

Run pattern mining on a schedule:

```python
# Nightly pattern mining job (2 AM)
schedule.every().day.at("02:00").do(mine_patterns_job)

async def mine_patterns_job():
    """
    Nightly pattern mining across all tenants
    """
    tenants = db.query(distinct(WorkflowExecution.tenant_id)).all()
    
    for tenant_id in tenants:
        job = PatternMiningJob(
            tenant_id=tenant_id,
            min_support=5,
            min_confidence=0.80,
            execution_window_days=30,
            top_k=50
        )
        
        await pattern_miner.mine_and_create_playbooks(job)
```

### Pattern Confidence Threshold

Adjust confidence thresholds based on category:

```python
CONFIDENCE_THRESHOLDS = {
    'security': 0.90,        # High confidence for security
    'deployment': 0.85,      # High for deployments
    'code_review': 0.80,     # Standard
    'data_analysis': 0.75,   # Lower for exploratory
    'documentation': 0.70    # Lower for creative tasks
}
```

### Multi-Tenant Isolation

Patterns are strictly isolated by organization:

```sql
-- Each tenant sees only their patterns
SELECT * FROM playbooks 
WHERE tenant_id = 'org-acme'
AND is_active = TRUE
ORDER BY confidence DESC;

-- Cross-tenant learning is NEVER allowed
-- Ensures data privacy and relevance
```

---

## Best Practices

### 1. Sufficient Historical Data

**Minimum Requirements**:
- At least 50 completed workflow executions
- Spanning 30+ days
- Multiple workflow categories

**Why**: Pattern mining needs volume to find statistically significant patterns.

### 2. Regular Mining Schedule

**Recommended**:
- Nightly pattern mining (during low-traffic hours)
- Weekly deep analysis
- Monthly pattern review and cleanup

### 3. Pattern Validation

**Before using a playbook**:
- Review historical executions
- Check confidence score (≥80%)
- Verify support count (≥5)
- Review average cost vs budget

### 4. Customization

**Always customize playbooks for**:
- Project-specific context
- Custom agent configurations
- Specific compliance requirements
- Budget constraints

---

## Troubleshooting

### No Patterns Discovered

**Problem**: Mining job returns 0 patterns

**Common Causes**:
1. Insufficient historical data (<50 workflows)
2. Min support too high (try lowering from 5 to 3)
3. Min confidence too high (try 0.70 instead of 0.80)
4. Workflows too diverse (no recurring patterns)

**Solutions**:
```python
# Lower thresholds
mine_patterns(
    min_support=3,           # Instead of 5
    min_confidence=0.70,     # Instead of 0.80
    execution_window_days=60 # Instead of 30
)
```

### Low Confidence Patterns

**Problem**: Patterns discovered but confidence <80%

**Analysis**:
```sql
-- Check pattern failures
SELECT 
    p.name,
    p.confidence,
    COUNT(CASE WHEN we.quality_score < 0.7 THEN 1 END) as failures,
    COUNT(*) as total
FROM playbooks p
JOIN workflow_executions we ON we.pattern_id = p.id
GROUP BY p.id
HAVING p.confidence < 0.80;
```

**Solutions**:
- Investigate failure causes
- Add error handling to pattern
- Consider removing unreliable agents
- Increase agent timeout limits

---

## Future Enhancements

### Planned Features

1. **Adaptive Learning** (Q2 2026)
   - Patterns improve automatically from feedback
   - Self-adjusting confidence thresholds

2. **Cross-Org Learning** (Q3 2026)
   - Opt-in pattern sharing
   - Privacy-preserving federated learning

3. **Visual Pattern Editor** (Q2 2026)
   - Drag-and-drop pattern modification
   - Visual flow builder

4. **Pattern Recommendations** (Q1 2026)
   - AI suggests playbooks for new workflows
   - Contextual recommendations based on goal

---

## FAQ

### Q: How often should I mine for patterns?

**A**: Daily is recommended. Pattern mining is fast (~15 seconds) and ensures playbooks stay current with recent successes.

### Q: Can I manually create a playbook?

**A**: Not directly. Playbooks are discovered patterns. However, you can create workflow templates manually which serve a similar purpose.

### Q: What if a playbook becomes outdated?

**A**: The system tracks playbook usage and success rates. Low-performing playbooks are automatically flagged for review and can be deactivated.

### Q: Are playbooks shared across organizations?

**A**: No. Playbooks are strictly isolated by tenant_id for privacy and relevance.

### Q: Can I export/import playbooks?

**A**: Yes! Use the export/import API:
```bash
# Export
GET /api/playbooks/42/export

# Import
POST /api/playbooks/import
```

---

## Next Steps

1. **🔄 [Workflow System Guide](WORKFLOW_SYSTEM_GUIDE.md)** - Understanding workflows
2. **🤖 [Agent System Guide](AGENT_SYSTEM_GUIDE.md)** - Agent combinations
3. **📊 [Benchmarking Guide](BENCHMARKING_GUIDE.md)** - Performance tracking
4. **🧠 [Memory & Knowledge Guide](MEMORY_KNOWLEDGE_GUIDE.md)** - Pattern storage

---

**Built with ❤️ based on PRD-12 (Playbooks - Automated Pattern Discovery & Learning)**

*Last updated: January 2025*

