#!/usr/bin/env python3
"""
Phase 2 Analysis - Compare Duplicate APIs
==========================================
"""

import re
from pathlib import Path

def extract_endpoints(file_path):
    """Extract all FastAPI endpoints from a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    endpoints = []
    
    # Find all @router decorators
    patterns = [
        r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
        r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            method = match.group(1).upper()
            path = match.group(2)
            endpoints.append(f"{method} {path}")
    
    return sorted(set(endpoints))

print("📊 PHASE 2 ANALYSIS - Duplicate API Investigation")
print("=" * 70)
print()

# Analyze Analytics APIs (all 3 are imported!)
print("1️⃣  ANALYTICS APIs (All 3 imported - which one does what?)")
print("-" * 70)

analytics_files = [
    ('analytics.py', 'api/analytics.py'),
    ('analytics_api.py', 'api/analytics_api.py'),
    ('analytics_real.py', 'api/analytics_real.py')
]

all_endpoints = {}
for name, path in analytics_files:
    file_path = Path(path)
    if file_path.exists():
        endpoints = extract_endpoints(file_path)
        all_endpoints[name] = endpoints
        size = file_path.stat().st_size
        print(f"\n📁 {name} ({size:,} bytes) - {len(endpoints)} endpoints:")
        for ep in endpoints[:10]:  # Show first 10
            print(f"   {ep}")
        if len(endpoints) > 10:
            print(f"   ... and {len(endpoints) - 10} more")

# Check for overlapping endpoints
print(f"\n🔍 Checking for endpoint overlap...")
set1 = set(all_endpoints.get('analytics.py', []))
set2 = set(all_endpoints.get('analytics_api.py', []))
set3 = set(all_endpoints.get('analytics_real.py', []))

overlap_1_2 = set1 & set2
overlap_1_3 = set1 & set3
overlap_2_3 = set2 & set3
all_overlap = set1 & set2 & set3

if overlap_1_2:
    print(f"  ⚠️  analytics.py ∩ analytics_api.py: {len(overlap_1_2)} overlapping endpoints")
if overlap_1_3:
    print(f"  ⚠️  analytics.py ∩ analytics_real.py: {len(overlap_1_3)} overlapping endpoints")
if overlap_2_3:
    print(f"  ⚠️  analytics_api.py ∩ analytics_real.py: {len(overlap_2_3)} overlapping endpoints")
if all_overlap:
    print(f"  🔴 All 3 files share: {len(all_overlap)} endpoints")

if not (overlap_1_2 or overlap_1_3 or overlap_2_3):
    print(f"  ✅ No overlapping endpoints - each file serves different purposes")

print()

# Analyze Knowledge Graph files
print("\n2️⃣  KNOWLEDGE GRAPH Files (None imported - why not?)")
print("-" * 70)

knowledge_files = [
    ('knowledge_graph.py', 'api/knowledge_graph.py'),
    ('knowledge_multimodal.py', 'api/knowledge_multimodal.py'),
]

for name, path in knowledge_files:
    file_path = Path(path)
    if file_path.exists():
        endpoints = extract_endpoints(file_path)
        size = file_path.stat().st_size
        
        # Check if functionality exists elsewhere
        with open(file_path, 'r') as f:
            content = f.read()
        
        has_multimodal = 'multimodal' in content.lower()
        has_image = 'image' in content.lower()
        has_audio = 'audio' in content.lower()
        
        print(f"\n📁 {name} ({size:,} bytes) - {len(endpoints)} endpoints")
        print(f"   Features: ", end="")
        features = []
        if has_multimodal:
            features.append("multimodal")
        if has_image:
            features.append("images")
        if has_audio:
            features.append("audio")
        print(", ".join(features) if features else "text-only")
        
        if endpoints:
            print(f"   Endpoints: {', '.join(endpoints[:5])}")

# Check if knowledge.py (which IS imported) covers this
knowledge_path = Path('api/knowledge.py')
if knowledge_path.exists():
    endpoints = extract_endpoints(knowledge_path)
    size = knowledge_path.stat().st_size
    print(f"\n✅ api/knowledge.py (IMPORTED) ({size:,} bytes) - {len(endpoints)} endpoints")
    print(f"   Might supersede the unused knowledge_graph files")

print()
print("=" * 70)
print()
