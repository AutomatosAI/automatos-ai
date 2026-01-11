"""
Script to identify missing logos
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database.database import SessionLocal

def main():
    db = SessionLocal()
    
    try:
        # Get all unique logo paths from tools
        result = db.execute(text("SELECT DISTINCT logo FROM mcp_tools WHERE logo IS NOT NULL"))
        tool_logos = {row[0] for row in result.fetchall()}
        
        # Get all unique logo paths from credential types
        result = db.execute(text("SELECT DISTINCT logo FROM credential_types WHERE logo IS NOT NULL"))
        cred_logos = {row[0] for row in result.fetchall()}
        
        # Combine
        all_logos = tool_logos | cred_logos
        
        print(f"📊 Total unique logo paths: {len(all_logos)}")
        
        # Extract brand names
        brands = set()
        for logo_path in all_logos:
            # Extract brand name from /logos/BrandName.png
            if logo_path and logo_path.startswith('/logos/'):
                brand = logo_path.replace('/logos/', '').replace('.png', '')
                brands.add(brand)
        
        print(f"📦 Total unique brands: {len(brands)}")
        
        # Check which logos exist
        logos_dir = "frontend/public/logos"
        existing_logos = set()
        if os.path.exists(logos_dir):
            existing_logos = {f.replace('.png', '') for f in os.listdir(logos_dir) if f.endswith('.png')}
        
        print(f"✅ Existing logos: {len(existing_logos)}")
        
        # Find missing
        missing = brands - existing_logos
        print(f"❌ Missing logos: {len(missing)}")
        
        if missing:
            print("\n📋 Missing brand logos:")
            for brand in sorted(missing):
                print(f"  - {brand}")
        
        # Save to file
        with open('missing_logos.txt', 'w') as f:
            for brand in sorted(missing):
                f.write(f"{brand}\n")
        
        print(f"\n💾 Saved missing logos list to missing_logos.txt")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
