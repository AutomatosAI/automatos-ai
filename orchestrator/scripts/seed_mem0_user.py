
import os
import sys
from sqlalchemy import create_engine, text

# Load env from orchestrator/.env manually
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key] = val.strip().strip('"').strip("'")
            
# Try to get URL from env
db_url = os.getenv("MEM0_DATABASE_URL", os.getenv("DATABASE_URL"))

if not db_url:
    print("Error: DATABASE_URL not set. Please set DATABASE_URL (or MEM0_DATABASE_URL) to the Mem0 Postgres connection string.")
    sys.exit(1)

# Mask password for printing
masked_url = db_url
if "@" in db_url:
    prefix = db_url.split("@")[0]
    suffix = db_url.split("@")[1]
    # rough masking
    masked_url = f"{prefix.split(':')[0]}:***@{suffix}"
    
print(f"Connecting to DB: {masked_url}")


try:
    engine = create_engine(db_url)
    with engine.connect() as conn:
        # Check all tables
        print("--- Tables ---")
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result]
        print(tables)
        
        if 'users' in tables:
            print("\n--- Columns in 'users' ---")
            result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users'"))
            for row in result:
                print(f"{row[0]}: {row[1]}")
                
        if 'memories' in tables:
             print("\n✅ 'memories' table found!")
        else:
             print("\n❌ 'memories' table NOT found. Wrong DB?")

except Exception as e:
    print(f"❌ Database error: {e}")

