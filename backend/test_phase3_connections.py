import os
from dotenv import load_dotenv
load_dotenv()

# ── Test Supabase ─────────────────────────────────────────────────────────────
print("Testing Supabase...")
try:
    from supabase import create_client
    client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    # Simple ping — list tables
    result = client.table("nonexistent").select("*").limit(1).execute()
    print("✅ Supabase connected")
except Exception as e:
    if "relation" in str(e).lower() or "does not exist" in str(e).lower():
        print("✅ Supabase connected (table doesn't exist yet — expected)")
    else:
        print(f"❌ Supabase error: {e}")

# ── Test Qdrant ───────────────────────────────────────────────────────────────
print("\nTesting Qdrant...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    collections = client.get_collections()
    print(f"✅ Qdrant connected — collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"❌ Qdrant error: {e}")
