#!/usr/bin/env python3
"""
Test script for the add_myths_bulk function.
"""

import numpy as np
from mythologizer_postgres.connectors import add_myths_bulk
from mythologizer_postgres.db import clear_all_rows
from mythologizer_postgres.connectors import insert_mythemes_bulk


def test_add_myths_bulk():
    """Test the add_myths_bulk function."""
    
    # Clear any existing data
    clear_all_rows()
    
    # Create some test mythemes first
    print("1. Creating test mythemes...")
    embedding_dim = 4
    n_mythemes = 5
    
    # Create random mytheme embeddings
    mytheme_embeddings = [np.random.rand(embedding_dim).tolist() for _ in range(n_mythemes)]
    mytheme_names = [f"mytheme_{i}" for i in range(n_mythemes)]
    
    # Insert mythemes
    mytheme_ids = insert_mythemes_bulk(mytheme_names, mytheme_embeddings)
    print(f"   Created {len(mytheme_ids)} mythemes with IDs: {mytheme_ids}")
    
    # Create test myths data
    print("\n2. Creating test myths data...")
    n_myths = 3
    
    myths_data = []
    for i in range(n_myths):
        # Each myth uses 2-3 mythemes
        n_mythemes_per_myth = np.random.randint(2, 4)
        embedding_ids = np.random.choice(mytheme_ids, n_mythemes_per_myth, replace=False).tolist()
        
        # Create random offsets and weights
        offsets = [np.random.rand(embedding_dim).astype(np.float32) for _ in range(n_mythemes_per_myth)]
        weights = np.random.rand(n_mythemes_per_myth).astype(np.float32)
        
        myths_data.append((embedding_ids, offsets, weights))
        print(f"   Myth {i}: {len(embedding_ids)} mythemes, embedding_ids={embedding_ids}")
    
    # Add myths in bulk
    print("\n3. Adding myths in bulk...")
    try:
        myth_ids = add_myths_bulk(myths_data)
        print(f"   Successfully added {len(myth_ids)} myths with IDs: {myth_ids}")
        
        # Verify the myths were added correctly
        print("\n4. Verifying myths in database...")
        from mythologizer_postgres.connectors import get_myths_bulk
        from mythologizer_postgres.db import psycopg_connection
        
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM myths")
                count = cur.fetchone()[0]
                print(f"   Total myths in database: {count}")
                
                # Get the myths we just created
                ids, main_embeddings, embedding_ids_list, offsets_list, weights_list, created_ats, updated_ats = get_myths_bulk(myth_ids)
                print(f"   Retrieved {len(ids)} myths from database")
                
                for i, myth_id in enumerate(ids):
                    print(f"   Myth {myth_id}: {len(embedding_ids_list[i])} embedding_ids, {len(offsets_list[i])} offsets, {len(weights_list[i])} weights")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_add_myths_bulk()


