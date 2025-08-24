#!/usr/bin/env python3
"""
Example usage of the memory store functions.
"""

import numpy as np
from mythologizer_postgres.connectors import get_myth_ids_and_retention_from_agents_memory
from mythologizer_postgres.db import get_engine, clear_all_rows
from sqlalchemy import text


def main():
    """Demonstrate the memory store functionality."""
    
    # Clear any existing data
    clear_all_rows()
    
    # 1. Insert an agent
    print("1. Inserting an agent...")
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Memory Agent', 10)"))
        conn.commit()
        
        # Get agent ID
        result = conn.execute(text("SELECT id FROM agents"))
        agent_id = result.fetchone()[0]
        print(f"   Inserted agent with ID: {agent_id}")
    
    # 2. Insert some myths
    print("\n2. Inserting myths...")
    embedding_dim = 4
    with engine.connect() as conn:
        for i in range(5):
            embedding = np.random.rand(embedding_dim).tolist()
            conn.execute(text("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights) 
                VALUES (:embedding, ARRAY[:id], ARRAY[]::vector[], ARRAY[]::double precision[])
            """), {"embedding": embedding, "id": i + 1})
        
        conn.commit()
        
        # Get myth IDs
        result = conn.execute(text("SELECT id FROM myths ORDER BY id"))
        myth_ids = [row[0] for row in result.fetchall()]
        print(f"   Inserted {len(myth_ids)} myths with IDs: {myth_ids}")
    
    # 3. Insert myths into agent's memory with different retentions
    print("\n3. Inserting myths into agent's memory...")
    retentions = [0.9, 0.7, 0.8, 0.6, 0.95]  # Different retention values
    
    with engine.connect() as conn:
        for i, (myth_id, retention) in enumerate(zip(myth_ids, retentions)):
            conn.execute(text("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention) 
                VALUES (:myth_id, :agent_id, :position, :retention)
            """), {"myth_id": myth_id, "agent_id": agent_id, "position": 1, "retention": retention})
        
        conn.commit()
        print("   Inserted all myths into agent's memory")
    
    # 4. Get the agent's memory
    print("\n4. Retrieving agent's memory...")
    retrieved_myth_ids, retrieved_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
    
    print(f"   Retrieved {len(retrieved_myth_ids)} myths from memory")
    print(f"   Myth IDs: {retrieved_myth_ids}")
    print(f"   Retentions: {retrieved_retentions}")
    
    # 5. Display memory in a more readable format
    print("\n5. Agent's memory contents:")
    print("   Position | Myth ID | Retention")
    print("   ---------|---------|----------")
    for i, (myth_id, retention) in enumerate(zip(retrieved_myth_ids, retrieved_retentions)):
        position = i + 1
        print(f"   {position:8d} | {myth_id:7d} | {retention:.2f}")
    
    # 6. Demonstrate analysis of the memory
    print("\n6. Memory analysis:")
    if retrieved_retentions:
        avg_retention = sum(retrieved_retentions) / len(retrieved_retentions)
        max_retention = max(retrieved_retentions)
        min_retention = min(retrieved_retentions)
        
        print(f"   Average retention: {avg_retention:.3f}")
        print(f"   Maximum retention: {max_retention:.3f}")
        print(f"   Minimum retention: {min_retention:.3f}")
        
        # Find myths with highest retention
        max_retention_idx = retrieved_retentions.index(max_retention)
        best_remembered_myth = retrieved_myth_ids[max_retention_idx]
        print(f"   Best remembered myth: ID {best_remembered_myth} (retention: {max_retention:.3f})")
    
    # 7. Test with non-existent agent
    print("\n7. Testing with non-existent agent...")
    empty_myth_ids, empty_retentions = get_myth_ids_and_retention_from_agents_memory(999)
    print(f"   Non-existent agent returned: {len(empty_myth_ids)} myths, {len(empty_retentions)} retentions")


if __name__ == "__main__":
    main()

