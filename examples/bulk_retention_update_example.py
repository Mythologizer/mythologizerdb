#!/usr/bin/env python3
"""
Example demonstrating the update_retentions_and_reorder function.
"""

from mythologizer_postgres.connectors import (
    update_retentions_and_reorder,
    get_myth_ids_and_retention_from_agents_memory,
)
from mythologizer_postgres.db import psycopg_connection
import numpy as np
import os


def main():
    """Example usage of bulk retention update."""
    print("=== Bulk Retention Update Example ===\n")
    
    # Create test agent and myths (simplified)
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Create agent
            cur.execute("INSERT INTO agents (name, memory_size) VALUES ('Test Agent', 10) RETURNING id")
            agent_id = cur.fetchone()[0]
            
            # Create myths
            myth_ids = []
            for i in range(3):
                embedding = np.random.rand(4).tolist()
                cur.execute("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights)
                    VALUES (%s, ARRAY[1], ARRAY[]::vector[], ARRAY[]::double precision[])
                    RETURNING id
                """, (embedding,))
                myth_ids.append(cur.fetchone()[0])
            
            # Insert into agent memory
            for myth_id in myth_ids:
                cur.execute("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention)
                    VALUES (%s, %s, 0, %s)
                """, (myth_id, agent_id, 0.3))
            
            conn.commit()
    
    print(f"Created agent {agent_id} with myths {myth_ids}")
    
    # Show initial state
    initial_myth_ids, initial_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
    print(f"\nInitial order: {list(zip(initial_myth_ids, initial_retentions))}")
    
    # Update retentions and reorder
    myth_retention_pairs = [
        (myth_ids[0], 0.9),  # highest retention
        (myth_ids[1], 0.5),  # medium retention
        (myth_ids[2], 0.1),  # lowest retention
    ]
    
    success = update_retentions_and_reorder(agent_id, myth_retention_pairs)
    print(f"\nUpdate successful: {success}")
    
    # Show final state
    final_myth_ids, final_retentions = get_myth_ids_and_retention_from_agents_memory(agent_id)
    print(f"Final order: {list(zip(final_myth_ids, final_retentions))}")
    
    # Verify highest retention is at position 0
    if final_retentions and final_retentions[0] == max(final_retentions):
        print("✅ Highest retention correctly at position 0!")


if __name__ == "__main__":
    main()
