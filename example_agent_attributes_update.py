#!/usr/bin/env python3
"""
Example usage of the agent attributes matrix store functions with update capability.
"""

import numpy as np
from mythologizer_postgres.connectors import (
    insert_agent_attribute_defs,
    get_agent_attribute_matrix,
    update_agent_attribute_matrix,
)
from mythologizer_postgres.db import get_engine, clear_all_rows
from sqlalchemy import text


def main():
    """Demonstrate the agent attributes matrix functionality with updates."""
    
    # Clear any existing data
    clear_all_rows()
    
    # 1. Insert attribute definitions
    print("1. Inserting attribute definitions...")
    defs = [
        {"name": "strength", "type": "float", "description": "Physical power"},
        {"name": "wisdom", "type": "int", "description": "Cognitive ability"},
        {"name": "luck", "type": "float", "description": "Luck factor"},
    ]
    insert_agent_attribute_defs(defs)
    print(f"   Inserted {len(defs)} attribute definitions")
    
    # 2. Insert some agents
    print("\n2. Inserting agents...")
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 1', 10)"))
        conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 2', 15)"))
        conn.execute(text("INSERT INTO agents (name, memory_size) VALUES ('Agent 3', 20)"))
        conn.commit()
        
        # Get agent IDs
        result = conn.execute(text("SELECT id FROM agents ORDER BY id"))
        agent_ids = [row[0] for row in result.fetchall()]
        print(f"   Inserted {len(agent_ids)} agents with IDs: {agent_ids}")
        
        # 3. Insert initial agent attributes
        print("\n3. Inserting initial agent attributes...")
        conn.execute(text("""
            INSERT INTO agent_attributes (agent_id, attribute_values) 
            VALUES (:agent_id, :attribute_values)
        """), {"agent_id": agent_ids[0], "attribute_values": [10.5, 20.0, 0.8]})
        
        conn.execute(text("""
            INSERT INTO agent_attributes (agent_id, attribute_values) 
            VALUES (:agent_id, :attribute_values)
        """), {"agent_id": agent_ids[1], "attribute_values": [15.2, 25.0, 0.3]})
        
        conn.execute(text("""
            INSERT INTO agent_attributes (agent_id, attribute_values) 
            VALUES (:agent_id, :attribute_values)
        """), {"agent_id": agent_ids[2], "attribute_values": [8.0, 18.0, 0.9]})
        
        conn.commit()
        print("   Inserted initial attributes for all agents")
    
    # 4. Get the initial matrix
    print("\n4. Getting initial agent attribute matrix...")
    matrix, agent_indices, attribute_name_to_col = get_agent_attribute_matrix()
    
    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Agent indices: {agent_indices}")
    print(f"   Attribute name to column mapping: {attribute_name_to_col}")
    print(f"   Initial matrix:\n{matrix}")
    
    # 5. Modify the matrix (e.g., increase all strength values by 5)
    print("\n5. Modifying the matrix (increasing strength by 5)...")
    strength_col = attribute_name_to_col["strength"]
    modified_matrix = matrix.copy()
    modified_matrix[:, strength_col] += 5.0
    
    print(f"   Modified matrix:\n{modified_matrix}")
    
    # 6. Update the database with the modified matrix
    print("\n6. Updating database with modified matrix...")
    update_agent_attribute_matrix(modified_matrix, agent_indices)
    print("   Database updated successfully")
    
    # 7. Verify the update by reading back
    print("\n7. Verifying the update...")
    updated_matrix, updated_agent_indices, updated_attribute_name_to_col = get_agent_attribute_matrix()
    
    print(f"   Updated matrix:\n{updated_matrix}")
    print(f"   Matrices match: {np.array_equal(modified_matrix, updated_matrix)}")
    
    # 8. Demonstrate more complex modifications
    print("\n8. Demonstrating more complex modifications...")
    
    # Apply a random boost to all attributes
    np.random.seed(42)  # For reproducible results
    boost_matrix = np.random.uniform(0.9, 1.1, matrix.shape)
    complex_modified_matrix = matrix * boost_matrix
    
    print(f"   Boost matrix:\n{boost_matrix}")
    print(f"   Complex modified matrix:\n{complex_modified_matrix}")
    
    # Update with the complex modifications
    update_agent_attribute_matrix(complex_modified_matrix, agent_indices)
    
    # Verify the complex update
    final_matrix, final_agent_indices, final_attribute_name_to_col = get_agent_attribute_matrix()
    print(f"   Final matrix:\n{final_matrix}")
    print(f"   Complex modification successful: {np.allclose(complex_modified_matrix, final_matrix)}")
    
    # 9. Demonstrate using the attribute name to column mapping for analysis
    print("\n9. Demonstrating analysis using attribute name to column mapping...")
    
    for attr_name, col_idx in attribute_name_to_col.items():
        values = final_matrix[:, col_idx]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"   {attr_name}: mean={mean_val:.2f}, std={std_val:.2f}")


if __name__ == "__main__":
    main()
