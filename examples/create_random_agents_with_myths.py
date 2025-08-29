#!/usr/bin/env python3
"""
Script to create 10 random agents with attributes, myths, and cultures.

This script demonstrates:
1. Creating 10 random agents with attributes
2. Setting memory size to 3 for each agent
3. Assigning 3 random myths to each agent
4. Assigning 2 random cultures to each agent
"""

import os
import numpy as np
import random
from typing import List, Tuple, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mythologizer_postgres.connectors import (
    get_myths_bulk,
    get_all_cultures,
    insert_agent_myth,
    get_agent_attribute_defs,
    insert_agent_attribute_defs,
)
from mythologizer_postgres.db import psycopg_connection


def get_embedding_dim():
    """Get the embedding dimension - hardcoded to 384 for now."""
    return 384


def create_random_agent_attributes() -> List[Union[int, float]]:
    """Create random agent attributes that respect the defined constraints."""
    
    # Get the attribute definitions
    attr_defs = get_agent_attribute_defs()
    
    if not attr_defs:
        print("Warning: No agent attribute definitions found. Creating basic ones...")
        # Create basic attribute definitions if none exist
        basic_defs = [
            {"name": "Age", "type": "int", "description": "Agent age", "min_val": 0, "max_val": None, "col_idx": 0},
            {"name": "Confidence", "type": "float", "description": "Agent confidence level", "min_val": 0, "max_val": 1, "col_idx": 1},
            {"name": "Emotionality", "type": "float", "description": "Agent emotionality level", "min_val": 0, "max_val": 1, "col_idx": 2}
        ]
        insert_agent_attribute_defs(basic_defs)
        attr_defs = get_agent_attribute_defs()
    
    # Generate attributes based on the definitions
    attributes = []
    for attr_def in attr_defs:
        attr_id, name, description, atype, min_val, max_val, col_idx = attr_def
        
        # Convert Decimal to float/int if needed
        if min_val is not None:
            min_val = float(min_val) if isinstance(min_val, (int, float)) else float(str(min_val))
        if max_val is not None:
            max_val = float(max_val) if isinstance(max_val, (int, float)) else float(str(max_val))
        
        if atype == "int":
            # Generate integer value
            if min_val is not None and max_val is not None:
                value = random.randint(int(min_val), int(max_val))
            elif min_val is not None:
                # No max value, use a reasonable upper bound
                value = random.randint(int(min_val), int(min_val) + 100)
            else:
                # No min/max, use reasonable range
                value = random.randint(0, 100)
        elif atype == "float":
            # Generate float value
            if min_val is not None and max_val is not None:
                value = random.uniform(min_val, max_val)
            elif min_val is not None:
                # No max value, use min + 1 as max
                value = random.uniform(min_val, min_val + 1.0)
            else:
                # No min/max, use 0 to 1
                value = random.uniform(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported attribute type: {atype}")
        
        attributes.append(value)
    
    return attributes


def create_random_agent(name: str, memory_size: int = 3) -> int:
    """Create a random agent and return its ID."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agents (name, memory_size)
                VALUES (%s, %s)
                RETURNING id
            """, (name, memory_size))
            agent_id = cur.fetchone()[0]
            conn.commit()
            return agent_id


def assign_agent_attributes(agent_id: int, attributes: List[Union[int, float]]) -> None:
    """Assign attributes to an agent."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Get the attribute definitions
            attr_defs = get_agent_attribute_defs()
            
            if not attr_defs:
                print("Warning: No agent attribute definitions found. Creating basic ones...")
                # Create basic attribute definitions if none exist
                basic_defs = [
                    {"name": "Age", "type": "int", "description": "Agent age", "min_val": 0, "max_val": None, "col_idx": 0},
                    {"name": "Confidence", "type": "float", "description": "Agent confidence level", "min_val": 0, "max_val": 1, "col_idx": 1},
                    {"name": "Emotionality", "type": "float", "description": "Agent emotionality level", "min_val": 0, "max_val": 1, "col_idx": 2}
                ]
                insert_agent_attribute_defs(basic_defs)
                attr_defs = get_agent_attribute_defs()
            
            # Create attribute values array in the correct order
            max_col_idx = max(attr_def[6] for attr_def in attr_defs) if attr_defs else 0
            attribute_values = [0.0] * (max_col_idx + 1)  # Initialize with zeros
            
            # Assign attributes to their correct positions
            for i, attr_def in enumerate(attr_defs):
                if i < len(attributes):
                    col_idx = attr_def[6]  # col_idx
                    attr_type = attr_def[3]  # atype
                    
                    # Convert to float for storage (PostgreSQL arrays need consistent types)
                    attribute_values[col_idx] = float(attributes[i])
            
            # Insert or update agent attributes
            cur.execute("""
                INSERT INTO agent_attributes (agent_id, attribute_values)
                VALUES (%s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                attribute_values = EXCLUDED.attribute_values
            """, (agent_id, attribute_values))
            
            conn.commit()


def assign_agent_cultures(agent_id: int, culture_ids: List[int]) -> None:
    """Assign cultures to an agent."""
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            for culture_id in culture_ids:
                cur.execute("""
                    INSERT INTO agent_cultures (agent_id, culture_id)
                    VALUES (%s, %s)
                """, (agent_id, culture_id))
            conn.commit()


def assign_agent_myths(agent_id: int, myth_ids: List[int]) -> None:
    """Assign myths to an agent's memory."""
    from mythologizer_postgres.connectors import insert_agent_myth_safe
    
    for myth_id in myth_ids:
        # Generate random retention value between 0.1 and 1.0
        retention = random.uniform(0.1, 1.0)
        # Use the safe function that handles position assignment and reordering
        success = insert_agent_myth_safe(myth_id, agent_id, retention)
        if not success:
            print(f"    WARNING: Failed to insert myth {myth_id} into agent {agent_id}")


def main():
    """Main function to create random agents with myths and cultures."""
    print("=== Creating Random Agents with Myths and Cultures ===\n")
    
    # Get existing myths and cultures
    print("\n1. Querying existing myths and cultures...")
    
    # Get all myths
    myth_ids, _, _, _, _, _, _ = get_myths_bulk()
    if not myth_ids:
        print("No myths found in the database. Please create some myths first.")
        return
    
    print(f"Found {len(myth_ids)} myths in the database")
    
    # Get all cultures
    cultures = get_all_cultures()
    if not cultures:
        print("No cultures found in the database. Please create some cultures first.")
        return
    
    culture_ids = [culture[0] for culture in cultures]  # Extract culture IDs
    print(f"Found {len(culture_ids)} cultures in the database")
    
    # Get attribute definitions
    attr_defs = get_agent_attribute_defs()
    print(f"Found {len(attr_defs)} attribute definitions")
    for attr_def in attr_defs:
        print(f"  - {attr_def[1]} ({attr_def[3]}): min={attr_def[4]}, max={attr_def[5]}")
    
    # Create 10 random agents
    print(f"\n2. Creating 10 random agents...")
    
    agent_names = [
        "Agent Alpha", "Agent Beta", "Agent Gamma", "Agent Delta", "Agent Epsilon",
        "Agent Zeta", "Agent Eta", "Agent Theta", "Agent Iota", "Agent Kappa"
    ]
    
    created_agents = []
    used_myth_ids = set()  # Track which myths have been used
    
    for i, name in enumerate(agent_names):
        print(f"  Creating agent {i+1}/10: {name}")
        
        # Create agent
        agent_id = create_random_agent(name, memory_size=3)
        
        # Create random attributes
        attributes = create_random_agent_attributes()
        assign_agent_attributes(agent_id, attributes)
        
        # Assign 2 random cultures
        selected_culture_ids = random.sample(culture_ids, 2)
        assign_agent_cultures(agent_id, selected_culture_ids)
        
        # Assign 3 random myths (avoiding already used myths)
        available_myth_ids = [mid for mid in myth_ids if mid not in used_myth_ids]
        if len(available_myth_ids) < 3:
            print(f"    WARNING: Only {len(available_myth_ids)} myths available, need 3")
            break
        
        selected_myth_ids = random.sample(available_myth_ids, 3)
        used_myth_ids.update(selected_myth_ids)  # Mark these myths as used
        assign_agent_myths(agent_id, selected_myth_ids)
        
        created_agents.append({
            'id': agent_id,
            'name': name,
            'cultures': selected_culture_ids,
            'myths': selected_myth_ids,
            'attributes': attributes
        })
        
        print(f"    - Agent ID: {agent_id}")
        print(f"    - Cultures: {selected_culture_ids}")
        print(f"    - Myths: {selected_myth_ids}")
        print(f"    - Attributes: {attributes}")
    
    print("\n✅ Successfully created 10 agents with myths and cultures!")
    
    # Display summary
    print("\n3. Agent creation summary:")
    print(f"  - Total agents created: 10")
    print(f"  - Memory size per agent: 3")
    print(f"  - Myths per agent: 3")
    print(f"  - Cultures per agent: 2")
    print(f"  - Attributes per agent: {len(attr_defs)}")
    
    # Show details of first few agents
    print("\n4. Sample of created agents:")
    for i, agent in enumerate(created_agents[:3]):
        print(f"  Agent {i+1}: {agent['name']} (ID: {agent['id']})")
        print(f"    - Cultures: {agent['cultures']}")
        print(f"    - Myths: {agent['myths']}")
        print(f"    - Attributes: {agent['attributes']}")
        print()
    
    print("=== Agent Creation Complete ===")


if __name__ == "__main__":
    main()
