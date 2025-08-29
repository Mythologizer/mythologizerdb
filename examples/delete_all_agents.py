#!/usr/bin/env python3
"""
Script to delete all agents and their associated data.
This will delete:
- agent_myths (myth references)
- agent_cultures (culture references) 
- agent_attributes (agent attributes)
- agents (the agents themselves)

Note: This preserves agent_attribute_defs and other reference data.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mythologizer_postgres.db import psycopg_connection

def delete_all_agents():
    """Delete all agents and their associated data."""
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            print("Deleting all agent data...")
            
            # Delete in order to respect foreign key constraints
            # 1. Delete agent_myths (myth references)
            cur.execute("DELETE FROM agent_myths")
            agent_myths_deleted = cur.rowcount
            print(f"Deleted {agent_myths_deleted} agent-myth references")
            
            # 2. Delete agent_cultures (culture references)
            cur.execute("DELETE FROM agent_cultures")
            agent_cultures_deleted = cur.rowcount
            print(f"Deleted {agent_cultures_deleted} agent-culture references")
            
            # 3. Delete agent_attributes (agent attributes)
            cur.execute("DELETE FROM agent_attributes")
            agent_attributes_deleted = cur.rowcount
            print(f"Deleted {agent_attributes_deleted} agent attributes")
            
            # 4. Delete agents
            cur.execute("DELETE FROM agents")
            agents_deleted = cur.rowcount
            print(f"Deleted {agents_deleted} agents")
            
            conn.commit()
            print("All agent data deleted successfully!")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check if we're using the test database
    db_name = os.getenv('POSTGRES_DB', 'mythologizer')
    if db_name == 'mythologizer_test':
        print("WARNING: You are about to delete all agents from the TEST database!")
        response = input("Are you sure? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    delete_all_agents()


