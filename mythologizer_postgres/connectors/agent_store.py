"""
Agent store implementation for managing agents.
"""

from typing import List, Dict, Any, Tuple
from mythologizer_postgres.db import psycopg_connection


def get_agents_bulk(agent_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Get multiple agents by their IDs in bulk.
    
    Args:
        agent_ids: List of agent IDs to retrieve
        
    Returns:
        List of dictionaries containing agent data, ordered by the input agent_ids
    """
    if not agent_ids:
        return []
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Use a parameterized query with IN clause
            placeholders = ','.join(['%s'] * len(agent_ids))
            cur.execute(f"""
                SELECT id, name, memory_size
                FROM agents
                WHERE id IN ({placeholders})
                ORDER BY id;
            """, agent_ids)
            
            rows = cur.fetchall()
            
            # Create a mapping for quick lookup
            agent_map = {row[0]: {'id': row[0], 'name': row[1], 'memory_size': row[2]} for row in rows}
            
            # Return results in the same order as input agent_ids
            result = []
            for agent_id in agent_ids:
                if agent_id in agent_map:
                    result.append(agent_map[agent_id])
            
            return result


def get_agent_cultures(agent_id: int) -> List[Tuple[int, str, str]]:
    """
    Get cultures for a single agent by ID.
    
    Args:
        agent_id: The ID of the agent
        
    Returns:
        List of tuples containing (culture_id, name, description) for the agent's cultures
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.name, c.description
                FROM cultures c
                INNER JOIN agent_cultures ac ON c.id = ac.culture_id
                WHERE ac.agent_id = %s
                ORDER BY c.id;
            """, (agent_id,))
            
            rows = cur.fetchall()
            return [(row[0], row[1], row[2]) for row in rows]


def get_agents_cultures_bulk(agent_ids: List[int]) -> List[List[Tuple[int, str, str]]]:
    """
    Get cultures for multiple agents by their IDs in bulk.
    
    Args:
        agent_ids: List of agent IDs to retrieve cultures for
        
    Returns:
        List of lists containing tuples (culture_id, name, description), ordered by the input agent_ids
    """
    if not agent_ids:
        return []
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Use a parameterized query with IN clause
            placeholders = ','.join(['%s'] * len(agent_ids))
            cur.execute(f"""
                SELECT ac.agent_id, c.id, c.name, c.description
                FROM cultures c
                INNER JOIN agent_cultures ac ON c.id = ac.culture_id
                WHERE ac.agent_id IN ({placeholders})
                ORDER BY ac.agent_id, c.id;
            """, agent_ids)
            
            rows = cur.fetchall()
            
            # Group results by agent_id
            agent_cultures_map = {}
            for row in rows:
                agent_id = row[0]
                culture_data = (row[1], row[2], row[3])  # (culture_id, name, description)
                
                if agent_id not in agent_cultures_map:
                    agent_cultures_map[agent_id] = []
                agent_cultures_map[agent_id].append(culture_data)
            
            # Return results in the same order as input agent_ids
            result = []
            for agent_id in agent_ids:
                if agent_id in agent_cultures_map:
                    result.append(agent_cultures_map[agent_id])
                else:
                    result.append([])
            
            return result
