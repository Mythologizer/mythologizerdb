"""
Agent store implementation for managing agents.
"""

from typing import List, Dict, Any, Tuple, Optional
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


def get_agent_myth(agent_id: int) -> Optional[Dict[str, Any]]:
    """
    Get the agent_myth entry for a specific agent.
    
    Args:
        agent_id: The ID of the agent
        
    Returns:
        Dictionary containing agent_myth data or None if not found
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT myth_id, agent_id, position, retention
                FROM agent_myths
                WHERE agent_id = %s
                LIMIT 1
            """, (agent_id,))
            
            row = cur.fetchone()
            if row:
                return {
                    'myth_id': row[0],
                    'agent_id': row[1],
                    'position': row[2],
                    'retention': row[3]
                }
            return None


def insert_agent_myth(myth_id: int, agent_id: int, position: int, retention: float) -> bool:
    """
    Insert a new agent_myth entry.
    
    Args:
        myth_id: The ID of the myth
        agent_id: The ID of the agent
        position: The position
        retention: The retention value
        
    Returns:
        True if inserted successfully, False otherwise
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO agent_myths (myth_id, agent_id, position, retention)
                    VALUES (%s, %s, %s, %s)
                """, (myth_id, agent_id, position, retention))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                return False


def update_agent_myth_retention(agent_id: int, myth_id: int, retention: float) -> bool:
    """
    Update the retention value for an agent_myth entry.
    
    Args:
        agent_id: The ID of the agent
        myth_id: The ID of the myth
        retention: The new retention value
        
    Returns:
        True if updated successfully, False otherwise
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE agent_myths 
                SET retention = %s
                WHERE myth_id = %s AND agent_id = %s
            """, (retention, myth_id, agent_id))
            conn.commit()
            return cur.rowcount > 0


def recalculate_agent_myth_positions_by_retention(agent_id: int) -> bool:
    """
    Manually trigger retention-based position recalculation for an agent.
    
    Args:
        agent_id: The ID of the agent
        
    Returns:
        True if successful, False otherwise
    """
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT recalculate_agent_myth_positions_by_retention(%s)", (agent_id,))
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                return False
