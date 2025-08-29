"""
Agent store implementation for managing agents.
"""

from typing import List, Dict, Any, Tuple, Optional
from mythologizer_postgres.db import psycopg_connection
import time


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


def get_agents_cultures_ids_bulk(agent_ids: List[int]) -> List[List[int]]:
    """
    Get culture IDs for multiple agents by their IDs in bulk.
    
    Args:
        agent_ids: List of agent IDs to retrieve culture IDs for
        
    Returns:
        List of lists containing culture IDs, ordered by the input agent_ids
    """
    if not agent_ids:
        return []
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            # Use a parameterized query with IN clause
            placeholders = ','.join(['%s'] * len(agent_ids))
            cur.execute(f"""
                SELECT ac.agent_id, c.id
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
                culture_id = row[1]
                
                if agent_id not in agent_cultures_map:
                    agent_cultures_map[agent_id] = []
                agent_cultures_map[agent_id].append(culture_id)
            
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


def insert_agent_myth_safe(myth_id: int, agent_id: int, retention: float, max_retries: int = 5) -> bool:
    """
    Safely insert a new agent_myth entry with proper position assignment and retention reordering.
    This function handles the position assignment that was previously done by the trigger.
    
    Args:
        myth_id: The ID of the myth
        agent_id: The ID of the agent
        retention: The retention value (must be > 0)
        max_retries: Maximum number of retry attempts for race conditions
        
    Returns:
        True if inserted successfully, False otherwise
    """
    import time
    
    # Validate retention value
    if retention <= 0:
        print(f"Error in insert_agent_myth_safe: retention must be > 0, got {retention}")
        return False
    
    for attempt in range(max_retries):
        try:
            with psycopg_connection() as conn:
                with conn.cursor() as cur:
                    # Start transaction with proper isolation level to handle race conditions
                    cur.execute("BEGIN")
                    
                    # Lock the agent row to prevent concurrent modifications
                    cur.execute("""
                        SELECT memory_size FROM agents WHERE id = %s FOR UPDATE
                    """, (agent_id,))
                    result = cur.fetchone()
                    if not result:
                        print(f"Error in insert_agent_myth_safe: agent {agent_id} not found")
                        conn.rollback()
                        return False
                    
                    max_size = result[0]
                    if max_size == 0:
                        print(f"Error in insert_agent_myth_safe: agent {agent_id} has memory_size = 0")
                        conn.rollback()
                        return False
                    
                    # Check if myth is already assigned to an agent
                    cur.execute("SELECT 1 FROM agent_myths WHERE myth_id = %s", (myth_id,))
                    if cur.fetchone():
                        print(f"Error in insert_agent_myth_safe: myth {myth_id} is already assigned to an agent")
                        conn.rollback()
                        return False
                    
                    # Get current count and handle memory size limit first
                    cur.execute("SELECT COUNT(*) FROM agent_myths WHERE agent_id = %s", (agent_id,))
                    cur_count = cur.fetchone()[0]
                    
                    # Handle memory size limit
                    if cur_count >= max_size:
                        # Evict highest position myth (bottom of stack)
                        cur.execute("""
                            DELETE FROM agent_myths 
                            WHERE myth_id = (
                                SELECT myth_id FROM agent_myths 
                                WHERE agent_id = %s 
                                ORDER BY position DESC 
                                LIMIT 1
                            )
                        """, (agent_id,))
                    
                    # Insert the new myth at position 0 and shift all existing myths down by 1
                    # Use a temporary offset to avoid conflicts during the shift
                    cur.execute("""
                        UPDATE agent_myths 
                        SET position = position + 10000 
                        WHERE agent_id = %s
                    """, (agent_id,))
                    
                    # Then insert the new myth at position 0
                    cur.execute("""
                        INSERT INTO agent_myths (myth_id, agent_id, position, retention)
                        VALUES (%s, %s, 0, %s)
                    """, (myth_id, agent_id, retention))
                    
                    # Now shift all existing myths (excluding the new one) down by 1
                    cur.execute("""
                        UPDATE agent_myths 
                        SET position = position - 9999 
                        WHERE agent_id = %s AND myth_id != %s
                    """, (agent_id, myth_id))
                    
                    # Now reorder all positions by retention (highest retention = position 0)
                    # Use a two-step approach to avoid unique constraint violations
                    
                    # Step 1: Assign temporary positions (offset by 10000 to avoid conflicts)
                    cur.execute("""
                        UPDATE agent_myths
                        SET position = new_pos + 10000
                        FROM (
                            SELECT myth_id, ROW_NUMBER() OVER (ORDER BY retention DESC, myth_id ASC) - 1 as new_pos
                            FROM agent_myths
                            WHERE agent_id = %s
                        ) reordered
                        WHERE agent_myths.myth_id = reordered.myth_id
                    """, (agent_id,))
                    
                    # Step 2: Assign final positions
                    cur.execute("""
                        UPDATE agent_myths
                        SET position = new_pos
                        FROM (
                            SELECT myth_id, ROW_NUMBER() OVER (ORDER BY position ASC) - 1 as new_pos
                            FROM agent_myths
                            WHERE agent_id = %s
                        ) reordered
                        WHERE agent_myths.myth_id = reordered.myth_id
                    """, (agent_id,))
                    
                    conn.commit()
                    return True
                
        except Exception as e:
            # Check if this is a unique constraint violation (race condition)
            if "duplicate key value violates unique constraint" in str(e) and attempt < max_retries - 1:
                print(f"DEBUG: Race condition detected on attempt {attempt + 1}, retrying...")
                time.sleep(0.01 * (2 ** attempt))  # Exponential backoff
                continue
            else:
                print(f"Error in insert_agent_myth_safe: {e}")
                return False
    
    print(f"Error in insert_agent_myth_safe: failed after {max_retries} attempts")
    return False


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


def insert_agent_myth_safe_with_session(session, myth_id: int, agent_id: int, retention: float) -> bool:
    """
    Safely insert a new agent_myth entry with proper position assignment and retention reordering.
    This version works with SQLAlchemy sessions for testing.
    
    Args:
        session: SQLAlchemy session
        myth_id: The ID of the myth
        agent_id: The ID of the agent
        retention: The retention value (must be > 0)
        
    Returns:
        True if inserted successfully, False otherwise
    """
    from sqlalchemy import text
    
    # Validate retention value
    if retention <= 0:
        print(f"Error in insert_agent_myth_safe_with_session: retention must be > 0, got {retention}")
        return False
    
    try:
        # Check if agent exists and get memory size
        result = session.execute(text("SELECT memory_size FROM agents WHERE id = :agent_id"), {"agent_id": agent_id})
        agent_data = result.fetchone()
        if not agent_data:
            print(f"Error in insert_agent_myth_safe_with_session: agent {agent_id} not found")
            return False
        
        max_size = agent_data[0]
        if max_size == 0:
            print(f"Error in insert_agent_myth_safe_with_session: agent {agent_id} has memory_size = 0")
            return False
        
        # Check if myth is already assigned to an agent
        result = session.execute(text("SELECT 1 FROM agent_myths WHERE myth_id = :myth_id"), {"myth_id": myth_id})
        if result.fetchone():
            print(f"Error in insert_agent_myth_safe_with_session: myth {myth_id} is already assigned to an agent")
            return False
        
        # Get current count
        result = session.execute(text("SELECT COUNT(*) FROM agent_myths WHERE agent_id = :agent_id"), {"agent_id": agent_id})
        cur_count = result.fetchone()[0]
        
        # Handle memory size limit
        if cur_count >= max_size:
            # Evict highest position myth (bottom of stack)
            session.execute(text("""
                DELETE FROM agent_myths 
                WHERE myth_id = (
                    SELECT myth_id FROM agent_myths 
                    WHERE agent_id = :agent_id 
                    ORDER BY position DESC 
                    LIMIT 1
                )
            """), {"agent_id": agent_id})
            
            # Reorder remaining positions to be contiguous starting from 0
            session.execute(text("""
                UPDATE agent_myths
                SET position = new_pos
                FROM (
                    SELECT myth_id, ROW_NUMBER() OVER (ORDER BY position ASC) - 1 as new_pos
                    FROM agent_myths
                    WHERE agent_id = :agent_id
                ) reordered
                WHERE agent_myths.myth_id = reordered.myth_id
            """), {"agent_id": agent_id})
        
        # Get the next available position
        result = session.execute(text("SELECT COALESCE(MAX(position), -1) + 1 FROM agent_myths WHERE agent_id = :agent_id"), {"agent_id": agent_id})
        next_position = result.fetchone()[0]
        
        # Insert the new myth at the next available position
        session.execute(text("""
            INSERT INTO agent_myths (myth_id, agent_id, position, retention)
            VALUES (:myth_id, :agent_id, :next_position, :retention)
        """), {"myth_id": myth_id, "agent_id": agent_id, "next_position": next_position, "retention": retention})
        
        # Now reorder positions by retention (highest retention = position 0)
        session.execute(text("""
            UPDATE agent_myths
            SET position = new_pos
            FROM (
                SELECT myth_id, ROW_NUMBER() OVER (ORDER BY retention DESC, myth_id ASC) - 1 as new_pos
                FROM agent_myths
                WHERE agent_id = :agent_id
            ) reordered
            WHERE agent_myths.myth_id = reordered.myth_id
        """), {"agent_id": agent_id})
        
        return True
        
    except Exception as e:
        print(f"Error in insert_agent_myth_safe_with_session: {e}")
        return False
