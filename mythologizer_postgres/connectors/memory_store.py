from typing import List, Tuple
from sqlalchemy import text
from mythologizer_postgres.db import session_scope, psycopg_connection


def get_myth_ids_and_retention_from_agents_memory(agent_id: int) -> Tuple[List[int], List[float]]:
    """
    Get all myth ids and retentions from the agents memory. Ordered by position (top to bottom).
    returns two lists, one for the myth ids and one for the retentions.
    
    Args:
        agent_id: The ID of the agent whose memory to retrieve
        
    Returns:
        Tuple containing:
        - myth_ids: List of myth IDs ordered by position (0 = top, highest = bottom)
        - retentions: List of retention values corresponding to the myth IDs
    """
    
    with session_scope() as session:
        result = session.execute(text("""
            SELECT myth_id, retention
            FROM agent_myths
            WHERE agent_id = :agent_id
            ORDER BY position ASC
        """), {"agent_id": agent_id})
        
        rows = result.fetchall()
        
        if not rows:
            return [], []
        
        myth_ids, retentions = zip(*rows)
        return list(myth_ids), list(retentions)


def update_retentions_and_reorder(agent_id: int, myth_retention_pairs: List[Tuple[int, float]]) -> bool:
    """
    Update multiple retention values for an agent and trigger position reordering.
    The myth with the highest retention will be at position 0.
    
    Args:
        agent_id: The ID of the agent
        myth_retention_pairs: List of tuples containing (myth_id, retention) pairs
        
    Returns:
        True if all updates were successful, False otherwise
    """
    if not myth_retention_pairs:
        return True
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Update all retention values
                for myth_id, retention in myth_retention_pairs:
                    cur.execute("""
                        UPDATE agent_myths 
                        SET retention = %s
                        WHERE myth_id = %s AND agent_id = %s
                    """, (retention, myth_id, agent_id))
                    
                    if cur.rowcount == 0:
                        # Myth not found for this agent
                        conn.rollback()
                        return False
                
                # Trigger position reordering based on retention
                cur.execute("SELECT recalculate_agent_myth_positions_by_retention(%s)", (agent_id,))
                
                conn.commit()
                return True
                
            except Exception:
                conn.rollback()
                return False
    