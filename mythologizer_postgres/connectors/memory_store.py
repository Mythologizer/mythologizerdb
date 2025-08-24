from typing import List, Tuple
from sqlalchemy import text
from mythologizer_postgres.db import session_scope


def get_myth_ids_and_retention_from_agents_memory(agent_id: int) -> Tuple[List[int], List[float]]:
    """
    Get all myth ids and retentions from the agents memory. Ordered by the queue design.
    returns two lists, one for the myth ids and one for the retentions.
    
    Args:
        agent_id: The ID of the agent whose memory to retrieve
        
    Returns:
        Tuple containing:
        - myth_ids: List of myth IDs ordered by position (1 = bottom, highest = top)
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
    