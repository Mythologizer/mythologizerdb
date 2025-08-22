"""
Events store implementation for managing events.
"""

from typing import Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine, psycopg_connection

EngineT = Engine


def insert_event(description: str) -> int:
    """
    Insert a single event.
    
    Args:
        description: Description of the event
    
    Returns:
        The ID of the inserted event
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO events (description, has_been_triggered)
                VALUES (%s, FALSE)
                RETURNING id;
            """, (description,))
            
            event_id = cur.fetchone()[0]
            conn.commit()
            return event_id


def get_next_event() -> Optional[Dict[str, Any]]:
    """
    Get the earliest event that has not been triggered yet.
    
    Returns:
        Dictionary containing event data or None if no untriggered events exist
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, description, has_been_triggered, created_at
                FROM events
                WHERE has_been_triggered = FALSE
                ORDER BY created_at ASC
                LIMIT 1;
            """)
            
            row = cur.fetchone()
            if row:
                return {
                    'id': row[0],
                    'description': row[1],
                    'has_been_triggered': row[2],
                    'created_at': row[3]
                }
            return None


def set_event_triggered(event_id: int) -> bool:
    """
    Set an event as triggered by its ID.
    
    Args:
        event_id: The ID of the event to mark as triggered
    
    Returns:
        True if the event was updated, False if not found
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE events
                SET has_been_triggered = TRUE
                WHERE id = %s
                RETURNING id;
            """, (event_id,))
            
            result = cur.fetchone()
            conn.commit()
            
            return result is not None
