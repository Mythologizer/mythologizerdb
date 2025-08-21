"""
Culture store implementation for managing cultures.
Uses SQLAlchemy for database operations.
"""

from typing import List, Optional, Tuple, Union, Sequence
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine

EngineT = Engine


def get_cultures_bulk(
    ids: Optional[List[int]] = None,
) -> List[Tuple[int, str, str]]:
    """
    Fetch cultures by id or all of them.
    Returns a list of tuples: (id, name, description).
    
    Args:
        ids: Optional list of culture IDs to fetch. If None, fetches all cultures.
    
    Returns:
        List of tuples (id, name, description)
    """
    engine: EngineT = get_engine()

    if ids:
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        sql_text = text(f"""
            SELECT id, name, description
              FROM public.cultures
             WHERE id IN ({placeholders})
             ORDER BY name
        """)
        bind = {f"id_{i}": val for i, val in enumerate(ids)}
        with engine.connect() as conn:
            rows = conn.execute(sql_text, bind).all()
    else:
        sql_text = text("""
            SELECT id, name, description 
            FROM public.cultures 
            ORDER BY name
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_text).all()

    return [(row[0], row[1], row[2]) for row in rows]


def get_culture(
    culture_id: int,
) -> Tuple[int, str, str]:
    """
    Fetch exactly one culture by id.
    
    Args:
        culture_id: The ID of the culture to fetch
    
    Returns:
        Tuple of (id, name, description)
    
    Raises:
        KeyError: If culture not found
    """
    cultures = get_cultures_bulk([culture_id])
    if not cultures:
        raise KeyError(f"culture {culture_id} not found")
    return cultures[0]


def insert_culture(
    name: str,
    description: str,
) -> int:
    """
    Insert a single culture.
    
    Args:
        name: Culture name
        description: Culture description
    
    Returns:
        The ID of the inserted culture
    """
    engine: EngineT = get_engine()

    sql_text = text("""
        INSERT INTO public.cultures (name, description)
        VALUES (:name, :description)
        RETURNING id
    """)

    with engine.begin() as conn:
        result = conn.execute(sql_text, {"name": name, "description": description})
        return result.fetchone()[0]


def insert_cultures_bulk(
    cultures: Sequence[Tuple[str, str]],
) -> List[int]:
    """
    Insert multiple cultures in one transaction.
    
    Args:
        cultures: Sequence of tuples (name, description)
    
    Returns:
        List of inserted culture IDs
    """
    engine: EngineT = get_engine()

    culture_ids = []
    
    with engine.begin() as conn:
        for name, description in cultures:
            sql_text = text("""
                INSERT INTO public.cultures (name, description)
                VALUES (:name, :description)
                RETURNING id
            """)
            result = conn.execute(sql_text, {"name": name, "description": description})
            culture_ids.append(result.fetchone()[0])
    
    return culture_ids


def update_culture(
    culture_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """
    Update a culture by ID.
    
    Args:
        culture_id: The ID of the culture to update
        name: New name (optional)
        description: New description (optional)
    
    Returns:
        True if culture was updated, False if not found
    """
    engine: EngineT = get_engine()

    # Build dynamic update query
    updates = []
    params = {"culture_id": culture_id}
    
    if name is not None:
        updates.append("name = :name")
        params["name"] = name
    
    if description is not None:
        updates.append("description = :description")
        params["description"] = description
    
    if not updates:
        return False  # Nothing to update
    
    sql_text = text(f"""
        UPDATE public.cultures 
        SET {", ".join(updates)}
        WHERE id = :culture_id
    """)

    with engine.begin() as conn:
        result = conn.execute(sql_text, params)
        return result.rowcount > 0


def delete_culture(
    culture_id: int,
) -> bool:
    """
    Delete a culture by ID.
    
    Args:
        culture_id: The ID of the culture to delete
    
    Returns:
        True if culture was deleted, False if not found
    """
    engine: EngineT = get_engine()

    sql_text = text("""
        DELETE FROM public.cultures 
        WHERE id = :culture_id
    """)

    with engine.begin() as conn:
        result = conn.execute(sql_text, {"culture_id": culture_id})
        return result.rowcount > 0


def get_cultures_by_name(
    name_pattern: str,
    exact_match: bool = False,
) -> List[Tuple[int, str, str]]:
    """
    Search for cultures by name pattern.
    
    Args:
        name_pattern: Name pattern to search for
        exact_match: If True, performs exact match. If False, performs LIKE search.
    
    Returns:
        List of tuples (id, name, description)
    """
    engine: EngineT = get_engine()

    if exact_match:
        sql_text = text("""
            SELECT id, name, description
            FROM public.cultures
            WHERE name = :name_pattern
            ORDER BY name
        """)
        params = {"name_pattern": name_pattern}
    else:
        sql_text = text("""
            SELECT id, name, description
            FROM public.cultures
            WHERE name ILIKE :name_pattern
            ORDER BY name
        """)
        params = {"name_pattern": f"%{name_pattern}%"}

    with engine.connect() as conn:
        rows = conn.execute(sql_text, params).all()

    return [(row[0], row[1], row[2]) for row in rows]
