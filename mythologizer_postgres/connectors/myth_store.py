"""
Myth store implementation for managing myths with nested embeddings.
Uses psycopg for proper multi-vector support.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Union, Sequence
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine, psycopg_connection

EngineT = Engine


def insert_myth(
    main_embedding: Union[np.ndarray, List[float]],
    embedding_ids: Union[List[int], np.ndarray],
    offsets: Union[List[np.ndarray], List[List[float]], np.ndarray],
    weights: Union[List[float], np.ndarray],
) -> int:
    """
    Insert a single myth with nested embeddings.
    
    Args:
        main_embedding: Single vector of shape (embedding_dim,) as numpy array or list
        embedding_ids: List of integer IDs for each nested embedding (list or numpy array)
        offsets: List of vectors, each of shape (embedding_dim,) as numpy arrays, lists, or 2D numpy array
        weights: List of float weights for each nested embedding (list or numpy array)
    
    Returns:
        The ID of the inserted myth
    """
    
    # Convert inputs to the correct format
    if isinstance(main_embedding, list):
        main_embedding = np.array(main_embedding, dtype=np.float32)
    else:
        main_embedding = main_embedding.astype(np.float32)
    
    if isinstance(embedding_ids, np.ndarray):
        embedding_ids = embedding_ids.tolist()
    
    if isinstance(weights, np.ndarray):
        weights = weights.tolist()
    
    # Handle offsets - can be list of arrays, list of lists, or 2D numpy array
    if isinstance(offsets, np.ndarray):
        if offsets.ndim == 2:
            # 2D array: convert to list of 1D arrays
            offsets = [offsets[i].astype(np.float32) for i in range(offsets.shape[0])]
        else:
            raise ValueError("offsets numpy array must be 2D")
    else:
        # List of arrays or lists
        offsets = [
            np.array(offset, dtype=np.float32) if isinstance(offset, list) else offset.astype(np.float32)
            for offset in offsets
        ]
    
    # Validate input shapes
    if len(embedding_ids) != len(offsets) or len(offsets) != len(weights):
        raise ValueError("embedding_ids, offsets, and weights must have the same length")
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO myths (embedding, embedding_ids, offsets, weights)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (main_embedding, embedding_ids, offsets, weights))
            
            myth_id = cur.fetchone()[0]
            conn.commit()
            return myth_id


def insert_myths_bulk(
    main_embeddings: Union[List[np.ndarray], List[List[float]], np.ndarray],
    embedding_ids_list: Union[List[List[int]], List[np.ndarray], np.ndarray],
    offsets_list: Union[List[List[np.ndarray]], List[List[List[float]]], List[np.ndarray], np.ndarray],
    weights_list: Union[List[List[float]], List[np.ndarray], np.ndarray],
) -> List[int]:
    """
    Insert multiple myths with nested embeddings in bulk.
    
    Args:
        main_embeddings: List of main embeddings, list of lists, or numpy array of shape (n_myths, embedding_dim)
        embedding_ids_list: List of lists, each containing embedding IDs for one myth (or 2D numpy array)
        offsets_list: List of lists, each containing offset vectors for one myth (or 3D numpy array)
        weights_list: List of lists, each containing weights for one myth (or 2D numpy array)
    
    Returns:
        List of inserted myth IDs
    """
    
    # Convert main_embeddings to list of numpy arrays
    if isinstance(main_embeddings, np.ndarray):
        if main_embeddings.ndim == 2:
            main_embeddings = [main_embeddings[i] for i in range(main_embeddings.shape[0])]
        else:
            raise ValueError("main_embeddings numpy array must be 2D")
    else:
        # List of arrays or lists
        main_embeddings = [
            np.array(emb, dtype=np.float32) if isinstance(emb, list) else emb.astype(np.float32)
            for emb in main_embeddings
        ]
    
    # Convert embedding_ids_list
    if isinstance(embedding_ids_list, np.ndarray):
        if embedding_ids_list.ndim == 2:
            embedding_ids_list = [embedding_ids_list[i].tolist() for i in range(embedding_ids_list.shape[0])]
        else:
            raise ValueError("embedding_ids_list numpy array must be 2D")
    else:
        # List of lists or arrays
        embedding_ids_list = [
            ids.tolist() if isinstance(ids, np.ndarray) else ids
            for ids in embedding_ids_list
        ]
    
    # Convert weights_list
    if isinstance(weights_list, np.ndarray):
        if weights_list.ndim == 2:
            weights_list = [weights_list[i].tolist() for i in range(weights_list.shape[0])]
        else:
            raise ValueError("weights_list numpy array must be 2D")
    else:
        # List of lists or arrays
        weights_list = [
            weights.tolist() if isinstance(weights, np.ndarray) else weights
            for weights in weights_list
        ]
    
    # Convert offsets_list - most complex case
    if isinstance(offsets_list, np.ndarray):
        if offsets_list.ndim == 3:
            # 3D array: (n_myths, n_offsets, embedding_dim)
            offsets_list = [
                [offsets_list[i, j] for j in range(offsets_list.shape[1])]
                for i in range(offsets_list.shape[0])
            ]
        else:
            raise ValueError("offsets_list numpy array must be 3D")
    else:
        # List of lists of arrays or lists
        offsets_list = [
            [
                np.array(offset, dtype=np.float32) if isinstance(offset, list) else offset.astype(np.float32)
                for offset in offsets
            ]
            for offsets in offsets_list
        ]
    
    if len(main_embeddings) != len(embedding_ids_list) or len(main_embeddings) != len(offsets_list) or len(main_embeddings) != len(weights_list):
        raise ValueError("All input lists must have the same length")
    
    myth_ids = []
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            for main_emb, ids, offsets, weights in zip(
                main_embeddings, embedding_ids_list, offsets_list, weights_list
            ):
                cur.execute("""
                    INSERT INTO myths (embedding, embedding_ids, offsets, weights)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (main_emb, ids, offsets, weights))
                
                myth_id = cur.fetchone()[0]
                myth_ids.append(myth_id)
            
            conn.commit()
    
    return myth_ids


def get_myth(myth_id: int) -> Optional[Dict]:
    """
    Retrieve a single myth by ID.
    
    Args:
        myth_id: The ID of the myth to retrieve
    
    Returns:
        Dictionary containing myth data or None if not found
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, embedding, embedding_ids, offsets, weights, created_at, updated_at
                FROM myths
                WHERE id = %s;
            """, (myth_id,))
            
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "embedding": row[1],
                    "embedding_ids": row[2],
                    "offsets": row[3],
                    "weights": row[4],
                    "created_at": row[5],
                    "updated_at": row[6]
                }
            return None


def get_myths_bulk(
    myth_ids: Optional[List[int]] = None,
    as_numpy: bool = True,
) -> Tuple[List[int], List[np.ndarray], List[List[int]], List[List[np.ndarray]], List[List[float]], List, List]:
    """
    Fetch myths by ID or all of them.
    
    Args:
        myth_ids: List of myth IDs to fetch, or None for all
        as_numpy: Whether to return embeddings as numpy arrays
    
    Returns:
        Tuple of (ids, main_embeddings, embedding_ids_list, offsets_list, weights_list, created_ats, updated_ats)
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            if myth_ids:
                placeholders = ", ".join(["%s"] * len(myth_ids))
                cur.execute(f"""
                    SELECT id, embedding, embedding_ids, offsets, weights, created_at, updated_at
                    FROM myths
                    WHERE id IN ({placeholders})
                    ORDER BY id;
                """, myth_ids)
            else:
                cur.execute("""
                    SELECT id, embedding, embedding_ids, offsets, weights, created_at, updated_at
                    FROM myths
                    ORDER BY id;
                """)
            
            rows = cur.fetchall()
    
    if not rows:
        return [], [], [], [], [], [], []
    
    ids, main_embeddings, embedding_ids_list, offsets_list, weights_list, created_ats, updated_ats = zip(*rows)
    
    if as_numpy:
        main_embeddings = [np.array(emb, dtype=np.float32) for emb in main_embeddings]
        offsets_list = [[np.array(offset, dtype=np.float32) for offset in offsets] for offsets in offsets_list]
    
    return list(ids), list(main_embeddings), list(embedding_ids_list), list(offsets_list), list(weights_list), list(created_ats), list(updated_ats)


def update_myth(
    myth_id: int,
    main_embedding: Optional[Union[np.ndarray, List[float]]] = None,
    embedding_ids: Optional[Union[List[int], np.ndarray]] = None,
    offsets: Optional[Union[List[np.ndarray], List[List[float]], np.ndarray]] = None,
    weights: Optional[Union[List[float], np.ndarray]] = None,
) -> bool:
    """
    Update a myth's data.
    
    Args:
        myth_id: The ID of the myth to update
        main_embedding: New main embedding as numpy array or list (optional)
        embedding_ids: New embedding IDs as list or numpy array (optional)
        offsets: New offset vectors as list of arrays, list of lists, or 2D numpy array (optional)
        weights: New weights as list or numpy array (optional)
    
    Returns:
        True if myth was updated, False if not found
    """
    
    # Build dynamic update query
    updates = []
    params = []
    
    if main_embedding is not None:
        if isinstance(main_embedding, list):
            main_embedding = np.array(main_embedding, dtype=np.float32)
        else:
            main_embedding = main_embedding.astype(np.float32)
        updates.append("embedding = %s")
        params.append(main_embedding)
    
    if embedding_ids is not None:
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = embedding_ids.tolist()
        updates.append("embedding_ids = %s")
        params.append(embedding_ids)
    
    if offsets is not None:
        if isinstance(offsets, np.ndarray):
            if offsets.ndim == 2:
                offsets = [offsets[i].astype(np.float32) for i in range(offsets.shape[0])]
            else:
                raise ValueError("offsets numpy array must be 2D")
        else:
            offsets = [
                np.array(offset, dtype=np.float32) if isinstance(offset, list) else offset.astype(np.float32)
                for offset in offsets
            ]
        updates.append("offsets = %s")
        params.append(offsets)
    
    if weights is not None:
        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        updates.append("weights = %s")
        params.append(weights)
    
    if not updates:
        return False  # Nothing to update
    
    params.append(myth_id)
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE myths
                SET {', '.join(updates)}
                WHERE id = %s;
            """, params)
            
            conn.commit()
            return cur.rowcount > 0


def update_myths_bulk(
    myth_ids: List[int],
    main_embeddings: Optional[Union[List[np.ndarray], List[List[float]], np.ndarray]] = None,
    embedding_ids_list: Optional[Union[List[List[int]], List[np.ndarray], np.ndarray]] = None,
    offsets_list: Optional[Union[List[List[np.ndarray]], List[List[List[float]]], List[np.ndarray], np.ndarray]] = None,
    weights_list: Optional[Union[List[List[float]], List[np.ndarray], np.ndarray]] = None,
) -> int:
    """
    Update multiple myths in bulk.
    
    Args:
        myth_ids: List of myth IDs to update
        main_embeddings: List of new main embeddings, list of lists, or 2D numpy array (optional)
        embedding_ids_list: List of new embedding ID lists or 2D numpy array (optional)
        offsets_list: List of new offset vector lists or 3D numpy array (optional)
        weights_list: List of new weight lists or 2D numpy array (optional)
    
    Returns:
        Number of myths updated
    """
    
    if len(myth_ids) == 0:
        return 0
    
    # Validate input lengths
    if main_embeddings is not None and len(main_embeddings) != len(myth_ids):
        raise ValueError("main_embeddings length must match myth_ids length")
    if embedding_ids_list is not None and len(embedding_ids_list) != len(myth_ids):
        raise ValueError("embedding_ids_list length must match myth_ids length")
    if offsets_list is not None and len(offsets_list) != len(myth_ids):
        raise ValueError("offsets_list length must match myth_ids length")
    if weights_list is not None and len(weights_list) != len(myth_ids):
        raise ValueError("weights_list length must match myth_ids length")
    
    # Convert main_embeddings
    if main_embeddings is not None:
        if isinstance(main_embeddings, np.ndarray):
            if main_embeddings.ndim == 2:
                main_embeddings = [main_embeddings[i] for i in range(main_embeddings.shape[0])]
            else:
                raise ValueError("main_embeddings numpy array must be 2D")
        else:
            main_embeddings = [
                np.array(emb, dtype=np.float32) if isinstance(emb, list) else emb.astype(np.float32)
                for emb in main_embeddings
            ]
    
    # Convert embedding_ids_list
    if embedding_ids_list is not None:
        if isinstance(embedding_ids_list, np.ndarray):
            if embedding_ids_list.ndim == 2:
                embedding_ids_list = [embedding_ids_list[i].tolist() for i in range(embedding_ids_list.shape[0])]
            else:
                raise ValueError("embedding_ids_list numpy array must be 2D")
        else:
            embedding_ids_list = [
                ids.tolist() if isinstance(ids, np.ndarray) else ids
                for ids in embedding_ids_list
            ]
    
    # Convert weights_list
    if weights_list is not None:
        if isinstance(weights_list, np.ndarray):
            if weights_list.ndim == 2:
                weights_list = [weights_list[i].tolist() for i in range(weights_list.shape[0])]
            else:
                raise ValueError("weights_list numpy array must be 2D")
        else:
            weights_list = [
                weights.tolist() if isinstance(weights, np.ndarray) else weights
                for weights in weights_list
            ]
    
    # Convert offsets_list
    if offsets_list is not None:
        if isinstance(offsets_list, np.ndarray):
            if offsets_list.ndim == 3:
                offsets_list = [
                    [offsets_list[i, j] for j in range(offsets_list.shape[1])]
                    for i in range(offsets_list.shape[0])
                ]
            else:
                raise ValueError("offsets_list numpy array must be 3D")
        else:
            offsets_list = [
                [
                    np.array(offset, dtype=np.float32) if isinstance(offset, list) else offset.astype(np.float32)
                    for offset in offsets
                ]
                for offsets in offsets_list
            ]
    
    updated_count = 0
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            for i, myth_id in enumerate(myth_ids):
                updates = []
                params = []
                
                if main_embeddings:
                    updates.append("embedding = %s")
                    params.append(main_embeddings[i])
                
                if embedding_ids_list:
                    updates.append("embedding_ids = %s")
                    params.append(embedding_ids_list[i])
                
                if offsets_list:
                    updates.append("offsets = %s")
                    params.append(offsets_list[i])
                
                if weights_list:
                    updates.append("weights = %s")
                    params.append(weights_list[i])
                
                if updates:
                    params.append(myth_id)
                    cur.execute(f"""
                        UPDATE myths
                        SET {', '.join(updates)}
                        WHERE id = %s;
                    """, params)
                    updated_count += cur.rowcount
            
            conn.commit()
    
    return updated_count


def delete_myth(myth_id: int) -> bool:
    """
    Delete a myth by ID.
    
    Args:
        myth_id: The ID of the myth to delete
    
    Returns:
        True if myth was deleted, False if not found
    """
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM myths WHERE id = %s;", (myth_id,))
            conn.commit()
            return cur.rowcount > 0


def delete_myths_bulk(myth_ids: List[int]) -> int:
    """
    Delete multiple myths by ID.
    
    Args:
        myth_ids: List of myth IDs to delete
    
    Returns:
        Number of myths deleted
    """
    
    if not myth_ids:
        return 0
    
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(myth_ids))
            cur.execute(f"DELETE FROM myths WHERE id IN ({placeholders});", myth_ids)
            conn.commit()
            return cur.rowcount
