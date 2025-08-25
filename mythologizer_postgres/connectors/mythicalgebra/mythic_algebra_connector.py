"""
Higher-level abstraction for myth algebra operations using the mythic algebra package.
Provides functions to work with myth matrices and embeddings.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from numpy.typing import NDArray

from mythicalgebra import (
    infer_embedding_dim,
    num_mythemes,
    decompose_myth_matrix,
    compose_myth_matrix,
    compute_myth_embedding,
)

from ..myth_store import get_myth, get_myths_bulk, insert_myth
from ..mytheme_store import get_mythemes_bulk
from mythologizer_postgres.db import psycopg_connection


def insert_myth_to_agent_memory(
    agent_id: int,
    myth_matrix: NDArray[np.floating],
    embedding_ids: List[int],
    embedding: Optional[NDArray[np.floating]] = None
) -> int:
    """
    Insert a new myth to an agent's memory.
    
    This function:
    1. Creates a new entry in the myths table with the provided myth_matrix and embedding_ids
    2. Creates an entry in agent_myths with retention = 1
    3. The position is automatically assigned by the push_agent_myth trigger
    
    Args:
        agent_id: The ID of the agent to add the myth to
        myth_matrix: The myth matrix (numpy array) containing embeddings, offsets, and weights
        embedding_ids: List of embedding IDs for the myth
        embedding: Optional main embedding. If not provided, it will be computed from the myth_matrix
    
    Returns:
        The ID of the newly created myth
    
    Raises:
        ValueError: If agent_id doesn't exist or if myth_matrix is invalid
    """
    
    # Decompose the myth matrix to get embeddings, offsets, and weights
    embeddings, offsets, weights = decompose_myth_matrix(myth_matrix)
    
    # Compute the main embedding if not provided
    if embedding is None:
        embedding = compute_myth_embedding(myth_matrix)
    
    # Insert the myth into the myths table
    myth_id = insert_myth(
        main_embedding=embedding,
        embedding_ids=embedding_ids,
        offsets=offsets,
        weights=weights
    )
    
    # Insert the myth into the agent's memory with retention = 1
    # The position will be automatically assigned by the push_agent_myth trigger
    with psycopg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO agent_myths (myth_id, agent_id, position, retention)
                VALUES (%s, %s, %s, %s)
            """, (myth_id, agent_id, 1, 1.0))  # position will be overridden by trigger
            
            conn.commit()
    
    return myth_id


def get_myth_embeddings(myth_ids: Union[int, List[int]]) -> Union[NDArray[np.floating], List[NDArray[np.floating]]]:
    """
    Get the main embeddings for one or more myths.
    
    Args:
        myth_ids: Single myth ID or list of myth IDs
    
    Returns:
        Single embedding array or list of embedding arrays
    """
    
    if isinstance(myth_ids, int):
        # Single myth
        myth = get_myth(myth_ids)
        if myth is None:
            raise ValueError(f"Myth {myth_ids} not found")
        return myth["embedding"]
    else:
        # Multiple myths
        if not myth_ids:
            return []
        
        ids, main_embeddings, _, _, _, _, _ = get_myths_bulk(myth_ids)
        if len(ids) != len(myth_ids):
            missing_ids = set(myth_ids) - set(ids)
            raise ValueError(f"Myths not found: {missing_ids}")
        
        return main_embeddings


def get_myth_matrices_and_embedding_ids(myth_ids: Union[int, List[int]]) -> Union[Tuple[NDArray[np.floating], List[int]], List[Tuple[NDArray[np.floating], List[int]]]]:
    """
    Get the myth matrices and embedding IDs for one or more myths.
    A myth matrix combines embeddings, offsets, and weights into a single array.
    
    Args:
        myth_ids: Single myth ID or list of myth IDs
    
    Returns:
        Single (myth_matrix, embedding_ids) tuple or list of (myth_matrix, embedding_ids) tuples
    """
    
    if isinstance(myth_ids, int):
        # Single myth
        myth = get_myth(myth_ids)
        if myth is None:
            raise ValueError(f"Myth {myth_ids} not found")
        
        # Get mytheme embeddings for the embedding IDs
        embedding_ids = myth["embedding_ids"]
        if not embedding_ids:
            raise ValueError(f"Myth {myth_ids} has no embedding IDs")
        
        mytheme_ids, _, mytheme_embeddings = get_mythemes_bulk(embedding_ids)
        if len(mytheme_ids) != len(embedding_ids):
            missing_ids = set(embedding_ids) - set(mytheme_ids)
            raise ValueError(f"Mythemes not found: {missing_ids}")
        
        # Convert to numpy arrays
        embeddings = np.array(mytheme_embeddings, dtype=np.float32)
        offsets = np.array(myth["offsets"], dtype=np.float32)
        weights = np.array(myth["weights"], dtype=np.float32)
        
        # Compose myth matrix
        myth_matrix = compose_myth_matrix(embeddings, offsets, weights)
        return (myth_matrix, embedding_ids)
        
    else:
        # Multiple myths
        if not myth_ids:
            return []
        
        # Get all myths
        ids, main_embeddings, embedding_ids_list, offsets_list, weights_list, created_ats, updated_ats = get_myths_bulk(myth_ids)
        if len(ids) != len(myth_ids):
            missing_ids = set(myth_ids) - set(ids)
            raise ValueError(f"Myths not found: {missing_ids}")
        
        # Collect all unique embedding IDs
        all_embedding_ids = []
        for embedding_ids in embedding_ids_list:
            all_embedding_ids.extend(embedding_ids)
        unique_embedding_ids = list(set(all_embedding_ids))
        
        # Get all mytheme embeddings
        mytheme_ids, _, mytheme_embeddings = get_mythemes_bulk(unique_embedding_ids)
        mytheme_embeddings_dict = dict(zip(mytheme_ids, mytheme_embeddings))
        
        # Build myth matrices and collect embedding IDs
        result = []
        for embedding_ids, offsets, weights in zip(embedding_ids_list, offsets_list, weights_list):
            # Get embeddings for this myth's embedding IDs
            embeddings = [mytheme_embeddings_dict[eid] for eid in embedding_ids]
            embeddings = np.array(embeddings, dtype=np.float32)
            offsets = np.array(offsets, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            
            # Compose myth matrix
            myth_matrix = compose_myth_matrix(embeddings, offsets, weights)
            result.append((myth_matrix, embedding_ids))
        
        return result


def recalc_and_update_myths(
    myth_data: Union[
        List[Tuple[int, NDArray[np.floating]]],  # List of (id, matrix) tuples
        List[int],  # List of myth IDs (will recalculate from existing data)
        NDArray[np.floating]  # Single myth matrix (will update all myths)
    ],
    myth_ids: Optional[List[int]] = None
) -> List[int]:
    """
    Recalculate myth embeddings and update the database entries.
    
    Args:
        myth_data: Can be:
            - List of (myth_id, myth_matrix) tuples
            - List of myth IDs (will recalculate from existing data)
            - Single myth matrix (will update all myths in myth_ids)
        myth_ids: Required if myth_data is a single matrix
    
    Returns:
        List of updated myth IDs
    """
    
    from ..myth_store import update_myth, update_myths_bulk
    
    if isinstance(myth_data, np.ndarray):
        # Single myth matrix - need myth_ids
        if myth_ids is None:
            raise ValueError("myth_ids must be provided when myth_data is a single matrix")
        
        # Decompose the matrix
        embeddings, offsets, weights = decompose_myth_matrix(myth_data)
        
        # Get original myths to preserve embedding IDs
        original_myths = []
        for myth_id in myth_ids:
            myth = get_myth(myth_id)
            if myth is not None:
                original_myths.append(myth)
        
        if not original_myths:
            return []
        
        # Update all myths with the same matrix, preserving original embedding IDs
        updated_count = update_myths_bulk(
            myth_ids=myth_ids,
            main_embeddings=[compute_myth_embedding(myth_data)] * len(myth_ids),
            embedding_ids_list=[myth["embedding_ids"] for myth in original_myths],
            offsets_list=[offsets.tolist() for _ in myth_ids],
            weights_list=[weights.tolist() for _ in myth_ids]
        )
        
        return myth_ids if updated_count > 0 else []
        
    elif isinstance(myth_data, list) and len(myth_data) > 0:
        if isinstance(myth_data[0], tuple):
            # List of (id, matrix) tuples
            updated_ids = []
            
            for myth_id, myth_matrix in myth_data:
                # Decompose the matrix
                embeddings, offsets, weights = decompose_myth_matrix(myth_matrix)
                
                # Compute new main embedding
                main_embedding = compute_myth_embedding(myth_matrix)
                
                # Get the original myth to preserve embedding IDs
                original_myth = get_myth(myth_id)
                if original_myth is None:
                    continue
                
                # Update the myth, preserving original embedding IDs
                success = update_myth(
                    myth_id=myth_id,
                    main_embedding=main_embedding,
                    embedding_ids=original_myth["embedding_ids"],
                    offsets=offsets,
                    weights=weights
                )
                
                if success:
                    updated_ids.append(myth_id)
            
            return updated_ids
            
        elif isinstance(myth_data[0], int):
            # List of myth IDs - recalculate from existing data
            myth_matrices_and_embedding_ids = get_myth_matrices_and_embedding_ids(myth_data)
            myth_matrices = [matrix for matrix, _ in myth_matrices_and_embedding_ids]
            updated_ids = []
            
            for myth_id, myth_matrix in zip(myth_data, myth_matrices):
                # Recompute the main embedding from the matrix
                main_embedding = compute_myth_embedding(myth_matrix)
                
                # Update only the main embedding
                success = update_myth(
                    myth_id=myth_id,
                    main_embedding=main_embedding
                )
                
                if success:
                    updated_ids.append(myth_id)
            
            return updated_ids
        else:
            raise ValueError("myth_data must be a list of tuples (id, matrix) or list of integers (myth_ids)")
    else:
        raise ValueError("myth_data must be a numpy array, list of tuples, or list of integers")


def update_myth_with_retention(
    agent_id: int,
    myth_id: int,
    myth_matrix: NDArray[np.floating],
    embedding_ids: List[int],
    retention: float,
    embedding: Optional[NDArray[np.floating]] = None
) -> bool:
    """
    Update a myth with new matrix data and update the agent_myths table with new retention.
    
    Args:
        agent_id: The ID of the agent
        myth_id: The ID of the myth to update
        myth_matrix: The myth matrix to update
        embedding_ids: List of embedding IDs
        retention: The retention value for the agent_myths table
        embedding: Optional embedding to use instead of computing from matrix
    
    Returns:
        True if myth was updated successfully, False otherwise
    """
    from ..myth_store import update_myth as update_myth_store
    from ..agent_store import update_agent_myth_retention
    
    # Decompose the matrix
    embeddings, offsets, weights = decompose_myth_matrix(myth_matrix)
    
    # Compute main embedding (use provided embedding if available)
    if embedding is not None:
        main_embedding = embedding
    else:
        main_embedding = compute_myth_embedding(myth_matrix)
    
    # Update the myth
    success = update_myth_store(
        myth_id=myth_id,
        main_embedding=main_embedding,
        embedding_ids=embedding_ids,
        offsets=offsets,
        weights=weights
    )
    
    if not success:
        return False
    
    # Update the retention in agent_myths table
    update_success = update_agent_myth_retention(agent_id, myth_id, retention)
    
    if update_success:
        # Manually trigger retention-based position recalculation
        from mythologizer_postgres.db import psycopg_connection
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT recalculate_agent_myth_positions_by_retention(%s)", (agent_id,))
                conn.commit()
    
    return update_success