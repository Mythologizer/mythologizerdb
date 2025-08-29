# Myth algebra connector functions
from .mythic_algebra_connector import (
    insert_myth_to_agent_memory,
    get_myth_embeddings,
    get_myth_matrices_and_embedding_ids,
    recalc_and_update_myths,
    update_myth_with_retention,
)

# Import mythicalgebra functions for testing
try:
    from mythicalgebra import (
        decompose_myth_matrix,
        compose_myth_matrix,
        compute_myth_embedding,
    )
except ImportError:
    # Mock these for testing if mythicalgebra is not available
    decompose_myth_matrix = None
    compose_myth_matrix = None
    compute_myth_embedding = None

__all__ = [
    "insert_myth_to_agent_memory",
    "get_myth_embeddings",
    "get_myth_matrices_and_embedding_ids",
    "recalc_and_update_myths",
    "update_myth_with_retention",
    "decompose_myth_matrix",
    "compose_myth_matrix",
    "compute_myth_embedding",
] 