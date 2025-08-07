# Myth algebra connector functions
from .mythic_algebra_connector import (
    get_myth_embeddings,
    get_myth_matrices,
    recalc_and_update_myths,
)
from ..myth_store import get_myth, get_myths_bulk, update_myth, update_myths_bulk
from ..mytheme_store import get_mythemes_bulk

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
    "get_myth",
    "get_myths_bulk",
    "get_mythemes_bulk",
    "update_myth",
    "update_myths_bulk",
    "get_myth_embeddings",
    "get_myth_matrices",
    "recalc_and_update_myths",
    "decompose_myth_matrix",
    "compose_myth_matrix",
    "compute_myth_embedding",
] 