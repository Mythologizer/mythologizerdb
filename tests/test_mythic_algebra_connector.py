import os
import pytest
import numpy as np
from typing import List, Tuple

from mythologizer_postgres.connectors.mythicalgebra.mythic_algebra_connector import (
    get_myth_embeddings,
    get_myth_matrices,
    recalc_and_update_myths,
)

from mythologizer_postgres.connectors.myth_store import (
    insert_myth,
    insert_myths_bulk,
    get_myth,
)

from mythologizer_postgres.connectors.mytheme_store import (
    insert_mythemes_bulk,
    get_mythemes_bulk,
)

from mythicalgebra import (
    infer_embedding_dim,
    num_mythemes,
    decompose_myth_matrix,
    compose_myth_matrix,
    compute_myth_embedding,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


def create_test_embedding(base_values, embedding_dim):
    """Create a test embedding array with the correct dimension."""
    # Extend base_values to at least embedding_dim length
    extended_values = base_values + [0.1 * i for i in range(len(base_values), embedding_dim)]
    return np.array(extended_values[:embedding_dim], dtype=np.float32)


# IMPORTANT: Data Integrity Testing
# 
# The tests in this file verify CRITICAL data integrity for myth algebra operations:
# - Myth matrix composition and decomposition
# - Myth embedding computation
# - Integration with myth_store and mytheme_store
# - Matrix operations using the mythicalgebra package
#
# We use np.testing.assert_array_almost_equal with decimal=7 to account for
# database floating-point precision differences while ensuring high precision
# data integrity for all vector components.


class TestMythicAlgebraConnector:
    """Test the mythic_algebra_connector module with real database operations."""
    
    @pytest.fixture(autouse=True)
    def cleanup_database(self):
        """Clean up the database before each test."""
        from mythologizer_postgres.db import clear_all_rows
        # Clean up before test
        try:
            clear_all_rows()
        except Exception:
            pass
        yield
        # Clean up after test
        try:
            clear_all_rows()
        except Exception:
            pass
    
    @pytest.mark.integration
    def test_mythicalgebra_package_functions(self):
        """Test the core mythicalgebra package functions."""
        embedding_dim = get_embedding_dim()
        
        # Create test data
        N = 3  # Number of mythemes
        embeddings = np.random.rand(N, embedding_dim).astype(np.float32)
        offsets = np.random.rand(N, embedding_dim).astype(np.float32)
        weights = np.random.rand(N).astype(np.float32)
        
        # Test compose_myth_matrix
        myth_matrix = compose_myth_matrix(embeddings, offsets, weights)
        
        # Verify matrix shape
        expected_shape = (N, 2 * embedding_dim + 1)
        assert myth_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {myth_matrix.shape}"
        
        # Test infer_embedding_dim
        inferred_dim = infer_embedding_dim(myth_matrix)
        assert inferred_dim == embedding_dim, f"Expected embedding dim {embedding_dim}, got {inferred_dim}"
        
        # Test num_mythemes
        num_themes = num_mythemes(myth_matrix)
        assert num_themes == N, f"Expected {N} mythemes, got {num_themes}"
        
        # Test decompose_myth_matrix
        decomp_embeddings, decomp_offsets, decomp_weights = decompose_myth_matrix(myth_matrix)
        
        # CRITICAL: Verify decomposition integrity
        np.testing.assert_array_almost_equal(
            decomp_embeddings, embeddings, decimal=7,
            err_msg="CRITICAL: Decomposed embeddings don't match original"
        )
        np.testing.assert_array_almost_equal(
            decomp_offsets, offsets, decimal=7,
            err_msg="CRITICAL: Decomposed offsets don't match original"
        )
        np.testing.assert_array_almost_equal(
            decomp_weights, weights, decimal=7,
            err_msg="CRITICAL: Decomposed weights don't match original"
        )
        
        # Test compute_myth_embedding
        computed_embedding = compute_myth_embedding(myth_matrix)
        
        # Verify computed embedding shape
        assert computed_embedding.shape == (embedding_dim,), f"Expected shape ({embedding_dim},), got {computed_embedding.shape}"
        
        # Verify computed embedding is correct: sum_i w_i * (e_i + o_i)
        expected_embedding = ((embeddings + offsets).T @ weights).ravel()
        np.testing.assert_array_almost_equal(
            computed_embedding, expected_embedding, decimal=7,
            err_msg="CRITICAL: Computed myth embedding doesn't match expected"
        )
    
    @pytest.mark.integration
    def test_get_myth_embeddings_single(self):
        """Test getting a single myth embedding."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert test data
        main_embedding = create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim)
        embedding_ids = [1, 2]
        offsets = [
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Get myth embedding
        retrieved_embedding = get_myth_embeddings(myth_id)
        
        # CRITICAL: Verify embedding integrity
        np.testing.assert_array_almost_equal(
            retrieved_embedding, main_embedding, decimal=7,
            err_msg="CRITICAL: Retrieved myth embedding doesn't match original"
        )
    
    @pytest.mark.integration
    def test_get_myth_embeddings_multiple(self):
        """Test getting multiple myth embeddings."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert test myths
        main_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        embedding_ids_list = [[1, 2], [3, 4]]
        offsets_list = [
            [
                create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
                create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
            ],
            [
                create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
                create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
            ]
        ]
        weights_list = [[0.5, 0.5], [0.5, 0.5]]
        
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Get myth embeddings
        retrieved_embeddings = get_myth_embeddings(myth_ids)
        
        # CRITICAL: Verify embeddings integrity
        assert len(retrieved_embeddings) == len(main_embeddings)
        for i, (retrieved, original) in enumerate(zip(retrieved_embeddings, main_embeddings)):
            np.testing.assert_array_almost_equal(
                retrieved, original, decimal=7,
                err_msg=f"CRITICAL: Retrieved myth embedding {i} doesn't match original"
            )
    
    @pytest.mark.integration
    def test_get_myth_embeddings_not_found(self):
        """Test error handling for non-existent myth."""
        with pytest.raises(ValueError, match="Myth 99999 not found"):
            get_myth_embeddings(99999)
    
    @pytest.mark.integration
    def test_get_myth_matrices_single(self):
        """Test getting a single myth matrix."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert mythemes first
        mytheme_sentences = ["Mytheme 1", "Mytheme 2"]
        mytheme_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert myth
        main_embedding = create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        embedding_ids = mytheme_ids[:2]  # Use first two mytheme IDs
        offsets = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Get myth matrix
        myth_matrix = get_myth_matrices(myth_id)
        
        # Verify matrix shape
        expected_shape = (2, 2 * embedding_dim + 1)  # 2 mythemes, 2D+1 columns
        assert myth_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {myth_matrix.shape}"
        
        # Decompose and verify integrity
        decomp_embeddings, decomp_offsets, decomp_weights = decompose_myth_matrix(myth_matrix)
        
        # CRITICAL: Verify mytheme embeddings integrity
        for i, (retrieved, original) in enumerate(zip(decomp_embeddings, mytheme_embeddings)):
            np.testing.assert_array_almost_equal(
                retrieved, original, decimal=7,
                err_msg=f"CRITICAL: Decomposed mytheme embedding {i} doesn't match original"
            )
        
        # CRITICAL: Verify offsets integrity
        for i, (retrieved, original) in enumerate(zip(decomp_offsets, offsets)):
            np.testing.assert_array_almost_equal(
                retrieved, original, decimal=7,
                err_msg=f"CRITICAL: Decomposed offset {i} doesn't match original"
            )
        
        # CRITICAL: Verify weights integrity
        np.testing.assert_array_almost_equal(
            decomp_weights, weights, decimal=7,
            err_msg="CRITICAL: Decomposed weights don't match original"
        )
    
    @pytest.mark.integration
    def test_get_myth_matrices_multiple(self):
        """Test getting multiple myth matrices."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert mythemes first
        mytheme_sentences = ["Mytheme 1", "Mytheme 2", "Mytheme 3", "Mytheme 4"]
        mytheme_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
            create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert myths
        main_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        embedding_ids_list = [mytheme_ids[:2], mytheme_ids[2:]]  # Use different mytheme sets
        offsets_list = [
            [
                create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
                create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
            ],
            [
                create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
                create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
            ]
        ]
        weights_list = [[0.5, 0.5], [0.6, 0.4]]
        
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Get myth matrices
        myth_matrices = get_myth_matrices(myth_ids)
        
        # Verify we got the expected number of matrices
        assert len(myth_matrices) == len(myth_ids)
        
        # Verify each matrix
        for i, myth_matrix in enumerate(myth_matrices):
            # Verify matrix shape
            expected_shape = (2, 2 * embedding_dim + 1)  # 2 mythemes per myth
            assert myth_matrix.shape == expected_shape, f"Matrix {i} has wrong shape"
            
            # Decompose and verify integrity
            decomp_embeddings, decomp_offsets, decomp_weights = decompose_myth_matrix(myth_matrix)
            
            # CRITICAL: Verify mytheme embeddings integrity
            start_idx = i * 2
            for j, (retrieved, original) in enumerate(zip(decomp_embeddings, mytheme_embeddings[start_idx:start_idx+2])):
                np.testing.assert_array_almost_equal(
                    retrieved, original, decimal=7,
                    err_msg=f"CRITICAL: Matrix {i}, mytheme embedding {j} doesn't match original"
                )
            
            # CRITICAL: Verify offsets integrity
            for j, (retrieved, original) in enumerate(zip(decomp_offsets, offsets_list[i])):
                np.testing.assert_array_almost_equal(
                    retrieved, original, decimal=7,
                    err_msg=f"CRITICAL: Matrix {i}, offset {j} doesn't match original"
                )
            
            # CRITICAL: Verify weights integrity
            np.testing.assert_array_almost_equal(
                decomp_weights, weights_list[i], decimal=7,
                err_msg=f"CRITICAL: Matrix {i} weights don't match original"
            )
    
    @pytest.mark.integration
    def test_get_myth_matrices_not_found(self):
        """Test error handling for non-existent myth."""
        with pytest.raises(ValueError, match="Myth 99999 not found"):
            get_myth_matrices(99999)
    
    @pytest.mark.integration
    def test_recalc_and_update_myths_with_matrices(self):
        """Test recalculating and updating myths with new matrices."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert mythemes first
        mytheme_sentences = ["Mytheme 1", "Mytheme 2"]
        mytheme_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert initial myth
        main_embedding = create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        embedding_ids = mytheme_ids[:2]
        offsets = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Create new myth matrix with different values
        new_embeddings = np.array(mytheme_embeddings, dtype=np.float32)
        new_offsets = np.array([
            create_test_embedding([0.2, 0.3, 0.4, 0.5], embedding_dim),
            create_test_embedding([0.6, 0.7, 0.8, 0.9], embedding_dim)
        ], dtype=np.float32)
        new_weights = np.array([0.6, 0.4], dtype=np.float32)
        
        new_myth_matrix = compose_myth_matrix(new_embeddings, new_offsets, new_weights)
        
        # Calculate expected new main embedding
        expected_new_embedding = compute_myth_embedding(new_myth_matrix)
        
        # Update myth with new matrix
        updated_ids = recalc_and_update_myths([(myth_id, new_myth_matrix)])
        
        # Verify update was successful
        assert len(updated_ids) == 1
        assert updated_ids[0] == myth_id
        
        # Retrieve updated myth
        updated_myth = get_myth(myth_id)
        
        # CRITICAL: Verify new main embedding
        np.testing.assert_array_almost_equal(
            updated_myth["embedding"], expected_new_embedding, decimal=7,
            err_msg="CRITICAL: Updated main embedding doesn't match expected"
        )
        
        # CRITICAL: Verify new offsets
        for i, (retrieved, original) in enumerate(zip(updated_myth["offsets"], new_offsets)):
            np.testing.assert_array_almost_equal(
                retrieved, original, decimal=7,
                err_msg=f"CRITICAL: Updated offset {i} doesn't match expected"
            )
        
        # CRITICAL: Verify new weights
        np.testing.assert_array_almost_equal(
            updated_myth["weights"], new_weights, decimal=7,
            err_msg="CRITICAL: Updated weights don't match expected"
        )
        
        # Verify embedding IDs were preserved
        assert updated_myth["embedding_ids"] == embedding_ids
    
    @pytest.mark.integration
    def test_recalc_and_update_myths_with_ids(self):
        """Test recalculating and updating myths from existing data."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert mythemes first
        mytheme_sentences = ["Mytheme 1", "Mytheme 2"]
        mytheme_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert myth
        main_embedding = create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        embedding_ids = mytheme_ids[:2]
        offsets = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Get the myth matrix to compute expected embedding
        myth_matrix = get_myth_matrices(myth_id)
        expected_embedding = compute_myth_embedding(myth_matrix)
        
        # Recalculate from existing data
        updated_ids = recalc_and_update_myths([myth_id])
        
        # Verify update was successful
        assert len(updated_ids) == 1
        assert updated_ids[0] == myth_id
        
        # Retrieve updated myth
        updated_myth = get_myth(myth_id)
        
        # CRITICAL: Verify recalculated main embedding
        np.testing.assert_array_almost_equal(
            updated_myth["embedding"], expected_embedding, decimal=7,
            err_msg="CRITICAL: Recalculated main embedding doesn't match expected"
        )
        
        # Verify other components were unchanged
        for i, (retrieved, original) in enumerate(zip(updated_myth["offsets"], offsets)):
            np.testing.assert_array_almost_equal(
                retrieved, original, decimal=7,
                err_msg=f"CRITICAL: Offset {i} should remain unchanged"
            )
        
        np.testing.assert_array_almost_equal(
            updated_myth["weights"], weights, decimal=7,
            err_msg="CRITICAL: Weights should remain unchanged"
        )
    
    @pytest.mark.integration
    def test_recalc_and_update_myths_with_single_matrix(self):
        """Test updating multiple myths with a single matrix."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert mythemes first
        mytheme_sentences = ["Mytheme 1", "Mytheme 2"]
        mytheme_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert two myths
        main_embeddings = [
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
            create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
        ]
        embedding_ids_list = [mytheme_ids[:2], mytheme_ids[:2]]
        offsets_list = [
            [
                create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
                create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
            ],
            [
                create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
                create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
            ]
        ]
        weights_list = [[0.5, 0.5], [0.6, 0.4]]
        
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Create new myth matrix
        new_embeddings = np.array(mytheme_embeddings, dtype=np.float32)
        new_offsets = np.array([
            create_test_embedding([0.2, 0.3, 0.4, 0.5], embedding_dim),
            create_test_embedding([0.6, 0.7, 0.8, 0.9], embedding_dim)
        ], dtype=np.float32)
        new_weights = np.array([0.7, 0.3], dtype=np.float32)
        
        new_myth_matrix = compose_myth_matrix(new_embeddings, new_offsets, new_weights)
        
        # Calculate expected new main embedding
        expected_new_embedding = compute_myth_embedding(new_myth_matrix)
        
        # Update both myths with the same matrix
        updated_ids = recalc_and_update_myths(new_myth_matrix, myth_ids)
        
        # Verify both myths were updated
        assert len(updated_ids) == 2
        assert set(updated_ids) == set(myth_ids)
        
        # Verify both myths have the same updated data
        for myth_id in myth_ids:
            updated_myth = get_myth(myth_id)
            
            # CRITICAL: Verify new main embedding
            np.testing.assert_array_almost_equal(
                updated_myth["embedding"], expected_new_embedding, decimal=7,
                err_msg=f"CRITICAL: Myth {myth_id} main embedding doesn't match expected"
            )
            
            # CRITICAL: Verify new offsets
            for i, (retrieved, original) in enumerate(zip(updated_myth["offsets"], new_offsets)):
                np.testing.assert_array_almost_equal(
                    retrieved, original, decimal=7,
                    err_msg=f"CRITICAL: Myth {myth_id}, offset {i} doesn't match expected"
                )
            
            # CRITICAL: Verify new weights
            np.testing.assert_array_almost_equal(
                updated_myth["weights"], new_weights, decimal=7,
                err_msg=f"CRITICAL: Myth {myth_id} weights don't match expected"
            )
    
    @pytest.mark.integration
    def test_complex_myth_matrix_operations(self):
        """Test complex myth matrix operations with larger structures."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert multiple mythemes
        mytheme_sentences = [f"Mytheme {i}" for i in range(5)]
        mytheme_embeddings = [
            create_test_embedding([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i], embedding_dim)
            for i in range(1, 6)
        ]
        insert_mythemes_bulk(mytheme_sentences, mytheme_embeddings)
        
        # Get mytheme IDs
        mytheme_ids, _, _ = get_mythemes_bulk()
        
        # Create and insert myths with different numbers of mythemes
        main_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
        ]
        embedding_ids_list = [mytheme_ids[:3], mytheme_ids[2:]]  # 3 and 3 mythemes
        offsets_list = [
            [
                create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
                create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
                create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
            ],
            [
                create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim),
                create_test_embedding([1.7, 1.8, 1.9, 2.0], embedding_dim),
                create_test_embedding([2.1, 2.2, 2.3, 2.4], embedding_dim)
            ]
        ]
        weights_list = [[0.4, 0.3, 0.3], [0.5, 0.3, 0.2]]
        
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Get myth matrices
        myth_matrices = get_myth_matrices(myth_ids)
        
        # Verify matrices
        assert len(myth_matrices) == 2
        
        # Test each matrix
        for i, myth_matrix in enumerate(myth_matrices):
            # Verify matrix properties
            inferred_dim = infer_embedding_dim(myth_matrix)
            assert inferred_dim == embedding_dim, f"Matrix {i} has wrong inferred dimension"
            
            num_themes = num_mythemes(myth_matrix)
            assert num_themes == 3, f"Matrix {i} has wrong number of mythemes"
            
            # Decompose and verify
            embeddings, offsets, weights = decompose_myth_matrix(myth_matrix)
            
            # CRITICAL: Verify embeddings match original mythemes
            # Get the expected mytheme embeddings for this myth
            expected_mytheme_indices = [0, 1, 2] if i == 0 else [2, 3, 4]  # First myth uses mythemes 0,1,2; second uses 2,3,4
            for j, (retrieved, expected_idx) in enumerate(zip(embeddings, expected_mytheme_indices)):
                original = mytheme_embeddings[expected_idx]
                np.testing.assert_array_almost_equal(
                    retrieved, original, decimal=7,
                    err_msg=f"CRITICAL: Matrix {i}, embedding {j} doesn't match original mytheme {expected_idx}"
                )
            
            # CRITICAL: Verify offsets and weights
            for j, (retrieved, original) in enumerate(zip(offsets, offsets_list[i])):
                np.testing.assert_array_almost_equal(
                    retrieved, original, decimal=7,
                    err_msg=f"CRITICAL: Matrix {i}, offset {j} doesn't match original"
                )
            
            np.testing.assert_array_almost_equal(
                weights, weights_list[i], decimal=7,
                err_msg=f"CRITICAL: Matrix {i} weights don't match original"
            )
            
            # Test myth embedding computation
            computed_embedding = compute_myth_embedding(myth_matrix)
            assert computed_embedding.shape == (embedding_dim,), f"Matrix {i} computed embedding has wrong shape" 