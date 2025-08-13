import os
import pytest
import numpy as np
from typing import List, Dict
from datetime import datetime, timezone
import time

from mythologizer_postgres.connectors.myth_store import (
    insert_myth,
    insert_myths_bulk,
    get_myth,
    get_myths_bulk,
    update_myth,
    update_myths_bulk,
    delete_myth,
    delete_myths_bulk,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


def create_test_embedding(base_values, embedding_dim):
    """Create a test embedding array with the correct dimension."""
    # Extend base_values to at least embedding_dim length
    extended_values = base_values + [0.1 * i for i in range(len(base_values), embedding_dim)]
    return np.array(extended_values[:embedding_dim], dtype=np.float32)


def assert_timestamps_valid(created_at, updated_at, expected_created_at=None):
    """
    Assert that timestamps are valid and properly set.
    
    Args:
        created_at: The created_at timestamp from database
        updated_at: The updated_at timestamp from database
        expected_created_at: Expected created_at value (for comparison after updates)
    """
    # Verify timestamps are not None
    assert created_at is not None, "created_at should not be None"
    assert updated_at is not None, "updated_at should not be None"
    
    # Verify they are datetime objects
    assert isinstance(created_at, datetime), "created_at should be a datetime object"
    assert isinstance(updated_at, datetime), "updated_at should be a datetime object"
    
    # Verify created_at is not in the future
    assert created_at <= datetime.now(timezone.utc), "created_at should not be in the future"
    
    # Verify updated_at is not in the future
    assert updated_at <= datetime.now(timezone.utc), "updated_at should not be in the future"
    
    # Verify created_at <= updated_at
    assert created_at <= updated_at, "created_at should be less than or equal to updated_at"
    
    # If we have an expected created_at, verify it matches
    if expected_created_at is not None:
        assert created_at == expected_created_at, "created_at should remain unchanged after updates"


# IMPORTANT: Data Integrity Testing
# 
# The tests in this file verify CRITICAL data integrity for complex nested structures:
# - main_embedding: Single vector with high precision
# - offsets: List of vectors (nested embeddings) with high precision  
# - weights: List of float weights with high precision
# - embedding_ids: List of integer IDs
# - created_at: Creation timestamp
# - updated_at: Last modification timestamp
#
# We use np.testing.assert_array_almost_equal with decimal=7 to account for
# database floating-point precision differences while ensuring high precision
# data integrity for all vector components.


class TestMythStore:
    """Test the myth_store module with real database operations."""
    
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
    def test_insert_and_retrieve_single_myth(self):
        """Test single myth insertion and retrieval with data integrity including timestamps."""
        embedding_dim = get_embedding_dim()
        
        # Create test data with known values for precise comparison
        main_embedding = create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim)
        embedding_ids = [1, 2, 3]
        offsets = [
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
            create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim)
        ]
        weights = [0.25, 0.5, 0.25]
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve myth
        retrieved_myth = get_myth(myth_id)
        
        # Verify myth was found
        assert retrieved_myth is not None, "Myth should be found"
        assert retrieved_myth["id"] == myth_id
        
        # CRITICAL: Verify main embedding integrity
        np.testing.assert_array_almost_equal(
            retrieved_myth["embedding"],
            main_embedding,
            decimal=7,
            err_msg="CRITICAL: Main embedding data integrity failed"
        )
        
        # CRITICAL: Verify embedding_ids integrity
        assert retrieved_myth["embedding_ids"] == embedding_ids, "Embedding IDs should match exactly"
        
        # CRITICAL: Verify offsets integrity (list of embeddings)
        assert len(retrieved_myth["offsets"]) == len(offsets), "Number of offsets should match"
        for i, (retrieved_offset, original_offset) in enumerate(zip(retrieved_myth["offsets"], offsets)):
            np.testing.assert_array_almost_equal(
                retrieved_offset,
                original_offset,
                decimal=7,
                err_msg=f"CRITICAL: Offset {i} data integrity failed"
            )
        
        # CRITICAL: Verify weights integrity
        np.testing.assert_array_almost_equal(
            retrieved_myth["weights"],
            weights,
            decimal=7,
            err_msg="CRITICAL: Weights data integrity failed"
        )
        
        # CRITICAL: Verify timestamp fields exist and are valid
        assert "created_at" in retrieved_myth, "created_at field should be present"
        assert "updated_at" in retrieved_myth, "updated_at field should be present"
        assert_timestamps_valid(retrieved_myth["created_at"], retrieved_myth["updated_at"])
    
    @pytest.mark.integration
    def test_insert_and_retrieve_myths_bulk(self):
        """Test bulk myth insertion and retrieval with data integrity including timestamps."""
        embedding_dim = get_embedding_dim()
        
        # Create test data for 3 myths
        main_embeddings = [
            create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        ]
        
        embedding_ids_list = [
            [1, 2],
            [3, 4, 5],
            [6]
        ]
        
        offsets_list = [
            [
                create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim),
                create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim)
            ],
            [
                create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim),
                create_test_embedding([1.3, 1.4, 1.5, 1.6], embedding_dim),
                create_test_embedding([1.7, 1.8, 1.9, 2.0], embedding_dim)
            ],
            [
                create_test_embedding([2.1, 2.2, 2.3, 2.4], embedding_dim)
            ]
        ]
        
        weights_list = [
            [0.5, 0.5],
            [0.33, 0.33, 0.34],
            [1.0]
        ]
        
        # Insert myths in bulk
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Verify we got the expected number of IDs
        assert len(myth_ids) == 3, f"Expected 3 myth IDs, got {len(myth_ids)}"
        
        # Retrieve all myths
        retrieved_ids, retrieved_main_embeddings, retrieved_embedding_ids_list, retrieved_offsets_list, retrieved_weights_list, created_ats, updated_ats = get_myths_bulk()
        
        # Verify we got the expected number of myths
        assert len(retrieved_ids) == 3, f"Expected 3 myths, got {len(retrieved_ids)}"
        assert len(created_ats) == 3, f"Expected 3 created_at timestamps, got {len(created_ats)}"
        assert len(updated_ats) == 3, f"Expected 3 updated_at timestamps, got {len(updated_ats)}"
        
        # CRITICAL: Verify data integrity for each myth
        for i in range(3):
            # Find the myth in retrieved data
            myth_idx = retrieved_ids.index(myth_ids[i])
            
            # Verify main embedding
            np.testing.assert_array_almost_equal(
                retrieved_main_embeddings[myth_idx],
                main_embeddings[i],
                decimal=7,
                err_msg=f"CRITICAL: Main embedding {i} data integrity failed"
            )
            
            # Verify embedding IDs
            assert retrieved_embedding_ids_list[myth_idx] == embedding_ids_list[i], f"Embedding IDs {i} should match"
            
            # Verify offsets
            assert len(retrieved_offsets_list[myth_idx]) == len(offsets_list[i]), f"Number of offsets {i} should match"
            for j, (retrieved_offset, original_offset) in enumerate(zip(retrieved_offsets_list[myth_idx], offsets_list[i])):
                np.testing.assert_array_almost_equal(
                    retrieved_offset,
                    original_offset,
                    decimal=7,
                    err_msg=f"CRITICAL: Offset {i}.{j} data integrity failed"
                )
            
            # Verify weights
            np.testing.assert_array_almost_equal(
                retrieved_weights_list[myth_idx],
                weights_list[i],
                decimal=7,
                err_msg=f"CRITICAL: Weights {i} data integrity failed"
            )
            
            # CRITICAL: Verify timestamp fields are valid
            assert_timestamps_valid(created_ats[myth_idx], updated_ats[myth_idx])
    
    @pytest.mark.integration
    def test_get_myth_single_not_found(self):
        """Test that get_myth returns None for non-existent ID."""
        result = get_myth(99999)
        assert result is None, "Should return None for non-existent myth"
    
    @pytest.mark.integration
    def test_get_myths_by_ids(self):
        """Test retrieving specific myths by their IDs with timestamp verification."""
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
        
        # Retrieve specific myths by IDs
        retrieved_ids, retrieved_main_embeddings, retrieved_embedding_ids_list, retrieved_offsets_list, retrieved_weights_list, created_ats, updated_ats = get_myths_bulk(myth_ids=myth_ids)
        
        # Verify we got the right number
        assert len(retrieved_ids) == len(myth_ids)
        assert len(created_ats) == len(myth_ids)
        assert len(updated_ats) == len(myth_ids)
        
        # Verify the IDs match
        assert set(retrieved_ids) == set(myth_ids)
        
        # Verify data integrity
        for i, myth_id in enumerate(retrieved_ids):
            original_idx = myth_ids.index(myth_id)
            
            # Verify main embedding
            np.testing.assert_array_almost_equal(
                retrieved_main_embeddings[i],
                main_embeddings[original_idx],
                decimal=7,
                err_msg=f"CRITICAL: Main embedding integrity failed for ID {myth_id}"
            )
            
            # Verify offsets
            for j, (retrieved_offset, original_offset) in enumerate(zip(retrieved_offsets_list[i], offsets_list[original_idx])):
                np.testing.assert_array_almost_equal(
                    retrieved_offset,
                    original_offset,
                    decimal=7,
                    err_msg=f"CRITICAL: Offset integrity failed for ID {myth_id}, offset {j}"
                )
            
            # Verify timestamps are valid
            assert_timestamps_valid(created_ats[i], updated_ats[i])
    
    @pytest.mark.integration
    def test_update_myth_with_timestamp_verification(self):
        """Test updating a myth with data integrity verification and timestamp changes."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert initial myth
        main_embedding = create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim)
        embedding_ids = [1, 2]
        offsets = [
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Get initial timestamps
        initial_myth = get_myth(myth_id)
        initial_created_at = initial_myth["created_at"]
        initial_updated_at = initial_myth["updated_at"]
        
        # Wait a moment to ensure timestamp difference
        time.sleep(0.1)
        
        # Update the myth
        new_main_embedding = create_test_embedding([1.1, 1.2, 1.3, 1.4], embedding_dim)
        new_offsets = [
            create_test_embedding([1.5, 1.6, 1.7, 1.8], embedding_dim),
            create_test_embedding([1.9, 2.0, 2.1, 2.2], embedding_dim)
        ]
        new_weights = [0.6, 0.4]
        
        success = update_myth(myth_id, main_embedding=new_main_embedding, offsets=new_offsets, weights=new_weights)
        assert success, "Update should succeed"
        
        # Retrieve updated myth
        updated_myth = get_myth(myth_id)
        
        # CRITICAL: Verify updated main embedding
        np.testing.assert_array_almost_equal(
            updated_myth["embedding"],
            new_main_embedding,
            decimal=7,
            err_msg="CRITICAL: Updated main embedding integrity failed"
        )
        
        # CRITICAL: Verify updated offsets
        for i, (retrieved_offset, original_offset) in enumerate(zip(updated_myth["offsets"], new_offsets)):
            np.testing.assert_array_almost_equal(
                retrieved_offset,
                original_offset,
                decimal=7,
                err_msg=f"CRITICAL: Updated offset {i} integrity failed"
            )
        
        # CRITICAL: Verify updated weights
        np.testing.assert_array_almost_equal(
            updated_myth["weights"],
            new_weights,
            decimal=7,
            err_msg="CRITICAL: Updated weights integrity failed"
        )
        
        # Verify embedding_ids remained unchanged
        assert updated_myth["embedding_ids"] == embedding_ids, "Embedding IDs should remain unchanged"
        
        # CRITICAL: Verify timestamp behavior
        # created_at should remain unchanged
        assert updated_myth["created_at"] == initial_created_at, "created_at should remain unchanged after update"
        
        # updated_at should be newer than the initial updated_at
        assert updated_myth["updated_at"] > initial_updated_at, "updated_at should be newer after update"
        
        # Verify timestamps are still valid
        assert_timestamps_valid(updated_myth["created_at"], updated_myth["updated_at"], initial_created_at)
    
    @pytest.mark.integration
    def test_update_myths_bulk_with_timestamp_verification(self):
        """Test bulk updating of myths with data integrity verification and timestamp changes."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert initial myths
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
        
        # Get initial timestamps
        initial_ids, _, _, _, _, initial_created_ats, initial_updated_ats = get_myths_bulk(myth_ids=myth_ids)
        
        # Wait a moment to ensure timestamp difference
        time.sleep(0.1)
        
        # Update myths in bulk
        new_main_embeddings = [
            create_test_embedding([1.1, 1.2, 1.3, 1.4], embedding_dim),
            create_test_embedding([1.5, 1.6, 1.7, 1.8], embedding_dim)
        ]
        new_weights_list = [[0.6, 0.4], [0.7, 0.3]]
        
        updated_count = update_myths_bulk(myth_ids, main_embeddings=new_main_embeddings, weights_list=new_weights_list)
        assert updated_count == 2, "Should update 2 myths"
        
        # Retrieve updated myths
        retrieved_ids, retrieved_main_embeddings, _, _, retrieved_weights_list, retrieved_created_ats, retrieved_updated_ats = get_myths_bulk(myth_ids=myth_ids)
        
        # CRITICAL: Verify updated data integrity
        for i, myth_id in enumerate(retrieved_ids):
            original_idx = myth_ids.index(myth_id)
            initial_idx = initial_ids.index(myth_id)
            
            # Verify updated main embedding
            np.testing.assert_array_almost_equal(
                retrieved_main_embeddings[i],
                new_main_embeddings[original_idx],
                decimal=7,
                err_msg=f"CRITICAL: Bulk updated main embedding {i} integrity failed"
            )
            
            # Verify updated weights
            np.testing.assert_array_almost_equal(
                retrieved_weights_list[i],
                new_weights_list[original_idx],
                decimal=7,
                err_msg=f"CRITICAL: Bulk updated weights {i} integrity failed"
            )
            
            # CRITICAL: Verify timestamp behavior
            # created_at should remain unchanged
            assert retrieved_created_ats[i] == initial_created_ats[initial_idx], f"created_at should remain unchanged for myth {myth_id}"
            
            # updated_at should be newer than the initial updated_at
            assert retrieved_updated_ats[i] > initial_updated_ats[initial_idx], f"updated_at should be newer for myth {myth_id}"
            
            # Verify timestamps are still valid
            assert_timestamps_valid(retrieved_created_ats[i], retrieved_updated_ats[i], initial_created_ats[initial_idx])
    
    @pytest.mark.integration
    def test_delete_myth(self):
        """Test deleting a myth."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert myth
        main_embedding = create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim)
        embedding_ids = [1, 2]
        offsets = [
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Verify myth exists
        assert get_myth(myth_id) is not None, "Myth should exist before deletion"
        
        # Delete myth
        success = delete_myth(myth_id)
        assert success, "Delete should succeed"
        
        # Verify myth is gone
        assert get_myth(myth_id) is None, "Myth should not exist after deletion"
    
    @pytest.mark.integration
    def test_delete_myths_bulk(self):
        """Test bulk deleting of myths."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert myths
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
        
        # Verify myths exist
        for myth_id in myth_ids:
            assert get_myth(myth_id) is not None, f"Myth {myth_id} should exist before deletion"
        
        # Delete myths in bulk
        deleted_count = delete_myths_bulk(myth_ids)
        assert deleted_count == 2, "Should delete 2 myths"
        
        # Verify myths are gone
        for myth_id in myth_ids:
            assert get_myth(myth_id) is None, f"Myth {myth_id} should not exist after deletion"
    
    @pytest.mark.integration
    def test_critical_nested_embedding_precision(self):
        """CRITICAL: Test high-precision nested embedding data integrity with timestamps."""
        embedding_dim = get_embedding_dim()
        
        # Create test data with high-precision values
        main_embedding = create_test_embedding([0.123456789012345, 0.987654321098765, 0.555555555555555, 0.111111111111111], embedding_dim)
        embedding_ids = [1, 2, 3]
        offsets = [
            create_test_embedding([0.222222222222222, 0.333333333333333, 0.444444444444444, 0.555555555555555], embedding_dim),
            create_test_embedding([0.666666666666666, 0.777777777777777, 0.888888888888888, 0.999999999999999], embedding_dim),
            create_test_embedding([0.101010101010101, 0.202020202020202, 0.303030303030303, 0.404040404040404], embedding_dim)
        ]
        weights = [0.123456789012345, 0.987654321098765, 0.555555555555555]
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve myth
        retrieved_myth = get_myth(myth_id)
        
        # CRITICAL: Verify main embedding precision
        np.testing.assert_array_almost_equal(
            retrieved_myth["embedding"],
            main_embedding,
            decimal=7,
            err_msg="CRITICAL: Main embedding precision failed"
        )
        
        # CRITICAL: Verify offsets precision (nested embeddings)
        for i, (retrieved_offset, original_offset) in enumerate(zip(retrieved_myth["offsets"], offsets)):
            np.testing.assert_array_almost_equal(
                retrieved_offset,
                original_offset,
                decimal=7,
                err_msg=f"CRITICAL: Offset {i} precision failed"
            )
        
        # CRITICAL: Verify weights precision
        np.testing.assert_array_almost_equal(
            retrieved_myth["weights"],
            weights,
            decimal=7,
            err_msg="CRITICAL: Weights precision failed"
        )
        
        # CRITICAL: Verify timestamp fields are valid
        assert_timestamps_valid(retrieved_myth["created_at"], retrieved_myth["updated_at"])
    
    @pytest.mark.integration
    def test_large_nested_structures(self):
        """Test with larger nested structures (4 < n < 20 as requested) including timestamps."""
        embedding_dim = get_embedding_dim()
        
        # Create 5 myths with varying numbers of nested embeddings
        main_embeddings = []
        embedding_ids_list = []
        offsets_list = []
        weights_list = []
        
        for i in range(5):
            # Each myth has i+1 nested embeddings
            num_nested = i + 1
            main_embeddings.append(np.random.rand(embedding_dim).astype(np.float32))
            embedding_ids_list.append(list(range(i*10, i*10 + num_nested)))
            
            # Create nested offsets
            myth_offsets = []
            myth_weights = []
            for j in range(num_nested):
                myth_offsets.append(np.random.rand(embedding_dim).astype(np.float32))
                myth_weights.append(float(np.random.rand()))
            
            offsets_list.append(myth_offsets)
            weights_list.append(myth_weights)
        
        # Insert myths in bulk
        myth_ids = insert_myths_bulk(main_embeddings, embedding_ids_list, offsets_list, weights_list)
        
        # Retrieve all myths
        retrieved_ids, retrieved_main_embeddings, retrieved_embedding_ids_list, retrieved_offsets_list, retrieved_weights_list, created_ats, updated_ats = get_myths_bulk()
        
        # Verify we got the expected number
        assert len(retrieved_ids) == 5, f"Expected 5 myths, got {len(retrieved_ids)}"
        assert len(created_ats) == 5, f"Expected 5 created_at timestamps, got {len(created_ats)}"
        assert len(updated_ats) == 5, f"Expected 5 updated_at timestamps, got {len(updated_ats)}"
        
        # CRITICAL: Verify data integrity for each myth
        for i, myth_id in enumerate(retrieved_ids):
            original_idx = myth_ids.index(myth_id)
            
            # Verify main embedding
            np.testing.assert_array_almost_equal(
                retrieved_main_embeddings[i],
                main_embeddings[original_idx],
                decimal=7,
                err_msg=f"CRITICAL: Large structure main embedding {i} integrity failed"
            )
            
            # Verify number of nested embeddings
            assert len(retrieved_offsets_list[i]) == len(offsets_list[original_idx]), f"Nested count mismatch for myth {i}"
            
            # Verify each nested embedding
            for j, (retrieved_offset, original_offset) in enumerate(zip(retrieved_offsets_list[i], offsets_list[original_idx])):
                np.testing.assert_array_almost_equal(
                    retrieved_offset,
                    original_offset,
                    decimal=7,
                    err_msg=f"CRITICAL: Large structure offset {i}.{j} integrity failed"
                )
            
            # Verify weights
            np.testing.assert_array_almost_equal(
                retrieved_weights_list[i],
                weights_list[original_idx],
                decimal=7,
                err_msg=f"CRITICAL: Large structure weights {i} integrity failed"
            )
            
            # CRITICAL: Verify timestamp fields are valid
            assert_timestamps_valid(created_ats[i], updated_ats[i])
    
    @pytest.mark.integration
    def test_timestamp_behavior_on_multiple_updates(self):
        """Test that timestamps behave correctly across multiple updates."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert initial myth
        main_embedding = create_test_embedding([0.1, 0.2, 0.3, 0.4], embedding_dim)
        embedding_ids = [1, 2]
        offsets = [
            create_test_embedding([0.5, 0.6, 0.7, 0.8], embedding_dim),
            create_test_embedding([0.9, 1.0, 1.1, 1.2], embedding_dim)
        ]
        weights = [0.5, 0.5]
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Get initial timestamps
        initial_myth = get_myth(myth_id)
        initial_created_at = initial_myth["created_at"]
        initial_updated_at = initial_myth["updated_at"]
        
        # Perform multiple updates
        update_timestamps = []
        for i in range(3):
            time.sleep(0.1)  # Ensure timestamp difference
            
            # Update with new weights
            new_weights = [0.3 + i*0.1, 0.7 - i*0.1]
            success = update_myth(myth_id, weights=new_weights)
            assert success, f"Update {i+1} should succeed"
            
            # Get updated myth and record timestamp
            updated_myth = get_myth(myth_id)
            update_timestamps.append(updated_myth["updated_at"])
            
            # Verify created_at remains unchanged
            assert updated_myth["created_at"] == initial_created_at, f"created_at should remain unchanged after update {i+1}"
            
            # Verify updated_at is newer than previous
            if i > 0:
                assert updated_myth["updated_at"] > update_timestamps[i-1], f"updated_at should be newer after update {i+1}"
        
        # Verify final timestamps are valid
        final_myth = get_myth(myth_id)
        assert_timestamps_valid(final_myth["created_at"], final_myth["updated_at"], initial_created_at)
        
        # Verify all update timestamps are newer than initial
        for i, update_timestamp in enumerate(update_timestamps):
            assert update_timestamp > initial_updated_at, f"Update {i+1} timestamp should be newer than initial"
    
    @pytest.mark.integration
    def test_empty_bulk_retrieval(self):
        """Test that bulk retrieval returns empty lists when no myths exist."""
        # Retrieve all myths when none exist
        retrieved_ids, retrieved_main_embeddings, retrieved_embedding_ids_list, retrieved_offsets_list, retrieved_weights_list, created_ats, updated_ats = get_myths_bulk()
        
        # Verify all lists are empty
        assert len(retrieved_ids) == 0, "Should return empty list for IDs"
        assert len(retrieved_main_embeddings) == 0, "Should return empty list for main embeddings"
        assert len(retrieved_embedding_ids_list) == 0, "Should return empty list for embedding IDs"
        assert len(retrieved_offsets_list) == 0, "Should return empty list for offsets"
        assert len(retrieved_weights_list) == 0, "Should return empty list for weights"
        assert len(created_ats) == 0, "Should return empty list for created_at timestamps"
        assert len(updated_ats) == 0, "Should return empty list for updated_at timestamps" 