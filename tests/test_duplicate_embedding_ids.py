#!/usr/bin/env python3
"""
Comprehensive tests for duplicate embedding ID handling behavior.

This test suite verifies that all myth insertion and update functions
properly handle duplicate embedding IDs by removing them along with
their corresponding offsets and weights, keeping only the first occurrence.
"""

import pytest
import numpy as np
from typing import List, Tuple

from mythologizer_postgres.connectors import (
    insert_myth,
    insert_myths_bulk,
    update_myth,
    update_myths_bulk,
    get_myth,
    get_myths_bulk,
    insert_mythemes_bulk,
    add_myths_bulk,
    update_myth_with_retention
)
from mythologizer_postgres.db import clear_all_rows, psycopg_connection
from mythologizer_postgres.connectors.mythicalgebra.mythic_algebra_connector import (
    compose_myth_matrix,
    compute_myth_embedding
)


class TestDuplicateEmbeddingIds:
    """Test duplicate embedding ID handling in all myth functions."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear database before and after each test."""
        clear_all_rows()
        yield
        clear_all_rows()
    
    @pytest.fixture
    def sample_mythemes(self):
        """Create sample mythemes for testing."""
        embedding_dim = 4
        n_mythemes = 10
        
        # Create random mytheme embeddings
        mytheme_embeddings = [np.random.rand(embedding_dim).tolist() for _ in range(n_mythemes)]
        mytheme_names = [f"mytheme_{i}" for i in range(n_mythemes)]
        
        # Insert mythemes
        insert_mythemes_bulk(mytheme_names, mytheme_embeddings)
        
        # Get the created mytheme IDs
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM mythemes ORDER BY id")
                mytheme_ids = [row[0] for row in cur.fetchall()]
        
        return mytheme_ids
    
    def create_myth_data_with_duplicates(self, mytheme_ids: List[int], duplicates: List[int]) -> Tuple[List[int], List[np.ndarray], List[float]]:
        """Create myth data with specified duplicate embedding IDs."""
        embedding_dim = 4
        
        # Create embedding_ids with duplicates
        embedding_ids = []
        for i in range(len(mytheme_ids)):
            embedding_ids.append(mytheme_ids[i])
            if i in duplicates:
                # Add duplicate
                embedding_ids.append(mytheme_ids[i])
        
        # Create corresponding offsets and weights
        offsets = []
        weights = []
        for i, emb_id in enumerate(embedding_ids):
            offsets.append(np.random.rand(embedding_dim).astype(np.float32))
            weights.append(np.float32(np.random.rand()))
        
        return embedding_ids, offsets, weights
    
    def test_insert_myth_removes_duplicates(self, sample_mythemes):
        """Test that insert_myth removes duplicate embedding IDs."""
        # Create myth data with duplicates (mytheme 0 appears twice)
        embedding_ids, offsets, weights = self.create_myth_data_with_duplicates(
            sample_mythemes[:5], [0]  # Duplicate the first mytheme
        )
        
        # Create main embedding - use the original length before deduplication for the matrix
        embeddings_array = np.array([np.random.rand(4) for _ in range(len(embedding_ids))], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        main_embedding = compute_myth_embedding(myth_matrix)
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve the myth
        myth = get_myth(myth_id)
        assert myth is not None
        
        # Verify duplicates were removed
        retrieved_embedding_ids = myth["embedding_ids"]
        assert len(retrieved_embedding_ids) == len(set(embedding_ids)), "Duplicates should be removed"
        assert len(retrieved_embedding_ids) == 5, "Should have 5 unique embedding IDs"
        
        # Verify the first occurrence was kept
        assert retrieved_embedding_ids[0] == embedding_ids[0], "First occurrence should be kept"
        
        # Verify corresponding offsets and weights were also adjusted
        retrieved_offsets = myth["offsets"]
        retrieved_weights = myth["weights"]
        assert len(retrieved_offsets) == len(retrieved_embedding_ids)
        assert len(retrieved_weights) == len(retrieved_embedding_ids)
    
    def test_insert_myths_bulk_removes_duplicates(self, sample_mythemes):
        """Test that insert_myths_bulk removes duplicate embedding IDs."""
        # Create multiple myths with duplicates
        myths_data = []
        main_embeddings = []
        
        for i in range(3):
            # Each myth has different duplicates
            embedding_ids, offsets, weights = self.create_myth_data_with_duplicates(
                sample_mythemes[i:i+4], [0, 2]  # Duplicate first and third mythemes
            )
            
            # Create main embedding - use the original length before deduplication for the matrix
            embeddings_array = np.array([np.random.rand(4) for _ in range(len(embedding_ids))], dtype=np.float32)
            myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
            main_embedding = compute_myth_embedding(myth_matrix)
            
            myths_data.append((embedding_ids, offsets, weights))
            main_embeddings.append(main_embedding)
        
        # Insert myths in bulk
        myth_ids = insert_myths_bulk(main_embeddings, 
                                   [myth[0] for myth in myths_data],
                                   [myth[1] for myth in myths_data],
                                   [myth[2] for myth in myths_data])
        
        # Verify all myths were inserted
        assert len(myth_ids) == 3
        
        # Check each myth for duplicate removal
        for i, myth_id in enumerate(myth_ids):
            myth = get_myth(myth_id)
            assert myth is not None
            
            original_embedding_ids = myths_data[i][0]
            retrieved_embedding_ids = myth["embedding_ids"]
            
            # Should have removed duplicates
            assert len(retrieved_embedding_ids) == len(set(original_embedding_ids))
            assert len(retrieved_embedding_ids) == 4, f"Myth {i} should have 4 unique embedding IDs"
    
    def test_update_myth_removes_duplicates(self, sample_mythemes):
        """Test that update_myth removes duplicate embedding IDs."""
        # First create a myth without duplicates
        embedding_ids = sample_mythemes[:3]
        offsets = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        weights = [np.float32(np.random.rand()) for _ in range(3)]
        
        embeddings_array = np.array([np.random.rand(4) for _ in range(3)], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        main_embedding = compute_myth_embedding(myth_matrix)
        
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Now update with duplicates
        new_embedding_ids = sample_mythemes[2:6] + [sample_mythemes[2]]  # Duplicate mytheme 2
        new_offsets = [np.random.rand(4).astype(np.float32) for _ in range(5)]
        new_weights = [np.float32(np.random.rand()) for _ in range(5)]
        
        success = update_myth(myth_id, embedding_ids=new_embedding_ids, 
                            offsets=new_offsets, weights=new_weights)
        assert success
        
        # Retrieve the updated myth
        myth = get_myth(myth_id)
        assert myth is not None
        
        # Verify duplicates were removed
        retrieved_embedding_ids = myth["embedding_ids"]
        assert len(retrieved_embedding_ids) == len(set(new_embedding_ids))
        assert len(retrieved_embedding_ids) == 4, "Should have 4 unique embedding IDs"
    
    def test_update_myths_bulk_removes_duplicates(self, sample_mythemes):
        """Test that update_myths_bulk removes duplicate embedding IDs."""
        # First create myths without duplicates
        myth_ids = []
        for i in range(2):
            embedding_ids = sample_mythemes[i:i+3]
            offsets = [np.random.rand(4).astype(np.float32) for _ in range(3)]
            weights = [np.float32(np.random.rand()) for _ in range(3)]
            
            embeddings_array = np.array([np.random.rand(4) for _ in range(3)], dtype=np.float32)
            myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
            main_embedding = compute_myth_embedding(myth_matrix)
            
            myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
            myth_ids.append(myth_id)
        
        # Now update with duplicates
        new_embedding_ids_list = [
            sample_mythemes[3:6] + [sample_mythemes[3]],  # Duplicate first mytheme
            sample_mythemes[6:9] + [sample_mythemes[6], sample_mythemes[7]]  # Duplicate first two mythemes
        ]
        new_offsets_list = [
            [np.random.rand(4).astype(np.float32) for _ in range(4)],
            [np.random.rand(4).astype(np.float32) for _ in range(5)]
        ]
        new_weights_list = [
            [np.float32(np.random.rand()) for _ in range(4)],
            [np.float32(np.random.rand()) for _ in range(5)]
        ]
        
        updated_count = update_myths_bulk(myth_ids, 
                                        embedding_ids_list=new_embedding_ids_list,
                                        offsets_list=new_offsets_list,
                                        weights_list=new_weights_list)
        assert updated_count == 2
        
        # Verify duplicates were removed
        for i, myth_id in enumerate(myth_ids):
            myth = get_myth(myth_id)
            assert myth is not None
            
            original_embedding_ids = new_embedding_ids_list[i]
            retrieved_embedding_ids = myth["embedding_ids"]
            
            # Should have removed duplicates
            assert len(retrieved_embedding_ids) == len(set(original_embedding_ids))
    
    def test_add_myths_bulk_removes_duplicates(self, sample_mythemes):
        """Test that add_myths_bulk removes duplicate embedding IDs."""
        # Create myth data with duplicates
        myths_data = []
        
        for i in range(2):
            # Create embedding_ids with duplicates
            embedding_ids = sample_mythemes[i:i+4] + [sample_mythemes[i]]  # Duplicate first mytheme
            offsets = [np.random.rand(4).astype(np.float32) for _ in range(5)]
            weights = [np.float32(np.random.rand()) for _ in range(5)]
            
            myths_data.append((embedding_ids, offsets, weights))
        
        # Add myths using add_myths_bulk
        myth_ids = add_myths_bulk(myths_data)
        assert len(myth_ids) == 2
        
        # Verify duplicates were removed
        for i, myth_id in enumerate(myth_ids):
            myth = get_myth(myth_id)
            assert myth is not None
            
            original_embedding_ids = myths_data[i][0]
            retrieved_embedding_ids = myth["embedding_ids"]
            
            # Should have removed duplicates
            assert len(retrieved_embedding_ids) == len(set(original_embedding_ids))
            assert len(retrieved_embedding_ids) == 4, f"Myth {i} should have 4 unique embedding IDs"
    
    def test_update_myth_with_retention_removes_duplicates(self, sample_mythemes):
        """Test that update_myth_with_retention removes duplicate embedding IDs."""
        # First create a myth
        embedding_ids = sample_mythemes[:3]
        offsets = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        weights = [np.float32(np.random.rand()) for _ in range(3)]
        
        embeddings_array = np.array([np.random.rand(4) for _ in range(3)], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        
        myth_id = insert_myth(compute_myth_embedding(myth_matrix), embedding_ids, offsets, weights)
        
        # Create agent for testing
        with psycopg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO agents (name, memory_size) VALUES (%s, %s) RETURNING id", 
                           ("test_agent", 100))
                agent_id = cur.fetchone()[0]
                conn.commit()
        
        # Add the myth to the agent's memory first
        from mythologizer_postgres.connectors.agent_store import insert_agent_myth_safe
        insert_agent_myth_safe(myth_id, agent_id, retention=1.0)
        
        # Now update with duplicates
        new_embedding_ids = sample_mythemes[3:6] + [sample_mythemes[3]]  # Duplicate first mytheme
        new_offsets = [np.random.rand(4).astype(np.float32) for _ in range(4)]
        new_weights = [np.float32(np.random.rand()) for _ in range(4)]
        
        new_embeddings_array = np.array([np.random.rand(4) for _ in range(4)], dtype=np.float32)
        new_myth_matrix = compose_myth_matrix(new_embeddings_array, new_offsets, new_weights)
        
        success = update_myth_with_retention(agent_id, myth_id, new_myth_matrix, 
                                           new_embedding_ids, retention=0.8)
        assert success
        
        # Verify duplicates were removed
        myth = get_myth(myth_id)
        assert myth is not None
        
        retrieved_embedding_ids = myth["embedding_ids"]
        assert len(retrieved_embedding_ids) == len(set(new_embedding_ids))
        assert len(retrieved_embedding_ids) == 3, "Should have 3 unique embedding IDs"
    
    def test_multiple_duplicates_handling(self, sample_mythemes):
        """Test handling of multiple duplicates in the same myth."""
        # Create embedding_ids with multiple duplicates
        embedding_ids = [
            sample_mythemes[0],  # First occurrence
            sample_mythemes[1],  # First occurrence
            sample_mythemes[0],  # Duplicate of 0
            sample_mythemes[2],  # First occurrence
            sample_mythemes[1],  # Duplicate of 1
            sample_mythemes[0],  # Another duplicate of 0
        ]
        
        offsets = [np.random.rand(4).astype(np.float32) for _ in range(6)]
        weights = [np.float32(np.random.rand()) for _ in range(6)]
        
        embeddings_array = np.array([np.random.rand(4) for _ in range(6)], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        main_embedding = compute_myth_embedding(myth_matrix)
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve the myth
        myth = get_myth(myth_id)
        assert myth is not None
        
        # Verify duplicates were removed
        retrieved_embedding_ids = myth["embedding_ids"]
        expected_unique_ids = [sample_mythemes[0], sample_mythemes[1], sample_mythemes[2]]
        
        assert len(retrieved_embedding_ids) == 3
        assert retrieved_embedding_ids == expected_unique_ids
        
        # Verify corresponding offsets and weights were adjusted
        retrieved_offsets = myth["offsets"]
        retrieved_weights = myth["weights"]
        assert len(retrieved_offsets) == 3
        assert len(retrieved_weights) == 3
    
    def test_no_duplicates_unchanged(self, sample_mythemes):
        """Test that myths without duplicates remain unchanged."""
        # Create myth data without duplicates
        embedding_ids = sample_mythemes[:4]
        offsets = [np.random.rand(4).astype(np.float32) for _ in range(4)]
        weights = [np.float32(np.random.rand()) for _ in range(4)]
        
        embeddings_array = np.array([np.random.rand(4) for _ in range(4)], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        main_embedding = compute_myth_embedding(myth_matrix)
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve the myth
        myth = get_myth(myth_id)
        assert myth is not None
        
        # Verify no changes were made
        retrieved_embedding_ids = myth["embedding_ids"]
        assert retrieved_embedding_ids == embedding_ids
        assert len(retrieved_embedding_ids) == 4
    
    def test_consecutive_duplicates(self, sample_mythemes):
        """Test handling of consecutive duplicates."""
        # Create embedding_ids with consecutive duplicates
        embedding_ids = [
            sample_mythemes[0],
            sample_mythemes[0],  # Consecutive duplicate
            sample_mythemes[1],
            sample_mythemes[1],  # Consecutive duplicate
            sample_mythemes[2],
        ]
        
        offsets = [np.random.rand(4).astype(np.float32) for _ in range(5)]
        weights = [np.float32(np.random.rand()) for _ in range(5)]
        
        embeddings_array = np.array([np.random.rand(4) for _ in range(5)], dtype=np.float32)
        myth_matrix = compose_myth_matrix(embeddings_array, offsets, weights)
        main_embedding = compute_myth_embedding(myth_matrix)
        
        # Insert myth
        myth_id = insert_myth(main_embedding, embedding_ids, offsets, weights)
        
        # Retrieve the myth
        myth = get_myth(myth_id)
        assert myth is not None
        
        # Verify duplicates were removed
        retrieved_embedding_ids = myth["embedding_ids"]
        expected_unique_ids = [sample_mythemes[0], sample_mythemes[1], sample_mythemes[2]]
        
        assert len(retrieved_embedding_ids) == 3
        assert retrieved_embedding_ids == expected_unique_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
