import os
import pytest
import numpy as np
from typing import List, Tuple

from mythologizer_postgres.connectors.mytheme_store import (
    insert_mythemes_bulk,
    get_mythemes_bulk,
    get_mytheme,
)


def get_embedding_dim():
    """Get embedding dimension from environment variable."""
    return int(os.getenv('EMBEDDING_DIM', '4'))


# IMPORTANT: Data Integrity Testing
# 
# The tests in this file verify CRITICAL embedding data integrity. We discovered that
# database storage and retrieval of floating-point embeddings introduces small precision
# differences on the order of 10^-8. This is normal behavior for database floating-point
# operations.
#
# We use np.testing.assert_array_almost_equal with decimal=7 to account for these
# differences while still ensuring high precision data integrity. This means we verify
# that embeddings are preserved to 7 decimal places, which is more than sufficient
# for machine learning applications while being realistic for database operations.


class TestMythemeStore:
    """Test the mytheme_store module with real database operations."""
    
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
    def test_insert_and_retrieve_mythemes_bulk(self):
        """Test bulk insertion and retrieval of mythemes."""
        embedding_dim = get_embedding_dim()
        
        # Create test data: 10 random embeddings and sentences
        num_mythemes = 10
        sentences = [f"Test sentence {i} for mytheme store" for i in range(num_mythemes)]
        embeddings = np.random.rand(num_mythemes, embedding_dim).tolist()
        
        # Insert mythemes in bulk
        insert_mythemes_bulk(sentences, embeddings)
        
        # Retrieve all mythemes
        ids, retrieved_sentences, retrieved_embeddings = get_mythemes_bulk()
        
        # Verify we got the expected number of mythemes
        assert len(ids) == num_mythemes, f"Expected {num_mythemes} mythemes, got {len(ids)}"
        assert len(retrieved_sentences) == num_mythemes
        assert retrieved_embeddings.shape == (num_mythemes, embedding_dim)
        
        # CRITICAL: Verify the data matches exactly
        for i in range(num_mythemes):
            assert retrieved_sentences[i] == sentences[i], f"Sentence mismatch at index {i}"
            
            # CRITICAL: Test with high precision (accounting for database floating-point precision)
            np.testing.assert_array_almost_equal(
                retrieved_embeddings[i], 
                embeddings[i], 
                decimal=7,  # High precision - accounts for typical database floating-point precision
                err_msg=f"CRITICAL: Embedding data integrity failed at index {i}. "
                       f"Original: {embeddings[i]}, Retrieved: {retrieved_embeddings[i]}"
            )
    
    @pytest.mark.integration
    def test_insert_and_retrieve_mythemes_single(self):
        """Test single insertion and retrieval of mythemes."""
        embedding_dim = get_embedding_dim()
        
        # Create test data for single insertions
        sentences = [
            "Single test sentence 1",
            "Single test sentence 2"
        ]
        embeddings = np.random.rand(2, embedding_dim).tolist()
        
        # Insert mythemes one by one (using bulk with single items)
        for sentence, embedding in zip(sentences, embeddings):
            insert_mythemes_bulk([sentence], [embedding])
        
        # Get all mythemes to find the IDs
        all_ids, all_sentences, all_embeddings = get_mythemes_bulk()
        
        # Find our test sentences
        test_ids = []
        for sentence in sentences:
            try:
                idx = all_sentences.index(sentence)
                test_ids.append(all_ids[idx])
            except ValueError:
                pytest.fail(f"Test sentence '{sentence}' not found in database")
        
        # Retrieve each mytheme individually and verify CRITICAL data integrity
        for i, theme_id in enumerate(test_ids):
            retrieved_id, retrieved_sentence, retrieved_embedding = get_mytheme(theme_id)
            
            assert retrieved_id == theme_id
            assert retrieved_sentence == sentences[i]
            
            # CRITICAL: Verify embedding match with high precision
            np.testing.assert_array_almost_equal(
                retrieved_embedding, 
                embeddings[i], 
                decimal=7,  # High precision - accounts for typical database floating-point precision
                err_msg=f"CRITICAL: Single retrieval embedding data integrity failed for ID {theme_id}. "
                       f"Original: {embeddings[i]}, Retrieved: {retrieved_embedding}"
            )
    
    @pytest.mark.integration
    def test_mixed_bulk_and_single_operations(self):
        """Test mixing bulk and single operations."""
        embedding_dim = get_embedding_dim()
        
        # Create test data
        bulk_sentences = [f"Bulk sentence {i}" for i in range(5)]
        bulk_embeddings = np.random.rand(5, embedding_dim).tolist()
        
        single_sentences = [f"Single sentence {i}" for i in range(2)]
        single_embeddings = np.random.rand(2, embedding_dim).tolist()
        
        # Insert bulk mythemes
        insert_mythemes_bulk(bulk_sentences, bulk_embeddings)
        
        # Insert single mythemes
        for sentence, embedding in zip(single_sentences, single_embeddings):
            insert_mythemes_bulk([sentence], [embedding])
        
        # Get all mythemes
        all_ids, all_sentences, all_embeddings = get_mythemes_bulk()
        
        # Verify total count
        expected_total = len(bulk_sentences) + len(single_sentences)
        assert len(all_ids) == expected_total, f"Expected {expected_total} mythemes, got {len(all_ids)}"
        
        # Verify all sentences are present
        all_expected_sentences = bulk_sentences + single_sentences
        for sentence in all_expected_sentences:
            assert sentence in all_sentences, f"Sentence '{sentence}' not found"
    
    @pytest.mark.integration
    def test_get_mythemes_by_ids(self):
        """Test retrieving specific mythemes by their IDs."""
        embedding_dim = get_embedding_dim()
        
        # Create and insert test data
        sentences = [f"ID test sentence {i}" for i in range(8)]
        embeddings = np.random.rand(8, embedding_dim).tolist()
        insert_mythemes_bulk(sentences, embeddings)
        
        # Get all IDs first
        all_ids, _, _ = get_mythemes_bulk()
        
        # Select some specific IDs to retrieve
        selected_ids = all_ids[1:4]  # Get IDs 1, 2, 3
        
        # Retrieve specific mythemes by IDs
        retrieved_ids, retrieved_sentences, retrieved_embeddings = get_mythemes_bulk(ids=selected_ids)
        
        # Verify we got the right number
        assert len(retrieved_ids) == len(selected_ids)
        
        # Verify the IDs match
        assert set(retrieved_ids) == set(selected_ids)
        
        # Verify the data corresponds to the correct IDs
        for i, theme_id in enumerate(retrieved_ids):
            original_idx = all_ids.index(theme_id)
            assert retrieved_sentences[i] == sentences[original_idx]
            np.testing.assert_array_almost_equal(
                retrieved_embeddings[i], 
                embeddings[original_idx], 
                decimal=6
            )
    
    @pytest.mark.integration
    def test_get_mytheme_single_not_found(self):
        """Test that get_mytheme raises KeyError for non-existent ID."""
        with pytest.raises(KeyError, match="mytheme 99999 not found"):
            get_mytheme(99999)
    
    @pytest.mark.integration
    def test_numpy_vs_list_embeddings(self):
        """Test that embeddings can be retrieved as both numpy arrays and lists."""
        embedding_dim = get_embedding_dim()
        
        # Create test data
        sentences = ["Numpy test sentence"]
        embeddings = np.random.rand(1, embedding_dim).tolist()
        
        # Insert test data
        insert_mythemes_bulk(sentences, embeddings)
        
        # Get all mythemes to find the ID
        all_ids, _, _ = get_mythemes_bulk()
        test_id = all_ids[-1]  # Get the last inserted ID
        
        # Test numpy format (default)
        id_np, sentence_np, embedding_np = get_mytheme(test_id, as_numpy=True)
        assert isinstance(embedding_np, np.ndarray)
        assert embedding_np.shape == (embedding_dim,)
        
        # Test list format
        id_list, sentence_list, embedding_list = get_mytheme(test_id, as_numpy=False)
        assert isinstance(embedding_list, list)
        assert len(embedding_list) == embedding_dim
        
        # Verify they contain the same data
        np.testing.assert_array_almost_equal(embedding_np, embedding_list, decimal=6)
    
    @pytest.mark.integration
    def test_bulk_retrieval_numpy_vs_list(self):
        """Test bulk retrieval with both numpy and list formats."""
        embedding_dim = get_embedding_dim()
        
        # Create test data
        sentences = [f"Bulk format test {i}" for i in range(3)]
        embeddings = np.random.rand(3, embedding_dim).tolist()
        
        # Insert test data
        insert_mythemes_bulk(sentences, embeddings)
        
        # Test numpy format (default)
        ids_np, sentences_np, embeddings_np = get_mythemes_bulk(as_numpy=True)
        assert isinstance(embeddings_np, np.ndarray)
        assert embeddings_np.shape == (len(sentences), embedding_dim)
        
        # Test list format
        ids_list, sentences_list, embeddings_list = get_mythemes_bulk(as_numpy=False)
        assert isinstance(embeddings_list, list)
        assert len(embeddings_list) == len(sentences)
        
        # Verify they contain the same data
        np.testing.assert_array_almost_equal(embeddings_np, embeddings_list, decimal=6)
    
    @pytest.mark.integration
    def test_large_bulk_operations(self):
        """Test operations with a larger number of mythemes."""
        embedding_dim = get_embedding_dim()
        
        # Create 15 mythemes (between 4 and 20 as requested)
        num_mythemes = 15
        sentences = [f"Large bulk test sentence {i}" for i in range(num_mythemes)]
        embeddings = np.random.rand(num_mythemes, embedding_dim).tolist()
        
        # Insert in bulk
        insert_mythemes_bulk(sentences, embeddings)
        
        # Retrieve all
        ids, retrieved_sentences, retrieved_embeddings = get_mythemes_bulk()
        
        # Verify count
        assert len(ids) >= num_mythemes, f"Expected at least {num_mythemes} mythemes"
        
        # Find our test sentences
        found_count = 0
        for sentence in sentences:
            if sentence in retrieved_sentences:
                found_count += 1
        
        assert found_count == num_mythemes, f"Expected to find {num_mythemes} test sentences, found {found_count}"
    
    @pytest.mark.integration
    def test_embedding_dimension_consistency(self):
        """Test that embeddings maintain correct dimensions throughout operations."""
        embedding_dim = get_embedding_dim()
        
        # Create test data with specific dimensions
        sentences = ["Dimension test sentence"]
        embeddings = np.random.rand(1, embedding_dim).tolist()
        
        # Insert
        insert_mythemes_bulk(sentences, embeddings)
        
        # Retrieve
        ids, _, retrieved_embeddings = get_mythemes_bulk()
        test_id = ids[-1]
        
        # Test single retrieval
        _, _, single_embedding = get_mytheme(test_id)
        
        # Verify dimensions
        assert single_embedding.shape == (embedding_dim,)
        assert retrieved_embeddings.shape[1] == embedding_dim
        
        # Test bulk retrieval by IDs
        _, _, bulk_embeddings = get_mythemes_bulk(ids=[test_id])
        assert bulk_embeddings.shape == (1, embedding_dim)
    
    @pytest.mark.integration
    def test_critical_embedding_data_integrity(self):
        """CRITICAL: Test that embeddings are stored and retrieved with high precision.
        
        Note: We use decimal=7 precision which accounts for typical database floating-point
        precision differences. The differences we observed are on the order of 10^-8,
        which is normal for database storage and retrieval of floating-point values.
        """
        embedding_dim = get_embedding_dim()
        
        # Create test data with known values for precise comparison
        sentences = [
            "Critical integrity test sentence 1",
            "Critical integrity test sentence 2",
            "Critical integrity test sentence 3"
        ]
        
        # Create embeddings with specific patterns for easy verification
        embeddings = []
        for i in range(len(sentences)):
            # Create embedding with pattern: [i, i+0.1, i+0.2, ..., i+0.1*(dim-1)]
            embedding = [i + 0.1 * j for j in range(embedding_dim)]
            embeddings.append(embedding)
        
        # Insert mythemes
        insert_mythemes_bulk(sentences, embeddings)
        
        # Retrieve all mythemes
        ids, retrieved_sentences, retrieved_embeddings = get_mythemes_bulk()
        
        # Find our test sentences and verify exact matches
        for i, sentence in enumerate(sentences):
            try:
                idx = retrieved_sentences.index(sentence)
                retrieved_embedding = retrieved_embeddings[idx]
                original_embedding = embeddings[i]
                
                # CRITICAL: Verify embedding match with high precision
                np.testing.assert_array_almost_equal(
                    retrieved_embedding, 
                    original_embedding,
                    decimal=7,  # High precision - accounts for typical database floating-point precision
                    err_msg=f"CRITICAL: Embedding data integrity failed for sentence '{sentence}'. "
                           f"Original: {original_embedding}, Retrieved: {retrieved_embedding}"
                )
                
                # Also test with single retrieval
                theme_id = ids[idx]
                _, _, single_embedding = get_mytheme(theme_id)
                np.testing.assert_array_almost_equal(
                    single_embedding,
                    original_embedding,
                    decimal=7,  # High precision - accounts for typical database floating-point precision
                    err_msg=f"CRITICAL: Single retrieval embedding integrity failed for ID {theme_id}. "
                           f"Original: {original_embedding}, Retrieved: {single_embedding}"
                )
                
            except ValueError:
                pytest.fail(f"Test sentence '{sentence}' not found in retrieved data")
    
    @pytest.mark.integration
    def test_embedding_precision_preservation(self):
        """Test that floating point precision is preserved in embeddings."""
        embedding_dim = get_embedding_dim()
        
        # Create embeddings with high precision values
        sentences = ["Precision test sentence"]
        embeddings = [[0.123456789012345, 0.987654321098765, 0.555555555555555, 0.111111111111111, 0.222222222222222, 0.333333333333333][:embedding_dim]]
        
        # Insert
        insert_mythemes_bulk(sentences, embeddings)
        
        # Retrieve
        ids, _, retrieved_embeddings = get_mythemes_bulk()
        test_id = ids[-1]
        
        # Test both bulk and single retrieval
        retrieved_bulk = retrieved_embeddings[-1]
        _, _, retrieved_single = get_mytheme(test_id)
        
        # Verify precision is preserved (using almost_equal with realistic precision)
        np.testing.assert_array_almost_equal(
            retrieved_bulk, 
            embeddings[0], 
            decimal=7,  # High precision - accounts for typical database floating-point precision
            err_msg="Bulk retrieval precision mismatch"
        )
        
        np.testing.assert_array_almost_equal(
            retrieved_single, 
            embeddings[0], 
            decimal=7,  # High precision - accounts for typical database floating-point precision
            err_msg="Single retrieval precision mismatch"
        ) 